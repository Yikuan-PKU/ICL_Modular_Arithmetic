import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset

from ICL.datasets.RHM import RandomHierarchyModel
from ICL.settings import DatasetConfig, create_base_parser, parse_dataset_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_dataset_parser():
    """Create argument parser for dataset generation."""
    parser = create_base_parser(require_model_type=False)
    parser.description = "Generate RHM dataset for hierarchical learning experiments"

    # Dataset-specific arguments
    dataset_group = parser.add_argument_group("Dataset Generation")
    dataset_group.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration without generating data"
    )

    return parser


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def generate_zipf_distribution(n: int, alpha: float = 1.0) -> list[float]:
    """Generate normalized Zipf distribution with n elements."""
    if n <= 0:
        raise ValueError("Number of elements must be positive")

    # Generate Zipf probabilities: 1/k^alpha for k=1,2,...,n
    ranks = np.arange(1, n + 1)
    raw_probs = 1.0 / (ranks**alpha)

    # Normalize to sum to 1
    normalized_probs = raw_probs / raw_probs.sum()

    return normalized_probs.tolist()


def create_rule_probabilities(
    rules: dict[int, torch.Tensor], distribution_type: str, **params
) -> dict[int, torch.Tensor] | None:
    """Create rule sampling probabilities for RHM based on distribution type."""
    if distribution_type == "uniform":
        return None  # RHM uses uniform probabilities by default

    if distribution_type == "zipf":
        alpha = params.get("zipf_alpha", 1.0)
        probability = {}

        for level, rule_tensor in rules.items():
            m = rule_tensor.shape[1]  # Number of synonymic rules at this level
            zipf_probs = generate_zipf_distribution(m, alpha)
            probability[level] = torch.tensor(zipf_probs, dtype=torch.float32)

        return probability

    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def generate_rhm_dataset_from_config(
    dataset_config: DatasetConfig, yaml_config: dict[str, Any]
) -> tuple[Dataset, dict[str, Any]]:
    """Generate RHM dataset using dataset config and YAML parameters."""
    # Extract configuration sections
    rhm_params = yaml_config["rhm_params"]
    configurations = yaml_config["configurations"]
    distribution_config = yaml_config["distribution"]

    logger.info(f"Generating dataset: {dataset_config.to_name()}")
    logger.info(f"Distribution type: {distribution_config['type']}")
    logger.info(f"Configurations: {len(configurations)} (L,m) pairs")

    # Initialize storage for all sequences and metadata
    all_sequences = []
    all_task_ids = []
    all_config_L = []
    all_config_m = []
    all_sequence_lengths = []
    all_rules = {}
    config_stats = {}

    total_sequences = 0

    # Generate data for each (L,m) configuration
    for task_id, config in enumerate(configurations):
        L, m = config["L"], config["m"]
        logger.info(f"Generating Task {task_id}: L={L} (depth), m={m} (multiplicity)")

        try:
            # Create initial RHM to get rules structure
            temp_rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                seed_rules=task_id + dataset_config.seed,  # Unique rules per task
                seed_sample=dataset_config.seed,
                train_size=1,  # Minimal size to get rules
                replacement=True,
                input_format="long",
            )

            # Create probability distribution for this configuration
            probability = create_rule_probabilities(
                temp_rhm.rules,
                distribution_config["type"],
                **{k: v for k, v in distribution_config.items() if k != "type"},
            )

            # Generate full dataset with proper probabilities
            rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                probability=probability,
                seed_rules=task_id + dataset_config.seed,
                seed_sample=dataset_config.seed,
                train_size=rhm_params["samples_per_config"],
                test_size=0,
                replacement=False,  # Required for custom probabilities
                input_format="long",
            )

            # Extract generated data
            sequences = rhm.features
            labels = rhm.labels
            rules = rhm.rules

            # Convert to lists for HuggingFace compatibility
            sequences_list = sequences.tolist() if hasattr(sequences, "tolist") else [list(seq) for seq in sequences]

            # Calculate statistics
            seq_lengths = [len(seq) for seq in sequences_list]
            config_stats[task_id] = {
                "L": L,
                "m": m,
                "num_sequences": len(sequences_list),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "avg_length": sum(seq_lengths) / len(seq_lengths),
                "total_tokens": sum(seq_lengths),
                "distribution_type": distribution_config["type"],
            }

            # Store rules and probability info
            all_rules[task_id] = {
                "L": L,
                "m": m,
                "rules_dict": rules,
                "probability_dict": probability,
                "distribution_type": distribution_config["type"],
                "vocab_range": f"1-{rhm_params['vocab_size']}",
                "num_sequences": len(sequences_list),
            }

            # Add to master dataset
            all_sequences.extend(sequences_list)
            all_task_ids.extend([task_id] * len(sequences_list))
            all_config_L.extend([L] * len(sequences_list))
            all_config_m.extend([m] * len(sequences_list))
            all_sequence_lengths.extend(seq_lengths)

            total_sequences += len(sequences_list)

            logger.info(f"  ✓ Generated {len(sequences_list)} sequences")
            logger.info(f"  ✓ Length range: {min(seq_lengths)}-{max(seq_lengths)} tokens")
            logger.info(f"  ✓ Total tokens: {sum(seq_lengths):,}")

        except Exception as e:
            logger.info(f"  ✗ Error generating task {task_id} (L={L}, m={m}): {e}")
            warnings.warn(f"Skipping configuration L={L}, m={m} due to error: {e}")
            continue

    # Create comprehensive metadata
    max_L = max([config["L"] for config in configurations]) if configurations else 0

    metadata = {
        "generation_params": rhm_params,
        "distribution_config": distribution_config,
        "configurations": [{"task_id": i, **config} for i, config in enumerate(configurations)],
        "config_stats": config_stats,
        "rules": all_rules,
        "derived_metadata": {
            "max_L": max_L,
            "total_rules_param": dataset_config.total_rules,  # From experiment name
            "actual_configs_generated": len(config_stats),
        },
        "dataset_stats": {
            "total_sequences": total_sequences,
            "total_configs": len(configurations),
            "successful_configs": len(config_stats),
            "min_seq_length": min(all_sequence_lengths) if all_sequence_lengths else 0,
            "max_seq_length": max(all_sequence_lengths) if all_sequence_lengths else 0,
            "avg_seq_length": sum(all_sequence_lengths) / len(all_sequence_lengths) if all_sequence_lengths else 0,
            "total_tokens": sum(all_sequence_lengths),
            "vocab_range": f"1-{rhm_params['vocab_size']} (0 reserved for special tokens)",
        },
    }

    # Create HuggingFace Dataset
    logger.info("Creating HuggingFace Dataset...")
    dataset_dict = {
        "input_ids": all_sequences,
        "task_id": all_task_ids,
        "config_L": all_config_L,
        "config_m": all_config_m,
        "length": all_sequence_lengths,
    }

    dataset = Dataset.from_dict(dataset_dict)

    return dataset, metadata


def save_dataset_with_metadata(dataset: Dataset, metadata: dict[str, Any], dataset_config: DatasetConfig, args) -> None:
    """Save dataset and metadata using dataset naming."""
    paths = dataset_config.get_dataset_paths()
    dataset_dir = paths["dataset_dir"]

    # Create output directory
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Check for existing data
    if (dataset_dir / "dataset").exists() and not args.overwrite:
        if args.resume:
            logger.info(f"Dataset already exists at {dataset_dir}, skipping generation")
            return
        raise FileExistsError(
            f"Dataset already exists at {dataset_dir}. Use --overwrite to replace or --resume to skip."
        )

    # Save HuggingFace dataset
    dataset.save_to_disk(str(dataset_dir / "dataset"))
    logger.info(f"✓ Saved HuggingFace dataset to {dataset_dir / 'dataset'}")

    # Save metadata
    with (dataset_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"✓ Saved metadata to {dataset_dir / 'metadata.pkl'}")

    # Save human-readable summary
    with (dataset_dir / "dataset_summary.txt").open("w") as f:
        f.write(f"RHM DATASET: {dataset_config.to_name()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Experiment Parameters:\n")
        f.write(f"  Dataset Type: {dataset_config.dataset_type}\n")
        f.write(f"  Mixture Type: {dataset_config.mixture_type}\n")
        f.write(f"  Total Rules: {dataset_config.total_rules}\n")
        f.write(f"  Seed: {dataset_config.seed}\n\n")

        f.write("Dataset Statistics:\n")
        f.write(f"  Total sequences: {metadata['dataset_stats']['total_sequences']:,}\n")
        f.write(f"  Total tokens: {metadata['dataset_stats']['total_tokens']:,}\n")
        f.write(f"  Vocabulary: {metadata['dataset_stats']['vocab_range']}\n")
        f.write(
            f"  Sequence length: {metadata['dataset_stats']['min_seq_length']}-{metadata['dataset_stats']['max_seq_length']}\n"
        )
        f.write(f"  Average length: {metadata['dataset_stats']['avg_seq_length']:.1f}\n")
        f.write(f"  Distribution: {metadata['distribution_config']['type']}\n\n")

        f.write("Configuration Details:\n")
        for task_id, stats in metadata["config_stats"].items():
            f.write(f"  Task {task_id}: L={stats['L']}, m={stats['m']}\n")
            f.write(f"    Sequences: {stats['num_sequences']:,}, Tokens: {stats['total_tokens']:,}\n")

    logger.info(f"✓ Saved summary to {dataset_dir / 'dataset_summary.txt'}")


def main():
    """Main dataset generation function."""
    parser = create_dataset_parser()
    args = parser.parse_args()

    # Convert to dataset config (no model type needed)
    dataset_config = parse_dataset_config(args)
    paths = dataset_config.get_dataset_paths()

    # Load dataset configuration from YAML
    config_path = paths["config_dir"] / "train_dataset.yaml"

    try:
        yaml_config = load_yaml_config(config_path)
    except FileNotFoundError:
        logger.info(f"Error: Configuration file not found at {config_path}")
        logger.info("Please ensure the YAML configuration exists before running dataset generation.")
        return 1

    if args.validate_only:
        logger.info(f"Configuration validation successful for {dataset_config.to_name()}")
        logger.info(f"Configurations to generate: {len(yaml_config['configurations'])}")
        return 0

    if args.verbose:
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(f"Config directory: {paths['config_dir']}")
        logger.info(f"Output directory: {paths['dataset_dir']}")

    # Generate dataset
    try:
        dataset, metadata = generate_rhm_dataset_from_config(dataset_config, yaml_config)
        save_dataset_with_metadata(dataset, metadata, dataset_config, args)

        logger.info(f"\n{'=' * 60}")
        logger.info("DATASET GENERATION COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(f"Total sequences: {len(dataset):,}")
        logger.info(f"Output directory: {paths['dataset_dir']}")
        logger.info(f"{'=' * 60}")

        return 0

    except Exception as e:
        logger.info(f"Error during dataset generation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
