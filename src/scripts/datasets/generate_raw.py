import json
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
from ICL.settings import PATH, DatasetConfig, create_base_parser, parse_dataset_config, validate_args

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_dataset_parser():
    """Create argument parser for dataset generation."""
    parser = create_base_parser(require_eval_flag=True)  # Allow eval flag for future use
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


def generate_rhm_dataset_for_config(
    L: int, m: int, random_seeds: list[int], yaml_config: dict[str, Any]
) -> tuple[dict[int, Dataset], dict[str, Any]]:
    """Generate RHM dataset for a specific (L,M) configuration using multiple random seeds.

    Returns:
        tuple: (dict of datasets per seed, combined metadata)

    """
    # Extract configuration sections
    rhm_params = yaml_config["rhm_params"]
    distribution_config = yaml_config["distribution"]

    logger.info(f"Generating dataset for L={L}, m={m} with {len(random_seeds)} seeds")
    logger.info(f"Seeds: {random_seeds}")

    # Initialize storage for datasets per seed and metadata
    seed_datasets = {}
    config_stats = {}
    rules_by_seed = {}

    total_sequences = 0
    all_sequence_lengths = []

    # Generate data for each random seed separately
    for seed_idx, random_seed in enumerate(random_seeds):
        logger.info(f"  Generating data with seed {random_seed} ({seed_idx + 1}/{len(random_seeds)})")

        try:
            # Create initial RHM to get rules structure
            temp_rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                seed_rules=random_seed,  # Use random seed for rules
                seed_sample=random_seed + 1,  # Slightly different seed for sampling
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
                seed_rules=random_seed,
                seed_sample=random_seed + 1,
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

            # Create individual dataset for this seed
            seed_dataset_dict = {
                "input_ids": sequences_list,
                "length": [len(seq) for seq in sequences_list],
            }

            seed_datasets[random_seed] = Dataset.from_dict(seed_dataset_dict)

            # Calculate statistics for this seed
            seq_lengths = [len(seq) for seq in sequences_list]
            config_stats[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "num_sequences": len(sequences_list),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "avg_length": sum(seq_lengths) / len(seq_lengths),
                "total_tokens": sum(seq_lengths),
                "distribution_type": distribution_config["type"],
            }

            # Store rules for this seed
            rules_by_seed[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "rules_dict": rules,
                "probability_dict": probability,
                "distribution_type": distribution_config["type"],
            }

            # Update totals
            total_sequences += len(sequences_list)
            all_sequence_lengths.extend(seq_lengths)

            logger.info(f"    ✓ Generated {len(sequences_list)} sequences")
            logger.info(f"    ✓ Length range: {min(seq_lengths)}-{max(seq_lengths)} tokens")

        except Exception as e:
            logger.error(f"    ✗ Error generating data with seed {random_seed}: {e}")
            warnings.warn(f"Skipping seed {random_seed} for L={L}, m={m} due to error: {e}")
            continue

    if total_sequences == 0:
        raise RuntimeError(f"Failed to generate any data for L={L}, m={m}")

    # Create comprehensive metadata for this configuration
    metadata = {
        "config_params": {
            "L": L,
            "m": m,
            "generation_params": rhm_params,
            "distribution_config": distribution_config,
        },
        "seed_info": {
            "random_seeds": random_seeds,
            "successful_seeds": [seed for seed in config_stats],
            "num_seeds": len(random_seeds),
            "successful_count": len(config_stats),
        },
        "config_stats": config_stats,
        "rules_by_seed": rules_by_seed,
        "dataset_stats": {
            "total_sequences": total_sequences,
            "min_seq_length": min(all_sequence_lengths),
            "max_seq_length": max(all_sequence_lengths),
            "avg_seq_length": sum(all_sequence_lengths) / len(all_sequence_lengths),
            "total_tokens": sum(all_sequence_lengths),
            "vocab_range": f"1-{rhm_params['vocab_size']} (0 reserved for special tokens)",
        },
    }

    logger.info(f"  ✓ Created {len(seed_datasets)} separate seed datasets with {total_sequences} total sequences")

    return seed_datasets, metadata


def save_dataset_with_metadata(
    seed_datasets: dict[int, Dataset], metadata: dict[str, Any], dataset_config: DatasetConfig, L: int, m: int, args
) -> None:
    """Save separate seed datasets and metadata for a specific (L,M) configuration."""
    paths = dataset_config.get_config_paths(L, m)
    dataset_dir = paths["dataset_dir"]
    config_base_dir = paths["config_base_dir"]

    # Create output directory
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config_base_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    if any((dataset_dir / f"seed_{seed}").exists() for seed in seed_datasets) and not args.overwrite:
        if args.resume:
            logger.info(f"Seed datasets already exist for L={L}, m={m} at {dataset_dir}, skipping")
            return
        raise FileExistsError(
            f"Seed datasets already exist for L={L}, m={m} at {dataset_dir}. Use --overwrite to replace or --resume to skip."
        )

    # Save each seed dataset separately
    for seed, dataset in seed_datasets.items():
        seed_dir = dataset_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(seed_dir / "dataset"))
        logger.info(f"  ✓ Saved seed {seed} dataset to {seed_dir / 'dataset'}")

    # Save combined metadata
    with (dataset_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"✓ Saved metadata to {config_base_dir / 'metadata.pkl'}")

    # Save seed index for easy discovery
    seed_index = {
        "available_seeds": list(seed_datasets.keys()),
        "num_seeds": len(seed_datasets),
        "dataset_paths": {seed: f"seed_{seed}/dataset" for seed in seed_datasets},
    }

    with (dataset_dir / "seed_index.json").open("w") as f:
        json.dump(seed_index, f, indent=2)
    logger.info(f"✓ Saved seed index to {config_base_dir / 'seed_index.json'}")

    # Save human-readable summary
    with (dataset_dir / "dataset_summary.txt").open("w") as f:
        f.write(f"RHM DATASET: L={L}, m={m}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration Parameters:\n")
        f.write(f"  Hierarchy Depth (L): {L}\n")
        f.write(f"  Multiplicity (m): {m}\n")
        f.write(f"  Dataset Type: {dataset_config.dataset_type}\n")
        f.write(f"  RNG Seed: {dataset_config.seed}\n")
        f.write(f"  Num Seeds: {dataset_config.num_seeds}\n\n")

        f.write("Seed Information:\n")
        f.write(f"  Random seeds used: {metadata['seed_info']['random_seeds']}\n")
        f.write(f"  Successful seeds: {metadata['seed_info']['successful_seeds']}\n")
        f.write(f"  Success rate: {metadata['seed_info']['successful_count']}/{metadata['seed_info']['num_seeds']}\n\n")

        f.write("Dataset Structure:\n")
        f.write("  datasets/\n")
        for seed in seed_datasets:
            f.write(f"    └── seed_{seed}/dataset/  # {metadata['config_stats'][seed]['num_sequences']} sequences\n")
        f.write("\n")

        f.write("Dataset Statistics (Combined):\n")
        stats = metadata["dataset_stats"]
        f.write(f"  Total sequences: {stats['total_sequences']:,}\n")
        f.write(f"  Total tokens: {stats['total_tokens']:,}\n")
        f.write(f"  Vocabulary: {stats['vocab_range']}\n")
        f.write(f"  Sequence length: {stats['min_seq_length']}-{stats['max_seq_length']}\n")
        f.write(f"  Average length: {stats['avg_seq_length']:.1f}\n")
        f.write(f"  Distribution: {metadata['config_params']['distribution_config']['type']}\n\n")

        f.write("Per-Seed Statistics:\n")
        for seed, seed_stats in metadata["config_stats"].items():
            f.write(f"  Seed {seed}:\n")
            f.write(f"    Path: seed_{seed}/dataset/\n")
            f.write(f"    Sequences: {seed_stats['num_sequences']:,}, Tokens: {seed_stats['total_tokens']:,}\n")
            f.write(
                f"    Length: {seed_stats['min_length']}-{seed_stats['max_length']} (avg: {seed_stats['avg_length']:.1f})\n"
            )

    logger.info(f"✓ Saved summary to {dataset_dir / 'dataset_summary.txt'}")

    # Log final structure
    logger.info(f"✓ Dataset structure for L={L}, m={m}:")
    for seed in seed_datasets:
        logger.info(f"    {dataset_dir / f'seed_{seed}' / 'dataset'}")


def extract_unique_configurations(yaml_config: dict[str, Any]) -> list[tuple[int, int]]:
    """Extract unique (L,M) configurations from YAML config."""
    configurations = yaml_config["configurations"]
    unique_configs = []
    seen = set()

    for config in configurations:
        L, m = config["L"], config["m"]
        if (L, m) not in seen:
            unique_configs.append((L, m))
            seen.add((L, m))
        else:
            logger.warning(f"Duplicate configuration L={L}, m={m} found, skipping")

    logger.info(f"Found {len(unique_configs)} unique (L,M) configurations: {unique_configs}")
    return unique_configs


def discover_configurations(dataset_config: DatasetConfig) -> list[tuple[int, int]]:
    """Discover available (L,M) configurations by scanning config directories."""
    import re

    # Try multiple patterns for flexibility
    patterns = [
        rf"{dataset_config.dataset_type}_{dataset_config.num_seeds}_L(\d+)_M(\d+)",  # Exact match
        rf"{dataset_config.dataset_type}_\d+_L(\d+)_M(\d+)",  # Any num_seeds
        r"L(\d+)_M(\d+)",  # Just L*_M*
    ]

    configurations = []

    if not PATH.conf_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found: {PATH.conf_dir}")

    for config_dir in PATH.conf_dir.iterdir():
        if config_dir.is_dir():
            for pattern in patterns:
                match = re.match(pattern, config_dir.name)
                if match:
                    L, m = int(match.group(1)), int(match.group(2))

                    # Check if train_dataset.yaml exists in this directory
                    yaml_file = config_dir / "generate_raw.yaml"
                    if yaml_file.exists():
                        configurations.append((L, m))
                        logger.info(f"Found configuration: L={L}, m={m} at {config_dir}")
                        break  # Stop checking other patterns for this directory
                    logger.warning(f"Config directory {config_dir} missing generate_raw.yaml")

    if not configurations:
        logger.error("No valid configurations found. Searched patterns:")
        for pattern in patterns:
            logger.error(f"  - {pattern}")
        logger.error(f"Available directories in {PATH.conf_dir}:")
        if PATH.conf_dir.exists():
            for item in PATH.conf_dir.iterdir():
                logger.error(f"  - {item.name}")
        raise FileNotFoundError(
            "No valid configurations found. Please create config directories matching one of the patterns above."
        )

    # Remove duplicates and sort
    configurations = sorted(list(set(configurations)))
    logger.info(f"Discovered {len(configurations)} configurations: {configurations}")
    return configurations


def load_config_for_LM(dataset_config: DatasetConfig, L: int, m: int) -> dict[str, Any]:
    """Load YAML configuration for a specific (L,M) pair."""
    paths = dataset_config.get_config_paths(L, m)
    config_path = paths["config_dir"] / "generate_raw.yaml"

    try:
        return load_yaml_config(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def main():
    """Main dataset generation function."""
    parser = create_dataset_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to dataset config
    dataset_config = parse_dataset_config(args)

    # Discover available (L,M) configurations from config directories
    try:
        unique_configs = discover_configurations(dataset_config)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please ensure the YAML configurations exist before running dataset generation.")
        logger.error(
            f"Expected pattern: conf/{dataset_config.dataset_type}_{dataset_config.num_seeds}_L{{L}}_M{{m}}/generate_raw.yaml"
        )
        return 1

    if args.validate_only:
        logger.info(f"Configuration validation successful for {dataset_config.to_name()}")
        logger.info(f"Found configurations: {len(unique_configs)}")
        logger.info(f"Configurations: {unique_configs}")
        logger.info(f"Random seeds to generate: {dataset_config.num_seeds}")

        # Validate each configuration file
        for L, m in unique_configs:
            try:
                yaml_config = load_config_for_LM(dataset_config, L, m)
                logger.info(f"  ✓ L={L}, m={m}: Valid YAML configuration")
            except Exception as e:
                logger.error(f"  ✗ L={L}, m={m}: Invalid configuration - {e}")
        return 0

    # Generate random seeds using the provided RNG seed
    logger.info(f"Generating {dataset_config.num_seeds} random seeds using RNG seed {dataset_config.seed}")
    np.random.seed(dataset_config.seed)
    random_seeds = np.random.randint(0, 2**31, size=dataset_config.num_seeds).tolist()
    logger.info(f"Generated random seeds: {random_seeds}")

    if args.verbose:
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(
            f"Output pattern: datasets/{dataset_config.dataset_type}_{dataset_config.num_seeds}_L{{L}}_M{{m}}/raw/seed_{{seed}}/dataset/"
        )

    # Generate datasets for each discovered (L,M) configuration
    total_configs = len(unique_configs)
    successful_configs = 0
    failed_configs = []

    logger.info(f"\n{'=' * 60}")
    logger.info("STARTING DATASET GENERATION")
    logger.info(f"{'=' * 60}")

    for config_idx, (L, m) in enumerate(unique_configs, 1):
        logger.info(f"\nProcessing configuration {config_idx}/{total_configs}: L={L}, m={m}")
        logger.info("-" * 40)

        try:
            # Load configuration for this specific (L,M) pair
            yaml_config = load_config_for_LM(dataset_config, L, m)

            # Generate separate datasets for each seed
            seed_datasets, metadata = generate_rhm_dataset_for_config(L, m, random_seeds, yaml_config)

            # Save separate seed datasets and metadata
            save_dataset_with_metadata(seed_datasets, metadata, dataset_config, L, m, args)

            successful_configs += 1
            logger.info(f"✓ Successfully completed L={L}, m={m}")

        except Exception as e:
            logger.error(f"✗ Failed to generate dataset for L={L}, m={m}: {e}")
            failed_configs.append((L, m, str(e)))
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    return 0 if successful_configs > 0 else 1


if __name__ == "__main__":
    exit(main())
