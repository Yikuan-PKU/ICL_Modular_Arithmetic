import json
import logging
import pickle
from typing import Any

from datasets import Dataset

from ICL.datasets.overlap_checker import generate_rhm_dataset_original, generate_rhm_dataset_with_deduplication
from ICL.datasets.utils import load_yaml_config
from ICL.settings import DatasetConfig, create_base_parser, parse_dataset_config, validate_args

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


def save_dataset_with_metadata(
    seed_datasets: dict[int, Dataset], metadata: dict[str, Any], dataset_config: DatasetConfig, args
) -> None:
    """Save separate seed datasets and metadata for the specified L,M configuration."""
    L, m = dataset_config.L, dataset_config.m
    paths = dataset_config.get_config_paths(L, m)
    dataset_dir = paths["dataset_dir"]
    config_base_dir = paths["config_base_dir"]

    # Determine if deduplication was enabled from metadata
    dedup_enabled = "deduplication_stats" in metadata and metadata["deduplication_stats"].get("enabled", False)

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
    logger.info(f"✓ Saved metadata to {dataset_dir / 'metadata.pkl'}")

    # Save seed index for easy discovery
    seed_index = {
        "available_seeds": list(seed_datasets.keys()),
        "num_seeds": len(seed_datasets),
        "dataset_paths": {seed: f"seed_{seed}/dataset" for seed in seed_datasets},
        "deduplication_enabled": dedup_enabled,
    }

    with (dataset_dir / "seed_index.json").open("w") as f:
        json.dump(seed_index, f, indent=2)
    logger.info(f"✓ Saved seed index to {dataset_dir / 'seed_index.json'}")

    # Save human-readable summary
    dedup_status = "WITH GLOBAL DEDUPLICATION" if dedup_enabled else "WITHOUT DEDUPLICATION"

    with (dataset_dir / "dataset_summary.txt").open("w") as f:
        f.write(f"RHM DATASET {dedup_status}: L={L}, m={m}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration Parameters:\n")
        f.write(f"  Hierarchy Depth (L): {L}\n")
        f.write(f"  Multiplicity (m): {m}\n")
        f.write(f"  Dataset Type: {dataset_config.dataset_type}\n")
        f.write(f"  RNG Seed: {dataset_config.seed}\n")
        f.write(f"  Num Seeds: {dataset_config.num_seeds}\n")
        f.write(f"  Deduplication: {'ENABLED' if dedup_enabled else 'DISABLED'}\n\n")

        if dedup_enabled and "deduplication_stats" in metadata:
            f.write("Deduplication Results:\n")
            dedup_stats = metadata["deduplication_stats"]
            f.write(f"  Global unique sequences: {dedup_stats['global_unique_sequences']:,}\n")
            # f.write(f"  Total collisions prevented: {dedup_stats['total_collisions']:,}\n")
            f.write(f"  Average generation efficiency: {dedup_stats['average_efficiency']:.2%}\n\n")

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
            if dedup_enabled and "efficiency" in seed_stats:
                f.write(f"    Generation efficiency: {seed_stats['efficiency']:.2%}\n")

    logger.info(f"✓ Saved summary to {dataset_dir / 'dataset_summary.txt'}")

    # Log final structure
    logger.info(f"✓ Dataset structure for L={L}, m={m}:")
    for seed in seed_datasets:
        logger.info(f"    {dataset_dir / f'seed_{seed}' / 'dataset'}")


def load_config_for_dataset(dataset_config: DatasetConfig) -> dict[str, Any]:
    """Load YAML configuration for the specified L,M configuration."""
    paths = dataset_config.get_config_paths(dataset_config.L, dataset_config.m)
    config_path = paths["config_dir"] / "generate_raw.yaml"

    try:
        config_data = load_yaml_config(config_path)

        # Set default deduplication configuration if not present
        if "deduplication" not in config_data:
            logger.info("No deduplication config found in YAML, using defaults")
            config_data["deduplication"] = {"enabled": True, "max_attempts": 10, "batch_multiplier": 2.0}

        # Ensure deduplication section has required fields with defaults
        dedup_defaults = {"enabled": True, "max_attempts": 10, "batch_multiplier": 2.0}

        for key, default_value in dedup_defaults.items():
            if key not in config_data["deduplication"]:
                config_data["deduplication"][key] = default_value

        # Validate that L,M from command line match any L,M in YAML (if present)
        yaml_L = config_data.get("L")
        yaml_m = config_data.get("m")

        if yaml_L is not None and yaml_L != dataset_config.L:
            logger.warning(f"YAML L={yaml_L} differs from command line L={dataset_config.L}. Using command line value.")

        if yaml_m is not None and yaml_m != dataset_config.m:
            logger.warning(f"YAML m={yaml_m} differs from command line m={dataset_config.m}. Using command line value.")

        return config_data

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def main():
    """Main dataset generation function."""
    parser = create_dataset_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to dataset config (L,M now come from command line)
    dataset_config = parse_dataset_config(args)

    # Log the configuration
    logger.info(f"Dataset configuration: {dataset_config.to_name()}")
    logger.info(f"Using L={dataset_config.L}, m={dataset_config.m} from command line")

    if args.validate_only:
        logger.info(f"Configuration validation successful for {dataset_config.to_name()}")
        return

    # Load YAML configuration for this dataset setup
    yaml_config = load_config_for_dataset(dataset_config)

    # Generate the actual list of random seeds
    random_seeds = generate_random_seeds(dataset_config.seed, dataset_config.num_seeds)

    # Check whether deduplication is enabled from YAML
    dedup_enabled = yaml_config.get("deduplication", {}).get("enabled", True)

    # Generate dataset using dedup or original method
    if dedup_enabled:
        logger.info("Deduplication ENABLED — using generate_rhm_dataset_with_deduplication()")
        seed_datasets, metadata = generate_rhm_dataset_with_deduplication(
            dataset_config.L,
            dataset_config.m,
            random_seeds,  # Now passing actual list of seeds
            yaml_config,
        )
    else:
        logger.info("Deduplication DISABLED — using generate_rhm_dataset_original()")
        seed_datasets, metadata = generate_rhm_dataset_original(
            dataset_config.L,
            dataset_config.m,
            random_seeds,  # Now passing actual list of seeds
            yaml_config,
        )

    # Save datasets and metadata to disk
    save_dataset_with_metadata(seed_datasets, metadata, dataset_config, args)

    logger.info(f"✅ Completed dataset generation for {dataset_config.to_name()}")
    logger.info("=" * 80)
    logger.info("Dataset successfully generated and saved.")


def generate_random_seeds(base_seed: int, num_seeds: int) -> list[int]:
    """Generate a list of random seeds from a base seed."""
    import random

    random.seed(base_seed)
    return [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]


if __name__ == "__main__":
    main()
