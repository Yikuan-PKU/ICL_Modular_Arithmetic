#!/usr/bin/env python3
"""Create train/validation splits from raw RHM datasets."""

import logging
import typing as t

from ICL.datasets.split import SplitDirectoryManager, check_split_exists, copy_raw_metadata_to_splits
from ICL.datasets.utils import (
    DatasetSplitter,
    load_seed_datasets,
    load_split_config_from_yaml,
    validate_split_consistency,
)
from ICL.settings import DatasetConfig, create_base_parser, parse_dataset_config, validate_args

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_split_parser():
    """Create argument parser for dataset splitting."""
    parser = create_base_parser(require_eval_flag=False)
    parser.description = "Create train/validation splits from raw RHM datasets"

    # Split-specific arguments
    split_group = parser.add_argument_group("Dataset Splitting")
    split_group.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration and check raw data"
    )
    split_group.add_argument("--config-path", type=str, help="Path to YAML config file (auto-detected if not provided)")

    return parser


def discover_raw_configurations(dataset_config: DatasetConfig) -> list[tuple[int, int]]:
    """Discover available (L,M) configurations with raw data."""
    import re

    from ICL.settings import PATH

    # Look for existing raw datasets
    pattern = rf"{dataset_config.dataset_type}_{dataset_config.num_seeds}_L(\d+)_M(\d+)"
    configurations = []

    datasets_dir = PATH.dataset_root
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            match = re.match(pattern, dataset_dir.name)
            if match:
                L, m = int(match.group(1)), int(match.group(2))

                # Check if raw data exists
                raw_dir = dataset_dir / "raw"
                if raw_dir.exists() and any(raw_dir.iterdir()):
                    configurations.append((L, m))
                    logger.info(f"Found raw data for L={L}, m={m}")

    if not configurations:
        raise FileNotFoundError(
            f"No raw datasets found matching pattern: {pattern}\n"
            f"Please run generate_raw.py first to create raw datasets."
        )

    return sorted(configurations)


def load_yaml_config_for_LM(dataset_config: DatasetConfig, L: int, m: int) -> dict[str, t.Any]:
    """Load YAML configuration for specific (L,M) configuration."""
    import yaml

    paths = dataset_config.get_config_paths(L, m)
    config_path = paths["config_dir"] / "create_split.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def split_single_configuration(L: int, m: int, dataset_config: DatasetConfig, overwrite: bool = False) -> bool:
    """Split datasets for a single (L,M) configuration.

    Returns:
        True if successful, False if skipped or failed

    """
    logger.info(f"Processing L={L}, m={m}")

    # Get paths for this configuration
    paths = dataset_config.get_config_paths(L, m)
    base_dir = paths["dataset_dir"].parent  # Remove /train to get base

    # Initialize directory manager
    dir_manager = SplitDirectoryManager(base_dir)

    # Validate raw directory exists
    dir_manager.validate_raw_directory()

    # Check if splits already exist
    if check_split_exists(base_dir) and not overwrite:
        logger.info(f"  Splits already exist for L={L}, m={m}, skipping")
        return True

    # Load YAML configuration
    yaml_config = load_yaml_config_for_LM(dataset_config, L, m)
    split_config = load_split_config_from_yaml(yaml_config)

    logger.info(f"  Split config: {split_config.validation_ratio:.1%} validation, {split_config.split_method} method")

    # Load raw seed datasets
    logger.info(f"  Loading raw datasets from {dir_manager.raw_dir}")
    seed_datasets = load_seed_datasets(dir_manager.raw_dir)
    logger.info(f"  Loaded {len(seed_datasets)} seed datasets")

    # Initialize splitter and perform splits
    splitter = DatasetSplitter(split_config)
    train_datasets, val_datasets, metadata = splitter.split_all_seed_datasets(seed_datasets)

    # Validate split integrity
    validate_split_consistency(seed_datasets, train_datasets, val_datasets)

    logger.info(f"  Split results: {len(train_datasets)} train seeds, {len(val_datasets)} val seeds")
    logger.info(
        f"  Total sequences: {metadata.total_stats['total_train_sequences']} train, "
        f"{metadata.total_stats['total_val_sequences']} val"
    )

    # Create split directories and save data
    dir_manager.create_split_directories(overwrite=overwrite)

    logger.info("  Saving train datasets...")
    dir_manager.save_seed_datasets(train_datasets, "train")

    logger.info("  Saving validation datasets...")
    dir_manager.save_seed_datasets(val_datasets, "validation")

    # Save metadata and summary files
    logger.info("  Saving metadata...")
    dir_manager.save_split_metadata_files(metadata, train_datasets, val_datasets)
    dir_manager.create_summary_files(metadata)

    # Copy relevant files from raw directory
    copy_raw_metadata_to_splits(dir_manager.raw_dir, dir_manager.train_dir, dir_manager.val_dir)

    logger.info(f"  ✓ Successfully created splits for L={L}, m={m}")
    logger.info(f"    Train: {dir_manager.train_dir}")
    logger.info(f"    Validation: {dir_manager.val_dir}")

    return True


def main():
    """Main splitting function."""
    parser = create_split_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to dataset config
    dataset_config = parse_dataset_config(args)

    # Discover available raw configurations
    try:
        configurations = discover_raw_configurations(dataset_config)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    if args.validate_only:
        logger.info(f"Validation successful for {dataset_config.to_name()}")
        logger.info(f"Found {len(configurations)} configurations with raw data:")
        for L, m in configurations:
            logger.info(f"  L={L}, m={m}")
        return 0

    if args.verbose:
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(f"Configurations to split: {configurations}")

    # Process each configuration
    total_configs = len(configurations)
    successful_configs = 0
    failed_configs = []

    logger.info(f"\n{'=' * 60}")
    logger.info("STARTING DATASET SPLITTING")
    logger.info(f"{'=' * 60}")

    for config_idx, (L, m) in enumerate(configurations, 1):
        logger.info(f"\nProcessing configuration {config_idx}/{total_configs}: L={L}, m={m}")
        logger.info("-" * 40)

        try:
            split_single_configuration(L, m, dataset_config, overwrite=args.overwrite)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("DATASET SPLITTING COMPLETE")

    return 0 if successful_configs > 0 else 1


if __name__ == "__main__":
    exit(main())
