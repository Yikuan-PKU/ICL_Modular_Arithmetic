#!/usr/bin/env python3
"""Create train/validation splits from raw RHM datasets."""

import logging
import typing as t

import yaml

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


def load_yaml_config(dataset_config: DatasetConfig) -> dict[str, t.Any]:
    """Load YAML configuration using L,M from DatasetConfig."""
    paths = dataset_config.get_config_paths(dataset_config.L, dataset_config.m)
    config_path = paths["config_dir"] / "create_split.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config_data = yaml.safe_load(f)

    # Validate that L,M from command line match any L,M in YAML (if present)
    yaml_L = config_data.get("L")
    yaml_m = config_data.get("m")

    if yaml_L is not None and yaml_L != dataset_config.L:
        logger.warning(f"YAML L={yaml_L} differs from command line L={dataset_config.L}. Using command line value.")

    if yaml_m is not None and yaml_m != dataset_config.m:
        logger.warning(f"YAML m={yaml_m} differs from command line m={dataset_config.m}. Using command line value.")

    return config_data


def split_single_configuration(dataset_config: DatasetConfig, overwrite: bool = False) -> bool:
    """Split datasets for the specified L,M configuration.

    Returns:
        True if successful, False if skipped or failed

    """
    L, m = dataset_config.L, dataset_config.m
    logger.info(f"Processing L={L}, m={m}")

    # Get paths for this configuration
    paths = dataset_config.get_config_paths(L, m)
    base_dir = paths["dataset_dir"].parent  # Remove /raw to get base

    # Initialize directory manager
    dir_manager = SplitDirectoryManager(base_dir)

    # Validate raw directory exists
    dir_manager.validate_raw_directory()

    # Check if splits already exist
    if check_split_exists(base_dir) and not overwrite:
        logger.info(f"  Splits already exist for L={L}, m={m}, skipping")
        return True

    # Load YAML configuration
    yaml_config = load_yaml_config(dataset_config)
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
    parser = create_split_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to dataset config (L,M now come from command line)
    dataset_config = parse_dataset_config(args)

    # Log the configuration
    logger.info(f"Dataset configuration: {dataset_config.to_name()}")
    logger.info(f"Using L={dataset_config.L}, m={dataset_config.m} from command line")

    if args.validate_only:
        logger.info(f"Validation successful for {dataset_config.to_name()}")

        # Check if config file exists
        try:
            yaml_config = load_yaml_config(dataset_config)
            logger.info("✓ Configuration file found and valid")

            # Check if raw data exists
            paths = dataset_config.get_config_paths(dataset_config.L, dataset_config.m)
            base_dir = paths["dataset_dir"].parent
            dir_manager = SplitDirectoryManager(base_dir)

            if dir_manager.raw_dir.exists():
                logger.info("✓ Raw data directory exists")
            else:
                logger.error(f"✗ Raw data directory not found: {dir_manager.raw_dir}")
                return 1

        except FileNotFoundError as e:
            logger.error(f"✗ Configuration validation failed: {e}")
            return 1

        return 0

    # Process the configuration
    success = split_single_configuration(dataset_config, overwrite=args.overwrite)

    if success:
        logger.info("DATASET SPLITTING COMPLETE")
        return 0
    logger.error("DATASET SPLITTING FAILED")
    return 1


if __name__ == "__main__":
    exit(main())
