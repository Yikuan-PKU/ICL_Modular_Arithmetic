#!/usr/bin/env python3
"""RHM Model Training Main Script
Simple training pipeline with hardcoded defaults and optional YAML overrides
"""

import argparse
import logging
import typing as t
from pathlib import Path

import yaml
from transformers import set_seed

from ICL import settings
from ICL.train.model import RHMTrainingConfig, create_rhm_training_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RHM models with hardcoded defaults and optional YAML overrides")
    parser.add_argument("--config", type=str, help="Path to YAML config file with parameter overrides (optional)")
    return parser.parse_args()


def load_yaml_overrides(config_path: str | Path | None) -> dict[str, t.Any]:
    """Load YAML configuration overrides if provided."""
    if config_path is None:
        logger.info("No YAML config provided, using hardcoded defaults")
        return {}

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using hardcoded defaults.")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            yaml_overrides = yaml.safe_load(file)

        if yaml_overrides:
            logger.info(f"Loaded YAML overrides from {config_path}")
            logger.info(f"YAML overrides: {yaml_overrides}")
            return yaml_overrides
        logger.warning(f"Config file {config_path} is empty. Using defaults.")
        return {}

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        logger.info("Falling back to hardcoded defaults")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        logger.info("Falling back to hardcoded defaults")
        return {}


def create_training_config(args: argparse.Namespace) -> RHMTrainingConfig:
    """Create training configuration with hardcoded defaults and YAML overrides."""
    # Start with hardcoded defaults from dataclass
    config = RHMTrainingConfig()
    logger.info("Using hardcoded defaults from RHMTrainingConfig")

    # Apply YAML overrides
    yaml_overrides = load_yaml_overrides(args.config)
    if yaml_overrides:
        # Convert dataclass to dict for merging
        config_dict = config.__dict__.copy()

        # Apply YAML overrides
        for key, value in yaml_overrides.items():
            if hasattr(config, key):
                config_dict[key] = value
                logger.info(f"YAML override: {key} = {value}")
            else:
                logger.warning(f"Unknown parameter in YAML: {key} (ignored)")

        # Create new config with overrides
        config = RHMTrainingConfig(**config_dict)

    return config


def main() -> dict:
    """Train RHM model using simplified configuration system."""
    # Parse arguments and create configuration
    args = parse_args()
    config = create_training_config(args)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Log final configuration
    logger.info("Final training configuration:")
    logger.info(f"  Task: {config.task_name}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Train batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Eval batch size: {config.per_device_eval_batch_size}")
    logger.info(f"  Vocab size: {config.vocab_size} (effective: {config.effective_vocab_size})")
    logger.info(f"  Max sequence length: {config.max_sequence_length}")
    logger.info(f"  Pack sequences: {config.pack_sequences}")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Run name: {config.run_name}")

    # Set dataset path (hardcoded default with YAML override option)
    dataset_path = getattr(config, "dataset_path", settings.PATH.train_dir / "raw")
    if hasattr(config, "dataset_path"):
        dataset_path = config.dataset_path
    else:
        dataset_path = settings.PATH.train_dir / "raw"
        logger.info(f"Using default dataset path: {dataset_path}")

    # Create trainer using factory function
    logger.info("Creating trainer and preparing datasets...")
    trainer, metadata = create_rhm_training_pipeline(
        dataset_path=str(dataset_path),
        training_config=config,
        train_split_ratio=getattr(config, "train_split_ratio", 0.8),
        filter_config_L=getattr(config, "filter_config_L", None),
        filter_config_m=getattr(config, "filter_config_m", None),
        max_samples=getattr(config, "max_samples", None),
    )

    # Log dataset info
    dataset_metadata = metadata["dataset_metadata"]
    logger.info("Dataset preparation completed:")
    logger.info(f"  Train size: {dataset_metadata.get('train_size', 'Unknown'):,}")
    logger.info(f"  Eval size: {dataset_metadata.get('eval_size', 'Unknown'):,}")
    logger.info(f"  Packing enabled: {dataset_metadata.get('packing_enabled', 'Unknown')}")

    # Train model
    logger.info(f"Starting training for {config.task_name.upper()} task...")
    results = trainer.train()
    logger.info(f"Training completed! Results saved to {config.output_dir}")

    return results


if __name__ == "__main__":
    main()
