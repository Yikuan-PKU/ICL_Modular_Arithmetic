import argparse
import logging
import typing as t
from pathlib import Path

import yaml

from ICL import settings
from ICL.datasets.hf import RHMDataLoaderFactory
from ICL.train.model import RHMTrainer, RHMTrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RHM models with hardcoded defaults and optional YAML overrides")
    # Configuration file - only CLI argument needed
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
    """Create training configuration with two-tier override system."""
    # Tier 1: Start with hardcoded defaults from dataclass
    config = RHMTrainingConfig()  # Use all hardcoded defaults
    logger.info("Using hardcoded defaults from RHMTrainingConfig")

    # Tier 2: Apply YAML overrides
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

    # Always set run_name with timestamp
    config_dict = config.__dict__.copy()
    return RHMTrainingConfig(**config_dict)


def main() -> dict:
    """Train RHM model using two-tier configuration system."""
    # Parse arguments and create configuration
    args = parse_args()
    config = create_training_config(args)

    # Log final configuration
    logger.info("Final training configuration:")
    logger.info(f"  Task: {config.task_name}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Train batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Eval batch size: {config.per_device_eval_batch_size}")
    logger.info(f"  Vocab size: {config.vocab_size}")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Run name: {config.run_name}")

    # Initialize factory and create DataLoaders
    factory = RHMDataLoaderFactory(settings.PATH.train_dir / "raw" / config., vocab_size=config.vocab_size)

    # Create train DataLoader
    train_dataloader, train_metadata = factory.create_dataloader(
        task_name=config.task_name,
        batch_size=config.per_device_train_batch_size,
        max_length=config.max_position_embeddings,
        batching_strategy="config_then_length",
        seed=config.seed,
    )

    # Create eval DataLoader (smaller sequences for faster evaluation)
    eval_dataloader, eval_metadata = factory.create_dataloader(
        task_name=config.task_name,
        batch_size=config.per_device_eval_batch_size,
        max_length=config.max_position_embeddings,
        batching_strategy="config_then_length",
        filter_max_length=512,  # Smaller sequences for eval
        seed=config.seed,
    )

    # Create and run trainer
    trainer = RHMTrainer(
        training_config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        dataloader_metadata=train_metadata,
    )

    # Train model
    logger.info(f"Starting training for {config.task_name.upper()} task...")
    results = trainer.train()
    logger.info(f"Training completed! Results saved to {config.output_dir}")

    return results


if __name__ == "__main__":
    main()
