#!/usr/bin/env python3
"""RHM Model Training Main Script
Integrated training pipeline for Random Hierarchy Model with HuggingFace components
"""

import argparse
import json
import logging
import sys
import typing as t
from pathlib import Path

import torch
import yaml
from transformers import set_seed

from ICL.train.model import (
    RHMTrainingConfig,
    create_rhm_training_pipeline,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("rhm_training.log")],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for RHM training."""
    parser = argparse.ArgumentParser(
        description="Train RHM models with hierarchical configurations using YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values using key=value format (e.g., learning_rate=1e-3 num_train_epochs=5)",
    )

    return parser.parse_args()


def load_config(config_path: str, overrides: list[str] = None) -> dict[str, t.Any]:
    """Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: List of key=value override strings

    Returns:
        Configuration dictionary

    """
    # Load base configuration
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    # Apply overrides
    if overrides:
        logger.info("Applying configuration overrides:")
        for override in overrides:
            if "=" not in override:
                logger.warning(f"Invalid override format: {override} (expected key=value)")
                continue

            key, value = override.split("=", 1)

            # Try to convert value to appropriate type
            original_value = config.get(key)
            try:
                if original_value is None:
                    # No existing value, try to infer type
                    if value.lower() in ["true", "false"]:
                        parsed_value = value.lower() == "true"
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                elif isinstance(original_value, bool):
                    parsed_value = value.lower() == "true"
                elif isinstance(original_value, int):
                    parsed_value = int(value)
                elif isinstance(original_value, float):
                    parsed_value = float(value)
                elif isinstance(original_value, list):
                    # Handle list values (comma-separated)
                    parsed_value = [item.strip() for item in value.split(",")]
                else:
                    parsed_value = value

                config[key] = parsed_value
                logger.info(f"  {key}: {original_value} -> {parsed_value}")

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse override {key}={value}: {e}")
                config[key] = value  # Use as string if parsing fails

    return config


def create_training_config(config_dict: dict[str, t.Any]) -> RHMTrainingConfig:
    """Create RHMTrainingConfig from configuration dictionary.

    Args:
        config_dict: Configuration dictionary from YAML

    Returns:
        RHMTrainingConfig instance

    """
    # Extract training-specific configuration
    training_config_dict = {}

    # Map config keys to RHMTrainingConfig fields
    config_mapping = {
        # Model configuration
        "model_name_or_path": "model_name_or_path",
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        # Training configuration
        "task_name": "task_name",
        "output_dir": "output_dir",
        "num_train_epochs": "num_train_epochs",
        "per_device_train_batch_size": "per_device_train_batch_size",
        "per_device_eval_batch_size": "per_device_eval_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio",
        "lr_scheduler_type": "lr_scheduler_type",
        # Sequence packing
        "max_sequence_length": "max_sequence_length",
        "pack_sequences": "pack_sequences",
        "separator_token_id": "separator_token_id",
        # MLM configuration
        "mlm_probability": "mlm_probability",
        "mask_strategy": "mask_strategy",
        # Special tokens
        "pad_token_id": "pad_token_id",
        "mask_token_id": "mask_token_id",
        "cls_token_id": "cls_token_id",
        "sep_token_id": "sep_token_id",
        # Checkpointing
        "save_strategy": "save_strategy",
        "save_steps": "save_steps",
        "save_total_limit": "save_total_limit",
        "load_best_model_at_end": "load_best_model_at_end",
        "metric_for_best_model": "metric_for_best_model",
        "greater_is_better": "greater_is_better",
        # Evaluation
        "evaluation_strategy": "evaluation_strategy",
        "eval_steps": "eval_steps",
        "eval_accumulation_steps": "eval_accumulation_steps",
        # Logging
        "logging_strategy": "logging_strategy",
        "logging_steps": "logging_steps",
        "report_to": "report_to",
        "run_name": "run_name",
        # Optimization
        "adam_beta1": "adam_beta1",
        "adam_beta2": "adam_beta2",
        "adam_epsilon": "adam_epsilon",
        "max_grad_norm": "max_grad_norm",
        # Early stopping
        "early_stopping": "early_stopping",
        "early_stopping_patience": "early_stopping_patience",
        "early_stopping_threshold": "early_stopping_threshold",
        # Mixed precision
        "fp16": "fp16",
        "bf16": "bf16",
        # Data loading
        "dataloader_num_workers": "dataloader_num_workers",
        "dataloader_pin_memory": "dataloader_pin_memory",
        "remove_unused_columns": "remove_unused_columns",
        # Hierarchical analysis
        "track_hierarchical_metrics": "track_hierarchical_metrics",
        "hierarchical_eval_frequency": "hierarchical_eval_frequency",
        # Reproducibility
        "seed": "seed",
    }

    # Map values from config dict
    for yaml_key, config_key in config_mapping.items():
        if yaml_key in config_dict:
            training_config_dict[config_key] = config_dict[yaml_key]

    # Create training configuration
    try:
        training_config = RHMTrainingConfig(**training_config_dict)
        logger.info("Successfully created RHMTrainingConfig")
        return training_config
    except TypeError as e:
        logger.error(f"Failed to create RHMTrainingConfig: {e}")
        logger.error("Available config keys:")
        for key in training_config_dict:
            logger.error(f"  {key}: {training_config_dict[key]}")
        raise


def extract_dataset_config(config_dict: dict[str, t.Any]) -> dict[str, t.Any]:
    """Extract dataset-specific configuration from config dictionary.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        Dataset configuration dictionary

    """
    dataset_config = {}

    # Dataset-specific keys
    dataset_keys = [
        "dataset_path",
        "filter_config_L",
        "filter_config_m",
        "max_samples",
        "train_split_ratio",
        "filter_min_length",
        "filter_max_length",
    ]

    for key in dataset_keys:
        if key in config_dict:
            dataset_config[key] = config_dict[key]

    return dataset_config


def validate_config(config_dict: dict[str, t.Any]) -> None:
    """Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    """
    # Required fields
    required_fields = ["dataset_path", "output_dir"]

    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required configuration field missing: {field}")

    # Validate dataset path exists
    dataset_path = Path(config_dict["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Validate task name
    if "task_name" in config_dict:
        valid_tasks = ["clm", "mlm"]
        if config_dict["task_name"] not in valid_tasks:
            raise ValueError(f"Invalid task_name: {config_dict['task_name']}. Must be one of {valid_tasks}")

    # Validate mask strategy for MLM
    if config_dict.get("task_name") == "mlm" and "mask_strategy" in config_dict:
        valid_strategies = ["random", "hierarchical", "level_specific"]
        if config_dict["mask_strategy"] not in valid_strategies:
            raise ValueError(
                f"Invalid mask_strategy: {config_dict['mask_strategy']}. Must be one of {valid_strategies}"
            )

    logger.info("Configuration validation passed")


def save_experiment_config(config_dict: dict[str, t.Any], output_dir: Path) -> None:
    """Save experiment configuration for reproducibility.

    Args:
        config_dict: Full configuration dictionary
        output_dir: Output directory for saving config

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original config
    config_path = output_dir / "experiment_config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    # Save JSON version for easier programmatic access
    json_path = output_dir / "experiment_config.json"
    with json_path.open("w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Experiment configuration saved to {config_path} and {json_path}")


def setup_logging(output_dir: Path, run_name: str | None = None) -> None:
    """Set up logging with file output.

    Args:
        output_dir: Directory for log files
        run_name: Optional run name for log file

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename
    log_filename = f"rhm_training_{run_name}.log" if run_name else "rhm_training.log"
    log_path = output_dir / log_filename

    # Configure logging
    logger = logging.getLogger()
    logger.handlers.clear()  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging configured. Log file: {log_path}")


def main() -> None:
    """Main training function."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Load configuration
        config_dict = load_config(args.config, args.override)

        # Validate configuration
        validate_config(config_dict)

        # Create output directory and setup logging
        output_dir = Path(config_dict["output_dir"])
        setup_logging(output_dir, config_dict.get("run_name"))

        # Save experiment configuration
        save_experiment_config(config_dict, output_dir)

        # Set seed for reproducibility
        seed = config_dict.get("seed", 42)
        set_seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")

        # Create training configuration
        training_config = create_training_config(config_dict)

        # Extract dataset configuration
        dataset_config = extract_dataset_config(config_dict)

        # Log experiment info
        logger.info("=" * 80)
        logger.info("RHM MODEL TRAINING EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Task: {training_config.task_name}")
        logger.info(f"Dataset: {dataset_config.get('dataset_path', 'Not specified')}")
        logger.info(f"Vocab size: {training_config.vocab_size} (effective: {training_config.effective_vocab_size})")
        logger.info(f"Model layers: {training_config.num_hidden_layers}")
        logger.info(f"Hidden size: {training_config.hidden_size}")
        logger.info(f"Sequence packing: {training_config.pack_sequences}")
        logger.info(f"Max sequence length: {training_config.max_sequence_length}")
        logger.info(f"Training epochs: {training_config.num_train_epochs}")
        logger.info(f"Batch size: {training_config.per_device_train_batch_size}")
        logger.info(f"Learning rate: {training_config.learning_rate}")

        if training_config.task_name == "mlm":
            logger.info(f"MLM probability: {training_config.mlm_probability}")
            logger.info(f"Mask strategy: {training_config.mask_strategy}")

        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"CUDA available: {gpu_count} GPU(s) - {gpu_name}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("CUDA not available - training on CPU")

        logger.info("=" * 80)

        # Create trainer using factory function
        logger.info("Creating trainer and preparing datasets...")
        trainer, metadata = create_rhm_training_pipeline(
            dataset_path=dataset_config["dataset_path"],
            training_config=training_config,
            **{k: v for k, v in dataset_config.items() if k != "dataset_path"},
        )

        # Log dataset info
        dataset_metadata = metadata["dataset_metadata"]
        logger.info("Dataset preparation completed:")
        logger.info(f"  Original size: {dataset_metadata.get('original_size', 'Unknown'):,}")
        logger.info(f"  Filtered size: {dataset_metadata.get('filtered_size', 'Unknown'):,}")
        logger.info(f"  Train size: {dataset_metadata.get('train_size', 'Unknown'):,}")
        logger.info(f"  Eval size: {dataset_metadata.get('eval_size', 'Unknown'):,}")
        logger.info(f"  Packing enabled: {dataset_metadata.get('packing_enabled', 'Unknown')}")

        # Start training
        logger.info("Starting training...")
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_time:
            start_time.record()

        # Train model
        training_results = trainer.train()

        if end_time and start_time:
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            training_time = training_results.get("training_time", 0)

        # Log final results
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total training time: {training_time:.2f} seconds ({training_time / 60:.1f} minutes)")
        logger.info(f"Final train loss: {training_results['train_result']['train_loss']:.4f}")
        logger.info(f"Final eval loss: {training_results['final_eval_metrics']['eval_loss']:.4f}")
        logger.info(f"Best model saved at: {training_config.output_dir}")
        logger.info(f"Model parameters: {training_results['model_size']:,}")

        # Save final experiment summary
        experiment_summary = {
            "experiment_config": config_dict,
            "training_results": training_results,
            "dataset_metadata": dataset_metadata,
            "training_time_seconds": training_time,
            "final_metrics": {
                "train_loss": training_results["train_result"]["train_loss"],
                "eval_loss": training_results["final_eval_metrics"]["eval_loss"],
                "model_parameters": training_results["model_size"],
            },
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
        }

        summary_path = output_dir / "experiment_summary.json"
        with summary_path.open("w") as f:
            json.dump(experiment_summary, f, indent=2, default=str)

        logger.info(f"Experiment summary saved to {summary_path}")
        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
