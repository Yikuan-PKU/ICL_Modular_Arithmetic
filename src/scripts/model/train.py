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
from ICL.train.model import RHMTrainingConfig
from ICL.train.train_pipeline import create_rhm_training_pipeline

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

        # Apply YAML overrides with type conversion
        for key, value in yaml_overrides.items():
            if hasattr(config, key):
                # Get the original type from the default config
                original_value = getattr(config, key)

                # Convert value to the correct type
                try:
                    if isinstance(original_value, bool):
                        converted_value = str(value).lower() in ["true", "1", "yes", "on"]
                    elif isinstance(original_value, int):
                        converted_value = int(value)
                    elif isinstance(original_value, float):
                        converted_value = float(value)
                    elif isinstance(original_value, list):
                        converted_value = value if isinstance(value, list) else [value]
                    else:
                        converted_value = value

                    config_dict[key] = converted_value
                    logger.info(f"YAML override: {key} = {converted_value} (type: {type(converted_value).__name__})")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {key}={value}: {e}. Using as string.")
                    config_dict[key] = value
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

    # Create model based on configuration
    logger.info("Creating model...")
    if config.model_name_or_path:
        # Load from existing model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        model_config.vocab_size = config.effective_vocab_size

        if config.task_name == "clm":
            model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, config=model_config)
        else:
            model = AutoModelForMaskedLM.from_pretrained(config.model_name_or_path, config=model_config)
    # Create new model
    elif config.task_name == "clm":
        from transformers import AutoModelForCausalLM, GPT2Config

        model_config = GPT2Config(
            vocab_size=config.effective_vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False,
            pad_token_id=config.pad_token_id,
        )
        model = AutoModelForCausalLM.from_config(model_config)

    elif config.task_name == "mlm":
        from transformers import AutoModelForMaskedLM, BertConfig

        model_config = BertConfig(
            vocab_size=config.effective_vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=config.pad_token_id,
            mask_token_id=config.mask_token_id,
            cls_token_id=config.cls_token_id,
            sep_token_id=config.sep_token_id,
        )
        model = AutoModelForMaskedLM.from_config(model_config)
    else:
        raise ValueError(f"Unknown task: {config.task_name}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer using your existing pipeline function
    logger.info("Creating training pipeline...")

    trainer, metadata = create_rhm_training_pipeline(
        dataset_path=str(dataset_path),
        model=model,
        training_config=config,  # Use 'training_config' instead of 'config'
        train_split_ratio=getattr(config, "train_split_ratio", 0.8),
        filter_config_L=getattr(config, "filter_config_L", None),
        filter_config_m=getattr(config, "filter_config_m", None),
        max_samples=getattr(config, "max_samples", None),
    )

    # Log dataset info
    logger.info("Dataset preparation completed:")
    results = trainer.train()
    trainer.save_model()  # This saves to output_dir
    trainer.save_state()  # This saves trainer state
    logger.info(f"Training completed! Results saved to {config.output_dir}")

    return results


if __name__ == "__main__":
    main()
