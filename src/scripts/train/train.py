#!/usr/bin/env python3
"""RHM Model Training Main Script with Explicit L,M Configuration"""

import argparse
import logging
import typing as t
from pathlib import Path

from transformers import set_seed

from ICL.settings import (
    ModelConfig,
    create_base_parser,
    load_experiment_config,
    parse_model_config,
    validate_args,
)
from ICL.train.model import RHMTrainingConfig
from ICL.train.train_pipeline import create_rhm_training_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# TODO: add multi-gpu config


def create_training_parser() -> argparse.ArgumentParser:
    """Create argument parser for training with explicit L,M configuration."""
    parser = create_base_parser(require_model_type=True)
    parser.description = "Train RHM models with explicit L,M configuration"

    # Training-specific arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--config-override", type=str, help="Path to YAML config file with parameter overrides (optional)"
    )
    training_group.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without starting training"
    )

    training_group.add_argument(
        "--save-by-steps", action="store_true", help="Save checkpoints by steps instead of epochs"
    )
    training_group.add_argument(
        "--save-steps-interval",
        type=int,
        default=500,
        help="Steps interval for saving when --save-by-steps is enabled (default: 500)",
    )
    training_group.add_argument(
        "--last-token-prediction", action="store_true", help="Enable last-token prediction masking (useful for CLM)"
    )
    training_group.add_argument(
        "--no-last-token-prediction", action="store_true", help="Explicitly disable last-token prediction masking"
    )

    return parser


def create_training_config(
    model_config: ModelConfig, config_override_path: str | None = None, args: argparse.Namespace | None = None
) -> RHMTrainingConfig:
    """Create training configuration from flattened configs and overrides."""
    # Start with defaults
    config = RHMTrainingConfig()
    logger.info("Starting with default RHMTrainingConfig")

    # Load model-specific config from flattened structure
    model_yaml = load_model_yaml_config(model_config)

    # Load override config if provided
    override_yaml = load_yaml_overrides(config_override_path)

    # Merge configurations: defaults < model_yaml < override_yaml < command_line_args
    all_overrides = {}

    # Apply model config
    if model_yaml:
        all_overrides.update(model_yaml)
        logger.info(f"Applied {len(model_yaml)} model-specific parameters from {model_config.model_type}.yaml")

    # Apply overrides
    if override_yaml:
        all_overrides.update(override_yaml)
        logger.info(f"Applied {len(override_yaml)} override parameters")

    # NEW: Apply command line argument overrides
    if args:
        cmd_overrides = {}

        # Handle save by steps
        if args.save_by_steps:
            cmd_overrides["save_by_steps"] = True
            cmd_overrides["save_steps_interval"] = args.save_steps_interval
            logger.info(f"Command line override: save_by_steps=True, save_steps_interval={args.save_steps_interval}")

        # Handle last token prediction
        if args.last_token_prediction:
            cmd_overrides["last_token_prediction"] = True
            logger.info("Command line override: last_token_prediction=True")
        elif args.no_last_token_prediction:
            cmd_overrides["last_token_prediction"] = False
            logger.info("Command line override: last_token_prediction=False")

        if cmd_overrides:
            all_overrides.update(cmd_overrides)
            logger.info(f"Applied {len(cmd_overrides)} command line overrides")

    # Set hierarchical paths (using train subdirectory)
    paths = model_config.get_model_paths()
    all_overrides["output_dir"] = str(paths["model_dir"])
    all_overrides["dataset_path"] = str(paths["dataset_dir"])  # Points to train subdirectory

    # Set model type and task name
    all_overrides["task_name"] = model_config.model_type

    # Apply all overrides with type conversion
    if all_overrides:
        config_dict = config.__dict__.copy()

        for key, value in all_overrides.items():
            if hasattr(config, key):
                original_value = getattr(config, key)

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
                    logger.info(f"Config override: {key} = {converted_value}")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {key}={value}: {e}")
                    config_dict[key] = value
            else:
                logger.warning(f"Unknown parameter: {key} (ignored)")

        config = RHMTrainingConfig(**config_dict)

    return config


def load_model_yaml_config(model_config: ModelConfig) -> dict[str, t.Any]:
    """Load unified training YAML configuration and extract task-specific settings."""
    L = model_config.dataset_config.L
    m = model_config.dataset_config.m

    logger.info(f"Using dataset configuration: L={L}, m={m}")

    # Load unified training config
    config_type = "training"  # Always load training.yaml now
    yaml_config = load_experiment_config(config_type, model_config.dataset_config, L=L, m=m)

    if not yaml_config:
        logger.warning(f"✗ No training config found for L={L}, m={m}")
        return {}

    logger.info(
        f"✓ Loaded training config from conf/{model_config.dataset_config.dataset_type}_{model_config.dataset_config.num_seeds}_L{L}_M{m}/training.yaml"
    )

    # Extract task-specific configuration
    task_name = model_config.model_type  # "clm" or "mlm"
    task_specific_config = yaml_config.get("task_specific", {}).get(task_name, {})

    if task_specific_config:
        logger.info(f"✓ Applied task-specific configuration for {task_name}")
        # Merge base config with task-specific overrides
        merged_config = yaml_config.copy()
        merged_config.update(task_specific_config)
        # Remove the task_specific section from the final config
        merged_config.pop("task_specific", None)
        return merged_config
    logger.warning(f"⚠ No task-specific configuration found for {task_name}, using base config")
    # Set the task_name from model_type
    yaml_config["task_name"] = task_name
    # Remove the task_specific section
    yaml_config.pop("task_specific", None)
    return yaml_config


def load_yaml_overrides(config_path: str | Path | None) -> dict[str, t.Any]:
    """Load additional YAML configuration overrides if provided."""
    if config_path is None:
        return {}

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Override config file {config_path} not found.")
        return {}

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as file:
            yaml_overrides = yaml.safe_load(file)

        if yaml_overrides:
            logger.info(f"Loaded YAML overrides from {config_path}")
            return yaml_overrides
        return {}

    except Exception as e:
        logger.error(f"Error loading override config {config_path}: {e}")
        return {}


def save_evaluation_metadata(output_dir: Path, training_metadata: dict[str, t.Any], model_config: ModelConfig) -> None:
    """Save metadata.json file for evaluation pipeline compatibility with new features."""
    metadata_file = output_dir / "metadata.json"

    train_meta = training_metadata.get("training_metadata", {})
    model_meta = training_metadata.get("model_config", {})

    # Get paths using new structure
    paths = model_config.get_model_paths()

    eval_metadata = {
        "config_L": train_meta.get("config_L", model_config.dataset_config.L),
        "config_m": train_meta.get("config_m", model_config.dataset_config.m),
        "n_train": train_meta.get("n_train", 0),
        "model_type": train_meta.get("model_type", "causal_lm"),
        "training_seed": 42,
        "eval_seed": 123,
        "tokenizer_class": "RHMTokenizer",
        "experiment_name": model_config.to_name(),
        "dataset_name": model_config.dataset_config.to_base_name(),
        "dataset_path": train_meta.get("dataset_path"),
        "output_dir": str(output_dir),
        "shuffling_enabled": train_meta.get("shuffling_enabled", False),
        "shuffle_strategy": train_meta.get("shuffle_strategy"),
        # NEW: Add new feature metadata
        "last_token_prediction": train_meta.get("last_token_prediction", False),
        "save_by_steps": train_meta.get("save_by_steps", False),
        "accuracy_tracking_enabled": True,  # Always enabled now
        "hierarchical_paths": {
            "base_dataset_dir": str(paths["base_dataset_dir"]),
            "train_dataset_dir": str(paths["dataset_dir"]),
            "model_dir": str(paths["model_dir"]),
            "config_dir": str(paths["config_dir"]),
        },
        "tokenizer_metadata": training_metadata.get("tokenizer_metadata", {}),
        "model_metadata": training_metadata.get("model_metadata", {}),
        "dataset_generation_params": training_metadata.get("dataset_generation_params", {}),
        # NEW: Add accuracy and training feature metadata
        "training_features": {
            "seed_balanced_batching": train_meta.get("seed_balanced_batching", True),
            "seed_sampling_strategy": train_meta.get("seed_sampling_strategy", "balanced"),
            "last_token_prediction": train_meta.get("last_token_prediction", False),
            "accuracy_tracking": True,
            "save_by_steps": train_meta.get("save_by_steps", False),
            "cross_config_shuffling": train_meta.get("shuffle_before_packing", False),
        },
    }

    import json

    with open(metadata_file, "w") as f:
        json.dump(eval_metadata, f, indent=2)
    logger.info(f"Saved evaluation metadata (including new features) to: {metadata_file}")


def safe_format_number(value, default="N/A"):
    """Safely format a number with commas, handling non-numeric values."""
    if isinstance(value, (int, float)) and value is not None:
        return f"{value:,}"
    return default


def main():
    """Train RHM model using explicit L,M configuration."""
    # Parse arguments
    parser = create_training_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to model config (L,M now come from command line)
    model_config = parse_model_config(args)

    # Log the configuration
    logger.info(f"Model configuration: {model_config.to_name()}")
    logger.info(f"Dataset configuration: {model_config.dataset_config.to_name()}")
    logger.info(f"Using L={model_config.dataset_config.L}, m={model_config.dataset_config.m} from command line")

    # Create training configuration with command line args
    training_config = create_training_config(
        model_config=model_config,
        config_override_path=args.config_override,
        args=args,  # Pass command line args for overrides
    )

    # Set seed for reproducibility
    set_seed(training_config.seed)

    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration validation only")
        logger.info(f"Model config: {model_config.to_name()}")
        logger.info(f"Dataset config: {model_config.dataset_config.to_name()}")
        logger.info(f"Using L={model_config.dataset_config.L}, M={model_config.dataset_config.m}")

        # Log new configuration options
        logger.info(f"Save by steps: {training_config.save_by_steps}")
        if training_config.save_by_steps:
            logger.info(f"Save steps interval: {training_config.save_steps_interval}")
        logger.info(f"Last token prediction: {training_config.last_token_prediction}")

        # Validate paths
        paths = model_config.get_model_paths()
        logger.info(f"Dataset path: {paths['dataset_dir']}")
        logger.info(f"Model output path: {paths['model_dir']}")
        logger.info(f"Config path: {paths['config_dir']}")

        # Check if paths exist
        if paths["dataset_dir"].exists():
            logger.info("✓ Dataset directory exists")
        else:
            logger.error(f"✗ Dataset directory not found: {paths['dataset_dir']}")
            return 1

        if paths["config_dir"].exists():
            logger.info("✓ Config directory exists")
        else:
            logger.warning(f"⚠ Config directory not found: {paths['config_dir']}")

        logger.info("✓ Configuration validation successful")
        return 0

    # Get paths for logging
    paths = model_config.get_model_paths()

    # Validate that required directories exist
    if not paths["dataset_dir"].exists():
        logger.error(f"Dataset directory not found: {paths['dataset_dir']}")
        logger.error("Please run dataset generation first.")
        return 1

    # Log final configuration including new features
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Experiment: {model_config.to_name()}")
    logger.info(f"Dataset: {model_config.dataset_config.to_name()}")
    logger.info(f"Configuration: L={model_config.dataset_config.L}, M={model_config.dataset_config.m}")
    logger.info(f"Model type: {model_config.model_type}")
    logger.info(f"Task: {training_config.task_name}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Epochs: {training_config.num_train_epochs}")

    # Log new features
    logger.info(f"Save by steps: {training_config.save_by_steps}")
    if training_config.save_by_steps:
        logger.info(f"Save steps interval: {training_config.save_steps_interval}")
    else:
        logger.info("Save strategy: every epoch")
    logger.info(f"Last token prediction: {training_config.last_token_prediction}")

    logger.info(f"Dataset path: {paths['dataset_dir']}")
    logger.info(f"Model output path: {training_config.output_dir}")
    logger.info(f"Config path: {paths['config_dir']}")
    logger.info("=" * 60)

    # [Model creation code remains the same...]
    # Create model based on configuration
    logger.info("Creating model...")
    if training_config.model_name_or_path:
        # Load from existing model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

        model_config_hf = AutoConfig.from_pretrained(training_config.model_name_or_path)
        model_config_hf.vocab_size = training_config.effective_vocab_size
        if training_config.task_name == "clm":
            model = AutoModelForCausalLM.from_pretrained(training_config.model_name_or_path, config=model_config_hf)
        else:
            model = AutoModelForMaskedLM.from_pretrained(training_config.model_name_or_path, config=model_config_hf)
    # Create new model
    elif training_config.task_name == "clm":
        from transformers import AutoModelForCausalLM, GPT2Config

        model_config_hf = GPT2Config(
            vocab_size=training_config.effective_vocab_size,
            n_positions=training_config.max_position_embeddings,
            n_embd=training_config.hidden_size,
            n_layer=training_config.num_hidden_layers,
            n_head=training_config.num_attention_heads,
            n_inner=training_config.intermediate_size,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False,
            pad_token_id=training_config.pad_token_id,
        )
        model = AutoModelForCausalLM.from_config(model_config_hf)

    elif training_config.task_name == "mlm":
        from transformers import AutoModelForMaskedLM, BertConfig

        model_config_hf = BertConfig(
            vocab_size=training_config.effective_vocab_size,
            hidden_size=training_config.hidden_size,
            num_hidden_layers=training_config.num_hidden_layers,
            num_attention_heads=training_config.num_attention_heads,
            intermediate_size=training_config.intermediate_size,
            max_position_embeddings=training_config.max_position_embeddings,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=training_config.pad_token_id,
            mask_token_id=training_config.mask_token_id,
            cls_token_id=training_config.cls_token_id,
            sep_token_id=training_config.sep_token_id,
        )
        model = AutoModelForMaskedLM.from_config(model_config_hf)
    else:
        raise ValueError(f"Unknown task: {training_config.task_name}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer using hierarchical pipeline
    logger.info("Creating training pipeline...")

    trainer, metadata = create_rhm_training_pipeline(
        model_config=model_config,  # Pass ModelConfig instead of dataset path
        model=model,
        training_config=training_config,
        max_samples_per_seed=getattr(training_config, "max_samples_per_seed", None),
    )

    # Log dataset info with new features
    dataset_meta = metadata.get("seed_dataset_metadata", {})
    logger.info("Dataset preparation completed:")
    logger.info(f"  Separate train/eval loading: {dataset_meta.get('separate_train_eval', True)}")
    logger.info(
        f"  Original train size: {safe_format_number(sum(dataset_meta.get('original_train_seed_datasets', {}).values()))}"
    )
    logger.info(
        f"  Original eval size: {safe_format_number(sum(dataset_meta.get('original_eval_seed_datasets', {}).values()))}"
    )
    logger.info(f"  Packed train size: {safe_format_number(dataset_meta.get('total_train_sequences'))}")
    logger.info(f"  Packed eval size: {safe_format_number(dataset_meta.get('total_eval_sequences'))}")
    logger.info(f"  Available seeds: {dataset_meta.get('available_seeds', [])}")
    logger.info(f"  Shuffling enabled: {dataset_meta.get('shuffling_enabled', False)}")
    if dataset_meta.get("shuffling_enabled"):
        logger.info(f"  Shuffle strategy: {dataset_meta.get('shuffle_strategy', 'N/A')}")

    # Log new features
    logger.info(f"  Last token prediction: {training_config.last_token_prediction}")
    logger.info("  Accuracy tracking: Enabled (computed alongside loss)")

    # Train the model
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    results = trainer.train()

    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    trainer.save_model()  # This saves to hierarchical output_dir
    trainer.save_state()  # This saves trainer state

    # Save evaluation metadata.json
    output_dir = Path(training_config.output_dir)
    save_evaluation_metadata(output_dir, metadata, model_config)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Experiment: {model_config.to_name()}")
    logger.info(f"Dataset base: {model_config.dataset_config.to_base_name()}")
    logger.info(f"Results saved to: {training_config.output_dir}")

    # Log training metadata including new features
    train_meta = metadata.get("training_metadata", {})
    logger.info("Training metadata:")
    logger.info(f"  config_L: {train_meta.get('config_L', model_config.dataset_config.L)}")
    logger.info(f"  config_m: {train_meta.get('config_m', model_config.dataset_config.m)}")
    logger.info(f"  total_train_samples: {train_meta.get('total_train_samples')}")
    logger.info(f"  total_eval_samples: {train_meta.get('total_eval_samples')}")
    logger.info(f"  model_type: {train_meta.get('model_type')}")
    logger.info(f"  separate_train_eval: {train_meta.get('separate_train_eval')}")
    logger.info(f"  shuffling_enabled: {train_meta.get('shuffling_enabled')}")
    logger.info(
        f"  last_token_prediction: {train_meta.get('last_token_prediction', training_config.last_token_prediction)}"
    )
    logger.info(f"  save_by_steps: {train_meta.get('save_by_steps', training_config.save_by_steps)}")

    # Log hierarchical paths
    logger.info("Hierarchical structure:")
    logger.info(f"  Base dataset: {paths['base_dataset_dir']}")
    logger.info(f"  Train dataset: {paths['base_dataset_dir'] / 'train'}")
    logger.info(f"  Eval dataset: {paths['base_dataset_dir'] / 'eval'}")
    logger.info(f"  Model output: {paths['model_dir']}")
    logger.info(f"  Config dir: {paths['config_dir']}")

    # Log final accuracy information
    logger.info("Accuracy tracking:")
    logger.info("  ✓ Token-level accuracy computed alongside loss")
    logger.info("  ✓ Per-seed accuracy tracking enabled")
    logger.info("  ✓ Accuracy metrics saved in seed_training_metrics.json")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
