#!/usr/bin/env python3
"""RHM Model Training Main Script with Flattened Config Support"""

import argparse
import json
import logging
import re
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
    """Create argument parser for training with flattened config support."""
    parser = create_base_parser(require_model_type=True)
    parser.description = "Train RHM models with flattened configuration"

    # Training-specific arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--config-override", type=str, help="Path to YAML config file with parameter overrides (optional)"
    )
    training_group.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without starting training"
    )

    return parser


def load_model_yaml_config(model_config: ModelConfig) -> dict[str, t.Any]:
    """Load model-specific YAML configuration from correct path structure."""
    # Extract L,M from the dataset path
    paths = model_config.get_model_paths()
    dataset_path = paths["dataset_dir"]

    try:
        L, m = extract_L_M_from_dataset_path(dataset_path)
        logger.info(f"Extracted L={L}, m={m} from dataset path: {dataset_path}")
    except ValueError as e:
        logger.error(f"Failed to extract L,M from dataset path: {e}")
        return {}

    # Load config using the correct L,M values
    config_type = model_config.model_type  # "clm" or "mlm"
    yaml_config = load_experiment_config(config_type, model_config.dataset_config, L=L, m=m)

    if yaml_config:
        logger.info(
            f"✓ Loaded {config_type} config from conf/{model_config.dataset_config.dataset_type}_{model_config.dataset_config.num_seeds}_L{L}_M{m}/{config_type}.yaml"
        )
    else:
        logger.warning(f"✗ No {config_type} config found for L={L}, m={m}")

    return yaml_config


def extract_L_M_from_dataset_path(dataset_path: str | Path) -> tuple[int, int]:
    """Extract L and M values from dataset directory path.

    Args:
        dataset_path: Path like datasets/uniform_10_L4_M2/train/

    Returns:
        tuple: (L, m) values

    """
    dataset_path = Path(dataset_path)

    # Look for pattern in the dataset directory name
    pattern = r".*_L(\d+)_M(\d+)"

    # Check the parent directory name (remove /train/ suffix)
    dir_name = dataset_path.parent.name if dataset_path.name == "train" else dataset_path.name

    match = re.search(pattern, dir_name)
    if match:
        L = int(match.group(1))
        m = int(match.group(2))
        return L, m

    raise ValueError(f"Could not extract L,M from dataset path: {dataset_path}")


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


def create_training_config(model_config: ModelConfig, config_override_path: str | None = None) -> RHMTrainingConfig:
    """Create training configuration from flattened configs and overrides."""
    # Start with defaults
    config = RHMTrainingConfig()
    logger.info("Starting with default RHMTrainingConfig")

    # Load model-specific config from flattened structure
    model_yaml = load_model_yaml_config(model_config)

    # Load override config if provided
    override_yaml = load_yaml_overrides(config_override_path)

    # Merge configurations: defaults < model_yaml < override_yaml
    all_overrides = {}

    # Apply model config
    if model_yaml:
        all_overrides.update(model_yaml)
        logger.info(f"Applied {len(model_yaml)} model-specific parameters from {model_config.model_type}.yaml")

    # Apply overrides
    if override_yaml:
        all_overrides.update(override_yaml)
        logger.info(f"Applied {len(override_yaml)} override parameters")

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


def save_evaluation_metadata(output_dir: Path, training_metadata: dict[str, t.Any], model_config: ModelConfig) -> None:
    """Save metadata.json file for evaluation pipeline compatibility."""
    metadata_file = output_dir / "metadata.json"

    train_meta = training_metadata.get("training_metadata", {})
    model_meta = training_metadata.get("model_config", {})

    # Get paths using new structure
    paths = model_config.get_model_paths()

    eval_metadata = {
        "config_L": train_meta.get("config_L", 2),
        "config_m": train_meta.get("config_m", 2),
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
        "hierarchical_paths": {
            "base_dataset_dir": str(paths["base_dataset_dir"]),
            "train_dataset_dir": str(paths["dataset_dir"]),
            "model_dir": str(paths["model_dir"]),
            "config_dir": str(paths["config_dir"]),
        },
        "tokenizer_metadata": training_metadata.get("tokenizer_metadata", {}),
        "model_metadata": training_metadata.get("model_metadata", {}),
        "dataset_generation_params": training_metadata.get("dataset_generation_params", {}),
    }

    with open(metadata_file, "w") as f:
        json.dump(eval_metadata, f, indent=2)

    logger.info(f"Saved evaluation metadata to: {metadata_file}")


def main():
    """Train RHM model using flattened configuration system."""
    # Parse arguments
    parser = create_training_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to model config (dataset_config.is_eval will be False by default)
    model_config = parse_model_config(args)

    # Create training configuration
    training_config = create_training_config(model_config=model_config, config_override_path=args.config_override)

    # Set seed for reproducibility
    set_seed(training_config.seed)

    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration validation only")
        logger.info(f"Model config: {model_config.to_name()}")
        logger.info(f"Dataset base: {model_config.dataset_config.to_base_name()}")
        logger.info(f"Model type: {model_config.model_type}")
        logger.info(f"Output directory: {training_config.output_dir}")
        logger.info(f"Dataset path: {training_config.dataset_path}")
        logger.info("✓ Configuration validation successful")
        return None

    # Get paths for logging
    paths = model_config.get_model_paths()

    # Log final configuration
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Experiment: {model_config.to_name()}")
    logger.info(f"Dataset base: {model_config.dataset_config.to_base_name()}")
    logger.info(f"Model type: {model_config.model_type}")
    logger.info(f"Task: {training_config.task_name}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Epochs: {training_config.num_train_epochs}")
    logger.info(f"Train batch size: {training_config.per_device_train_batch_size}")
    logger.info(f"Eval batch size: {training_config.per_device_eval_batch_size}")
    logger.info(f"Vocab size: {training_config.vocab_size} (effective: {training_config.effective_vocab_size})")
    logger.info(f"Max sequence length: {training_config.max_sequence_length}")
    logger.info(f"Pack sequences: {training_config.pack_sequences}")
    logger.info(f"Shuffle before packing: {training_config.shuffle_before_packing}")
    if training_config.shuffle_before_packing:
        logger.info(f"Shuffle strategy: {training_config.shuffle_strategy}")
    logger.info(f"Output dir: {training_config.output_dir}")
    logger.info(f"Dataset path: {training_config.dataset_path}")
    logger.info(f"Base dataset dir: {paths['base_dataset_dir']}")
    logger.info(f"Config dir: {paths['config_dir']}")
    logger.info("=" * 60)

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
        train_split_ratio=getattr(training_config, "train_split_ratio", 0.8),
        filter_config_L=getattr(training_config, "filter_config_L", None),
        filter_config_m=getattr(training_config, "filter_config_m", None),
        max_samples=getattr(training_config, "max_samples", None),
    )

    # Log dataset info
    dataset_meta = metadata.get("dataset_metadata", {})
    logger.info("Dataset preparation completed:")
    logger.info(f"  Original size: {dataset_meta.get('original_size', 'N/A'):,}")
    logger.info(f"  Filtered size: {dataset_meta.get('filtered_size', 'N/A'):,}")
    logger.info(f"  Packed size: {dataset_meta.get('packed_size', 'N/A'):,}")
    logger.info(f"  Train size: {dataset_meta.get('train_size', 'N/A'):,}")
    logger.info(f"  Eval size: {dataset_meta.get('eval_size', 'N/A'):,}")
    logger.info(f"  Shuffling enabled: {dataset_meta.get('shuffling_enabled', False)}")
    if dataset_meta.get("shuffling_enabled"):
        logger.info(f"  Shuffle strategy: {dataset_meta.get('shuffle_strategy', 'N/A')}")

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

    # Log evaluation-ready info
    train_meta = metadata.get("training_metadata", {})
    logger.info("Training metadata:")
    logger.info(f"  config_L: {train_meta.get('config_L')}")
    logger.info(f"  config_m: {train_meta.get('config_m')}")
    logger.info(f"  n_train: {train_meta.get('n_train')}")
    logger.info(f"  model_type: {train_meta.get('model_type')}")
    logger.info(f"  shuffling_enabled: {train_meta.get('shuffling_enabled')}")

    # Log hierarchical paths
    model_meta = metadata.get("model_config", {})
    logger.info("Hierarchical structure:")
    logger.info(f"  Base dataset: {paths['base_dataset_dir']}")
    logger.info(f"  Train dataset: {paths['dataset_dir']}")
    logger.info(f"  Model output: {paths['model_dir']}")
    logger.info(f"  Config dir: {paths['config_dir']}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
