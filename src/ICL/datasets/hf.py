import logging
import random
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from transformers import (
    TrainingArguments,
)

from ICL.datasets.gen import UnifiedRHMDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RHMTrainingConfig:
    """Configuration for RHM training with version compatibility."""

    # Model configuration
    model_name_or_path: str | None = None
    vocab_size: int = 37
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048

    # Training configuration
    task_name: str = "clm"
    output_dir: str = "./rhm_training_output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"

    # Checkpoint configuration
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Evaluation configuration
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: int | None = None

    # Logging configuration
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str | None = None

    # Optimization configuration
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Data configuration
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True

    # Hierarchical analysis
    track_hierarchical_metrics: bool = True
    hierarchical_eval_frequency: int = 1000

    # Reproducibility
    seed: int = 42

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments with version compatibility."""
        # Create arguments dict without problematic parameters
        training_args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "eval_accumulation_steps": self.eval_accumulation_steps,
            "logging_strategy": self.logging_strategy,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "remove_unused_columns": self.remove_unused_columns,
            "seed": self.seed,
        }

        # Only add accelerator_config if it's supported in this version
        try:
            # Test if accelerator_config is supported
            TrainingArguments(output_dir="test", accelerator_config=None)
            training_args_dict["accelerator_config"] = None
            logger.info("Using accelerator_config=None for compatibility")
        except TypeError:
            # accelerator_config not supported in this version
            logger.info("accelerator_config not supported in this transformers version - skipping")

        return TrainingArguments(**training_args_dict)


def prepare_packed_dataset(
    dataset_path: str,
    config: RHMTrainingConfig,
    train_split_ratio: float = 0.8,
    filter_config_L: int | None = None,
    filter_config_m: int | None = None,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Prepare packed dataset for HuggingFace training.

    Args:
        dataset_path: Path to unified RHM dataset
        config: Training configuration
        train_split_ratio: Ratio for train/eval split
        filter_config_L: Filter by hierarchy depth
        filter_config_m: Filter by multiplicity
        max_samples: Maximum samples to use

    Returns:
        Tuple of (train_dataset, eval_dataset, metadata)

    """
    print("=" * 60)
    print("PREPARING PACKED DATASET")
    print("=" * 60)

    # Load unified dataset
    unified_dataset = UnifiedRHMDataset(dataset_path)
    dataset = unified_dataset.get_dataset()

    print(f"Original dataset size: {len(dataset):,}")

    # Apply filters if specified
    if filter_config_L is not None or filter_config_m is not None:
        dataset = unified_dataset.filter_by_config(L=filter_config_L, m=filter_config_m)
        print(f"After config filtering: {len(dataset):,}")

    # Limit samples if specified
    if max_samples is not None and len(dataset) > max_samples:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = dataset.select(indices[:max_samples])
        print(f"After sampling: {len(dataset):,}")

    # Prepare sequences for packing
    if config.pack_sequences:
        packed_dataset = _pack_sequences(dataset, config)
    else:
        packed_dataset = _prepare_individual_sequences(dataset, config)

    # Split into train/eval
    split_idx = int(len(packed_dataset) * train_split_ratio)
    train_dataset = packed_dataset.select(range(split_idx))
    eval_dataset = packed_dataset.select(range(split_idx, len(packed_dataset)))

    metadata = {
        "original_size": len(unified_dataset.get_dataset()),
        "filtered_size": len(dataset),
        "packed_size": len(packed_dataset),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "config": config,
        "vocab_info": unified_dataset.get_vocab_info(),
    }

    print(f"Train dataset: {len(train_dataset):,}")
    print(f"Eval dataset: {len(eval_dataset):,}")
    print("=" * 60)

    return train_dataset, eval_dataset, metadata


def _pack_sequences(dataset: Dataset, config: RHMTrainingConfig) -> Dataset:
    """Pack multiple sequences into longer training examples."""
    print("Packing sequences...")

    packed_examples = []
    current_sequence = []
    current_length = 0

    # Reserve space for separator tokens
    effective_max_length = config.max_sequence_length - 10

    for example in dataset:
        sequence = example["input_ids"]

        # Skip empty sequences
        if not sequence:
            continue

        # If adding this sequence would exceed max length, finalize current packed sequence
        if current_length + len(sequence) + 1 > effective_max_length and current_sequence:
            packed_examples.append(
                {
                    "input_ids": current_sequence,
                    "length": len(current_sequence),
                }
            )
            current_sequence = []
            current_length = 0

        # Add separator if this isn't the first sequence in the pack
        if current_sequence:
            current_sequence.append(config.separator_token_id)
            current_length += 1

        # Add the sequence
        current_sequence.extend(sequence)
        current_length += len(sequence)

    # Add the last packed sequence if it exists
    if current_sequence:
        packed_examples.append(
            {
                "input_ids": current_sequence,
                "length": len(current_sequence),
            }
        )

    print(f"Packed {len(dataset)} sequences into {len(packed_examples)} training examples")
    avg_length = sum(ex["length"] for ex in packed_examples) / len(packed_examples)
    print(f"Average packed sequence length: {avg_length:.1f}")

    return Dataset.from_list(packed_examples)


def _prepare_individual_sequences(dataset: Dataset, config: RHMTrainingConfig) -> Dataset:
    """Prepare individual sequences without packing."""
    print("Preparing individual sequences...")

    examples = []
    for example in dataset:
        sequence = example["input_ids"]

        # Skip empty sequences
        if not sequence:
            continue

        # Truncate if too long
        if len(sequence) > config.max_sequence_length:
            sequence = sequence[: config.max_sequence_length]

        examples.append(
            {
                "input_ids": sequence,
                "length": len(sequence),
            }
        )

    print(f"Prepared {len(examples)} individual sequences")
    return Dataset.from_list(examples)
