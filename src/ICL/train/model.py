"""RHM Training Configuration with Custom Tokenizer Support"""

import logging
import os
import random
import typing as t
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import TrainingArguments

from ICL.train.tokenizer import RHMTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RHMTrainingConfig:
    """Configuration for RHM training with custom tokenizer support."""

    # Model configuration
    model_name_or_path: str | None = None
    vocab_size: int = 37  # RHM vocabulary size
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048

    # Tokenizer configuration
    pad_token: str = "<pad>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    mask_token: str = "<mask>"
    unk_token: str = "<unk>"

    # Training configuration
    task_name: str = "mlm"  # "mlm" or "clm"
    output_dir: str = "./rhm_training_output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"

    # Add these special token configuration fields:
    pad_token_id: int = 0
    mask_token_id: int = vocab_size + 1
    cls_token_id: int = vocab_size + 2
    sep_token_id: int = vocab_size + 3

    # Sequence processing
    max_sequence_length: int = 512
    pack_sequences: bool = True
    mlm: bool = True
    mlm_probability: float = 0.15

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

    # Dataset configuration
    dataset_path: str | None = None
    train_split_ratio: float = 0.8
    filter_config_L: int | None = None
    filter_config_m: int | None = None
    max_samples: int | None = None

    # Hierarchical analysis
    track_hierarchical_metrics: bool = True
    hierarchical_eval_frequency: int = 1000

    # Reproducibility
    seed: int = 42

    def create_tokenizer(self) -> RHMTokenizer:
        """Create RHM tokenizer with current configuration."""
        return RHMTokenizer(
            vocab_size=self.vocab_size,
            pad_token=self.pad_token,
            eos_token=self.eos_token,
            sep_token=self.sep_token,
            mask_token=self.mask_token,
            unk_token=self.unk_token,
        )

    @property
    def effective_vocab_size(self) -> int:
        """Get effective vocabulary size including special tokens."""
        return self.vocab_size + 5  # RHM tokens + 5 special tokens

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments with version compatibility."""
        training_args_dict = {
            "output_dir": self.output_dir,
            "output_dir": self.output_dir,
            "save_strategy": "steps",  # or "epoch"
            "save_steps": 500,  # adjust as needed
            "save_total_limit": 3,  # keep only last 3 checkpoints
            "load_best_model_at_end": True,
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
        if "wandb" in self.report_to:
            os.environ["WANDB_DIR"] = self.output_dir

        # Try creating TrainingArguments, remove problematic parameters if they fail
        try:
            return TrainingArguments(**training_args_dict)
        except TypeError as e:
            logger.warning(f"TrainingArguments creation failed: {e}")
            logger.info("Trying with minimal parameters for version compatibility...")

            # Minimal set of parameters that should work across versions
            minimal_args = {
                "output_dir": self.output_dir,
                "num_train_epochs": self.num_train_epochs,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "per_device_eval_batch_size": self.per_device_eval_batch_size,
                "learning_rate": self.learning_rate,
                "evaluation_strategy": self.evaluation_strategy,
                "eval_steps": self.eval_steps,
                "save_steps": self.save_steps,
                "logging_steps": self.logging_steps,
                "seed": self.seed,
            }

            return TrainingArguments(**minimal_args)


def prepare_packed_dataset(
    dataset_path: str,
    tokenizer: RHMTokenizer,
    config: RHMTrainingConfig,
    train_split_ratio: float = 0.8,
    filter_config_L: int | None = None,
    filter_config_m: int | None = None,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset, dict[str, t.Any]]:
    """Prepare packed dataset for HuggingFace training with custom tokenizer.

    Args:
        dataset_path: Path to unified RHM dataset
        tokenizer: RHM tokenizer instance
        config: Training configuration
        train_split_ratio: Ratio for train/eval split
        filter_config_L: Filter by hierarchy depth
        filter_config_m: Filter by multiplicity
        max_samples: Maximum samples to use

    Returns:
        Tuple of (train_dataset, eval_dataset, metadata)

    """
    print("=" * 60)
    print("PREPARING PACKED DATASET WITH CUSTOM TOKENIZER")
    print("=" * 60)

    # Import here to avoid circular dependencies
    from ICL.datasets.gen import UnifiedRHMDataset

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
        packed_dataset = _pack_sequences(dataset, tokenizer, config)
    else:
        packed_dataset = _prepare_individual_sequences(dataset, tokenizer, config)

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
        "packing_enabled": config.pack_sequences,
        "config": config,
        "vocab_info": unified_dataset.get_vocab_info(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }

    print(f"Train dataset: {len(train_dataset):,}")
    print(f"Eval dataset: {len(eval_dataset):,}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print("=" * 60)

    return train_dataset, eval_dataset, metadata


def _pack_sequences(dataset: Dataset, tokenizer: RHMTokenizer, config: RHMTrainingConfig) -> Dataset:
    """Pack multiple sequences into longer training examples."""
    print("Packing sequences with custom tokenizer...")

    packed_examples = []
    current_sequence = []
    current_length = 0

    # Reserve space for special tokens
    effective_max_length = config.max_sequence_length - 10

    for example in dataset:
        # Convert RHM sequence to token IDs using tokenizer
        if "input_ids" in example:
            sequence = example["input_ids"]
        elif "sequence" in example:
            sequence = tokenizer.encode_sequence(example["sequence"])
        else:
            continue

        # Skip empty sequences
        if not sequence:
            continue

        # If adding this sequence would exceed max length, finalize current packed sequence
        if current_length + len(sequence) + 1 > effective_max_length and current_sequence:
            # Add EOS token to end of packed sequence
            current_sequence.append(tokenizer.eos_token_id)
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
            current_sequence.append(tokenizer.sep_token_id)
            current_length += 1

        # Add the sequence
        current_sequence.extend(sequence)
        current_length += len(sequence)

    # Add the last packed sequence if it exists
    if current_sequence:
        current_sequence.append(tokenizer.eos_token_id)
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


def _prepare_individual_sequences(dataset: Dataset, tokenizer: RHMTokenizer, config: RHMTrainingConfig) -> Dataset:
    """Prepare individual sequences without packing."""
    print("Preparing individual sequences with custom tokenizer...")

    examples = []
    for example in dataset:
        # Convert RHM sequence to token IDs using tokenizer
        if "input_ids" in example:
            sequence = example["input_ids"]
        elif "sequence" in example:
            sequence = tokenizer.encode_sequence(example["sequence"])
        else:
            continue

        # Skip empty sequences
        if not sequence:
            continue

        # Add EOS token
        sequence.append(tokenizer.eos_token_id)

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
