import random
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from transformers import (
    TrainingArguments,
)

from ICL.datasets.gen import UnifiedRHMDataset


@dataclass
class RHMTrainingConfig:
    """Configuration for RHM training with HuggingFace integration."""

    # Basic training parameters
    output_dir: str = "./rhm_training_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500

    # Sequence packing parameters
    max_sequence_length: int = 512
    pack_sequences: bool = True
    separator_token_id: int = 0  # Use pad token as separator

    # Vocabulary parameters
    vocab_size: int = 32
    pad_token_id: int = 0
    mask_token_id: int = 33  # vocab_size + 1

    # Task-specific parameters
    mlm: bool = True  # True for MLM, False for CLM
    mlm_probability: float = 0.15

    # Evaluation parameters
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Other parameters
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.seed,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            report_to=None,  # Disable wandb by default
        )


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
