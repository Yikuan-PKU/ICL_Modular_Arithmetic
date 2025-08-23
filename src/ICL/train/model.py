"""RHM Training Configuration with Custom Tokenizer Support"""

import logging
from dataclasses import dataclass, field

from transformers import TrainingArguments

from ICL.train.tokenizer import RHMTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RHMTrainingConfig:
    """Configuration for RHM training with seed-based batching support."""

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

    # Special token IDs
    pad_token_id: int = 0
    mask_token_id: int = vocab_size + 1
    cls_token_id: int = vocab_size + 2
    sep_token_id: int = vocab_size + 3

    # Sequence processing (FIXED - always pack)
    max_sequence_length: int = 512
    pack_sequences: bool = True  # Always True, no longer configurable
    mlm: bool = True
    mlm_probability: float = 0.15

    # Cross-configuration shuffling (before packing)
    shuffle_before_packing: bool = False
    shuffle_seed: int = 42
    shuffle_strategy: str = "global"  # "global", "balanced"

    # NEW: Seed-based batching (during training)
    seed_balanced_batching: bool = True
    seed_sampling_strategy: str = "balanced"  # "balanced", "random", "weighted"
    seeds_per_batch: int | None = None  # None = use all available seeds

    # NEW: Last-token prediction configuration
    last_token_prediction: bool = False  # Enable last-token prediction masking

    # UPDATED: Checkpoint configuration with steps option
    save_by_steps: bool = False  # NEW: If True, save by steps instead of epochs
    save_steps_interval: int = 500  # NEW: Steps interval when save_by_steps=True
    save_strategy: str = "epoch"  # Will be set based on save_by_steps
    save_steps: int = 1  # Will be set based on save_by_steps
    save_total_limit: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Evaluation configuration - UPDATED for epoch-based evaluation
    evaluation_strategy: str = "epoch"  # Changed from "steps" to "epoch"
    eval_steps: int = 1  # Evaluate every epoch
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
    max_samples_per_seed: int | None = None  # Limit samples per seed

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

    def get_model_suffix(self) -> str:
        """Generate model name suffix based on configuration."""
        parts = [self.task_name]  # clm or mlm

        # Add shuffling info
        if self.shuffle_before_packing:
            parts.append(f"{self.shuffle_strategy}shuffle")
        else:
            parts.append("noshuffle")

        # Add seed batching info
        if self.seed_balanced_batching:
            parts.append(f"seed{self.seed_sampling_strategy}")
        else:
            parts.append("noseed")

        # Add last token prediction info
        if self.last_token_prediction:
            parts.append("lasttoken")

        return "_".join(parts)

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments with flexible checkpointing."""
        # Determine save strategy based on save_by_steps
        if self.save_by_steps:
            save_strategy = "steps"
            save_steps = self.save_steps_interval
        else:
            save_strategy = "epoch"
            save_steps = 1

        training_args_dict = {
            "output_dir": self.output_dir,
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
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
            import os

            os.environ["WANDB_DIR"] = self.output_dir

        try:
            return TrainingArguments(**training_args_dict)
        except TypeError as e:
            logger.warning(f"TrainingArguments creation failed: {e}")
            logger.info("Trying with minimal parameters for version compatibility...")

            minimal_args = {
                "output_dir": self.output_dir,
                "num_train_epochs": self.num_train_epochs,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "per_device_eval_batch_size": self.per_device_eval_batch_size,
                "learning_rate": self.learning_rate,
                "evaluation_strategy": self.evaluation_strategy,
                "eval_steps": self.eval_steps,
                "save_strategy": save_strategy,
                "save_steps": save_steps,
                "logging_steps": self.logging_steps,
                "seed": self.seed,
            }

            return TrainingArguments(**minimal_args)
