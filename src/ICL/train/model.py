"""RHM Training Configuration with Version Compatibility"""

import logging
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ICL.datasets.hf import RHMTrainingConfig, prepare_packed_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RHMDataCollator(DataCollatorForLanguageModeling):
    """Extended data collator for RHM with separator token handling."""

    def __init__(
        self,
        tokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        separator_token_id: int = 0,
        return_tensors: str = "pt",
    ):
        """Initialize RHM data collator.

        Args:
            tokenizer: Tokenizer (can be None for simple integer sequences)
            mlm: Whether to use masked language modeling
            mlm_probability: Probability of masking tokens
            separator_token_id: ID of separator token
            return_tensors: Format of returned tensors

        """
        # Initialize with dummy tokenizer if None provided
        if tokenizer is None:
            from transformers import PreTrainedTokenizer

            tokenizer = PreTrainedTokenizer()
            tokenizer.pad_token_id = 0

        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            return_tensors=return_tensors,
        )
        self.separator_token_id = separator_token_id

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Process batch with separator token awareness."""
        # Convert to format expected by parent class
        batch = []
        for example in examples:
            if isinstance(example["input_ids"], list):
                batch.append({"input_ids": torch.tensor(example["input_ids"])})
            else:
                batch.append({"input_ids": example["input_ids"]})

        # Use parent's processing but handle separators specially
        result = super().torch_call(batch)

        # Ensure separator tokens are never masked in MLM
        if self.mlm and "labels" in result:
            separator_mask = result["input_ids"] == self.separator_token_id
            result["labels"][separator_mask] = -100  # Don't compute loss on separators

        return result


class RHMTrainer(Trainer):
    """Simplified RHM trainer using standard HuggingFace components."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator: RHMDataCollator | None = None,
        config: RHMTrainingConfig | None = None,
        **kwargs,
    ):
        """Initialize RHM trainer.

        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            config: RHM training configuration
            **kwargs: Additional arguments for Trainer

        """
        self.rhm_config = config

        # Create default data collator if none provided
        if data_collator is None:
            data_collator = RHMDataCollator(
                tokenizer=None,  # We handle raw integer sequences
                mlm=config.mlm if config else True,
                mlm_probability=config.mlm_probability if config else 0.15,
                separator_token_id=config.separator_token_id if config else 0,
            )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with special handling for RHM sequences."""
        # Standard loss computation - the data collator handles separator masking
        return super().compute_loss(model, inputs, return_outputs)


class HierarchicalMetricsCallback(TrainerCallback):
    """Callback for computing hierarchical-specific metrics during training."""

    def __init__(self, eval_dataset: Dataset, config: RHMTrainingConfig):
        """Initialize metrics callback.

        Args:
            eval_dataset: Evaluation dataset for computing metrics
            config: Training configuration

        """
        self.eval_dataset = eval_dataset
        self.config = config
        self.step_count = 0

    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Compute additional metrics during evaluation."""
        if logs is None:
            return

        # Add custom metrics here
        # For example: perplexity per hierarchy level, separator token accuracy, etc.

        # Simple example: track evaluation frequency
        self.step_count += 1
        logs["hierarchical_eval_count"] = self.step_count

        print(f"Hierarchical evaluation #{self.step_count} completed")
        print(f"Current eval loss: {logs.get('eval_loss', 'N/A'):.4f}")


def create_rhm_training_pipeline(
    dataset_path: str, model, config: RHMTrainingConfig, **dataset_kwargs
) -> tuple[RHMTrainer, dict[str, Any]]:
    """Create complete RHM training pipeline.

    Args:
        dataset_path: Path to unified RHM dataset
        model: Model to train
        config: Training configuration
        **dataset_kwargs: Additional arguments for dataset preparation

    Returns:
        Tuple of (trainer, metadata)

    """
    print("Creating RHM training pipeline...")

    # Prepare datasets
    train_dataset, eval_dataset, metadata = prepare_packed_dataset(
        dataset_path=dataset_path, config=config, **dataset_kwargs
    )

    # Create training arguments
    training_args = config.to_training_arguments()

    # Create data collator
    data_collator = RHMDataCollator(
        tokenizer=None,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
        separator_token_id=config.separator_token_id,
    )

    # Create callbacks
    callbacks = [HierarchicalMetricsCallback(eval_dataset, config)]

    # Create trainer
    trainer = RHMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        config=config,
        callbacks=callbacks,
    )

    print("Training pipeline created successfully!")
    return trainer, metadata
