"""RHM Training Pipeline with Custom Tokenizer Support"""

import logging
import typing as t
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ICL.train.model import RHMTrainingConfig, prepare_packed_dataset
from ICL.train.tokenizer import RHMTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RHMDataCollator(DataCollatorForLanguageModeling):
    """Data collator for RHM with custom tokenizer and separator token handling."""

    def __init__(
        self,
        tokenizer: RHMTokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        return_tensors: str = "pt",
    ):
        """Initialize RHM data collator.

        Args:
            tokenizer: RHM tokenizer instance
            mlm: Whether to use masked language modeling
            mlm_probability: Probability of masking tokens
            return_tensors: Format of returned tensors

        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            return_tensors=return_tensors,
        )
        self.rhm_tokenizer = tokenizer

    def torch_call(self, examples: list[dict[str, t.Any]]) -> dict[str, torch.Tensor]:
        """Process batch with RHM-specific token handling."""
        # Convert to format expected by parent class
        batch = []
        for example in examples:
            if isinstance(example["input_ids"], list):
                batch.append({"input_ids": torch.tensor(example["input_ids"])})
            else:
                batch.append({"input_ids": example["input_ids"]})

        # Use parent's processing for padding and masking
        result = super().torch_call(batch)

        # Apply RHM-specific masking rules
        if self.mlm and "labels" in result:
            # Never mask special tokens
            special_token_ids = {
                self.rhm_tokenizer.pad_token_id,
                self.rhm_tokenizer.eos_token_id,
                self.rhm_tokenizer.sep_token_id,
                self.rhm_tokenizer.unk_token_id,
            }

            for special_token_id in special_token_ids:
                special_mask = result["input_ids"] == special_token_id
                result["labels"][special_mask] = -100  # Don't compute loss on special tokens

        return result


class RHMTrainer(Trainer):
    """RHM trainer with custom tokenizer support and hierarchical metrics."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        tokenizer: RHMTokenizer | None = None,
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
            tokenizer: RHM tokenizer instance
            data_collator: Data collator
            config: RHM training configuration
            **kwargs: Additional arguments for Trainer

        """
        self.rhm_config = config
        self.rhm_tokenizer = tokenizer

        # Create default data collator if none provided
        if data_collator is None:
            if tokenizer is None:
                raise ValueError("Either tokenizer or data_collator must be provided")

            data_collator = RHMDataCollator(
                tokenizer=tokenizer,
                mlm=config.mlm if config else True,
                mlm_probability=config.mlm_probability if config else 0.15,
            )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,  # Pass tokenizer to parent
            data_collator=data_collator,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with RHM-specific handling."""
        # Standard loss computation with special token masking handled by data collator
        return super().compute_loss(model, inputs, return_outputs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with RHM-specific metrics."""
        # Standard evaluation
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Add RHM-specific metrics if enabled
        if self.rhm_config and self.rhm_config.track_hierarchical_metrics:
            hierarchical_metrics = self._compute_hierarchical_metrics(eval_dataset)
            results.update(hierarchical_metrics)

        return results

    def _compute_hierarchical_metrics(self, eval_dataset=None) -> dict[str, float]:
        """Compute hierarchical-specific metrics."""
        # Placeholder for hierarchical metrics
        # In practice, you'd analyze model predictions on hierarchical patterns
        return {
            "eval_hierarchical_accuracy": 0.0,
            "eval_separator_token_accuracy": 0.0,
            "eval_special_token_perplexity": 0.0,
        }


class HierarchicalMetricsCallback(TrainerCallback):
    """Callback for computing hierarchical-specific metrics during training."""

    def __init__(self, eval_dataset: Dataset, config: RHMTrainingConfig, tokenizer: RHMTokenizer):
        """Initialize metrics callback.

        Args:
            eval_dataset: Evaluation dataset for computing metrics
            config: Training configuration
            tokenizer: RHM tokenizer instance

        """
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.step_count = 0

    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Compute additional metrics during evaluation."""
        if logs is None:
            return

        self.step_count += 1

        # Add hierarchical evaluation metrics
        logs["hierarchical_eval_count"] = self.step_count

        # Example: Track special token usage
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            special_token_norms = {}

            special_tokens = {
                "pad": self.tokenizer.pad_token_id,
                "eos": self.tokenizer.eos_token_id,
                "sep": self.tokenizer.sep_token_id,
                "mask": self.tokenizer.mask_token_id,
            }

            for token_name, token_id in special_tokens.items():
                if token_id < embeddings.num_embeddings:
                    norm = torch.norm(embeddings.weight[token_id]).item()
                    logs[f"special_token_{token_name}_norm"] = norm

        logger.info(f"Hierarchical evaluation #{self.step_count} completed")
        logger.info(f"Current eval loss: {logs.get('eval_loss', 'N/A'):.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Log tokenizer information at training start."""
        logger.info(f"Training with RHM tokenizer (vocab_size: {self.tokenizer.vocab_size})")
        logger.info(
            f"Special tokens - PAD: {self.tokenizer.pad_token_id}, "
            f"EOS: {self.tokenizer.eos_token_id}, "
            f"SEP: {self.tokenizer.sep_token_id}, "
            f"MASK: {self.tokenizer.mask_token_id}"
        )


def create_rhm_training_pipeline(
    dataset_path: str,
    model,
    training_config: RHMTrainingConfig,
    **dataset_kwargs,
) -> tuple[RHMTrainer, dict[str, t.Any]]:
    """Create complete RHM training pipeline with custom tokenizer.

    Args:
        dataset_path: Path to unified RHM dataset
        model: Model to train
        training_config: Training configuration
        **dataset_kwargs: Additional arguments for dataset preparation

    Returns:
        Tuple of (trainer, metadata)

    """
    logger.info("Creating RHM training pipeline with custom tokenizer...")

    # Load training metadata to get config info
    from ICL.datasets.utils import load_training_metadata

    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "metadata.pkl"

    dataset_metadata = None
    config_L = None
    config_m = None

    if metadata_file.exists():
        try:
            dataset_metadata = load_training_metadata(metadata_file)
            # Get first configuration as default
            if dataset_metadata.config_list:
                config_L, config_m = dataset_metadata.config_list[0]
                logger.info(f"Loaded dataset config: L={config_L}, m={config_m}")
            else:
                logger.info("Warning: No configurations found in dataset metadata")
        except Exception as e:
            logger.info(f"Warning: Could not load dataset metadata: {e}")
    else:
        logger.info(f"Warning: Dataset metadata not found at {metadata_file}")

    # Create tokenizer
    tokenizer = training_config.create_tokenizer()
    logger.info(f"Created RHM tokenizer with vocab size: {tokenizer.vocab_size}")

    # Prepare datasets
    train_dataset, eval_dataset, metadata = prepare_packed_dataset(
        dataset_path=str(dataset_path), tokenizer=tokenizer, config=training_config, **dataset_kwargs
    )

    # Calculate n_train from actual dataset
    n_train = len(train_dataset) if train_dataset else 0
    logger.info(f"Training dataset size: {n_train}")

    # Update model vocab size if needed
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(tokenizer.vocab_size)
        logger.info(f"Resized model token embeddings to {tokenizer.vocab_size}")

    # Create training arguments
    training_args = training_config.to_training_arguments()

    # Save tokenizer to output directory
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer to: {output_dir}")

    # Create data collator
    data_collator = RHMDataCollator(
        tokenizer=tokenizer,
        mlm=training_config.mlm,
        mlm_probability=training_config.mlm_probability,
    )

    # Create callbacks
    callbacks = [HierarchicalMetricsCallback(eval_dataset, training_config, tokenizer)]

    # Create trainer
    trainer = RHMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        config=training_config,
        callbacks=callbacks,
    )

    # Enhanced metadata
    enhanced_metadata = {
        "dataset_metadata": metadata,
        "training_metadata": {
            "config_L": config_L,
            "config_m": config_m,
            "n_train": n_train,
            "model_type": "causal_lm" if training_config.task_name == "clm" else "mlm",
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
        },
        "tokenizer_metadata": {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "mask_token_id": tokenizer.mask_token_id,
            "tokenizer_class": "RHMTokenizer",
            "special_tokens": {
                "pad": tokenizer.pad_token,
                "eos": tokenizer.eos_token,
                "sep": tokenizer.sep_token,
                "mask": tokenizer.mask_token,
                "unk": tokenizer.unk_token,
            },
        },
        "model_metadata": {
            "model_vocab_size": model.config.vocab_size if hasattr(model, "config") else None,
            "hidden_size": model.config.hidden_size if hasattr(model, "config") else None,
        },
        "dataset_generation_params": dataset_metadata.generation_params if dataset_metadata else {},
    }

    logger.info("Training pipeline created successfully!")
    logger.info(f"Tokenizer: {tokenizer.__class__.__name__}")
    logger.info(f"Data collator: {data_collator.__class__.__name__}")
    logger.info(f"Trainer: {trainer.__class__.__name__}")

    return trainer, enhanced_metadata
