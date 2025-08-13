"""RHM Training Configuration with Version Compatibility"""

import json
import logging
import time
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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


class HierarchicalMetricsCallback(TrainerCallback):
    """Callback to track hierarchical-specific metrics."""

    def __init__(self, config: RHMTrainingConfig, dataloader_metadata: dict[str, t.Any]):
        self.config = config
        self.dataloader_metadata = dataloader_metadata
        self.hierarchical_metrics: list[dict[str, t.Any]] = []

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called after evaluation."""
        if not self.config.track_hierarchical_metrics:
            return

        if state.global_step % self.config.hierarchical_eval_frequency == 0:
            metrics = self._compute_hierarchical_metrics(model, eval_dataloader)
            self.hierarchical_metrics.append({"step": state.global_step, "epoch": state.epoch, "metrics": metrics})

            # Log metrics
            logger.info(f"Hierarchical metrics at step {state.global_step}: {metrics}")

    def _compute_hierarchical_metrics(self, model, dataloader) -> dict[str, float]:
        """Compute metrics specific to hierarchical configurations."""
        model.eval()

        config_losses: dict[str, float] = {}
        config_counts: dict[str, int] = {}

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                device = next(model.parameters()).device

                # Prepare batch for model
                model_inputs = {}
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key in batch:
                        if hasattr(batch[key], "to"):
                            model_inputs[key] = batch[key].to(device)
                        else:
                            model_inputs[key] = batch[key]

                outputs = model(**model_inputs)
                losses = outputs.loss

                # Group by configuration if available
                if "config_L" in batch and "config_m" in batch:
                    for i, (L, m) in enumerate(zip(batch["config_L"], batch["config_m"], strict=False)):
                        config_key = f"L{L}_m{m}"

                        if config_key not in config_losses:
                            config_losses[config_key] = 0.0
                            config_counts[config_key] = 0

                        # Individual sample loss (approximation)
                        if len(losses.shape) == 0:  # Scalar loss
                            sample_loss = losses.item()
                        else:
                            sample_loss = losses[i].item() if len(losses) > i else losses.mean().item()

                        config_losses[config_key] += sample_loss
                        config_counts[config_key] += 1

        # Compute average losses per configuration
        hierarchical_metrics = {}
        for config_key in config_losses:
            avg_loss = config_losses[config_key] / config_counts[config_key]
            hierarchical_metrics[f"loss_{config_key}"] = avg_loss

        model.train()
        return hierarchical_metrics


class CheckpointCallback(TrainerCallback):
    """Enhanced checkpoint callback with hierarchical metadata."""

    def __init__(self, config: RHMTrainingConfig, dataloader_metadata: dict[str, t.Any]):
        self.config = config
        self.dataloader_metadata = dataloader_metadata
        self.checkpoint_history: list[dict[str, t.Any]] = []

    def on_save(self, args, state, control, model, tokenizer, **kwargs):
        """Called when saving checkpoint."""
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        # Save enhanced metadata
        enhanced_metadata = {
            "training_config": self.config.__dict__,
            "dataloader_metadata": self.dataloader_metadata,
            "training_state": {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
                "train_loss": state.log_history[-1].get("train_loss", 0) if state.log_history else 0,
                "eval_loss": state.log_history[-1].get("eval_loss", 0) if state.log_history else 0,
            },
            "model_config": model.config.to_dict() if hasattr(model.config, "to_dict") else str(model.config),
            "timestamp": datetime.now().isoformat(),
        }

        # Save metadata
        metadata_path = checkpoint_dir / "training_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w") as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)

        # Track checkpoint
        self.checkpoint_history.append(
            {
                "step": state.global_step,
                "epoch": state.epoch,
                "path": str(checkpoint_dir),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Enhanced checkpoint saved to {checkpoint_dir}")


class RHMTrainer:
    """Main trainer class for RHM models with version compatibility."""

    def __init__(
        self,
        training_config: RHMTrainingConfig,
        train_dataloader,
        eval_dataloader,
        dataloader_metadata: dict[str, t.Any],
    ):
        """Initialize RHM trainer.

        Args:
            training_config: Training configuration
            train_dataloader: Training DataLoader
            eval_dataloader: Evaluation DataLoader
            dataloader_metadata: Metadata from DataLoader creation

        """
        self.config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.dataloader_metadata = dataloader_metadata

        # Set up output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = self._create_model()

        # Set up training arguments
        self.training_args = self.config.to_training_arguments()

        # Set up callbacks
        self.callbacks = self._setup_callbacks()

        # Initialize trainer placeholder
        self.trainer: Trainer | None = None

        logger.info(f"RHM Trainer initialized for task: {self.config.task_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        logger.info(f"Training on device: {device_name}")

    def _create_model(self):
        """Create model based on task and configuration."""
        # Create model configuration
        if self.config.model_name_or_path:
            # Load from existing model
            model_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
            model_config.vocab_size = self.config.vocab_size
        # Create new configuration
        elif self.config.task_name == "clm":
            from transformers import GPT2Config

            model_config = GPT2Config(
                vocab_size=self.config.vocab_size,
                n_positions=self.config.max_position_embeddings,
                n_embd=self.config.hidden_size,
                n_layer=self.config.num_hidden_layers,
                n_head=self.config.num_attention_heads,
                n_inner=self.config.intermediate_size,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                use_cache=False,  # Disable for training
            )
        elif self.config.task_name == "mlm":
            from transformers import BertConfig

            model_config = BertConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_hidden_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                max_position_embeddings=self.config.max_position_embeddings,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )
        else:
            raise ValueError(f"Unknown task: {self.config.task_name}")

        # Create model
        if self.config.model_name_or_path:
            if self.config.task_name == "clm":
                model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, config=model_config)
            else:
                model = AutoModelForMaskedLM.from_pretrained(self.config.model_name_or_path, config=model_config)
        elif self.config.task_name == "clm":
            model = AutoModelForCausalLM.from_config(model_config)
        else:
            model = AutoModelForMaskedLM.from_config(model_config)

        # Move to device
        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def _setup_callbacks(self) -> list[TrainerCallback]:
        """Set up training callbacks."""
        callbacks = []

        # Hierarchical metrics callback
        if self.config.track_hierarchical_metrics:
            callbacks.append(HierarchicalMetricsCallback(self.config, self.dataloader_metadata))

        # Enhanced checkpoint callback
        callbacks.append(CheckpointCallback(self.config, self.dataloader_metadata))

        # Early stopping callback
        if self.config.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )

        return callbacks

    def train(self) -> dict[str, t.Any]:
        """Run training with improved error handling."""
        logger.info("Starting training...")
        logger.info(f"Task: {self.config.task_name}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataloader.dataset)}")
        logger.info(f"Epochs: {self.config.num_train_epochs}")
        logger.info(f"Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")

        # Save initial configuration
        self._save_training_setup()

        # Initialize trainer with better error handling
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.eval_dataloader.dataset,
            data_collator=getattr(self.train_dataloader, "collate_fn", None),
            callbacks=self.callbacks,
        )

        # Start training
        start_time = time.time()

        try:
            train_result = self.trainer.train()
            training_time = time.time() - start_time

            # Save final model
            self.trainer.save_model()

            # Collect final metrics
            final_metrics = {
                "train_result": train_result.metrics,
                "training_time": training_time,
                "final_eval_metrics": self.trainer.evaluate(),
                "model_size": sum(p.numel() for p in self.model.parameters()),
                "dataset_info": self.dataloader_metadata,
            }

            # Save final metrics
            metrics_path = self.output_dir / "final_metrics.json"
            with metrics_path.open("w") as f:
                json.dump(final_metrics, f, indent=2, default=str)

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final evaluation loss: {final_metrics['final_eval_metrics']['eval_loss']:.4f}")

            return final_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save error information
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
                "training_config": self.config.__dict__,
                "dataloader_metadata": self.dataloader_metadata,
            }

            error_path = self.output_dir / "training_error.json"
            with error_path.open("w") as f:
                json.dump(error_info, f, indent=2, default=str)

            raise

    def _save_training_setup(self) -> None:
        """Save complete training setup for reproducibility."""
        setup_info = {
            "training_config": self.config.__dict__,
            "dataloader_metadata": self.dataloader_metadata,
            "model_config": (
                self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else str(self.model.config)
            ),
            "training_arguments": self.training_args.to_dict(),
            "device_info": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device_name": (torch.cuda.get_device_name() if torch.cuda.is_available() else None),
            },
            "pytorch_version": torch.__version__,
            "timestamp": datetime.now().isoformat(),
        }

        setup_path = self.output_dir / "training_setup.json"
        with setup_path.open("w") as f:
            json.dump(setup_info, f, indent=2, default=str)

        logger.info(f"Training setup saved to {setup_path}")
