"""RHM Training Pipeline with Custom Tokenizer Support"""

import logging
import random
import typing as t
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ICL.settings import ModelConfig
from ICL.train.model import RHMTrainingConfig
from ICL.train.seed_dataset_loading import (
    analyze_batching_strategy,
    prepare_seed_based_dataset,
    validate_dataset_structure,
)
from ICL.train.tokenizer import RHMTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RHMTrainer(Trainer):
    """RHM trainer with seed-aware batching and hierarchical metrics."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_seed_datasets: dict[int, Dataset] | None = None,
        eval_seed_datasets: dict[int, Dataset] | None = None,
        tokenizer: RHMTokenizer | None = None,
        config: RHMTrainingConfig | None = None,
        **kwargs,
    ):
        """Initialize RHM trainer with seed-aware datasets.

        Args:
            model: The model to train
            args: Training arguments
            train_seed_datasets: Dict mapping seed -> training Dataset
            eval_seed_datasets: Dict mapping seed -> evaluation Dataset
            tokenizer: RHM tokenizer instance
            config: RHM training configuration
            **kwargs: Additional arguments for Trainer

        """
        self.rhm_config = config
        self.rhm_tokenizer = tokenizer
        self.train_seed_datasets = train_seed_datasets or {}
        self.eval_seed_datasets = eval_seed_datasets or {}

        # We'll create combined datasets for the parent Trainer
        # The actual seed-aware batching is handled by our custom dataloader
        train_dataset = CombinedSeedDataset(train_seed_datasets) if train_seed_datasets else None

        eval_dataset = CombinedSeedDataset(eval_seed_datasets) if eval_seed_datasets else None

        # Initialize parent without data_collator - we'll override get_train_dataloader
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=None,  # We'll create this in get_train_dataloader
            **kwargs,
        )

        # Track seed-specific metrics
        self.seed_metrics = defaultdict(list)
        self.epoch_seed_stats = defaultdict(dict)

    def get_train_dataloader(self):
        """Create training dataloader with seed-aware batching."""
        if self.train_seed_datasets is None or len(self.train_seed_datasets) == 0:
            raise ValueError("No training seed datasets provided")

        logger.info("Creating seed-aware training dataloader...")

        dataloader = create_seed_aware_dataloader(
            seed_datasets=self.train_seed_datasets,
            config=self.rhm_config,
            tokenizer=self.rhm_tokenizer,
            is_training=True,
        )

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        """Create evaluation dataloader with seed-aware batching."""
        if eval_dataset is None and (self.eval_seed_datasets is None or len(self.eval_seed_datasets) == 0):
            return None

        logger.info("Creating seed-aware evaluation dataloader...")

        dataloader = create_seed_aware_dataloader(
            seed_datasets=self.eval_seed_datasets,
            config=self.rhm_config,
            tokenizer=self.rhm_tokenizer,
            is_training=False,
        )

        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with seed tracking."""
        # Extract seed information if available
        seeds = inputs.pop("seeds", None)

        # Standard loss computation
        loss_outputs = super().compute_loss(model, inputs, return_outputs)

        if return_outputs:
            loss, outputs = loss_outputs
        else:
            loss = loss_outputs
            outputs = None

        # Track per-seed loss if seed information is available
        if seeds is not None and hasattr(self, "_current_epoch"):
            self._track_seed_loss(seeds, loss)

        # Return in the same format as parent
        if return_outputs:
            return loss, outputs
        return loss

    def _track_seed_loss(self, seeds: torch.Tensor, loss: torch.Tensor):
        """Track loss per seed for analysis."""
        current_epoch = getattr(self, "_current_epoch", 0)

        # Convert to CPU and detach
        seeds_cpu = seeds.cpu().detach().numpy()
        loss_cpu = loss.cpu().detach().item()

        # Group by seed
        for seed in seeds_cpu:
            seed = int(seed)
            if current_epoch not in self.epoch_seed_stats:
                self.epoch_seed_stats[current_epoch] = {}
            if seed not in self.epoch_seed_stats[current_epoch]:
                self.epoch_seed_stats[current_epoch][seed] = []

            self.epoch_seed_stats[current_epoch][seed].append(loss_cpu)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track epoch beginning for seed-specific metrics."""
        self._current_epoch = int(state.epoch) if state.epoch is not None else 0
        logger.info(f"Starting epoch {self._current_epoch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log seed-specific metrics at epoch end."""
        current_epoch = getattr(self, "_current_epoch", 0)

        if current_epoch in self.epoch_seed_stats:
            logger.info(f"Epoch {current_epoch} seed statistics:")

            for seed, losses in self.epoch_seed_stats[current_epoch].items():
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    logger.info(f"  Seed {seed}: avg_loss={avg_loss:.4f} (n={len(losses)})")

                    # Store for later analysis
                    self.seed_metrics[seed].append(
                        {"epoch": current_epoch, "avg_loss": avg_loss, "num_batches": len(losses)}
                    )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with seed-specific metrics."""
        # Standard evaluation
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Add seed-specific metrics if available
        if self.rhm_config and self.rhm_config.track_hierarchical_metrics:
            hierarchical_metrics = self._compute_hierarchical_metrics()
            results.update(hierarchical_metrics)

        # Add seed-specific evaluation metrics
        current_epoch = getattr(self, "_current_epoch", 0)
        if current_epoch in self.epoch_seed_stats:
            for seed in self.train_seed_datasets.keys():
                if seed in self.epoch_seed_stats[current_epoch]:
                    losses = self.epoch_seed_stats[current_epoch][seed]
                    if losses:
                        results[f"eval_seed_{seed}_loss"] = sum(losses) / len(losses)
                        results[f"eval_seed_{seed}_count"] = len(losses)

        return results

    def _compute_hierarchical_metrics(self) -> dict[str, float]:
        """Compute hierarchical-specific metrics."""
        # Enhanced hierarchical metrics
        metrics = {
            "eval_hierarchical_accuracy": 0.0,
            "eval_separator_token_accuracy": 0.0,
            "eval_special_token_perplexity": 0.0,
        }

        # Add seed diversity metrics
        current_epoch = getattr(self, "_current_epoch", 0)
        if current_epoch in self.epoch_seed_stats:
            seed_losses = []
            for seed in self.train_seed_datasets.keys():
                if seed in self.epoch_seed_stats[current_epoch]:
                    losses = self.epoch_seed_stats[current_epoch][seed]
                    if losses:
                        seed_losses.append(sum(losses) / len(losses))

            if len(seed_losses) > 1:
                import statistics

                metrics["eval_seed_loss_variance"] = statistics.variance(seed_losses)
                metrics["eval_seed_loss_std"] = statistics.stdev(seed_losses)

            metrics["eval_active_seeds"] = len(seed_losses)

        return metrics

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with enhanced metadata including seed information."""
        # Save using parent method
        super().save_model(output_dir, _internal_call)

        # Save seed-specific training metrics
        if output_dir is None:
            output_dir = self.args.output_dir

        output_path = Path(output_dir)

        # Save seed metrics
        seed_metrics_file = output_path / "seed_training_metrics.json"
        import json

        seed_metrics_data = {
            "seed_metrics": dict(self.seed_metrics),
            "epoch_seed_stats": {
                epoch: {
                    seed: {"avg_loss": sum(losses) / len(losses) if losses else 0, "num_batches": len(losses)}
                    for seed, losses in seed_data.items()
                }
                for epoch, seed_data in self.epoch_seed_stats.items()
            },
            "training_config": {
                "seed_balanced_batching": self.rhm_config.seed_balanced_batching,
                "seed_sampling_strategy": self.rhm_config.seed_sampling_strategy,
                "seeds_per_batch": self.rhm_config.seeds_per_batch,
                "shuffle_before_packing": self.rhm_config.shuffle_before_packing,
                "shuffle_strategy": self.rhm_config.shuffle_strategy,
            },
            "available_seeds": list(self.train_seed_datasets.keys()),
        }

        with seed_metrics_file.open("w") as f:
            json.dump(seed_metrics_data, f, indent=2)

        logger.info(f"Saved seed training metrics to: {seed_metrics_file}")


class SeedMetricsCallback(TrainerCallback):
    """Callback for enhanced seed-specific metrics tracking."""

    def __init__(self, config: RHMTrainingConfig, tokenizer: RHMTokenizer):
        """Initialize seed metrics callback."""
        self.config = config
        self.tokenizer = tokenizer
        self.epoch_count = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log epoch beginning with seed configuration."""
        self.epoch_count += 1
        logger.info(f"=== EPOCH {self.epoch_count} BEGIN ===")
        logger.info(f"Seed balanced batching: {self.config.seed_balanced_batching}")
        logger.info(f"Seed sampling strategy: {self.config.seed_sampling_strategy}")
        logger.info(f"Seeds per batch: {self.config.seeds_per_batch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch completion with checkpoint information."""
        logger.info(f"=== EPOCH {self.epoch_count} END ===")
        logger.info(f"Checkpoint saved at: {args.output_dir}")

        # Log current learning rate
        if hasattr(state, "log_history") and state.log_history:
            last_log = state.log_history[-1]
            if "learning_rate" in last_log:
                logger.info(f"Learning rate: {last_log['learning_rate']:.2e}")

    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Enhanced evaluation logging with seed information."""
        if logs is None:
            return

        logger.info(f"Evaluation at epoch {self.epoch_count}:")
        logger.info(f"  Eval loss: {logs.get('eval_loss', 'N/A'):.4f}")

        # Log seed-specific metrics if available
        seed_metrics = {k: v for k, v in logs.items() if k.startswith("eval_seed_")}
        if seed_metrics:
            logger.info("  Seed-specific metrics:")
            for metric, value in seed_metrics.items():
                logger.info(f"    {metric}: {value:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Log training configuration at start."""
        logger.info("=== TRAINING BEGIN ===")
        logger.info("Seed-based training configuration:")
        logger.info(f"  Tokenizer vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"  Seed balanced batching: {self.config.seed_balanced_batching}")
        logger.info(f"  Seed sampling strategy: {self.config.seed_sampling_strategy}")
        logger.info(f"  Cross-config shuffling: {self.config.shuffle_before_packing}")
        logger.info(f"  Shuffle strategy: {self.config.shuffle_strategy}")
        logger.info(f"  Checkpointing: {args.save_strategy} every {args.save_steps}")

    def on_train_end(self, args, state, control, **kwargs):
        """Log training completion."""
        logger.info("=== TRAINING COMPLETE ===")
        logger.info(f"Total epochs completed: {self.epoch_count}")
        logger.info(f"Final model saved to: {args.output_dir}")


class HierarchicalMetricsCallback(TrainerCallback):
    """Callback for computing hierarchical-specific metrics during training."""

    def __init__(self, eval_dataset: Dataset, config: RHMTrainingConfig, tokenizer: RHMTokenizer):
        """Initialize metrics callback."""
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
    model_config: ModelConfig,
    model,
    training_config: RHMTrainingConfig,
    **dataset_kwargs,
) -> tuple[RHMTrainer, dict[str, t.Any]]:
    """Create complete RHM training pipeline with proper config loading and seed-aware batching."""
    logger.info("Creating RHM training pipeline with seed-aware batching...")

    # Get paths from model config
    paths = model_config.get_model_paths()
    dataset_path = paths["dataset_dir"]
    output_dir = paths["model_dir"]

    # Validate dataset structure and extract L,M
    logger.info(f"Validating dataset structure at: {dataset_path}")
    validation = validate_dataset_structure(dataset_path)

    if not validation["valid"]:
        logger.error("Dataset validation failed:")
        for error in validation["errors"]:
            logger.error(f"  ✗ {error}")
        raise RuntimeError(f"Invalid dataset structure at {dataset_path}")

    L, m = validation["L"], validation["m"]
    available_seeds = validation["seed_datasets_found"]
    logger.info(f"✓ Dataset validation passed: L={L}, m={m}, seeds={available_seeds}")

    # Update training config output directory and model naming
    model_suffix = training_config.get_model_suffix()
    enhanced_output_dir = output_dir.parent / f"{output_dir.name}_{model_suffix}"
    training_config.output_dir = str(enhanced_output_dir)

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Configuration: L={L}, m={m}")
    logger.info(f"Enhanced model output path: {enhanced_output_dir}")
    logger.info(f"Model configuration suffix: {model_suffix}")

    # Create tokenizer
    tokenizer = training_config.create_tokenizer()
    logger.info(f"Created RHM tokenizer with vocab size: {tokenizer.vocab_size}")

    # Prepare seed-based datasets
    logger.info("Preparing seed-based datasets...")
    train_seed_datasets, eval_seed_datasets, metadata = prepare_seed_based_dataset(
        dataset_path=str(dataset_path),
        tokenizer=tokenizer,
        config=training_config,
        train_split_ratio=dataset_kwargs.get("train_split_ratio", 0.8),
        max_samples_per_seed=dataset_kwargs.get("max_samples_per_seed"),
    )

    # Calculate total training samples
    total_train_samples = sum(len(dataset) for dataset in train_seed_datasets.values())
    total_eval_samples = sum(len(dataset) for dataset in eval_seed_datasets.values())

    logger.info(f"Training samples: {total_train_samples} across {len(train_seed_datasets)} seeds")
    logger.info(f"Evaluation samples: {total_eval_samples} across {len(eval_seed_datasets)} seeds")

    # Log seed-specific statistics
    for seed in train_seed_datasets.keys():
        train_size = len(train_seed_datasets[seed])
        eval_size = len(eval_seed_datasets[seed])
        logger.info(f"  Seed {seed}: {train_size} train, {eval_size} eval")

    # Validate that loaded seeds match expected seeds
    loaded_seeds = set(train_seed_datasets.keys())
    expected_seeds = set(available_seeds)
    if loaded_seeds != expected_seeds:
        logger.warning(f"Loaded seeds {loaded_seeds} don't match expected seeds {expected_seeds}")

    # Update model vocab size if needed
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(tokenizer.vocab_size)
        logger.info(f"Resized model token embeddings to {tokenizer.vocab_size}")

    # Create training arguments
    training_args = training_config.to_training_arguments()

    # Create enhanced output directory
    enhanced_output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer to enhanced output directory
    tokenizer.save_pretrained(enhanced_output_dir)
    logger.info(f"Saved tokenizer to: {enhanced_output_dir}")

    # Create callbacks
    callbacks = [SeedMetricsCallback(training_config, tokenizer)]

    # Create trainer with seed-aware datasets
    trainer = RHMTrainer(
        model=model,
        args=training_args,
        train_seed_datasets=train_seed_datasets,
        eval_seed_datasets=eval_seed_datasets,
        tokenizer=tokenizer,
        config=training_config,
        callbacks=callbacks,
    )

    # Enhanced metadata with seed and hierarchical info
    enhanced_metadata = {
        "model_config": {
            "dataset_name": model_config.dataset_config.to_name(),
            "model_name": model_config.to_name(),
            "enhanced_model_name": f"{model_config.to_name()}_{model_suffix}",
            "model_type": model_config.model_type,
            "dataset_path": str(dataset_path),
            "model_output_path": str(enhanced_output_dir),
            "model_suffix": model_suffix,
            # Add discovered L,M information
            "discovered_L": L,
            "discovered_m": m,
        },
        "seed_dataset_metadata": metadata,
        "dataset_validation": validation,  # Include validation results
        "training_metadata": {
            "total_train_samples": total_train_samples,
            "total_eval_samples": total_eval_samples,
            "available_seeds": list(train_seed_datasets.keys()),
            "expected_seeds": available_seeds,
            "num_seeds": len(train_seed_datasets),
            "train_samples_per_seed": {seed: len(dataset) for seed, dataset in train_seed_datasets.items()},
            "eval_samples_per_seed": {seed: len(dataset) for seed, dataset in eval_seed_datasets.items()},
            "model_type": "causal_lm" if training_config.task_name == "clm" else "mlm",
            "dataset_path": str(dataset_path),
            "output_dir": str(enhanced_output_dir),
            "enhanced_naming": True,
            # Configuration parameters
            "config_L": L,
            "config_m": m,
            # Seed-based batching configuration
            "seed_balanced_batching": training_config.seed_balanced_batching,
            "seed_sampling_strategy": training_config.seed_sampling_strategy,
            "seeds_per_batch": training_config.seeds_per_batch,
            # Cross-configuration shuffling
            "shuffle_before_packing": training_config.shuffle_before_packing,
            "shuffle_strategy": training_config.shuffle_strategy if training_config.shuffle_before_packing else None,
            # Checkpointing strategy
            "save_strategy": training_config.save_strategy,
            "evaluation_strategy": training_config.evaluation_strategy,
        },
        "tokenizer_metadata": {
            "vocab_size": tokenizer.vocab_size,
            "rhm_vocab_size": tokenizer.rhm_vocab_size,
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
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
        "dataset_generation_params": metadata.get("dataset_metadata", {}),
        "batching_analysis": analyze_batching_strategy(train_seed_datasets, training_config),
    }

    logger.info("Training pipeline created successfully!")
    logger.info(f"Trainer: {trainer.__class__.__name__} with seed-aware batching")
    logger.info(f"Enhanced model name: {enhanced_metadata['model_config']['enhanced_model_name']}")
    logger.info(f"Configuration discovered: L={L}, m={m}")
    logger.info(f"Checkpointing: {training_config.save_strategy} every {training_config.save_steps}")
    logger.info(f"Evaluation: {training_config.evaluation_strategy} every {training_config.eval_steps}")

    return trainer, enhanced_metadata


##########################################


class SeedBalancedSampler(Sampler):
    """Sampler that ensures balanced representation of seeds within each batch."""

    def __init__(
        self,
        seed_datasets: dict[int, Dataset],
        batch_size: int,
        seed_sampling_strategy: str = "balanced",
        seeds_per_batch: int | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Initialize seed-balanced sampler.

        Args:
            seed_datasets: Dict mapping seed -> Dataset
            batch_size: Target batch size
            seed_sampling_strategy: How to sample seeds ("balanced", "random", "weighted")
            seeds_per_batch: Number of seeds per batch (None = all seeds)
            shuffle: Whether to shuffle within seeds
            generator: Random number generator

        """
        self.seed_datasets = seed_datasets
        self.batch_size = batch_size
        self.seed_sampling_strategy = seed_sampling_strategy
        self.seeds_per_batch = seeds_per_batch or len(seed_datasets)
        self.shuffle = shuffle
        self.generator = generator

        # Validate parameters
        if self.seeds_per_batch > len(seed_datasets):
            self.seeds_per_batch = len(seed_datasets)
            logger.warning(f"seeds_per_batch reduced to {self.seeds_per_batch} (available seeds)")

        if batch_size % self.seeds_per_batch != 0:
            logger.warning(
                f"batch_size ({batch_size}) not divisible by seeds_per_batch ({self.seeds_per_batch}). "
                f"Some batches may have uneven seed distribution."
            )

        # Create global index mapping
        self.global_indices = []
        self.seed_ranges = {}
        current_idx = 0

        for seed, dataset in seed_datasets.items():
            start_idx = current_idx
            end_idx = current_idx + len(dataset)
            self.seed_ranges[seed] = (start_idx, end_idx)

            # Add indices with seed information
            for local_idx in range(len(dataset)):
                self.global_indices.append((seed, local_idx, current_idx))
                current_idx += 1

        self.total_size = len(self.global_indices)
        logger.info(
            f"SeedBalancedSampler initialized: {self.total_size} total samples across {len(seed_datasets)} seeds"
        )

    def __iter__(self):
        """Generate indices for seed-balanced batching."""
        available_seeds = list(self.seed_datasets.keys())

        # Create per-seed iterators
        seed_iterators = {}
        for seed in available_seeds:
            indices = list(range(len(self.seed_datasets[seed])))
            if self.shuffle:
                if self.generator is not None:
                    generator_state = self.generator.get_state()
                    torch.manual_seed(hash((seed, generator_state)) % (2**32))
                random.shuffle(indices)
            seed_iterators[seed] = iter(indices)

        # Generate batches
        samples_per_seed = self.batch_size // self.seeds_per_batch
        remainder = self.batch_size % self.seeds_per_batch

        batch = []
        active_seeds = available_seeds.copy()

        while active_seeds:
            # Select seeds for this batch
            if self.seed_sampling_strategy == "balanced":
                # Use all available seeds in round-robin fashion
                selected_seeds = active_seeds[: self.seeds_per_batch]
            elif self.seed_sampling_strategy == "random":
                # Randomly select seeds
                if len(active_seeds) >= self.seeds_per_batch:
                    selected_seeds = random.sample(active_seeds, self.seeds_per_batch)
                else:
                    selected_seeds = active_seeds
            else:  # weighted or fallback
                # For now, fallback to balanced
                selected_seeds = active_seeds[: self.seeds_per_batch]

            # Sample from each selected seed
            batch_seeds_used = set()
            for i, seed in enumerate(selected_seeds):
                target_samples = samples_per_seed
                if i < remainder:  # Distribute remainder
                    target_samples += 1

                samples_added = 0
                try:
                    for _ in range(target_samples):
                        local_idx = next(seed_iterators[seed])
                        global_idx = self.seed_ranges[seed][0] + local_idx
                        batch.append(global_idx)
                        samples_added += 1
                        batch_seeds_used.add(seed)

                except StopIteration:
                    # This seed is exhausted
                    if seed in active_seeds:
                        active_seeds.remove(seed)
                    logger.debug(f"Seed {seed} exhausted after contributing {samples_added} samples to current batch")

            # Yield batch when full or when we can't fill it anymore
            if len(batch) >= self.batch_size or not active_seeds:
                if batch:
                    logger.debug(f"Yielding batch of size {len(batch)} with seeds: {batch_seeds_used}")
                    yield from batch
                    batch = []

        # Yield any remaining samples
        if batch:
            logger.debug(f"Yielding final batch of size {len(batch)}")
            yield from batch

    def __len__(self):
        """Return total number of samples."""
        return self.total_size


class CombinedSeedDataset:
    """Dataset that combines multiple seed datasets while preserving seed information."""

    def __init__(self, seed_datasets: dict[int, Dataset]):
        """Initialize combined dataset.

        Args:
            seed_datasets: Dict mapping seed -> Dataset

        """
        self.seed_datasets = seed_datasets

        # Create a single combined dataset with seed information
        all_items = []
        for seed, dataset in seed_datasets.items():
            for item in dataset:
                # Add seed information to each item
                if isinstance(item, dict):
                    item_with_seed = item.copy()
                    item_with_seed["seed"] = seed
                else:
                    item_with_seed = {"data": item, "seed": seed}
                all_items.append(item_with_seed)

        # Create HuggingFace Dataset from the combined items
        self.combined_dataset = Dataset.from_list(all_items)
        self.total_size = len(self.combined_dataset)

        logger.info(f"CombinedSeedDataset: {self.total_size} total samples from {len(seed_datasets)} seeds")

    def __len__(self):
        """Return total number of samples."""
        return self.total_size

    def __getitem__(self, idx):
        """Get item by global index - delegates to HuggingFace Dataset."""
        return self.combined_dataset[idx]

    def select(self, indices):
        """Select subset of items by indices."""
        return self.combined_dataset.select(indices)

    @property
    def column_names(self):
        """Get column names from the underlying dataset."""
        return self.combined_dataset.column_names

    @property
    def features(self):
        """Get features from the underlying dataset."""
        return self.combined_dataset.features


class SeedAwareDataCollator(DataCollatorForLanguageModeling):
    """Data collator that handles seed-based batching while preserving packed sequence integrity."""

    def __init__(
        self,
        tokenizer: RHMTokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        return_tensors: str = "pt",
        seed_balanced_batching: bool = True,
        seed_sampling_strategy: str = "balanced",
    ):
        """Initialize seed-aware data collator."""
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            return_tensors=return_tensors,
        )
        self.rhm_tokenizer = tokenizer
        self.seed_balanced_batching = seed_balanced_batching
        self.seed_sampling_strategy = seed_sampling_strategy

    def torch_call(self, examples: list[dict[str, t.Any]]) -> dict[str, torch.Tensor]:
        """Process batch with seed-aware handling and RHM-specific token masking."""
        # Log seed composition of the batch
        if "seed" in examples[0]:
            seed_counts = defaultdict(int)
            for example in examples:
                seed_counts[example["seed"]] += 1

            logger.debug(f"Batch seed composition: {dict(seed_counts)}")

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

        # Add seed information to the batch if available
        if "seed" in examples[0]:
            result["seeds"] = torch.tensor([example["seed"] for example in examples])

        return result


class SeedBalancedSampler(Sampler):
    """Sampler that ensures balanced representation of seeds within each batch."""

    def __init__(
        self,
        seed_datasets: dict[int, Dataset],
        batch_size: int,
        seed_sampling_strategy: str = "balanced",
        seeds_per_batch: int | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Initialize seed-balanced sampler."""
        self.seed_datasets = seed_datasets
        self.batch_size = batch_size
        self.seed_sampling_strategy = seed_sampling_strategy
        self.seeds_per_batch = seeds_per_batch or len(seed_datasets)
        self.shuffle = shuffle
        self.generator = generator

        # Validate parameters
        if self.seeds_per_batch > len(seed_datasets):
            self.seeds_per_batch = len(seed_datasets)
            logger.warning(f"seeds_per_batch reduced to {self.seeds_per_batch} (available seeds)")

        if batch_size % self.seeds_per_batch != 0:
            logger.warning(
                f"batch_size ({batch_size}) not divisible by seeds_per_batch ({self.seeds_per_batch}). "
                f"Some batches may have uneven seed distribution."
            )

        # Create global index mapping
        self.global_indices = []
        self.seed_ranges = {}
        current_idx = 0

        for seed, dataset in seed_datasets.items():
            start_idx = current_idx
            end_idx = current_idx + len(dataset)
            self.seed_ranges[seed] = (start_idx, end_idx)

            # Add indices with seed information
            for local_idx in range(len(dataset)):
                self.global_indices.append((seed, local_idx, current_idx))
                current_idx += 1

        self.total_size = len(self.global_indices)
        logger.info(
            f"SeedBalancedSampler initialized: {self.total_size} total samples across {len(seed_datasets)} seeds"
        )

    def __iter__(self):
        """Generate indices for seed-balanced batching."""
        available_seeds = list(self.seed_datasets.keys())

        # Create per-seed iterators
        seed_iterators = {}
        for seed in available_seeds:
            indices = list(range(len(self.seed_datasets[seed])))
            if self.shuffle:
                if self.generator is not None:
                    generator_state = self.generator.get_state()
                    torch.manual_seed(hash((seed, generator_state)) % (2**32))
                random.shuffle(indices)
            seed_iterators[seed] = iter(indices)

        # Generate batches
        samples_per_seed = self.batch_size // self.seeds_per_batch
        remainder = self.batch_size % self.seeds_per_batch

        batch = []
        active_seeds = available_seeds.copy()

        while active_seeds:
            # Select seeds for this batch
            if self.seed_sampling_strategy == "balanced":
                # Use all available seeds in round-robin fashion
                selected_seeds = active_seeds[: self.seeds_per_batch]
            elif self.seed_sampling_strategy == "random":
                # Randomly select seeds
                if len(active_seeds) >= self.seeds_per_batch:
                    selected_seeds = random.sample(active_seeds, self.seeds_per_batch)
                else:
                    selected_seeds = active_seeds
            else:  # weighted or fallback
                # For now, fallback to balanced
                selected_seeds = active_seeds[: self.seeds_per_batch]

            # Sample from each selected seed
            batch_seeds_used = set()
            for i, seed in enumerate(selected_seeds):
                target_samples = samples_per_seed
                if i < remainder:  # Distribute remainder
                    target_samples += 1

                samples_added = 0
                try:
                    for _ in range(target_samples):
                        local_idx = next(seed_iterators[seed])
                        global_idx = self.seed_ranges[seed][0] + local_idx
                        batch.append(global_idx)
                        samples_added += 1
                        batch_seeds_used.add(seed)

                except StopIteration:
                    # This seed is exhausted
                    if seed in active_seeds:
                        active_seeds.remove(seed)
                    logger.debug(f"Seed {seed} exhausted after contributing {samples_added} samples to current batch")

            # Yield batch when full or when we can't fill it anymore
            if len(batch) >= self.batch_size or not active_seeds:
                if batch:
                    logger.debug(f"Yielding batch of size {len(batch)} with seeds: {batch_seeds_used}")
                    yield from batch
                    batch = []

        # Yield any remaining samples
        if batch:
            logger.debug(f"Yielding final batch of size {len(batch)}")
            yield from batch

    def __len__(self):
        """Return total number of samples."""
        return self.total_size


def create_seed_aware_dataloader(
    seed_datasets: dict[int, Dataset],
    config: RHMTrainingConfig,
    tokenizer: RHMTokenizer,
    is_training: bool = True,
) -> DataLoader:
    """Create a dataloader with seed-aware batching."""
    # Combine seed datasets
    combined_dataset = CombinedSeedDataset(seed_datasets)

    # Create data collator
    data_collator = SeedAwareDataCollator(
        tokenizer=tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
        seed_balanced_batching=config.seed_balanced_batching,
        seed_sampling_strategy=config.seed_sampling_strategy,
    )

    # Determine batch size
    batch_size = config.per_device_train_batch_size if is_training else config.per_device_eval_batch_size
    # Create sampler if seed-balanced batching is enabled
    sampler = None
    shuffle = False

    if config.seed_balanced_batching and is_training:
        sampler = SeedBalancedSampler(
            seed_datasets=seed_datasets,
            batch_size=batch_size,
            seed_sampling_strategy=config.seed_sampling_strategy,
            seeds_per_batch=config.seeds_per_batch,
            shuffle=True,
        )
        shuffle = False  # Don't shuffle when using custom sampler
    else:
        shuffle = is_training  # Use standard shuffling for eval or when seed batching disabled

    # Create DataLoader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )

    logger.info(f"Created {'training' if is_training else 'evaluation'} dataloader:")
    logger.info(f"  Total samples: {len(combined_dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Seed balanced batching: {config.seed_balanced_batching}")
    logger.info(f"  Seed sampling strategy: {config.seed_sampling_strategy}")
    logger.info(f"  Seeds per batch: {config.seeds_per_batch}")
    logger.info(f"  Number of seeds: {len(seed_datasets)}")

    return dataloader
