"""Enhanced evaluator with minimal model config integration."""

import json
import logging
import time
import typing as t
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import psutil
import torch
from tqdm import tqdm

from ICL.eval.collection.ckpt_manager import CheckpointManager
from ICL.eval.collection.data_schema import (
    AttentionRecord,
    DataSchemaManager,
    ICLPerformanceRecord,
    ModelMetadata,
    determine_training_phase,
)
from ICL.eval.collection.eval_utils import (
    create_control_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class CollectionProgress:
    """Track collection progress for resumption."""

    completed_models: list[str] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    current_model_idx: int = 0
    total_models: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_save_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "completed_models": self.completed_models,
            "failed_models": self.failed_models,
            "current_model_idx": self.current_model_idx,
            "total_models": self.total_models,
            "start_time": self.start_time.isoformat(),
            "last_save_time": self.last_save_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "CollectionProgress":
        """Load from dictionary."""
        return cls(
            completed_models=data.get("completed_models", []),
            failed_models=data.get("failed_models", []),
            current_model_idx=data.get("current_model_idx", 0),
            total_models=data.get("total_models", 0),
            start_time=datetime.fromisoformat(data.get("start_time", datetime.now().isoformat())),
            last_save_time=datetime.fromisoformat(data.get("last_save_time", datetime.now().isoformat())),
        )


class CollectionEvaluator:
    """Enhanced evaluator with minimal model config integration."""

    def __init__(self, config):
        """Initialize enhanced evaluator."""
        self.config = config
        self.config.validate()
        self.config.create_output_structure()

        self.checkpoint_manager = CheckpointManager(config)
        self.evaluation_engine = ICLEvaluationEngine(config.device)
        self.evaluation_dataset = None

        # Progress tracking
        self.progress = CollectionProgress()
        self.consecutive_failures = 0

        # Results storage
        self.all_results: list[ICLPerformanceRecord] = []
        self.attention_records: list[AttentionRecord] = []

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for collection phase."""
        log_dir = self.config.output_dir / "logs"
        log_file = log_dir / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

        logger.info(f"Collection evaluator initialized. Logs: {log_file}")

    def load_evaluation_dataset(self) -> dict[str, t.Any]:
        """Load HuggingFace evaluation dataset and convert to internal format."""
        logger.info(f"Loading evaluation dataset from {self.config.eval_dataset_path}")

        from datasets import load_from_disk

        # Load HuggingFace dataset
        hf_dataset = load_from_disk(str(self.config.eval_dataset_path))

        # Convert to internal JSON format
        converted_dataset = self._convert_hf_to_internal_format(hf_dataset)

        self.evaluation_dataset = converted_dataset

        # Print dataset summary
        total_sequences = len(hf_dataset)
        logger.info(
            f"Loaded evaluation dataset with {total_sequences} sequences for eval_type: {self.config.eval_type}"
        )

        return converted_dataset

    def _convert_hf_to_internal_format(self, hf_dataset) -> dict[str, t.Any]:
        """Convert HuggingFace dataset to internal JSON format."""
        # Group sequences by context size for easier access
        sequences_by_k = defaultdict(list)

        for i, example in enumerate(hf_dataset):
            # Extract sequence data
            sequence = {
                "context_features": example.get("context_features", []),
                "context_labels": example.get("context_labels", []),
                "query_features": example.get("query_features", []),
                "query_label": example.get("query_label", 0),
                "context_size": example.get("context_size", len(example.get("context_features", []))),
                "sequence_id": i,
                # Evaluation type tracking
                "eval_type": self.config.eval_type,
                "appears_in_training": self._determine_training_appearance(example, self.config.eval_type),
                "source_seeds": example.get("source_seeds", []),
                # Target configuration
                "target_config_L": example.get("target_config_L", self.config.config_L),
                "target_config_m": example.get("target_config_m", self.config.config_m),
            }

            context_size = sequence["context_size"]
            sequences_by_k[context_size].append(sequence)

        # Create internal format
        converted_dataset = {
            "metadata": {
                "dataset_type": self.config.dataset_type,
                "num_seeds": self.config.num_seeds,
                "seed": self.config.seed,
                "config_L": self.config.config_L,
                "config_m": self.config.config_m,
                "eval_type": self.config.eval_type,
                "model_variant": self.config.model_variant,
                "total_sequences": len(hf_dataset),
                "conversion_timestamp": datetime.now().isoformat(),
            },
            "sequences": dict(sequences_by_k),
        }

        return converted_dataset

    def _determine_training_appearance(self, example: dict, eval_type: str) -> bool:
        """Determine if sequence appeared in training based on eval_type."""
        if eval_type == "memorization":
            return True
        if eval_type in ["id_generalization", "ood_same_rule", "ood_transfer"]:
            return False
        # Check if explicitly marked in the dataset
        return example.get("appears_in_training", False)

    def discover_and_validate_checkpoints(self) -> list[ModelMetadata]:
        """Discover and validate checkpoints for the model variant."""
        logger.info(f"Discovering checkpoints for model variant: {self.config.model_variant}")

        all_metadata = self.checkpoint_manager.discover_checkpoints()

        if not all_metadata:
            raise RuntimeError(f"No valid checkpoints found for {self.config.model_variant}")

        # Validate completeness
        validation_results = self.checkpoint_manager.validate_checkpoint_completeness(all_metadata)
        self._log_validation_results(validation_results)

        return all_metadata

    def _log_validation_results(self, validation_results: dict[str, t.Any]) -> None:
        """Log checkpoint validation results."""
        logger.info("\nCheckpoint Validation Results:")
        logger.info(f"  Model variant: {validation_results['model_variant']}")
        logger.info(f"  Total checkpoints: {validation_results['total_checkpoints']}")

        step_coverage = validation_results.get("step_coverage", {})
        if step_coverage:
            logger.info(f"  Step range: {step_coverage['min_step']} - {step_coverage['max_step']}")
            logger.info(f"  Step count: {step_coverage['step_count']}")

        training_phases = validation_results.get("training_phases", {})
        if training_phases:
            logger.info("  Training phase distribution:")
            for phase, count in training_phases.items():
                logger.info(f"    {phase}: {count} checkpoints")

    def check_resume_capability(self, checkpoint_metadata: list[ModelMetadata]) -> bool:
        """Check if we can resume from previous run."""
        resume_path = self.config.get_resume_path()

        if not resume_path.exists():
            return False

        try:
            with open(resume_path) as f:
                resume_data = json.load(f)

            # Load progress
            self.progress = CollectionProgress.from_dict(resume_data["progress"])

            # Load existing results if available
            results_path = self.config.output_dir / "intermediate" / "partial_results.parquet"
            if results_path.exists():
                import pandas as pd

                df = pd.read_parquet(results_path)
                logger.info(f"Loaded {len(df)} existing evaluation results")

            logger.info(f"Resuming from model {self.progress.current_model_idx}/{self.progress.total_models}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load resume data: {e}")
            return False

    def save_resume_checkpoint(self) -> None:
        """Save current progress for resumption."""
        resume_path = self.config.get_resume_path()

        resume_data = {
            "progress": self.progress.to_dict(),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(resume_path, "w") as f:
            json.dump(resume_data, f, indent=2)

        self.progress.last_save_time = datetime.now()

    def run_comprehensive_collection(self) -> dict[str, t.Any]:
        """Run the complete collection pipeline with resume capability."""
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("STARTING ICL COLLECTION FOR SINGLE EVAL TYPE")
        logger.info("=" * 80)
        logger.info(f"Start time: {start_time}")
        logger.info(
            f"Dataset: {self.config.dataset_type}_{self.config.num_seeds}_L{self.config.config_L}_M{self.config.config_m}"
        )
        logger.info(f"Model variant: {self.config.model_variant}")
        logger.info(f"Evaluation type: {self.config.eval_type}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Device: {self.config.device}")

        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset()

        # Discover checkpoints
        checkpoint_metadata = self.discover_and_validate_checkpoints()
        self.progress.total_models = len(checkpoint_metadata)

        # Check for resume capability
        resumed = False
        if hasattr(self.config, "resume") and self.config.resume:
            resumed = self.check_resume_capability(checkpoint_metadata)

        if not resumed:
            logger.info("Starting fresh collection run")
            self.progress = CollectionProgress(total_models=len(checkpoint_metadata))

        # Create evaluation manifest
        manifest = self._create_evaluation_manifest(start_time)
        manifest_path = self.config.output_dir / "metadata" / "experiment_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save model registry
        model_registry_df = DataSchemaManager.metadata_to_dataframe(checkpoint_metadata)
        registry_path = self.config.output_dir / "metadata" / "model_registry.parquet"
        model_registry_df.to_parquet(registry_path, index=False)
        logger.info(f"Saved model registry: {registry_path}")

        # Run evaluations with progress tracking
        try:
            self._run_evaluation_loop(checkpoint_metadata, eval_dataset)
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user. Saving progress...")
            self.save_resume_checkpoint()
            self._save_intermediate_results()
            raise
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            self.save_resume_checkpoint()
            self._save_intermediate_results()
            raise

        # Save final results
        logger.info("Saving final results...")
        results_summary = self._save_final_results(checkpoint_metadata)

        # Update manifest
        end_time = datetime.now()
        manifest["end_time"] = end_time.isoformat()
        manifest["status"] = "completed"
        manifest["results_summary"] = results_summary

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Clean up resume checkpoint on successful completion
        resume_path = self.config.get_resume_path()
        if resume_path.exists():
            resume_path.unlink()

        logger.info("=" * 80)
        logger.info("COLLECTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Duration: {end_time - start_time}")
        logger.info(f"Total evaluations: {len(self.all_results)}")
        logger.info(f"Results saved to: {self.config.output_dir}")

        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_evaluations": len(self.all_results),
            "total_models": len(checkpoint_metadata),
            "results_summary": results_summary,
            "output_dir": self.config.output_dir,
        }

    def _create_evaluation_manifest(self, start_time: datetime) -> dict[str, t.Any]:
        """Create evaluation manifest with minimal config tracking."""
        return {
            "experiment_id": f"{self.config.get_shared_identifier()}_{self.config.model_variant}_{self.config.eval_type}_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "dataset_type": self.config.dataset_type,
            "num_seeds": self.config.num_seeds,
            "seed": self.config.seed,
            "config_L": self.config.config_L,
            "config_m": self.config.config_m,
            "model_variant": self.config.model_variant,
            "eval_type": self.config.eval_type,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "status": "running",
            "config": {
                "context_sizes": self.config.context_sizes,
                "control_types": self.config.control_types,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "max_sequences_per_condition": self.config.max_sequences_per_condition,
                "capture_attention": self.config.capture_attention,
            },
            "data_schema_version": "3.0",
            "output_files": {
                "icl_performance": "raw_evaluations/icl_performance.parquet",
                "model_registry": "metadata/model_registry.parquet",
                "attention_data": "raw_evaluations/attention_data/" if self.config.capture_attention else None,
            },
        }

    def _run_evaluation_loop(self, checkpoint_metadata: list[ModelMetadata], eval_dataset: dict[str, t.Any]) -> None:
        """Run the main evaluation loop with error handling."""
        models_to_evaluate = checkpoint_metadata[self.progress.current_model_idx :]

        with tqdm(
            models_to_evaluate,
            desc=f"Evaluating {self.config.eval_type}",
            initial=self.progress.current_model_idx,
            total=self.progress.total_models,
        ) as pbar:
            for model_idx, metadata in enumerate(models_to_evaluate, start=self.progress.current_model_idx):
                # Check if model already completed
                if metadata.model_id in self.progress.completed_models:
                    pbar.update(1)
                    continue

                # Check memory usage
                self._check_memory_usage()

                pbar.set_description(f"Evaluating {metadata.model_id}")
                logger.info(f"[{model_idx + 1}/{self.progress.total_models}] Evaluating {metadata.model_id}")

                try:
                    # Evaluate single model
                    model_results, model_attention = self._evaluate_single_model_safe(metadata, eval_dataset)

                    self.all_results.extend(model_results)
                    self.attention_records.extend(model_attention)

                    # Mark as completed
                    self.progress.completed_models.append(metadata.model_id)
                    self.progress.current_model_idx = model_idx + 1
                    self.consecutive_failures = 0

                    # Save intermediate results periodically
                    if (model_idx + 1) % self.config.intermediate_save_frequency == 0:
                        self._save_intermediate_results()
                        self.save_resume_checkpoint()

                    # Clear cache periodically
                    if (model_idx + 1) % self.config.clear_cache_frequency == 0:
                        self._clear_memory_cache()

                except Exception as e:
                    logger.error(f"Failed to evaluate model {metadata.model_id}: {e}")

                    # Track failures
                    self.progress.failed_models.append(metadata.model_id)
                    self.consecutive_failures += 1

                    # Check if we should stop due to too many failures
                    if self.consecutive_failures >= self.config.max_model_failures:
                        logger.error(f"Too many consecutive failures ({self.consecutive_failures}). Stopping.")
                        break

                    # Retry if configured
                    if self.config.retry_failed_models:
                        logger.info(f"Retrying model {metadata.model_id} in {self.config.failure_retry_delay}s...")
                        time.sleep(self.config.failure_retry_delay)

                        try:
                            model_results, model_attention = self._evaluate_single_model_safe(metadata, eval_dataset)
                            self.all_results.extend(model_results)
                            self.attention_records.extend(model_attention)
                            self.progress.completed_models.append(metadata.model_id)
                            self.consecutive_failures = 0
                            logger.info(f"Retry successful for {metadata.model_id}")
                        except Exception as retry_e:
                            logger.error(f"Retry failed for {metadata.model_id}: {retry_e}")

                    # Continue with next model
                    self.progress.current_model_idx = model_idx + 1

                pbar.update(1)

    def _evaluate_single_model_safe(
        self, metadata: ModelMetadata, eval_dataset: dict[str, t.Any]
    ) -> tuple[list[ICLPerformanceRecord], list[AttentionRecord]]:
        """Safely evaluate a single model with proper cleanup."""
        model = None
        tokenizer = None

        try:
            # Load model
            model, tokenizer = self.checkpoint_manager.load_model_checkpoint(metadata)

            # Evaluate on this specific eval_type
            model_results, model_attention = self._evaluate_single_model(model, tokenizer, metadata, eval_dataset)

            return model_results, model_attention

        finally:
            # Ensure cleanup even if evaluation fails
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _evaluate_single_model(
        self, model: t.Any, tokenizer: t.Any, metadata: ModelMetadata, eval_dataset: dict[str, t.Any]
    ) -> tuple[list[ICLPerformanceRecord], list[AttentionRecord]]:
        """Evaluate a single model on the specified eval_type."""
        results = []
        attention_records = []

        # Get sequences grouped by context size
        sequences_by_k = eval_dataset.get("sequences", {})

        # Evaluate across context sizes and control types
        for context_size in self.config.context_sizes:
            context_sequences = sequences_by_k.get(context_size, [])

            if not context_sequences:
                logger.warning(f"No sequences found for context size {context_size}")
                continue

            # Limit sequences if specified
            if self.config.max_sequences_per_condition > 0:
                context_sequences = context_sequences[: self.config.max_sequences_per_condition]

            for control_type in self.config.control_types:
                for seq_idx, base_sequence in enumerate(context_sequences):
                    try:
                        # Create control sequence
                        sequence = create_control_sequence(base_sequence, control_type, context_sequences)

                        # Evaluate sequence
                        is_correct, attention_data = self.evaluation_engine.evaluate_icl_sequence(
                            model, tokenizer, sequence, capture_attention=self.config.capture_attention
                        )

                        # Extract target configuration
                        target_config_L = sequence.get("target_config_L", metadata.config_L)
                        target_config_m = sequence.get("target_config_m", metadata.config_m)

                        # Determine training phase
                        max_step = max(meta.checkpoint_step for meta in [metadata])  # Single model context
                        training_phase = (
                            determine_training_phase(metadata.checkpoint_step, max_step) if max_step > 0 else "unknown"
                        )

                        # Create performance record with minimal model config
                        record = ICLPerformanceRecord(
                            dataset_type=metadata.dataset_type,
                            num_seeds=metadata.num_seeds,
                            seed=metadata.seed,
                            config_L=metadata.config_L,
                            config_m=metadata.config_m,
                            task_name=metadata.task_name,
                            model_variant=metadata.model_variant,
                            checkpoint_step=metadata.checkpoint_step,
                            model_id=metadata.model_id,
                            eval_type=self.config.eval_type,
                            context_size=context_size,
                            control_type=control_type,
                            sequence_id=seq_idx,
                            target_config_L=target_config_L,
                            target_config_m=target_config_m,
                            source_seeds=sequence.get("source_seeds", []),
                            appears_in_training=sequence.get("appears_in_training", False),
                            accuracy=float(is_correct),
                            num_correct=int(is_correct),
                            evaluation_timestamp=datetime.now(),
                            training_phase=training_phase,
                            shuffle_before_packing=metadata.shuffle_before_packing,
                            seed_balanced_batching=metadata.seed_balanced_batching,
                        )

                        results.append(record)

                        # Store attention data if captured
                        if attention_data and self.config.capture_attention:
                            for attention_key, attention_matrix in attention_data.items():
                                layer_idx, head_idx = self._parse_attention_key(attention_key)

                                attention_record = AttentionRecord(
                                    dataset_type=metadata.dataset_type,
                                    num_seeds=metadata.num_seeds,
                                    seed=metadata.seed,
                                    config_L=metadata.config_L,
                                    config_m=metadata.config_m,
                                    model_variant=metadata.model_variant,
                                    checkpoint_step=metadata.checkpoint_step,
                                    model_id=metadata.model_id,
                                    eval_type=self.config.eval_type,
                                    layer_idx=layer_idx,
                                    head_idx=head_idx,
                                    context_size=context_size,
                                    sequence_id=seq_idx,
                                    attention_matrix=attention_matrix,
                                    evaluation_timestamp=datetime.now(),
                                )

                                attention_records.append(attention_record)

                    except Exception as e:
                        logger.warning(f"Failed to evaluate sequence {seq_idx}: {e}")
                        continue

        return results, attention_records

    def _parse_attention_key(self, attention_key: str) -> tuple[int, int]:
        """Parse layer and head indices from attention key."""
        parts = attention_key.split("_")
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        return layer_idx, head_idx

    def _check_memory_usage(self) -> None:
        """Check and log memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB

            if gpu_allocated > self.config.max_memory_usage_gb:
                logger.warning(f"High GPU memory usage: {gpu_allocated:.2f}GB allocated")
                torch.cuda.empty_cache()

        # Check system memory
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 80:
            logger.warning(f"High system memory usage: {system_memory.percent:.1f}%")

    def _clear_memory_cache(self) -> None:
        """Clear memory caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear checkpoint manager cache
        self.checkpoint_manager.clear_cache()

        logger.info("Cleared memory caches")

    def _save_intermediate_results(self) -> None:
        """Save intermediate results."""
        if self.all_results:
            df = DataSchemaManager.records_to_dataframe(self.all_results)
            intermediate_path = self.config.output_dir / "intermediate" / "partial_results.parquet"
            df.to_parquet(intermediate_path, index=False)
            logger.info(f"Saved {len(self.all_results)} intermediate results")

        if self.attention_records and self.config.capture_attention:
            self._save_attention_data(self.attention_records, intermediate=True)

    def _save_final_results(self, checkpoint_metadata: list[ModelMetadata]) -> dict[str, t.Any]:
        """Save final evaluation results."""
        # Save ICL performance results
        if self.all_results:
            df = DataSchemaManager.records_to_dataframe(self.all_results)
            results_path = self.config.output_dir / "raw_evaluations" / "icl_performance.parquet"
            df.to_parquet(results_path, index=False)
            logger.info(f"Saved ICL performance results: {results_path} ({len(self.all_results)} records)")

        # Save attention data
        if self.attention_records and self.config.capture_attention:
            self._save_attention_data(self.attention_records)
            logger.info(f"Saved attention data: {len(self.attention_records)} records")

        return {
            "total_evaluations": len(self.all_results),
            "total_models": len(checkpoint_metadata),
            "total_attention_records": len(self.attention_records),
            "completed_models": len(self.progress.completed_models),
            "failed_models": len(self.progress.failed_models),
            "eval_type": self.config.eval_type,
            "model_variant": self.config.model_variant,
            "context_sizes_evaluated": list(set(r.context_size for r in self.all_results)),
            "control_types_evaluated": list(set(r.control_type for r in self.all_results)),
        }

    def _save_attention_data(self, attention_records: list[AttentionRecord], intermediate: bool = False) -> None:
        """Save attention data to individual files organized by eval_type."""
        import numpy as np

        attention_dir = self.config.output_dir / "raw_evaluations" / "attention_data"
        if intermediate:
            attention_dir = attention_dir / "intermediate"
            attention_dir.mkdir(exist_ok=True)

        # Group by model for efficient storage
        by_model = defaultdict(list)
        for record in attention_records:
            by_model[record.model_id].append(record)

        for model_id, model_records in by_model.items():
            # Create model subdirectory
            model_dir = attention_dir / model_id
            model_dir.mkdir(exist_ok=True)

            for record in model_records:
                filename = record.get_filename()
                filepath = model_dir / filename

                np.savez_compressed(
                    filepath,
                    attention_matrix=record.attention_matrix,
                    metadata={
                        "dataset_type": record.dataset_type,
                        "num_seeds": record.num_seeds,
                        "seed": record.seed,
                        "config_L": record.config_L,
                        "config_m": record.config_m,
                        "model_variant": record.model_variant,
                        "checkpoint_step": record.checkpoint_step,
                        "model_id": record.model_id,
                        "eval_type": record.eval_type,
                        "layer_idx": record.layer_idx,
                        "head_idx": record.head_idx,
                        "context_size": record.context_size,
                        "sequence_id": record.sequence_id,
                        "timestamp": record.evaluation_timestamp.isoformat(),
                    },
                )


class ICLEvaluationEngine:
    """Core ICL evaluation functionality."""

    def __init__(self, device: str = "cuda"):
        """Initialize evaluation engine."""
        self.device = torch.device(device)

    def evaluate_icl_sequence(
        self, model: t.Any, tokenizer: t.Any, sequence: dict[str, t.Any], capture_attention: bool = False
    ) -> tuple[bool, dict[str, t.Any] | None]:
        """Evaluate a single ICL sequence and optionally capture attention."""
        try:
            # Prepare context and query
            context_features = sequence["context_features"]
            context_labels = sequence["context_labels"]
            query_features = sequence["query_features"]
            true_label = sequence["query_label"]

            # Format as ICL prompt
            prompt = self._format_icl_prompt(context_features, context_labels, query_features, tokenizer)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )

            # Get predictions and attention
            with torch.no_grad():
                if capture_attention:
                    outputs = model(**inputs, output_attentions=True)
                    attention_data = self._extract_attention_patterns(outputs.attentions)
                else:
                    outputs = model(**inputs)
                    attention_data = None

                # Get prediction
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                predicted_label = self._extract_prediction(logits, tokenizer, true_label)
                is_correct = predicted_label == true_label

            return is_correct, attention_data

        except Exception as e:
            warnings.warn(f"Evaluation failed for sequence: {e}")
            return False, None

    def _format_icl_prompt(
        self, context_features: list[list[int]], context_labels: list[int], query_features: list[int], tokenizer: t.Any
    ) -> str:
        """Format features and labels as ICL prompt."""
        prompt_parts = []

        # Add context examples
        for features, label in zip(context_features, context_labels, strict=False):
            feature_str = " ".join(map(str, features))
            prompt_parts.append(f"Input: {feature_str} Output: {label}")

        # Add query
        query_str = " ".join(map(str, query_features))
        prompt_parts.append(f"Input: {query_str} Output:")

        return "\n".join(prompt_parts)

    def _extract_prediction(self, logits: torch.Tensor, tokenizer: t.Any, true_label: int) -> int:
        """Extract predicted label from model logits."""
        # Get logits for the last token (where prediction should be)
        last_token_logits = logits[0, -1, :]

        # Get top prediction
        predicted_token_id = torch.argmax(last_token_logits).item()
        predicted_token = tokenizer.decode([predicted_token_id]).strip()

        # Try to convert to integer
        try:
            predicted_label = int(predicted_token)
            return predicted_label
        except ValueError:
            # If can't convert, try to extract digit
            import re

            digits = re.findall(r"\d+", predicted_token)
            if digits:
                return int(digits[0])
            # Return random guess if no valid prediction
            return -1

    def _extract_attention_patterns(self, attention_tensors: tuple[torch.Tensor, ...]) -> dict[str, t.Any]:
        """Extract and process attention patterns."""
        attention_data = {}

        for layer_idx, layer_attention in enumerate(attention_tensors):
            # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
            layer_attention = layer_attention.squeeze(0)  # Remove batch dimension

            for head_idx in range(layer_attention.size(0)):
                head_attention = layer_attention[head_idx].cpu().numpy()
                key = f"layer_{layer_idx}_head_{head_idx}"
                attention_data[key] = head_attention

        return attention_data
