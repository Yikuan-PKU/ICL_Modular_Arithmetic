"""Enhanced evaluator for collection phase with resume capabilities."""

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
from ICL.eval.collection.collection_config import CollectionConfig
from ICL.eval.collection.data_schema import (
    AttentionRecord,
    DataSchemaManager,
    ICLPerformanceRecord,
    ModelMetadata,
    create_evaluation_manifest,
)
from ICL.eval.collection.eval_utils import (
    create_control_sequence,
    extract_target_config_from_sequence,
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
    """Enhanced evaluator with resume capabilities and better error handling."""

    def __init__(self, config: CollectionConfig):
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
        """Load evaluation dataset and validate structure."""
        logger.info(f"Loading evaluation dataset from {self.config.eval_dataset_path}")

        with open(self.config.eval_dataset_path) as f:
            dataset = json.load(f)

        self.evaluation_dataset = dataset

        # Validate dataset structure
        self._validate_dataset_structure(dataset)

        # Print dataset summary
        conditions = dataset.get("conditions", {})
        logger.info(f"Loaded evaluation dataset with {len(conditions)} conditions")
        for condition_name, condition_data in conditions.items():
            if isinstance(condition_data, list):
                logger.info(f"  {condition_name}: {len(condition_data)} models")
            elif isinstance(condition_data, dict):
                total_models = sum(len(models) for models in condition_data.values())
                logger.info(f"  {condition_name}: {len(condition_data)} configs, {total_models} total models")

        return dataset

    def _validate_dataset_structure(self, dataset: dict[str, t.Any]) -> None:
        """Validate evaluation dataset structure."""
        required_keys = ["metadata", "conditions"]
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"Missing required key in evaluation dataset: {key}")

        # Check conditions structure
        conditions = dataset["conditions"]
        expected_conditions = ["within_config", "depth_transfer", "synonym_transfer", "full_transfer"]

        missing_conditions = [cond for cond in expected_conditions if cond not in conditions]
        if missing_conditions:
            logger.warning(f"Missing evaluation conditions: {missing_conditions}")

    def discover_and_validate_checkpoints(self) -> list[ModelMetadata]:
        """Discover and validate checkpoints with filtering."""
        logger.info("Discovering model checkpoints...")

        all_metadata = self.checkpoint_manager.discover_checkpoints()

        if not all_metadata:
            raise RuntimeError("No valid checkpoints found")

        # Filter by target configurations and diversity levels
        filtered_metadata = self._filter_checkpoints(all_metadata)

        if not filtered_metadata:
            raise RuntimeError("No checkpoints match the target configurations")

        # Validate completeness
        validation_results = self.checkpoint_manager.validate_checkpoint_completeness(filtered_metadata)
        self._log_validation_results(validation_results)

        return filtered_metadata

    def _filter_checkpoints(self, metadata_list: list[ModelMetadata]) -> list[ModelMetadata]:
        """Filter checkpoints by target configurations and settings."""
        filtered = []

        for metadata in metadata_list:
            config = (metadata.config_L, metadata.config_m)

            # Check target configurations
            if self.config.target_configs and config not in self.config.target_configs:
                continue

            # Check diversity levels
            if metadata.n_train not in self.config.diversity_levels:
                continue

            # Check model types
            if metadata.model_type not in self.config.model_types:
                continue

            filtered.append(metadata)

        logger.info(f"Filtered {len(metadata_list)} -> {len(filtered)} checkpoints based on target configs")
        return filtered

    def _log_validation_results(self, validation_results: dict[str, t.Any]) -> None:
        """Log checkpoint validation results."""
        logger.info("\nCheckpoint Validation Results:")
        logger.info(f"  Total checkpoints: {validation_results['total_checkpoints']}")
        logger.info(f"  Configs found: {len(validation_results['configs_found'])}")

        if validation_results["missing_configs"]:
            logger.warning(f"  Missing configs: {validation_results['missing_configs']}")

        if validation_results["incomplete_diversity"]:
            logger.warning("  Incomplete diversity coverage:")
            for config, missing in validation_results["incomplete_diversity"]:
                logger.warning(f"    {config}: missing {missing}")

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
                # Convert back to records (simplified - would need full conversion)
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
        logger.info("STARTING COMPREHENSIVE ICL COLLECTION")
        logger.info("=" * 80)
        logger.info(f"Start time: {start_time}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Device: {self.config.device}")

        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset()

        # Discover checkpoints
        checkpoint_metadata = self.discover_and_validate_checkpoints()
        self.progress.total_models = len(checkpoint_metadata)

        # Check for resume capability
        resumed = False
        if self.config.resume:
            resumed = self.check_resume_capability(checkpoint_metadata)

        if not resumed:
            logger.info("Starting fresh collection run")
            self.progress = CollectionProgress(total_models=len(checkpoint_metadata))

        # Create evaluation manifest
        manifest = create_evaluation_manifest(self.config, start_time)
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

    def _run_evaluation_loop(self, checkpoint_metadata: list[ModelMetadata], eval_dataset: dict[str, t.Any]) -> None:
        """Run the main evaluation loop with error handling."""
        models_to_evaluate = checkpoint_metadata[self.progress.current_model_idx :]

        with tqdm(
            models_to_evaluate,
            desc="Evaluating models",
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

            # Evaluate on all conditions
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
        """Evaluate a single model on all evaluation conditions."""
        results = []
        attention_records = []

        # Get model's training configuration
        model_config = (metadata.config_L, metadata.config_m)

        # Evaluate on all transfer conditions
        for transfer_condition in self.config.transfer_conditions:
            try:
                condition_results, condition_attention = self._evaluate_transfer_condition(
                    model, tokenizer, metadata, eval_dataset, transfer_condition, model_config
                )
                results.extend(condition_results)
                attention_records.extend(condition_attention)

            except Exception as e:
                logger.warning(f"Failed to evaluate {transfer_condition} for {metadata.model_id}: {e}")
                continue

        return results, attention_records

    def _evaluate_transfer_condition(
        self,
        model: t.Any,
        tokenizer: t.Any,
        metadata: ModelMetadata,
        eval_dataset: dict[str, t.Any],
        transfer_condition: str,
        model_config: tuple[int, int],
    ) -> tuple[list[ICLPerformanceRecord], list[AttentionRecord]]:
        """Evaluate model on a specific transfer condition."""
        results = []
        attention_records = []

        # Get appropriate evaluation sequences
        eval_sequences = self._get_sequences_for_condition(eval_dataset, transfer_condition, model_config)

        if not eval_sequences:
            logger.warning(f"No sequences found for {transfer_condition} condition")
            return results, attention_records

        # Limit sequences if specified
        if self.config.max_sequences_per_condition > 0:
            eval_sequences = eval_sequences[: self.config.max_sequences_per_condition]

        # Evaluate across context sizes and control types
        for context_size in self.config.context_sizes:
            # Filter sequences by context size
            context_sequences = [seq for seq in eval_sequences if seq.get("context_size") == context_size]

            if not context_sequences:
                continue

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
                        target_config = extract_target_config_from_sequence(sequence)

                        # Create performance record
                        record = ICLPerformanceRecord(
                            model_id=metadata.model_id,
                            config_L=metadata.config_L,
                            config_m=metadata.config_m,
                            n_train=metadata.n_train,
                            checkpoint_step=metadata.checkpoint_step,
                            context_size=context_size,
                            transfer_condition=transfer_condition,
                            target_config_L=target_config[0],
                            target_config_m=target_config[1],
                            accuracy=float(is_correct),
                            sequence_id=seq_idx,
                            control_type=control_type,
                            evaluation_timestamp=datetime.now(),
                            num_correct=int(is_correct),
                        )

                        results.append(record)

                        # Store attention data if captured
                        if attention_data and self.config.capture_attention:
                            for attention_key, attention_matrix in attention_data.items():
                                layer_idx, head_idx = self._parse_attention_key(attention_key)

                                attention_record = AttentionRecord(
                                    model_id=metadata.model_id,
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

    def _get_sequences_for_condition(
        self, eval_dataset: dict[str, t.Any], transfer_condition: str, model_config: tuple[int, int]
    ) -> list[dict[str, t.Any]]:
        """Get evaluation sequences for a specific transfer condition."""
        conditions = eval_dataset.get("conditions", {})

        if transfer_condition == "within_config":
            within_config_data = conditions.get("within_config", [])
            for model_data in within_config_data:
                if tuple(model_data["config"]) == model_config:
                    sequences = []
                    for k_sequences in model_data["sequences"].values():
                        sequences.extend(k_sequences)
                    return sequences

        else:
            # Map transfer conditions to dataset keys
            condition_mapping = {
                "cross_L": "depth_transfer",
                "cross_m": "synonym_transfer",
                "cross_config": "full_transfer",
            }

            condition_key = condition_mapping.get(transfer_condition)
            if not condition_key:
                return []

            transfer_data = conditions.get(condition_key, {})
            sequences = []

            for config_key, config_models in transfer_data.items():
                for model_data in config_models:
                    for k_sequences in model_data["sequences"].values():
                        # Add config information for target extraction
                        for seq in k_sequences:
                            if "config" not in seq:
                                seq["config"] = model_data.get("config", (2, 2))
                        sequences.extend(k_sequences)

            return sequences

        return []

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
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

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
            "unique_model_configs": len(set((m.config_L, m.config_m) for m in checkpoint_metadata)),
            "context_sizes_evaluated": list(set(r.context_size for r in self.all_results)),
            "transfer_conditions_evaluated": list(set(r.transfer_condition for r in self.all_results)),
        }

    def _save_attention_data(self, attention_records: list[AttentionRecord], intermediate: bool = False) -> None:
        """Save attention data to individual files."""
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
            model_dir = attention_dir / model_id
            model_dir.mkdir(exist_ok=True)

            for record in model_records:
                filename = record.get_filename()
                filepath = model_dir / filename

                np.savez_compressed(
                    filepath,
                    attention_matrix=record.attention_matrix,
                    metadata={
                        "model_id": record.model_id,
                        "layer_idx": record.layer_idx,
                        "head_idx": record.head_idx,
                        "context_size": record.context_size,
                        "sequence_id": record.sequence_id,
                        "timestamp": record.evaluation_timestamp.isoformat(),
                    },
                )


"""Comprehensive evaluation pipeline for all ICL experiments."""


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
