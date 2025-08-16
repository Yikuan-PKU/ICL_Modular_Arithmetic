"""Comprehensive evaluation pipeline for all ICL experiments."""

import json
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ICL.eval.collection.ckpt_manager import CheckpointManager
from ICL.eval.collection.data_schema import (
    AttentionRecord,
    ControlType,
    DataSchemaManager,
    EvaluationConfig,
    ICLPerformanceRecord,
    ModelMetadata,
    TransferCondition,
    create_evaluation_manifest,
)

##################################################
# ICLEvaluationEngine
##################################################


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
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

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


##################################################
# ComprehensiveEvaluator
##################################################


class ComprehensiveEvaluator:
    """Main comprehensive evaluation pipeline."""

    def __init__(self, config: EvaluationConfig):
        """Initialize comprehensive evaluator."""
        self.config = config
        self.config.validate()

        self.checkpoint_manager = CheckpointManager(config)
        self.evaluation_engine = ICLEvaluationEngine(config.device)
        self.evaluation_dataset = None
        self._setup_output_directories()

    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            self.config.output_dir / "metadata",
            self.config.output_dir / "raw_evaluations",
            self.config.output_dir / "raw_evaluations" / "attention_data",
            self.config.output_dir / "intermediate",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_evaluation_dataset(self) -> dict[str, t.Any]:
        """Load the comprehensive evaluation dataset."""
        print(f"Loading evaluation dataset from {self.config.eval_dataset_path}")

        with open(self.config.eval_dataset_path) as f:
            dataset = json.load(f)

        self.evaluation_dataset = dataset

        # Print dataset summary
        conditions = dataset.get("conditions", {})
        print(f"Loaded evaluation dataset with {len(conditions)} conditions:")
        for condition_name, condition_data in conditions.items():
            if isinstance(condition_data, list):
                print(f"  {condition_name}: {len(condition_data)} models")
            elif isinstance(condition_data, dict):
                total_models = sum(len(models) for models in condition_data.values())
                print(f"  {condition_name}: {len(condition_data)} configs, {total_models} total models")

        return dataset

    def discover_and_validate_checkpoints(self) -> list[ModelMetadata]:
        """Discover available checkpoints and validate completeness."""
        print("Discovering model checkpoints...")
        metadata_list = self.checkpoint_manager.discover_checkpoints()

        if not metadata_list:
            raise RuntimeError("No valid checkpoints found")

        # Validate completeness
        validation_results = self.checkpoint_manager.validate_checkpoint_completeness(metadata_list)

        print("\nCheckpoint Validation:")
        print(f"  Total checkpoints: {validation_results['total_checkpoints']}")
        print(f"  Configs found: {len(validation_results['configs_found'])}")

        if validation_results["missing_configs"]:
            warnings.warn(f"Missing configs: {validation_results['missing_configs']}")

        if validation_results["incomplete_diversity"]:
            print("  Incomplete diversity coverage:")
            for config, missing in validation_results["incomplete_diversity"]:
                print(f"    {config}: missing {missing}")

        return metadata_list

    def run_comprehensive_evaluation(self) -> dict[str, t.Any]:
        """Run the complete evaluation pipeline."""
        start_time = datetime.now()
        print("=" * 80)
        print("STARTING COMPREHENSIVE ICL EVALUATION")
        print("=" * 80)
        print(f"Start time: {start_time}")
        print(f"Output directory: {self.config.output_dir}")
        print()

        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset()

        # Discover checkpoints
        checkpoint_metadata = self.discover_and_validate_checkpoints()

        # Create evaluation manifest
        manifest = create_evaluation_manifest(self.config, start_time)
        manifest_path = self.config.output_dir / "metadata" / "experiment_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save model registry
        model_registry_df = DataSchemaManager.metadata_to_dataframe(checkpoint_metadata)
        registry_path = self.config.output_dir / "metadata" / "model_registry.parquet"
        model_registry_df.to_parquet(registry_path, index=False)
        print(f"Saved model registry: {registry_path}")

        # Run evaluations
        all_results = []
        attention_records = []

        total_models = len(checkpoint_metadata)

        for model_idx, metadata in enumerate(tqdm(checkpoint_metadata, desc="Evaluating models")):
            print(f"\n[{model_idx + 1}/{total_models}] Evaluating {metadata.model_id}")

            try:
                # Load model
                model, tokenizer = self.checkpoint_manager.load_model_checkpoint(metadata)

                # Evaluate on all conditions
                model_results, model_attention = self._evaluate_single_model(model, tokenizer, metadata, eval_dataset)

                all_results.extend(model_results)
                attention_records.extend(model_attention)

                # Save intermediate results periodically
                if self.config.save_intermediate and (model_idx + 1) % 5 == 0:
                    self._save_intermediate_results(all_results, attention_records)

                # Clear model from memory
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                warnings.warn(f"Failed to evaluate model {metadata.model_id}: {e}")
                continue

        # Save final results
        print("\nSaving final results...")
        results_summary = self._save_final_results(all_results, attention_records, checkpoint_metadata)

        # Update manifest
        end_time = datetime.now()
        manifest["end_time"] = end_time.isoformat()
        manifest["status"] = "completed"
        manifest["results_summary"] = results_summary

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print("=" * 80)
        print("COMPREHENSIVE EVALUATION COMPLETED")
        print("=" * 80)
        print(f"End time: {end_time}")
        print(f"Duration: {end_time - start_time}")
        print(f"Total evaluations: {len(all_results)}")
        print(f"Results saved to: {self.config.output_dir}")

        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_evaluations": len(all_results),
            "total_models": len(checkpoint_metadata),
            "results_summary": results_summary,
            "output_dir": self.config.output_dir,
        }

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
            condition_results, condition_attention = self._evaluate_transfer_condition(
                model, tokenizer, metadata, eval_dataset, transfer_condition, model_config
            )
            results.extend(condition_results)
            attention_records.extend(condition_attention)

        return results, attention_records

    def _evaluate_transfer_condition(
        self,
        model: t.Any,
        tokenizer: t.Any,
        metadata: ModelMetadata,
        eval_dataset: dict[str, t.Any],
        transfer_condition: TransferCondition,
        model_config: tuple[int, int],
    ) -> tuple[list[ICLPerformanceRecord], list[AttentionRecord]]:
        """Evaluate model on a specific transfer condition."""
        results = []
        attention_records = []

        # Get appropriate evaluation sequences based on transfer condition
        eval_sequences = self._get_sequences_for_condition(eval_dataset, transfer_condition, model_config)

        if not eval_sequences:
            warnings.warn(f"No sequences found for {transfer_condition} condition")
            return results, attention_records

        # Limit sequences if specified
        if self.config.max_sequences_per_condition > 0:
            eval_sequences = eval_sequences[: self.config.max_sequences_per_condition]

        # Evaluate across all context sizes and control types
        for context_size in self.config.context_sizes:
            for control_type in self.config.control_types:
                sequences = self._prepare_sequences_for_evaluation(eval_sequences, context_size, control_type)

                for seq_idx, sequence in enumerate(sequences):
                    try:
                        # Evaluate sequence
                        is_correct, attention_data = self.evaluation_engine.evaluate_icl_sequence(
                            model, tokenizer, sequence, capture_attention=self.config.capture_attention
                        )

                        # Create performance record
                        target_config = self._get_target_config(sequence, transfer_condition, model_config)

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
                            num_sequences=1,
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
                        warnings.warn(f"Failed to evaluate sequence {seq_idx}: {e}")
                        continue

        return results, attention_records

    def _get_sequences_for_condition(
        self, eval_dataset: dict[str, t.Any], transfer_condition: TransferCondition, model_config: tuple[int, int]
    ) -> list[dict[str, t.Any]]:
        """Get evaluation sequences for a specific transfer condition."""
        conditions = eval_dataset.get("conditions", {})

        if transfer_condition == "within_config":
            # Use within-config evaluation data
            within_config_data = conditions.get("within_config", [])
            for model_data in within_config_data:
                if tuple(model_data["config"]) == model_config:
                    sequences = []
                    for k_sequences in model_data["sequences"].values():
                        sequences.extend(k_sequences)
                    return sequences

        elif transfer_condition in ["cross_L", "cross_m", "cross_config"]:
            # Use appropriate transfer condition data
            condition_key = {
                "cross_L": "depth_transfer",
                "cross_m": "synonym_transfer",
                "cross_config": "full_transfer",
            }[transfer_condition]

            transfer_data = conditions.get(condition_key, {})
            sequences = []

            for config_key, config_models in transfer_data.items():
                for model_data in config_models:
                    for k_sequences in model_data["sequences"].values():
                        sequences.extend(k_sequences)

            return sequences

        return []

    def _prepare_sequences_for_evaluation(
        self, sequences: list[dict[str, t.Any]], context_size: int, control_type: ControlType
    ) -> list[dict[str, t.Any]]:
        """Prepare sequences for evaluation with specified context size and control type."""
        prepared_sequences = []

        for sequence in sequences:
            # Filter by context size
            if sequence.get("context_size") == context_size:
                if control_type == "normal":
                    prepared_sequences.append(sequence)
                elif control_type == "shuffled_context":
                    # Create shuffled version
                    shuffled_seq = self._create_shuffled_sequence(sequence)
                    prepared_sequences.append(shuffled_seq)
                elif control_type == "random_context":
                    # Create random context version
                    random_seq = self._create_random_context_sequence(sequence, sequences)
                    if random_seq:
                        prepared_sequences.append(random_seq)

        return prepared_sequences

    def _create_shuffled_sequence(self, sequence: dict[str, t.Any]) -> dict[str, t.Any]:
        """Create a sequence with shuffled context order."""
        import random

        shuffled_seq = sequence.copy()

        # Shuffle context pairs
        context_pairs = list(zip(shuffled_seq["context_features"], shuffled_seq["context_labels"], strict=False))
        random.shuffle(context_pairs)

        shuffled_seq["context_features"] = [pair[0] for pair in context_pairs]
        shuffled_seq["context_labels"] = [pair[1] for pair in context_pairs]

        return shuffled_seq

    def _create_random_context_sequence(
        self, sequence: dict[str, t.Any], all_sequences: list[dict[str, t.Any]]
    ) -> dict[str, t.Any] | None:
        """Create a sequence with random context from other sequences."""
        import random

        # Find other sequences with same context size
        same_k_sequences = [
            seq for seq in all_sequences if seq.get("context_size") == sequence.get("context_size") and seq != sequence
        ]

        if len(same_k_sequences) < sequence.get("context_size", 0):
            return None

        # Sample random contexts
        random_contexts = random.sample(same_k_sequences, sequence.get("context_size", 0))

        random_seq = sequence.copy()
        random_seq["context_features"] = [ctx["query_features"] for ctx in random_contexts]
        random_seq["context_labels"] = [ctx["query_label"] for ctx in random_contexts]

        return random_seq

    def _get_target_config(
        self, sequence: dict[str, t.Any], transfer_condition: TransferCondition, model_config: tuple[int, int]
    ) -> tuple[int, int]:
        """Get target configuration for the sequence."""
        # For within-config, target is same as model config
        if transfer_condition == "within_config":
            return model_config

        # For transfer conditions, try to infer from sequence metadata
        # This would need to be stored in the evaluation dataset
        # For now, return model config as fallback
        return model_config

    def _parse_attention_key(self, attention_key: str) -> tuple[int, int]:
        """Parse layer and head indices from attention key."""
        # Expected format: "layer_{layer_idx}_head_{head_idx}"
        parts = attention_key.split("_")
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        return layer_idx, head_idx

    def _save_intermediate_results(
        self, results: list[ICLPerformanceRecord], attention_records: list[AttentionRecord]
    ) -> None:
        """Save intermediate results to disk."""
        if results:
            df = DataSchemaManager.records_to_dataframe(results)
            intermediate_path = self.config.output_dir / "intermediate" / "partial_results.parquet"
            df.to_parquet(intermediate_path, index=False)

        if attention_records and self.config.capture_attention:
            self._save_attention_data(attention_records, intermediate=True)

    def _save_final_results(
        self,
        results: list[ICLPerformanceRecord],
        attention_records: list[AttentionRecord],
        checkpoint_metadata: list[ModelMetadata],
    ) -> dict[str, t.Any]:
        """Save final evaluation results."""
        # Save ICL performance results
        if results:
            df = DataSchemaManager.records_to_dataframe(results)
            results_path = self.config.output_dir / "raw_evaluations" / "icl_performance.parquet"
            df.to_parquet(results_path, index=False)
            print(f"Saved ICL performance results: {results_path} ({len(results)} records)")

        # Save attention data
        if attention_records and self.config.capture_attention:
            self._save_attention_data(attention_records)
            print(f"Saved attention data: {len(attention_records)} records")

        # Generate and save aggregated metrics
        aggregated_metrics = self._compute_aggregated_metrics(results)
        if aggregated_metrics:
            metrics_path = self.config.output_dir / "intermediate" / "aggregated_metrics.parquet"
            aggregated_metrics.to_parquet(metrics_path, index=False)
            print(f"Saved aggregated metrics: {metrics_path}")

        return {
            "total_evaluations": len(results),
            "total_models": len(checkpoint_metadata),
            "total_attention_records": len(attention_records),
            "unique_model_configs": len(set((m.config_L, m.config_m) for m in checkpoint_metadata)),
            "context_sizes_evaluated": list(set(r.context_size for r in results)),
            "transfer_conditions_evaluated": list(set(r.transfer_condition for r in results)),
        }

    def _save_attention_data(self, attention_records: list[AttentionRecord], intermediate: bool = False) -> None:
        """Save attention data to individual files."""
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

    def _compute_aggregated_metrics(self, results: list[ICLPerformanceRecord]) -> pd.DataFrame | None:
        """Compute aggregated metrics from evaluation results."""
        if not results:
            return None

        # Convert to DataFrame for easier aggregation
        df = DataSchemaManager.records_to_dataframe(results)

        # Group by model and compute aggregated metrics
        aggregated = []

        groupby_cols = ["model_id", "config_L", "config_m", "n_train", "checkpoint_step"]

        for name, group in df.groupby(groupby_cols):
            model_id, config_L, config_m, n_train, checkpoint_step = name

            # Compute emergence threshold (minimum k for >0.5 accuracy on normal sequences)
            normal_group = group[group["control_type"] == "normal"]
            emergence_threshold = self._compute_emergence_threshold(normal_group)

            # Compute max accuracy across all context sizes
            max_accuracy = normal_group["accuracy"].max() if len(normal_group) > 0 else 0.0

            # Compute transfer degradation
            within_config_acc = normal_group[normal_group["transfer_condition"] == "within_config"]["accuracy"].mean()

            transfer_acc = normal_group[normal_group["transfer_condition"] != "within_config"]["accuracy"].mean()

            transfer_degradation = (
                within_config_acc - transfer_acc
                if not pd.isna(within_config_acc) and not pd.isna(transfer_acc)
                else 0.0
            )

            # Compute baseline gap (normal vs shuffled/random)
            normal_acc = group[group["control_type"] == "normal"]["accuracy"].mean()
            control_acc = group[group["control_type"] != "normal"]["accuracy"].mean()
            baseline_gap = normal_acc - control_acc if not pd.isna(normal_acc) and not pd.isna(control_acc) else 0.0

            aggregated.append(
                {
                    "model_id": model_id,
                    "config_L": config_L,
                    "config_m": config_m,
                    "n_train": n_train,
                    "checkpoint_step": checkpoint_step,
                    "emergence_threshold": emergence_threshold,
                    "max_context_accuracy": max_accuracy,
                    "transfer_degradation": transfer_degradation,
                    "baseline_gap": baseline_gap,
                    "total_evaluations": len(group),
                }
            )

        return pd.DataFrame(aggregated)

    def _compute_emergence_threshold(self, group: pd.DataFrame) -> float:
        """Compute emergence threshold (minimum k for >0.5 accuracy)."""
        threshold = 0.5

        # Group by context size and compute mean accuracy
        by_context = group.groupby("context_size")["accuracy"].mean().sort_index()

        # Find first context size with accuracy > threshold
        for context_size, accuracy in by_context.items():
            if accuracy > threshold:
                return float(context_size)

        # If no context size achieves threshold, return max context size + 1
        return float(max(group["context_size"]) + 1) if len(group) > 0 else float("inf")
