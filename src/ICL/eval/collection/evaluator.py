"""Simplified evaluator - basic evaluation loop only."""

import logging
import random
from math import ceil

import torch
from datasets import load_from_disk
from tqdm import tqdm

from ICL.eval.collection.ckpt_manager import CheckpointManager
from ICL.eval.collection.data_schema import (
    create_attention_record,
    create_performance_record,
    records_to_dataframe,
    save_attention_record_immediately,
)

logger = logging.getLogger(__name__)


class CollectionEvaluator:
    """Simplified evaluator with memory management."""

    def __init__(self, config):
        """Simple initialization."""
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.evaluation_engine = ICLEvaluationEngine(config.device, config)

        # Simple result storage (performance only, attention streamed)
        self.performance_records = []

    def run_comprehensive_collection(self) -> dict:
        """Simplified collection pipeline with memory management."""
        # check whether the target file already exsits
        exist_file, results_path = self._check_existence()
        if self.config.resume and exist_file:
            logger.info(f"Exist {results_path}. Skip!")
            return None

        logger.info("Starting collection...")
        # Load dataset in chunks
        eval_dataset = self._load_evaluation_dataset()

        # Discover checkpoints
        checkpoints = self.checkpoint_manager.discover_checkpoints()
        logger.info(f"Found {len(checkpoints)} checkpoints")

        # Run evaluation with memory management
        self._run_evaluation_loop(checkpoints, eval_dataset)

        # Save results
        return self._save_results()

    def _load_evaluation_dataset(self) -> dict:
        """Load dataset with chunking support."""
        logger.info(f"Loading dataset from {self.config.eval_dataset_path}")

        hf_dataset = load_from_disk(str(self.config.eval_dataset_path))

        # Convert to simple format but don't load all at once
        total_sequences = len(hf_dataset)
        logger.info(f"Dataset contains {total_sequences} sequences")

        return {
            "hf_dataset": hf_dataset,
            "total_sequences": total_sequences,
        }

    def _get_training_appearance(self, example):
        """Simple training appearance logic."""
        if self.config.eval_type == "memorization":
            return True
        if self.config.eval_type in ["id_generalization", "ood_same_rule", "ood_transfer"]:
            return False
        return example.get("appears_in_training", False)

    def _run_evaluation_loop(self, checkpoints, eval_dataset):
        """Memory-managed evaluation loop."""
        for checkpoint_idx, checkpoint in enumerate(tqdm(checkpoints, desc="Evaluating models")):
            try:
                # Load model
                model, tokenizer = self.checkpoint_manager.load_model_checkpoint(checkpoint)

                # Evaluate on sequences with chunking
                self._evaluate_model_chunked(model, tokenizer, checkpoint, eval_dataset)

                # Offload or cleanup model
                if self.config.offload_models and checkpoint_idx < len(checkpoints) - 1:
                    self.checkpoint_manager.offload_model_to_cpu(model)
                else:
                    self.checkpoint_manager.cleanup_model(model, tokenizer)

            except Exception as e:
                logger.warning(f"Failed to evaluate {checkpoint['model_id']}: {e}")
                # Ensure cleanup even on failure
                try:
                    if "model" in locals():
                        self.checkpoint_manager.cleanup_model(model, tokenizer)
                except:
                    pass
                continue

    def _evaluate_model_chunked(self, model, tokenizer, checkpoint, eval_dataset):
        """Evaluate model using chunked sequence processing."""
        hf_dataset = eval_dataset["hf_dataset"]

        for context_size in self.config.context_sizes:
            # Filter sequence indices by context size (don't load sequences yet)
            valid_indices = []
            for i in range(len(hf_dataset)):
                example = hf_dataset[i]
                if len(example.get("context_features", [])) == context_size:
                    valid_indices.append(i)

            if not valid_indices:
                continue

            # Limit sequences if needed
            if self.config.max_sequences_per_condition > 0:
                valid_indices = valid_indices[: self.config.max_sequences_per_condition]

            # Process in chunks
            chunk_size = self.config.sequence_chunk_size
            num_chunks = ceil(len(valid_indices) / chunk_size)

            for control_type in self.config.control_types:
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(valid_indices))
                    chunk_indices = valid_indices[start_idx:end_idx]

                    # Load and process chunk
                    self._process_sequence_chunk(
                        model, tokenizer, checkpoint, hf_dataset, chunk_indices, context_size, control_type, start_idx
                    )

                    # Clear any intermediate tensors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _process_sequence_chunk(
        self, model, tokenizer, checkpoint, hf_dataset, chunk_indices, context_size, control_type, base_seq_id
    ):
        """Process a chunk of sequences."""
        for local_idx, global_idx in enumerate(chunk_indices):
            seq_id = base_seq_id + local_idx

            try:
                # Load single sequence
                example = hf_dataset[global_idx]
                sequence = {
                    "context_features": example.get("context_features", []),
                    "context_labels": example.get("context_labels", []),
                    "query_features": example.get("query_features", []),
                    "query_label": example.get("query_label", 0),
                    "context_size": len(example.get("context_features", [])),
                    "sequence_id": seq_id,
                    "target_config_L": example.get("target_config_L", self.config.config_L),
                    "target_config_m": example.get("target_config_m", self.config.config_m),
                    "appears_in_training": self._get_training_appearance(example),
                }

                # Create control sequence
                control_seq = self._create_control_sequence(sequence, control_type)

                # Determine if we should capture attention for this sequence
                should_capture = self.config.capture_attention and random.random() < self.config.attention_sampling_rate

                # Evaluate with memory management
                with torch.no_grad():
                    is_correct, attention_data = self.evaluation_engine.evaluate_icl_sequence(
                        model, tokenizer, control_seq, should_capture
                    )

                # Store performance result
                eval_context = {
                    "eval_type": self.config.eval_type,
                    "context_size": context_size,
                    "control_type": control_type,
                    "sequence_id": seq_id,
                }

                record = create_performance_record(checkpoint, eval_context, control_seq, is_correct)
                self.performance_records.append(record)

                # Stream attention data immediately if captured
                if attention_data and should_capture:
                    self._stream_attention_data(checkpoint, eval_context, attention_data)

            except Exception as e:
                logger.warning(f"Failed sequence {seq_id}: {e}")
                continue

    def _stream_attention_data(self, checkpoint, eval_context, attention_data):
        """Stream attention data immediately to disk."""
        for layer_idx, layer_data in attention_data.items():
            if isinstance(layer_data, dict):
                for head_idx, attention_matrix in layer_data.items():
                    attention_record = create_attention_record(
                        checkpoint, eval_context, layer_idx, head_idx, attention_matrix
                    )
                    save_attention_record_immediately(attention_record, self.config.output_dir)

    def _create_control_sequence(self, sequence, control_type):
        """Simple control sequence creation."""
        if control_type == "normal":
            return sequence
        if control_type == "shuffled_context":
            seq_copy = sequence.copy()
            context_pairs = list(zip(seq_copy["context_features"], seq_copy["context_labels"], strict=False))
            random.shuffle(context_pairs)
            seq_copy["context_features"] = [p[0] for p in context_pairs]
            seq_copy["context_labels"] = [p[1] for p in context_pairs]
            return seq_copy
        if control_type == "random_context":
            # Just return original for simplicity
            return sequence
        return sequence

    def _check_existence(self) -> dict:
        """Simple result saving."""
        self.config.create_output_structure()
        results_path = self.config.output_dir / "raw_evaluations" / "icl_performance.parquet"
        return results_path.exists(), results_path

    def _save_results(self) -> dict:
        """Simple result saving."""
        self.config.create_output_structure()

        # Save performance results
        if self.performance_records:
            df = records_to_dataframe(self.performance_records)
            results_path = self.config.output_dir / "raw_evaluations" / "icl_performance.parquet"
            df.to_parquet(results_path, index=False)
            logger.info(f"Saved {len(self.performance_records)} performance records")

        return {
            "total_evaluations": len(self.performance_records),
            "eval_type": self.config.eval_type,
            "model_variant": self.config.model_variant,
            "attention_streamed": self.config.capture_attention,
        }


class ICLEvaluationEngine:
    """Simple ICL evaluation engine."""

    def __init__(self, device: str, config=None):
        self.device = torch.device(device)
        self.config = config

    def evaluate_icl_sequence(self, model, tokenizer, sequence, capture_attention=False):
        """Simple sequence evaluation."""
        try:
            # Format prompt
            prompt = self._format_icl_prompt(
                sequence["context_features"], sequence["context_labels"], sequence["query_features"], tokenizer
            )

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )

            # Get prediction
            with torch.no_grad():
                if capture_attention:
                    outputs = model(**inputs, output_attentions=True)
                    attention_data = self._extract_attention(outputs.attentions)
                else:
                    outputs = model(**inputs)
                    attention_data = None

                # Get prediction
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                predicted_label = self._extract_prediction(logits, tokenizer)
                is_correct = predicted_label == sequence["query_label"]

            return is_correct, attention_data

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return False, None

    def _format_icl_prompt(self, context_features, context_labels, query_features, tokenizer):
        """Simple prompt formatting."""
        prompt_parts = []

        # Add context examples
        for features, label in zip(context_features, context_labels, strict=False):
            feature_str = " ".join(map(str, features))
            prompt_parts.append(f"Input: {feature_str} Output: {label}")

        # Add query
        query_str = " ".join(map(str, query_features))
        prompt_parts.append(f"Input: {query_str} Output:")

        return "\n".join(prompt_parts)

    def _extract_prediction(self, logits, tokenizer):
        """Simple prediction extraction."""
        last_token_logits = logits[0, -1, :]
        predicted_token_id = torch.argmax(last_token_logits).item()
        predicted_token = tokenizer.decode([predicted_token_id]).strip()

        try:
            return int(predicted_token)
        except ValueError:
            import re

            digits = re.findall(r"\d+", predicted_token)
            return int(digits[0]) if digits else -1

    def _extract_attention(self, attention_tensors):
        """Simple attention extraction."""
        attention_data = {}

        for layer_idx, layer_attention in enumerate(attention_tensors):
            layer_attention = layer_attention.squeeze(0)  # Remove batch dimension
            layer_data = {}

            for head_idx in range(layer_attention.size(0)):
                head_attention = layer_attention[head_idx].cpu().numpy()
                layer_data[head_idx] = head_attention

            attention_data[layer_idx] = layer_data

        return attention_data
