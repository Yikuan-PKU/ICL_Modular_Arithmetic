"""Minimal data schemas - only parsing essential fields from model configs."""

import typing as t
from collections import defaultdict

import numpy as np
import pandas as pd

# Simple type aliases
ControlType = t.Literal["normal", "shuffled_context", "random_context"]
EvalType = t.Literal["memorization", "id_generalization", "ood_same_rule", "ood_transfer"]


def create_performance_record(model_metadata: dict, eval_context: dict, sequence_data: dict, result: bool) -> dict:
    """Create simple performance record dict."""
    return {
        # Basic identifiers
        "model_variant": model_metadata["model_variant"],
        "checkpoint_step": model_metadata["checkpoint_step"],
        "model_id": model_metadata["model_id"],
        "config_L": model_metadata["config_L"],
        "config_m": model_metadata["config_m"],
        # Evaluation context
        "eval_type": eval_context["eval_type"],
        "context_size": eval_context["context_size"],
        "control_type": eval_context["control_type"],
        "sequence_id": eval_context["sequence_id"],
        # Target config
        "target_config_L": sequence_data.get("target_config_L", model_metadata["config_L"]),
        "target_config_m": sequence_data.get("target_config_m", model_metadata["config_m"]),
        # Results
        "accuracy": float(result),
        "appears_in_training": sequence_data.get("appears_in_training", False),
    }


def create_attention_record(
    model_metadata: dict, eval_context: dict, layer_idx: int, head_idx: int, attention_matrix
) -> dict:
    """Create simple attention record dict."""
    return {
        "model_id": model_metadata["model_id"],
        "eval_type": eval_context["eval_type"],
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "context_size": eval_context["context_size"],
        "sequence_id": eval_context["sequence_id"],
        "attention_matrix": attention_matrix,
        "filename": f"{model_metadata['model_id']}_layer{layer_idx}_head{head_idx}_k{eval_context['context_size']}_seq{eval_context['sequence_id']}.npz",
    }


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Simple conversion to DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def save_attention_data(attention_records: list[dict], output_dir) -> None:
    """Simple attention data saving."""
    attention_dir = output_dir / "raw_evaluations" / "attention_data"
    attention_dir.mkdir(parents=True, exist_ok=True)

    # Group by model
    by_model = defaultdict(list)
    for record in attention_records:
        by_model[record["model_id"]].append(record)

    for model_id, model_records in by_model.items():
        model_dir = attention_dir / model_id
        model_dir.mkdir(exist_ok=True)

        for record in model_records:
            filepath = model_dir / record["filename"]
            np.savez_compressed(filepath, attention_matrix=record["attention_matrix"])
