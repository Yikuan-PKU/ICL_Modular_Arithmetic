"""Minimal data schemas with evaluation type validation."""

import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

# Simple type aliases
ControlType = t.Literal["normal", "shuffled_context", "random_context"]
EvalType = t.Literal["memorization", "id_generalization", "ood_same_rule", "ood_transfer"]


def validate_eval_dataset(dataset_path: Path, expected_eval_type: str) -> bool:
    """Validate that dataset matches expected evaluation type."""
    try:
        if not dataset_path.exists():
            return False

        # Load small sample to check metadata
        dataset = load_from_disk(str(dataset_path))

        # Fallback: check first few examples for characteristics
        sample_size = min(10, len(dataset))
        samples = dataset.select(range(sample_size))

        # Basic heuristics for evaluation type validation
        if expected_eval_type == "memorization":
            # Should have training appearance indicators
            return any(example.get("appears_in_training", True) for example in samples)
        if expected_eval_type in ["id_generalization", "ood_same_rule", "ood_transfer"]:
            # Should have no training appearance or explicit False
            return any(not example.get("appears_in_training", True) for example in samples)

        return True  # Default to valid if unsure

    except Exception:
        return False


def extract_dataset_metadata(dataset_path: Path) -> dict[str, t.Any]:
    """Extract metadata from evaluation dataset."""
    try:
        dataset = load_from_disk(str(dataset_path))

        metadata = {
            "num_examples": len(dataset),
            "features": list(dataset.features.keys()) if hasattr(dataset, "features") else [],
            "eval_type_hint": None,
        }

        # Try to infer eval type from content
        if len(dataset) > 0:
            first_example = dataset[0]
            if "appears_in_training" in first_example:
                appears_in_training = first_example["appears_in_training"]
                metadata["eval_type_hint"] = "memorization" if appears_in_training else "generalization"

        return metadata

    except Exception:
        return {"num_examples": 0, "features": [], "eval_type_hint": None}


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


def save_attention_record_immediately(attention_record: dict, output_dir: Path) -> None:
    """Save single attention record immediately to disk."""
    attention_dir = output_dir / "raw_evaluations" / "attention_data"
    attention_dir.mkdir(parents=True, exist_ok=True)

    model_dir = attention_dir / attention_record["model_id"]
    model_dir.mkdir(exist_ok=True)

    filepath = model_dir / attention_record["filename"]
    np.savez_compressed(filepath, attention_matrix=attention_record["attention_matrix"])


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
