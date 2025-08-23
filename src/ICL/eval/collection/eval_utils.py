"""Simplified utility functions for evaluation."""

import random
import typing as t


def extract_target_config_from_sequence(sequence: dict[str, t.Any]) -> tuple[int, int]:
    """Simple target configuration extraction."""
    # Try direct fields first
    if "target_config_L" in sequence and "target_config_m" in sequence:
        return (sequence["target_config_L"], sequence["target_config_m"])

    # Try config field
    if "config" in sequence:
        config = sequence["config"]
        if isinstance(config, (list, tuple)) and len(config) == 2:
            return tuple(config)

    # Default fallback
    return (2, 2)


def create_control_sequence(sequence: dict[str, t.Any], control_type: str) -> dict[str, t.Any]:
    """Simple control sequence creation."""
    if control_type == "normal":
        return sequence

    if control_type == "shuffled_context":
        shuffled_seq = sequence.copy()
        context_features = shuffled_seq.get("context_features", [])
        context_labels = shuffled_seq.get("context_labels", [])

        if len(context_features) == len(context_labels):
            # Shuffle context pairs
            context_pairs = list(zip(context_features, context_labels, strict=False))
            random.shuffle(context_pairs)
            shuffled_seq["context_features"] = [pair[0] for pair in context_pairs]
            shuffled_seq["context_labels"] = [pair[1] for pair in context_pairs]

        return shuffled_seq

    if control_type == "random_context":
        # For simplicity, just return original sequence
        # In practice, you might want to sample from other sequences
        return sequence

    return sequence


def validate_sequence_structure(sequence: dict[str, t.Any]) -> bool:
    """Basic sequence validation."""
    required_fields = ["context_features", "context_labels", "query_features", "query_label"]

    # Check required fields exist
    for field in required_fields:
        if field not in sequence:
            return False

    # Check context consistency
    context_features = sequence["context_features"]
    context_labels = sequence["context_labels"]

    if not isinstance(context_features, list) or not isinstance(context_labels, list):
        return False

    if len(context_features) != len(context_labels):
        return False

    return True
