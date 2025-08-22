"""Utility functions for comprehensive evaluation pipeline."""

import random
import typing as t
from collections import defaultdict

from ICL.eval.collection.data_schema import ControlType


def extract_target_config_from_sequence(sequence: dict[str, t.Any]) -> tuple[int, int]:
    """Extract target configuration (L, m) from sequence metadata.

    Args:
        sequence: Evaluation sequence containing metadata and features

    Returns:
        Tuple of (L, m) representing target configuration

    """
    # Try to get from new target_config fields
    if "target_config_L" in sequence and "target_config_m" in sequence:
        return (sequence["target_config_L"], sequence["target_config_m"])

    # Try to get from sequence metadata if available
    if "target_config" in sequence:
        config = sequence["target_config"]
        if isinstance(config, (list, tuple)) and len(config) == 2:
            return tuple(config)

    # Try to get from general config field
    if "config" in sequence:
        config = sequence["config"]
        if isinstance(config, (list, tuple)) and len(config) == 2:
            return tuple(config)

    # Try to infer from context complexity patterns
    context_features = sequence.get("context_features", [])
    if context_features:
        # Estimate depth based on context structure
        estimated_L = min(len(context_features), 4)  # Cap at reasonable depth

        # Estimate multiplicity based on feature diversity
        if context_features:
            unique_features = set()
            for features in context_features:
                if isinstance(features, (list, tuple)):
                    unique_features.update(features)
                else:
                    unique_features.add(features)
            estimated_m = min(len(unique_features) // 2, 4)  # Rough heuristic
            estimated_m = max(estimated_m, 2)  # Minimum m=2
        else:
            estimated_m = 2

        return (estimated_L, estimated_m)

    # Final fallback to default configuration
    return (2, 2)


def create_control_sequence(
    sequence: dict[str, t.Any], control_type: ControlType, all_sequences: list[dict[str, t.Any]] | None = None
) -> dict[str, t.Any]:
    """Create control sequence based on specified control type.

    Args:
        sequence: Original sequence to create control from
        control_type: Type of control ("normal", "shuffled_context", "random_context")
        all_sequences: All available sequences for random context selection

    Returns:
        Control sequence with appropriate modifications

    """
    if control_type == "normal":
        return sequence

    if control_type == "shuffled_context":
        return _create_shuffled_context_sequence(sequence)

    if control_type == "random_context":
        if all_sequences:
            return _create_random_context_sequence(sequence, all_sequences)
        # Fallback to original if no sequences available
        return sequence

    # Unknown control type, return original
    return sequence


def _create_shuffled_context_sequence(sequence: dict[str, t.Any]) -> dict[str, t.Any]:
    """Create sequence with shuffled context order."""
    shuffled_seq = sequence.copy()

    # Get context features and labels
    context_features = shuffled_seq.get("context_features", [])
    context_labels = shuffled_seq.get("context_labels", [])

    if len(context_features) != len(context_labels):
        return sequence  # Return original if inconsistent

    # Create paired list and shuffle
    context_pairs = list(zip(context_features, context_labels, strict=False))
    random.shuffle(context_pairs)

    # Unpack shuffled pairs
    shuffled_seq["context_features"] = [pair[0] for pair in context_pairs]
    shuffled_seq["context_labels"] = [pair[1] for pair in context_pairs]

    return shuffled_seq


def _create_random_context_sequence(
    sequence: dict[str, t.Any], all_sequences: list[dict[str, t.Any]]
) -> dict[str, t.Any]:
    """Create sequence with random context from other sequences."""
    random_seq = sequence.copy()

    target_context_size = sequence.get("context_size", len(sequence.get("context_features", [])))

    # Find other sequences with same context size
    same_k_sequences = [
        seq for seq in all_sequences if seq.get("context_size") == target_context_size and seq != sequence
    ]

    # Need enough sequences to sample from
    if len(same_k_sequences) < target_context_size:
        return sequence  # Return original if not enough sequences

    # Sample random contexts
    try:
        random_contexts = random.sample(same_k_sequences, target_context_size)
        random_seq["context_features"] = [ctx["query_features"] for ctx in random_contexts]
        random_seq["context_labels"] = [ctx["query_label"] for ctx in random_contexts]
        return random_seq
    except (KeyError, ValueError):
        # Return original if sampling fails
        return sequence


def standardize_model_metadata(metadata_dict: dict[str, t.Any]) -> dict[str, t.Any]:
    """Standardize model metadata to consistent format.

    Args:
        metadata_dict: Raw metadata dictionary from various sources

    Returns:
        Standardized metadata dictionary with required fields

    """
    standardized = {}

    # Required fields with validation
    required_fields = {"config_L": int, "config_m": int, "n_train": int, "model_type": str}

    for field, expected_type in required_fields.items():
        if field not in metadata_dict:
            raise ValueError(f"Missing required field: {field}")

        try:
            standardized[field] = expected_type(metadata_dict[field])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {field}: {metadata_dict[field]}") from e

    # Validate model_type
    if standardized["model_type"] not in ["causal_lm", "mlm"]:
        raise ValueError(f"Invalid model_type: {standardized['model_type']}")

    # Optional fields with defaults
    optional_fields = {
        "checkpoint_step": (int, 0),
        "training_seed": (int, None),
        "eval_seed": (int, None),
    }

    for field, (field_type, default) in optional_fields.items():
        if field in metadata_dict:
            try:
                standardized[field] = field_type(metadata_dict[field])
            except (ValueError, TypeError):
                standardized[field] = default
        else:
            standardized[field] = default

    return standardized


def group_sequences_by_condition(eval_dataset: dict[str, t.Any]) -> dict[str, list[dict[str, t.Any]]]:
    """Group evaluation sequences by transfer condition for easier access.

    Args:
        eval_dataset: Complete evaluation dataset

    Returns:
        Dictionary mapping condition names to lists of sequences

    """
    grouped_sequences = defaultdict(list)
    conditions = eval_dataset.get("conditions", {})

    # Handle within_config (list format)
    within_config_data = conditions.get("within_config", [])
    for model_data in within_config_data:
        sequences = []
        for k_sequences in model_data.get("sequences", {}).values():
            sequences.extend(k_sequences)
        grouped_sequences["within_config"].extend(sequences)

    # Handle transfer conditions (nested dict format)
    transfer_conditions = {"depth_transfer": "cross_L", "synonym_transfer": "cross_m", "full_transfer": "cross_config"}

    for dataset_key, condition_name in transfer_conditions.items():
        transfer_data = conditions.get(dataset_key, {})
        sequences = []

        for config_key, config_models in transfer_data.items():
            for model_data in config_models:
                for k_sequences in model_data.get("sequences", {}).values():
                    # Add config information to sequences
                    for seq in k_sequences:
                        if "config" not in seq and "config" in model_data:
                            seq["config"] = model_data["config"]
                    sequences.extend(k_sequences)

        grouped_sequences[condition_name].extend(sequences)

    return dict(grouped_sequences)


def validate_sequence_structure(sequence: dict[str, t.Any]) -> bool:
    """Validate that sequence has required structure for evaluation.

    Args:
        sequence: Sequence dictionary to validate

    Returns:
        True if sequence is valid, False otherwise

    """
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

    # Check query structure
    return not (not sequence["query_features"] or sequence["query_label"] is None)


def compute_sequence_statistics(sequences: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    """Compute basic statistics for a collection of sequences.

    Args:
        sequences: List of evaluation sequences

    Returns:
        Dictionary containing sequence statistics

    """
    if not sequences:
        return {"total_sequences": 0}

    # Basic counts
    stats = {
        "total_sequences": len(sequences),
        "valid_sequences": sum(1 for seq in sequences if validate_sequence_structure(seq)),
    }

    # Context size distribution
    context_sizes = []
    for seq in sequences:
        if validate_sequence_structure(seq):
            context_sizes.append(len(seq["context_features"]))

    if context_sizes:
        stats.update(
            {
                "context_size_min": min(context_sizes),
                "context_size_max": max(context_sizes),
                "context_size_mean": sum(context_sizes) / len(context_sizes),
                "context_size_distribution": {size: context_sizes.count(size) for size in set(context_sizes)},
            }
        )

    # Configuration distribution
    configs = []
    for seq in sequences:
        config = extract_target_config_from_sequence(seq)
        configs.append(config)

    if configs:
        config_counts = defaultdict(int)
        for config in configs:
            config_counts[config] += 1

        stats["config_distribution"] = dict(config_counts)

    return stats
