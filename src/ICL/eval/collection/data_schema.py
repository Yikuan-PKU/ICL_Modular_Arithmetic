"""Minimal data schemas - only parsing essential fields from model configs."""

import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# Type aliases
ControlType = t.Literal["normal", "shuffled_context", "random_context"]
EvalType = t.Literal["memorization", "id_generalization", "ood_same_rule", "ood_transfer"]
TrainingPhase = t.Literal["early", "mid", "late"]


@dataclass
class ModelMetadata:
    """Minimal metadata - only essential fields from model configs."""

    # Core experiment identification (from shared args)
    dataset_type: str  # "uniform"
    num_seeds: int  # 10
    seed: int  # 42
    config_L: int  # 4 (auto-discovered from directory structure)
    config_m: int  # 2 (auto-discovered from directory structure)

    # Model variant identification (minimal from config)
    task_name: str  # "clm" or "mlm" (from config["task_name"])
    model_variant: str  # "clm_noshuffle_seedbalanced" (generated)

    # Only the config params needed for variant generation
    shuffle_before_packing: bool  # from config["shuffle_before_packing"]
    seed_balanced_batching: bool  # from config["seed_balanced_batching"]

    # Checkpoint information
    checkpoint_step: int
    checkpoint_path: Path
    model_id: str  # "{model_variant}_step{checkpoint_step}_hash"

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_type": self.dataset_type,
            "num_seeds": self.num_seeds,
            "seed": self.seed,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "task_name": self.task_name,
            "model_variant": self.model_variant,
            "shuffle_before_packing": self.shuffle_before_packing,
            "seed_balanced_batching": self.seed_balanced_batching,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": str(self.checkpoint_path),
            "model_id": self.model_id,
        }


@dataclass
class ICLPerformanceRecord:
    """ICL evaluation result with minimal model config tracking."""

    # Core experiment identification
    dataset_type: str
    num_seeds: int
    seed: int
    config_L: int
    config_m: int

    # Model identification
    task_name: str  # "clm" or "mlm"
    model_variant: str  # "clm_noshuffle_seedbalanced"
    checkpoint_step: int
    model_id: str

    # Evaluation context
    eval_type: EvalType
    context_size: int
    control_type: ControlType
    sequence_id: int

    # Target configuration (for analysis)
    target_config_L: int
    target_config_m: int
    source_seeds: list[int]
    appears_in_training: bool

    # Results
    accuracy: float
    num_correct: int
    evaluation_timestamp: datetime

    # Training dynamics
    training_phase: TrainingPhase

    # Minimal model configuration (for analysis)
    shuffle_before_packing: bool
    seed_balanced_batching: bool

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "dataset_type": self.dataset_type,
            "num_seeds": self.num_seeds,
            "seed": self.seed,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "task_name": self.task_name,
            "model_variant": self.model_variant,
            "checkpoint_step": self.checkpoint_step,
            "model_id": self.model_id,
            "eval_type": self.eval_type,
            "context_size": self.context_size,
            "control_type": self.control_type,
            "sequence_id": self.sequence_id,
            "target_config_L": self.target_config_L,
            "target_config_m": self.target_config_m,
            "source_seeds": self.source_seeds,
            "appears_in_training": self.appears_in_training,
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "evaluation_timestamp": self.evaluation_timestamp,
            "training_phase": self.training_phase,
            "shuffle_before_packing": self.shuffle_before_packing,
            "seed_balanced_batching": self.seed_balanced_batching,
        }


@dataclass
class AttentionRecord:
    """Attention pattern record - minimal identification."""

    # Core experiment identification
    dataset_type: str
    num_seeds: int
    seed: int
    config_L: int
    config_m: int

    # Model identification
    model_variant: str
    checkpoint_step: int
    model_id: str
    eval_type: EvalType

    # Attention-specific fields
    layer_idx: int
    head_idx: int
    context_size: int
    sequence_id: int
    attention_matrix: t.Any  # numpy array
    evaluation_timestamp: datetime

    def get_filename(self) -> str:
        """Generate filename for attention data."""
        return (
            f"{self.model_id}_layer{self.layer_idx}_head{self.head_idx}_k{self.context_size}_seq{self.sequence_id}.npz"
        )


def generate_model_variant_name(model_config: dict[str, t.Any]) -> str:
    """Generate model variant name from minimal config fields."""
    # Only parse the 3 essential fields
    task_name = model_config.get("task_name", "unknown")
    shuffle_before_packing = model_config.get("shuffle_before_packing", False)
    seed_balanced_batching = model_config.get("seed_balanced_batching", True)

    # Generate name: clm_noshuffle_seedbalanced
    shuffle_suffix = "shuffle" if shuffle_before_packing else "noshuffle"
    seedbalanced_suffix = "seedbalanced" if seed_balanced_batching else "noseedbalanced"

    return f"{task_name}_{shuffle_suffix}_{seedbalanced_suffix}"


def load_minimal_model_config(config_path: Path) -> dict[str, t.Any]:
    """Load only the essential fields from model configuration."""
    import yaml

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    # Extract only the 3 fields we need
    minimal_config = {
        "task_name": full_config.get("task_name", "unknown"),
        "shuffle_before_packing": full_config.get("shuffle_before_packing", False),
        "seed_balanced_batching": full_config.get("seed_balanced_batching", True),
    }

    return minimal_config


def parse_shared_identifier(shared_id: str) -> dict[str, t.Any]:
    """Parse shared identifier: uniform_10_L4_M2 → components."""
    parts = shared_id.split("_")

    if len(parts) < 4:
        raise ValueError(f"Invalid shared identifier format: {shared_id}")

    dataset_type = parts[0]  # "uniform"
    num_seeds = int(parts[1])  # 10

    # Find L and M
    L = None
    M = None
    for part in parts[2:]:
        if part.startswith("L"):
            L = int(part[1:])  # L4 → 4
        elif part.startswith("M"):
            M = int(part[1:])  # M2 → 2

    if L is None or M is None:
        raise ValueError(f"Could not parse L and M from: {shared_id}")

    return {
        "dataset_type": dataset_type,
        "num_seeds": num_seeds,
        "L": L,
        "M": M,
    }


def determine_training_phase(checkpoint_step: int, max_step: int) -> TrainingPhase:
    """Determine training phase based on checkpoint step."""
    if max_step == 0:
        return "unknown"

    progress = checkpoint_step / max_step
    if progress <= 0.33:
        return "early"
    if progress <= 0.66:
        return "mid"
    return "late"


# Simplified DataFrame schemas
class DataSchemaManager:
    """Simplified DataFrame management."""

    @staticmethod
    def create_icl_performance_schema() -> dict[str, str]:
        """Minimal ICL performance schema."""
        return {
            # Core experiment
            "dataset_type": "string",
            "num_seeds": "int32",
            "seed": "int32",
            "config_L": "int32",
            "config_m": "int32",
            # Model identification
            "task_name": "string",
            "model_variant": "string",
            "checkpoint_step": "int32",
            "model_id": "string",
            # Evaluation context
            "eval_type": "category",
            "context_size": "int32",
            "control_type": "category",
            "sequence_id": "int32",
            # Target configuration
            "target_config_L": "int32",
            "target_config_m": "int32",
            "source_seeds": "object",  # List[int]
            "appears_in_training": "bool",
            # Results
            "accuracy": "float64",
            "num_correct": "int32",
            "evaluation_timestamp": "datetime64[ns]",
            # Training dynamics
            "training_phase": "category",
            # Minimal model config
            "shuffle_before_packing": "bool",
            "seed_balanced_batching": "bool",
        }

    @staticmethod
    def create_model_metadata_schema() -> dict[str, str]:
        """Minimal model metadata schema."""
        return {
            "dataset_type": "string",
            "num_seeds": "int32",
            "seed": "int32",
            "config_L": "int32",
            "config_m": "int32",
            "task_name": "string",
            "model_variant": "string",
            "shuffle_before_packing": "bool",
            "seed_balanced_batching": "bool",
            "checkpoint_step": "int32",
            "checkpoint_path": "string",
            "model_id": "string",
        }

    @staticmethod
    def records_to_dataframe(records: list[ICLPerformanceRecord]) -> pd.DataFrame:
        """Convert records to DataFrame with minimal schema."""
        if not records:
            schema = DataSchemaManager.create_icl_performance_schema()
            return pd.DataFrame().astype(schema)

        data = [record.to_dict() for record in records]
        df = pd.DataFrame(data)

        # Apply schema
        schema = DataSchemaManager.create_icl_performance_schema()
        for col, dtype in schema.items():
            if col in df.columns:
                if dtype == "object":  # Handle list columns
                    continue
                df[col] = df[col].astype(dtype)

        return df

    @staticmethod
    def metadata_to_dataframe(metadata: list[ModelMetadata]) -> pd.DataFrame:
        """Convert metadata to DataFrame with minimal schema."""
        if not metadata:
            schema = DataSchemaManager.create_model_metadata_schema()
            return pd.DataFrame().astype(schema)

        data = [meta.to_dict() for meta in metadata]
        df = pd.DataFrame(data)

        # Apply schema
        schema = DataSchemaManager.create_model_metadata_schema()
        for col, dtype in schema.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        return df
