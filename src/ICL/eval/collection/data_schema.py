"""Data schemas for comprehensive evaluation results."""

import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Type aliases
ConfigTuple = tuple[int, int]
TransferCondition = t.Literal["within_config", "cross_L", "cross_m", "cross_config"]
ControlType = t.Literal["normal", "shuffled_context", "random_context"]


@dataclass
class ModelMetadata:
    """Metadata for a single model checkpoint."""

    model_id: str
    config_L: int
    config_m: int
    n_train: int
    checkpoint_step: int
    checkpoint_path: Path
    model_type: str  # "causal_lm" or "mlm"
    training_seed: int | None = None
    eval_seed: int | None = None

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "n_train": self.n_train,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": str(self.checkpoint_path),
            "model_type": self.model_type,
            "training_seed": self.training_seed,
            "eval_seed": self.eval_seed,
        }


@dataclass
class ICLPerformanceRecord:
    """Single ICL evaluation result record."""

    model_id: str
    config_L: int
    config_m: int
    n_train: int
    checkpoint_step: int
    context_size: int
    transfer_condition: TransferCondition
    target_config_L: int
    target_config_m: int
    accuracy: float
    sequence_id: int
    control_type: ControlType
    evaluation_timestamp: datetime
    num_sequences: int = 0
    num_correct: int = 0

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "model_id": self.model_id,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "n_train": self.n_train,
            "checkpoint_step": self.checkpoint_step,
            "context_size": self.context_size,
            "transfer_condition": self.transfer_condition,
            "target_config_L": self.target_config_L,
            "target_config_m": self.target_config_m,
            "accuracy": self.accuracy,
            "sequence_id": self.sequence_id,
            "control_type": self.control_type,
            "evaluation_timestamp": self.evaluation_timestamp,
            "num_sequences": self.num_sequences,
            "num_correct": self.num_correct,
        }


@dataclass
class AttentionRecord:
    """Single attention pattern record."""

    model_id: str
    layer_idx: int
    head_idx: int
    context_size: int
    sequence_id: int
    attention_matrix: np.ndarray
    evaluation_timestamp: datetime

    def get_filename(self) -> str:
        """Generate filename for attention data."""
        return (
            f"{self.model_id}_layer{self.layer_idx}_head{self.head_idx}_k{self.context_size}_seq{self.sequence_id}.npz"
        )


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""

    # Model and data paths
    checkpoint_base_dirs: list[Path]
    eval_dataset_path: Path
    output_dir: Path

    # Evaluation parameters
    context_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])
    transfer_conditions: list[TransferCondition] = field(
        default_factory=lambda: ["within_config", "cross_L", "cross_m", "cross_config"]
    )
    control_types: list[ControlType] = field(default_factory=lambda: ["normal", "shuffled_context", "random_context"])

    # Model configurations to evaluate
    target_configs: list[ConfigTuple] = field(default_factory=list)
    diversity_levels: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    model_types: list[str] = field(default_factory=lambda: ["causal_lm", "mlm"])

    # Computational parameters
    device: str = "cuda"
    batch_size: int = 32
    max_sequences_per_condition: int = 200
    capture_attention: bool = True
    capture_representations: bool = False

    # Output control
    save_intermediate: bool = True
    overwrite_existing: bool = False

    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check paths exist
        for checkpoint_dir in self.checkpoint_base_dirs:
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        if not self.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.eval_dataset_path}")

        # Validate parameters
        if not self.context_sizes or any(k <= 0 for k in self.context_sizes):
            raise ValueError("Context sizes must be positive integers")

        if not self.target_configs:
            raise ValueError("Must specify target configurations to evaluate")

        return True


class DataSchemaManager:
    """Manages data schemas and DataFrame operations."""

    @staticmethod
    def create_icl_performance_schema() -> dict[str, str]:
        """Define ICL performance DataFrame schema."""
        return {
            "model_id": "string",
            "config_L": "int32",
            "config_m": "int32",
            "n_train": "int32",
            "checkpoint_step": "int32",
            "context_size": "int32",
            "transfer_condition": "category",
            "target_config_L": "int32",
            "target_config_m": "int32",
            "accuracy": "float64",
            "sequence_id": "int32",
            "control_type": "category",
            "evaluation_timestamp": "datetime64[ns]",
            "num_sequences": "int32",
            "num_correct": "int32",
        }

    @staticmethod
    def create_model_metadata_schema() -> dict[str, str]:
        """Define model metadata DataFrame schema."""
        return {
            "model_id": "string",
            "config_L": "int32",
            "config_m": "int32",
            "n_train": "int32",
            "checkpoint_step": "int32",
            "checkpoint_path": "string",
            "model_type": "category",
            "training_seed": "Int32",  # Nullable integer
            "eval_seed": "Int32",  # Nullable integer
        }

    @staticmethod
    def records_to_dataframe(records: list[ICLPerformanceRecord]) -> pd.DataFrame:
        """Convert ICL performance records to typed DataFrame."""
        if not records:
            # Return empty DataFrame with correct schema
            schema = DataSchemaManager.create_icl_performance_schema()
            return pd.DataFrame().astype(schema)

        data = [record.to_dict() for record in records]
        df = pd.DataFrame(data)

        # Apply schema
        schema = DataSchemaManager.create_icl_performance_schema()
        for col, dtype in schema.items():
            if col in df.columns:
                if dtype == "category":
                    df[col] = df[col].astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)

        return df

    @staticmethod
    def metadata_to_dataframe(metadata: list[ModelMetadata]) -> pd.DataFrame:
        """Convert model metadata to typed DataFrame."""
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


def create_evaluation_manifest(
    config: EvaluationConfig, start_time: datetime, end_time: datetime | None = None, status: str = "running"
) -> dict[str, t.Any]:
    """Create evaluation run manifest."""
    return {
        "experiment_id": f"comprehensive_eval_{start_time.strftime('%Y%m%d_%H%M%S')}",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "status": status,
        "config": {
            "context_sizes": config.context_sizes,
            "transfer_conditions": config.transfer_conditions,
            "control_types": config.control_types,
            "target_configs": config.target_configs,
            "diversity_levels": config.diversity_levels,
            "model_types": config.model_types,
            "device": config.device,
            "batch_size": config.batch_size,
            "max_sequences_per_condition": config.max_sequences_per_condition,
            "capture_attention": config.capture_attention,
            "capture_representations": config.capture_representations,
        },
        "data_schema_version": "1.0",
        "output_files": {
            "icl_performance": "raw_evaluations/icl_performance.parquet",
            "model_registry": "metadata/model_registry.parquet",
            "attention_data": "raw_evaluations/attention_data/",
            "intermediate_metrics": "intermediate/aggregated_metrics.parquet",
        },
    }
