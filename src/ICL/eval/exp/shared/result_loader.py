"""Standardized loader for Phase 1 collection results."""

import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


class Phase1ResultLoader:
    """Standardized loader for Phase 1 collection results."""

    def __init__(self, collection_results_dir: Path):
        """Initialize loader with collection results directory."""
        self.collection_results_dir = collection_results_dir
        self.icl_performance_path = collection_results_dir / "raw_evaluations" / "icl_performance.parquet"
        self.model_registry_path = collection_results_dir / "metadata" / "model_registry.parquet"
        self.attention_data_dir = collection_results_dir / "raw_evaluations" / "attention_data"

    def load_icl_performance(self) -> pd.DataFrame:
        """Load and validate ICL performance data."""
        if not self.icl_performance_path.exists():
            raise FileNotFoundError(f"ICL performance data not found: {self.icl_performance_path}")

        df = pd.read_parquet(self.icl_performance_path)

        # Validate required columns
        required_columns = [
            "model_id",
            "config_L",
            "config_m",
            "n_train",
            "checkpoint_step",
            "context_size",
            "transfer_condition",
            "target_config_L",
            "target_config_m",
            "accuracy",
            "sequence_id",
            "control_type",
            "evaluation_timestamp",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in ICL performance data: {missing_columns}")

        print(f"Loaded ICL performance data: {len(df)} records")
        print(f"  Transfer conditions: {sorted(df['transfer_condition'].unique())}")
        print(f"  Control types: {sorted(df['control_type'].unique())}")
        print(f"  Context sizes: {sorted(df['context_size'].unique())}")
        print(f"  Unique models: {df['model_id'].nunique()}")

        return df

    def load_model_registry(self) -> pd.DataFrame:
        """Load model registry with metadata."""
        if not self.model_registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self.model_registry_path}")

        df = pd.read_parquet(self.model_registry_path)

        # Validate required columns
        required_columns = [
            "model_id",
            "config_L",
            "config_m",
            "n_train",
            "checkpoint_step",
            "checkpoint_path",
            "model_type",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in model registry: {missing_columns}")

        print(f"Loaded model registry: {len(df)} models")
        print(f"  Model types: {sorted(df['model_type'].unique())}")
        print(f"  Configurations: {len(df.groupby(['config_L', 'config_m']))}")
        print(f"  Training sizes: {sorted(df['n_train'].unique())}")

        return df

    def load_attention_data(self, model_ids: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Load attention data if available."""
        if not self.attention_data_dir.exists():
            warnings.warn(f"Attention data directory not found: {self.attention_data_dir}")
            return {}

        attention_data = {}

        # Get model directories
        model_dirs = [d for d in self.attention_data_dir.iterdir() if d.is_dir()]

        if model_ids:
            model_dirs = [d for d in model_dirs if d.name in model_ids]

        for model_dir in model_dirs:
            model_id = model_dir.name
            model_attention = {}

            # Load attention files for this model
            attention_files = list(model_dir.glob("*.npz"))

            for attention_file in attention_files:
                try:
                    data = np.load(attention_file)
                    attention_matrix = data["attention_matrix"]
                    metadata = data["metadata"].item()  # Convert 0-d array to dict

                    # Create key from metadata
                    key = f"layer_{metadata['layer_idx']}_head_{metadata['head_idx']}_k{metadata['context_size']}_seq{metadata['sequence_id']}"
                    model_attention[key] = {"attention_matrix": attention_matrix, "metadata": metadata}

                except Exception as e:
                    warnings.warn(f"Failed to load attention file {attention_file}: {e}")
                    continue

            if model_attention:
                attention_data[model_id] = model_attention

        if attention_data:
            print(f"Loaded attention data for {len(attention_data)} models")
            total_matrices = sum(len(model_data) for model_data in attention_data.values())
            print(f"  Total attention matrices: {total_matrices}")
        else:
            print("No attention data found")

        return attention_data

    def validate_data_completeness(self, df: pd.DataFrame) -> dict[str, t.Any]:
        """Validate completeness of loaded data."""
        validation = {
            "total_records": len(df),
            "unique_models": df["model_id"].nunique(),
            "transfer_conditions": sorted(df["transfer_condition"].unique()),
            "control_types": sorted(df["control_type"].unique()),
            "context_sizes": sorted(df["context_size"].unique()),
            "configurations": len(df.groupby(["config_L", "config_m"])),
            "missing_data_issues": [],
        }

        # Check expected transfer conditions
        expected_conditions = ["within_config", "cross_L", "cross_m", "cross_config"]
        missing_conditions = [c for c in expected_conditions if c not in validation["transfer_conditions"]]
        if missing_conditions:
            validation["missing_data_issues"].append(f"Missing transfer conditions: {missing_conditions}")

        # Check expected control types
        expected_controls = ["normal", "shuffled_context", "random_context"]
        missing_controls = [c for c in expected_controls if c not in validation["control_types"]]
        if missing_controls:
            validation["missing_data_issues"].append(f"Missing control types: {missing_controls}")

        # Check context size coverage
        expected_context_sizes = [1, 2, 3, 4, 5, 6, 8]
        missing_context_sizes = [k for k in expected_context_sizes if k not in validation["context_sizes"]]
        if missing_context_sizes:
            validation["missing_data_issues"].append(f"Missing context sizes: {missing_context_sizes}")

        # Check model completeness
        incomplete_models = []
        for model_id in df["model_id"].unique():
            model_data = df[df["model_id"] == model_id]
            model_conditions = set(model_data["transfer_condition"].unique())

            if "within_config" not in model_conditions:
                incomplete_models.append(model_id)

        if incomplete_models:
            validation["missing_data_issues"].append(
                f"Models missing within_config data: {len(incomplete_models)} models"
            )

        return validation

    def filter_for_analysis(self, df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
        """Filter data for specific analysis type."""
        if analysis_type == "emergence":
            # Emergence analysis: within-config normal sequences only
            filtered = df[(df["transfer_condition"] == "within_config") & (df["control_type"] == "normal")].copy()

        elif analysis_type == "transfer":
            # Transfer analysis: all conditions, normal sequences only
            filtered = df[df["control_type"] == "normal"].copy()

        elif analysis_type == "scaling":
            # Scaling analysis: within-config normal sequences only
            filtered = df[(df["transfer_condition"] == "within_config") & (df["control_type"] == "normal")].copy()

        elif analysis_type == "attention":
            # Attention analysis: all data (need controls for comparison)
            filtered = df.copy()

        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        print(f"Filtered data for {analysis_type} analysis: {len(filtered)} records")
        return filtered

    def get_baseline_performance(self, df: pd.DataFrame) -> dict[str, float]:
        """Get baseline performance metrics for comparison."""
        baselines = {}

        # Random baseline (shuffled/random context)
        control_data = df[df["control_type"].isin(["shuffled_context", "random_context"])]
        if len(control_data) > 0:
            baselines["control_accuracy"] = control_data["accuracy"].mean()

        # Single context performance
        single_context = df[df["context_size"] == 1]
        if len(single_context) > 0:
            baselines["single_context_accuracy"] = single_context["accuracy"].mean()

        # Within-config normal performance
        within_normal = df[(df["transfer_condition"] == "within_config") & (df["control_type"] == "normal")]
        if len(within_normal) > 0:
            baselines["within_config_accuracy"] = within_normal["accuracy"].mean()

        return baselines
