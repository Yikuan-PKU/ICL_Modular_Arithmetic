"""Metrics for transfer learning analysis (Experiment 2)."""

import typing as t
from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class TransferMetrics:
    """Metrics for transfer learning analysis."""

    model_id: str
    train_config_L: int
    train_config_m: int
    test_config_L: int
    test_config_m: int
    n_train: int
    within_config_accuracy: float
    transfer_accuracy: float
    transfer_degradation: float
    transfer_type: str  # "depth", "synonym", "full"
    depth_increase: int
    synonym_increase: int
    transfer_success: bool
    transfer_scaling_slope: float

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "model_id": self.model_id,
            "train_config_L": self.train_config_L,
            "train_config_m": self.train_config_m,
            "test_config_L": self.test_config_L,
            "test_config_m": self.test_config_m,
            "n_train": self.n_train,
            "within_config_accuracy": self.within_config_accuracy,
            "transfer_accuracy": self.transfer_accuracy,
            "transfer_degradation": self.transfer_degradation,
            "transfer_type": self.transfer_type,
            "depth_increase": self.depth_increase,
            "synonym_increase": self.synonym_increase,
            "transfer_success": self.transfer_success,
            "transfer_scaling_slope": self.transfer_scaling_slope,
        }


class TransferMetricsCalculator:
    """Calculator for transfer learning metrics."""

    def __init__(self, success_threshold: float = 0.5):
        """Initialize calculator with success threshold."""
        self.success_threshold = success_threshold

    def create_transfer_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create within-config vs transfer performance pairs."""
        # Separate within-config and transfer conditions
        within_config = data[data["transfer_condition"] == "within_config"].copy()
        transfer_conditions = data[data["transfer_condition"] != "within_config"].copy()

        print(f"Within-config records: {len(within_config)}")
        print(f"Transfer records: {len(transfer_conditions)}")

        # Create pairs by matching models and context sizes
        pairs = []
        for _, transfer_row in transfer_conditions.iterrows():
            # Find matching within-config performance
            matching_within = within_config[
                (within_config["model_id"] == transfer_row["model_id"])
                & (within_config["context_size"] == transfer_row["context_size"])
                & (within_config["config_L"] == transfer_row["config_L"])
                & (within_config["config_m"] == transfer_row["config_m"])
            ]

            if len(matching_within) > 0:
                within_row = matching_within.iloc[0]  # Take first match
                pair = {
                    "model_id": transfer_row["model_id"],
                    "train_config_L": transfer_row["config_L"],
                    "train_config_m": transfer_row["config_m"],
                    "test_config_L": transfer_row["target_config_L"],
                    "test_config_m": transfer_row["target_config_m"],
                    "n_train": transfer_row["n_train"],
                    "context_size": transfer_row["context_size"],
                    "within_accuracy": within_row["accuracy"],
                    "transfer_accuracy": transfer_row["accuracy"],
                    "transfer_condition": transfer_row["transfer_condition"],
                    "checkpoint_step": transfer_row["checkpoint_step"],
                }
                pairs.append(pair)

        transfer_pairs = pd.DataFrame(pairs)
        print(f"Created {len(transfer_pairs)} transfer pairs")

        return transfer_pairs

    def compute_transfer_metrics(self, transfer_pairs: pd.DataFrame, data: pd.DataFrame) -> list[TransferMetrics]:
        """Compute transfer degradation metrics for each transfer pair."""
        metrics_list = []

        for _, pair in transfer_pairs.iterrows():
            metrics = self._compute_single_transfer_metrics(pair, data)
            if metrics:
                metrics_list.append(metrics)

        print(f"Computed transfer metrics for {len(metrics_list)} transfer pairs")
        return metrics_list

    def _compute_single_transfer_metrics(self, pair: pd.Series, data: pd.DataFrame) -> TransferMetrics | None:
        """Compute transfer metrics for a single transfer pair."""
        try:
            # Basic transfer metrics
            transfer_degradation = pair["within_accuracy"] - pair["transfer_accuracy"]
            transfer_success = pair["transfer_accuracy"] > self.success_threshold

            # Determine transfer type and increases
            train_L, train_m = pair["train_config_L"], pair["train_config_m"]
            test_L, test_m = pair["test_config_L"], pair["test_config_m"]

            depth_increase = test_L - train_L
            synonym_increase = test_m - train_m

            transfer_type = self._classify_transfer_type(pair["transfer_condition"], depth_increase, synonym_increase)

            # Compute transfer scaling slope (accuracy vs context size for this transfer)
            transfer_scaling_slope = self._compute_transfer_scaling_slope(
                data, pair["model_id"], train_L, train_m, test_L, test_m
            )

            return TransferMetrics(
                model_id=pair["model_id"],
                train_config_L=train_L,
                train_config_m=train_m,
                test_config_L=test_L,
                test_config_m=test_m,
                n_train=pair["n_train"],
                within_config_accuracy=pair["within_accuracy"],
                transfer_accuracy=pair["transfer_accuracy"],
                transfer_degradation=transfer_degradation,
                transfer_type=transfer_type,
                depth_increase=depth_increase,
                synonym_increase=synonym_increase,
                transfer_success=transfer_success,
                transfer_scaling_slope=transfer_scaling_slope,
            )

        except Exception as e:
            print(f"Failed to compute transfer metrics for pair: {e}")
            return None

    def _classify_transfer_type(self, transfer_condition: str, depth_increase: int, synonym_increase: int) -> str:
        """Classify transfer type based on condition and configuration changes."""
        if transfer_condition == "cross_L" or (depth_increase > 0 and synonym_increase == 0):
            return "depth"
        if transfer_condition == "cross_m" or (depth_increase == 0 and synonym_increase > 0):
            return "synonym"
        if transfer_condition == "cross_config" or (depth_increase > 0 and synonym_increase > 0):
            return "full"
        return "unknown"

    def _compute_transfer_scaling_slope(
        self, data: pd.DataFrame, model_id: str, train_L: int, train_m: int, test_L: int, test_m: int
    ) -> float:
        """Compute transfer scaling slope across context sizes."""
        # Get transfer data for this specific model and config pair
        transfer_data = data[
            (data["model_id"] == model_id)
            & (data["config_L"] == train_L)
            & (data["config_m"] == train_m)
            & (data["target_config_L"] == test_L)
            & (data["target_config_m"] == test_m)
            & (data["transfer_condition"] != "within_config")
        ]

        if len(transfer_data) < 3:  # Need sufficient points
            return 0.0

        # Compute slope across context sizes
        context_accuracies = transfer_data.groupby("context_size")["accuracy"].mean()
        if len(context_accuracies) < 2:
            return 0.0

        context_sizes = context_accuracies.index.values
        accuracies = context_accuracies.values

        try:
            slope, _, _, _, _ = stats.linregress(context_sizes, accuracies)
            return slope
        except:
            return 0.0

    def analyze_transfer_patterns(self, metrics_list: list[TransferMetrics]) -> dict[str, t.Any]:
        """Analyze transfer patterns across different transfer types."""
        if not metrics_list:
            return {}

        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame([m.to_dict() for m in metrics_list])

        patterns = {
            "transfer_type_analysis": self._analyze_transfer_types(metrics_df),
            "degradation_patterns": self._analyze_degradation_patterns(metrics_df),
            "success_rates": self._analyze_success_rates(metrics_df),
            "scaling_effects": self._analyze_scaling_effects(metrics_df),
            "config_effects": self._analyze_config_transfer_effects(metrics_df),
        }

        return patterns

    def _analyze_transfer_types(self, metrics_df: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze performance by transfer type."""
        type_analysis = {}

        for transfer_type in ["depth", "synonym", "full"]:
            type_data = metrics_df[metrics_df["transfer_type"] == transfer_type]
            if len(type_data) > 0:
                type_analysis[transfer_type] = {
                    "count": len(type_data),
                    "mean_degradation": type_data["transfer_degradation"].mean(),
                    "std_degradation": type_data["transfer_degradation"].std(),
                    "success_rate": type_data["transfer_success"].mean(),
                    "mean_within_accuracy": type_data["within_config_accuracy"].mean(),
                    "mean_transfer_accuracy": type_data["transfer_accuracy"].mean(),
                }

        return type_analysis

    def _analyze_degradation_patterns(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze transfer degradation patterns."""
        patterns = {
            "overall_mean_degradation": metrics_df["transfer_degradation"].mean(),
            "overall_degradation_std": metrics_df["transfer_degradation"].std(),
            "positive_degradation_rate": (metrics_df["transfer_degradation"] > 0).mean(),
            "severe_degradation_rate": (metrics_df["transfer_degradation"] > 0.2).mean(),
            "improvement_rate": (metrics_df["transfer_degradation"] < 0).mean(),
            "median_degradation": metrics_df["transfer_degradation"].median(),
        }

        return patterns

    def _analyze_success_rates(self, metrics_df: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze transfer success rates."""
        success_rates = {"overall_success_rate": metrics_df["transfer_success"].mean()}

        # Success by training size
        training_sizes = sorted(metrics_df["n_train"].unique())
        success_by_training = {}
        for n_train in training_sizes:
            train_data = metrics_df[metrics_df["n_train"] == n_train]
            success_by_training[n_train] = train_data["transfer_success"].mean()
        success_rates["success_by_training_size"] = success_by_training

        # Success by depth increase
        depth_increases = sorted(metrics_df["depth_increase"].unique())
        success_by_depth = {}
        for depth_inc in depth_increases:
            depth_data = metrics_df[metrics_df["depth_increase"] == depth_inc]
            if len(depth_data) > 0:
                success_by_depth[depth_inc] = depth_data["transfer_success"].mean()
        success_rates["success_by_depth_increase"] = success_by_depth

        # Success by synonym increase
        synonym_increases = sorted(metrics_df["synonym_increase"].unique())
        success_by_synonym = {}
        for syn_inc in synonym_increases:
            syn_data = metrics_df[metrics_df["synonym_increase"] == syn_inc]
            if len(syn_data) > 0:
                success_by_synonym[syn_inc] = syn_data["transfer_success"].mean()
        success_rates["success_by_synonym_increase"] = success_by_synonym

        return success_rates

    def _analyze_scaling_effects(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze transfer scaling effects."""
        scaling_effects = {
            "mean_transfer_scaling_slope": metrics_df["transfer_scaling_slope"].mean(),
            "positive_scaling_rate": (metrics_df["transfer_scaling_slope"] > 0).mean(),
            "strong_scaling_rate": (metrics_df["transfer_scaling_slope"] > 0.05).mean(),
            "scaling_vs_degradation_correlation": metrics_df["transfer_scaling_slope"].corr(
                metrics_df["transfer_degradation"]
            ),
        }

        return scaling_effects

    def _analyze_config_transfer_effects(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze how configuration differences affect transfer."""
        config_effects = {}

        # Effect of depth increase on degradation
        depth_corr = metrics_df["depth_increase"].corr(metrics_df["transfer_degradation"])
        config_effects["depth_increase_degradation_correlation"] = depth_corr

        # Effect of synonym increase on degradation
        syn_corr = metrics_df["synonym_increase"].corr(metrics_df["transfer_degradation"])
        config_effects["synonym_increase_degradation_correlation"] = syn_corr

        # Effect of total complexity change
        metrics_df["complexity_change"] = (
            metrics_df["test_config_L"] * metrics_df["test_config_m"]
            - metrics_df["train_config_L"] * metrics_df["train_config_m"]
        )
        complexity_corr = metrics_df["complexity_change"].corr(metrics_df["transfer_degradation"])
        config_effects["complexity_change_degradation_correlation"] = complexity_corr

        return config_effects

    def create_transfer_matrix(self, metrics_list: list[TransferMetrics]) -> dict[str, pd.DataFrame]:
        """Create train→test configuration transfer performance matrix."""
        if not metrics_list:
            return {}

        # Create matrix data
        matrix_data = []
        for metric in metrics_list:
            train_config = f"({metric.train_config_L}, {metric.train_config_m})"
            test_config = f"({metric.test_config_L}, {metric.test_config_m})"
            matrix_data.append(
                {
                    "train_config": train_config,
                    "test_config": test_config,
                    "transfer_accuracy": metric.transfer_accuracy,
                    "transfer_degradation": metric.transfer_degradation,
                    "n_samples": 1,
                }
            )

        matrix_df = pd.DataFrame(matrix_data)

        # Aggregate by config pairs (average across models)
        aggregated = (
            matrix_df.groupby(["train_config", "test_config"])
            .agg({"transfer_accuracy": "mean", "transfer_degradation": "mean", "n_samples": "sum"})
            .reset_index()
        )

        # Pivot to create transfer matrices
        accuracy_matrix = aggregated.pivot(index="train_config", columns="test_config", values="transfer_accuracy")

        degradation_matrix = aggregated.pivot(
            index="train_config", columns="test_config", values="transfer_degradation"
        )

        print(f"Created transfer matrix: {accuracy_matrix.shape}")

        return {"accuracy": accuracy_matrix, "degradation": degradation_matrix, "raw_data": aggregated}
