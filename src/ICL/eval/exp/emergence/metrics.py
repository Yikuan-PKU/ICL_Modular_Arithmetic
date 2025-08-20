"""Metrics for ICL emergence analysis (Experiment 1)."""

import typing as t
from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class EmergenceMetrics:
    """Metrics for ICL emergence analysis."""

    model_id: str
    config_L: int
    config_m: int
    n_train: int
    emergence_threshold: float
    max_accuracy: float
    context_scaling_slope: float
    baseline_gap: float
    emergence_step: int | None
    training_size_effect: float
    config_complexity_effect: float

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "model_id": self.model_id,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "n_train": self.n_train,
            "emergence_threshold": self.emergence_threshold,
            "max_accuracy": self.max_accuracy,
            "context_scaling_slope": self.context_scaling_slope,
            "baseline_gap": self.baseline_gap,
            "emergence_step": self.emergence_step,
            "training_size_effect": self.training_size_effect,
            "config_complexity_effect": self.config_complexity_effect,
        }


class EmergenceMetricsCalculator:
    """Calculator for emergence metrics."""

    def __init__(self, emergence_threshold: float = 0.5):
        """Initialize calculator with emergence threshold."""
        self.emergence_threshold = emergence_threshold

    def compute_single_model_metrics(
        self, model_data: pd.DataFrame, control_data: pd.DataFrame | None = None
    ) -> EmergenceMetrics | None:
        """Compute emergence metrics for a single model."""
        try:
            # Extract model info
            model_id = model_data["model_id"].iloc[0]
            config_L = model_data["config_L"].iloc[0]
            config_m = model_data["config_m"].iloc[0]
            n_train = model_data["n_train"].iloc[0]

            # Get accuracy by context size
            context_accuracies = model_data.groupby("context_size")["accuracy"].mean().sort_index()
            if len(context_accuracies) < 3:  # Need sufficient data points
                return None

            # Emergence threshold (first context size > threshold accuracy)
            emergence_k = self._compute_emergence_threshold(context_accuracies)

            # Max accuracy across all context sizes
            max_accuracy = context_accuracies.max()

            # Context scaling slope using linear regression
            context_sizes = context_accuracies.index.values
            accuracies = context_accuracies.values
            slope, _, _, _, _ = stats.linregress(context_sizes, accuracies)

            # Baseline gap (compare with controls)
            baseline_gap = self._compute_baseline_gap(model_data, control_data)

            # Emergence step (steepest improvement)
            emergence_step = self._find_emergence_step(context_accuracies)

            # Effects (computed later in cross-model analysis)
            training_size_effect = 0.0
            config_complexity_effect = 0.0

            return EmergenceMetrics(
                model_id=model_id,
                config_L=config_L,
                config_m=config_m,
                n_train=n_train,
                emergence_threshold=emergence_k,
                max_accuracy=max_accuracy,
                context_scaling_slope=slope,
                baseline_gap=baseline_gap,
                emergence_step=emergence_step,
                training_size_effect=training_size_effect,
                config_complexity_effect=config_complexity_effect,
            )

        except Exception as e:
            print(f"Failed to compute metrics for model: {e}")
            return None

    def _compute_emergence_threshold(self, context_accuracies: pd.Series) -> float:
        """Compute emergence threshold (minimum k for >threshold accuracy)."""
        for context_size, accuracy in context_accuracies.items():
            if accuracy > self.emergence_threshold:
                return float(context_size)
        # No emergence observed
        return float("inf")

    def _compute_baseline_gap(self, model_data: pd.DataFrame, control_data: pd.DataFrame | None) -> float:
        """Compute gap between normal and control conditions."""
        if control_data is None or len(control_data) == 0:
            return 0.0

        normal_acc = model_data["accuracy"].mean()
        control_acc = control_data["accuracy"].mean()
        return normal_acc - control_acc

    def _find_emergence_step(self, context_accuracies: pd.Series) -> int | None:
        """Find context size with steepest accuracy improvement."""
        if len(context_accuracies) < 2:
            return None

        improvements = context_accuracies.diff().dropna()
        if len(improvements) == 0:
            return None

        max_improvement_idx = improvements.idxmax()
        return int(max_improvement_idx)

    def analyze_emergence_patterns(self, metrics_list: list[EmergenceMetrics]) -> dict[str, t.Any]:
        """Analyze cross-model emergence patterns."""
        if not metrics_list:
            return {}

        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame([m.to_dict() for m in metrics_list])

        patterns = {
            "config_effects": self._analyze_config_effects(metrics_df),
            "training_size_effects": self._analyze_training_size_effects(metrics_df),
            "emergence_statistics": self._compute_emergence_statistics(metrics_df),
            "scaling_patterns": self._analyze_scaling_patterns(metrics_df),
        }

        return patterns

    def _analyze_config_effects(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze how configuration affects emergence."""
        config_effects = {}

        # Effect of depth (L) on emergence threshold
        valid_thresholds = metrics_df[metrics_df["emergence_threshold"] != float("inf")]
        if len(valid_thresholds) > 0:
            corr_L = valid_thresholds["config_L"].corr(valid_thresholds["emergence_threshold"])
            config_effects["depth_threshold_correlation"] = corr_L

        # Effect of multiplicity (m) on max accuracy
        corr_m = metrics_df["config_m"].corr(metrics_df["max_accuracy"])
        config_effects["multiplicity_accuracy_correlation"] = corr_m

        # Complexity (L*m) effects
        metrics_df["complexity"] = metrics_df["config_L"] * metrics_df["config_m"]
        corr_complexity = metrics_df["complexity"].corr(metrics_df["max_accuracy"])
        config_effects["complexity_accuracy_correlation"] = corr_complexity

        return config_effects

    def _analyze_training_size_effects(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze how training size affects emergence."""
        training_effects = {}

        # Training size vs emergence threshold
        valid_thresholds = metrics_df[metrics_df["emergence_threshold"] != float("inf")]
        if len(valid_thresholds) > 0:
            corr_train = valid_thresholds["n_train"].corr(valid_thresholds["emergence_threshold"])
            training_effects["training_size_threshold_correlation"] = corr_train

        # Training size vs scaling slope
        corr_slope = metrics_df["n_train"].corr(metrics_df["context_scaling_slope"])
        training_effects["training_size_slope_correlation"] = corr_slope

        # Training size vs max accuracy
        corr_max_acc = metrics_df["n_train"].corr(metrics_df["max_accuracy"])
        training_effects["training_size_accuracy_correlation"] = corr_max_acc

        return training_effects

    def _compute_emergence_statistics(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Compute basic emergence statistics."""
        valid_thresholds = metrics_df[metrics_df["emergence_threshold"] != float("inf")]

        stats = {
            "emergence_rate": len(valid_thresholds) / len(metrics_df),
            "mean_emergence_threshold": valid_thresholds["emergence_threshold"].mean()
            if len(valid_thresholds) > 0
            else float("inf"),
            "median_emergence_threshold": valid_thresholds["emergence_threshold"].median()
            if len(valid_thresholds) > 0
            else float("inf"),
            "mean_max_accuracy": metrics_df["max_accuracy"].mean(),
            "std_max_accuracy": metrics_df["max_accuracy"].std(),
            "mean_baseline_gap": metrics_df["baseline_gap"].mean(),
            "mean_scaling_slope": metrics_df["context_scaling_slope"].mean(),
            "positive_slope_rate": (metrics_df["context_scaling_slope"] > 0).mean(),
        }

        return stats

    def _analyze_scaling_patterns(self, metrics_df: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze context scaling patterns."""
        patterns = {
            "positive_slope_rate": (metrics_df["context_scaling_slope"] > 0).mean(),
            "strong_scaling_rate": (metrics_df["context_scaling_slope"] > 0.1).mean(),
            "negative_slope_rate": (metrics_df["context_scaling_slope"] < 0).mean(),
            "slope_distribution": {
                "mean": metrics_df["context_scaling_slope"].mean(),
                "std": metrics_df["context_scaling_slope"].std(),
                "median": metrics_df["context_scaling_slope"].median(),
                "q25": metrics_df["context_scaling_slope"].quantile(0.25),
                "q75": metrics_df["context_scaling_slope"].quantile(0.75),
            },
        }

        return patterns
