"""Metrics for context scaling analysis (Experiment 3)."""

import typing as t
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


@dataclass
class ScalingMetrics:
    """Metrics for context length scaling analysis."""

    model_id: str
    config_L: int
    config_m: int
    n_train: int
    complexity_score: float

    # Optimal context sizes for different thresholds
    optimal_context_50: float
    optimal_context_70: float
    optimal_context_90: float

    # Scaling law parameters
    exponential_a: float
    exponential_b: float
    exponential_r2: float
    power_a: float
    power_b: float
    power_r2: float
    log_a: float
    log_b: float
    log_r2: float

    # Best fitting model
    best_model: str
    best_r2: float

    # Scaling characteristics
    saturation_point: float
    max_accuracy: float
    scaling_efficiency: float

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "model_id": self.model_id,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "n_train": self.n_train,
            "complexity_score": self.complexity_score,
            "optimal_context_50": self.optimal_context_50,
            "optimal_context_70": self.optimal_context_70,
            "optimal_context_90": self.optimal_context_90,
            "exponential_a": self.exponential_a,
            "exponential_b": self.exponential_b,
            "exponential_r2": self.exponential_r2,
            "power_a": self.power_a,
            "power_b": self.power_b,
            "power_r2": self.power_r2,
            "log_a": self.log_a,
            "log_b": self.log_b,
            "log_r2": self.log_r2,
            "best_model": self.best_model,
            "best_r2": self.best_r2,
            "saturation_point": self.saturation_point,
            "max_accuracy": self.max_accuracy,
            "scaling_efficiency": self.scaling_efficiency,
        }


class ScalingMetricsCalculator:
    """Calculator for context scaling metrics."""

    def __init__(self, performance_thresholds: list[float] = None, scaling_laws: list[str] = None):
        """Initialize calculator with performance thresholds and scaling laws."""
        self.performance_thresholds = performance_thresholds or [0.5, 0.7, 0.9]
        self.scaling_laws = scaling_laws or ["exponential", "power", "logarithmic"]

    def detect_optimal_context_sizes(self, data: pd.DataFrame) -> dict[str, t.Any]:
        """Detect optimal context sizes for different performance thresholds."""
        optimal_results = {"model_thresholds": [], "threshold_statistics": {}, "complexity_correlations": {}}

        # Group by model for individual analysis
        model_groups = data.groupby(["model_id", "config_L", "config_m", "n_train", "checkpoint_step"])

        for (model_id, config_L, config_m, n_train, checkpoint_step), group in model_groups:
            # Get context-accuracy curve for this model
            context_accuracies = group.groupby("context_size")["accuracy"].mean().sort_index()
            if len(context_accuracies) < 3:  # Need sufficient data points
                continue

            model_thresholds = {
                "model_id": model_id,
                "config_L": config_L,
                "config_m": config_m,
                "n_train": n_train,
                "complexity_score": config_L * config_m,
            }

            # Find optimal context size for each threshold
            for threshold in self.performance_thresholds:
                optimal_k = self._find_optimal_context_size(context_accuracies, threshold)
                threshold_key = f"optimal_context_{int(threshold * 100)}"
                model_thresholds[threshold_key] = optimal_k

            # Additional metrics
            model_thresholds["max_accuracy"] = context_accuracies.max()
            model_thresholds["saturation_point"] = self._find_saturation_point(context_accuracies)

            optimal_results["model_thresholds"].append(model_thresholds)

        # Compute threshold statistics
        if optimal_results["model_thresholds"]:
            threshold_df = pd.DataFrame(optimal_results["model_thresholds"])

            for threshold in self.performance_thresholds:
                thresh_key = f"optimal_context_{int(threshold * 100)}"
                valid_thresholds = threshold_df[threshold_df[thresh_key] != float("inf")]

                optimal_results["threshold_statistics"][f"threshold_{threshold}"] = {
                    "mean_optimal_k": valid_thresholds[thresh_key].mean()
                    if len(valid_thresholds) > 0
                    else float("inf"),
                    "std_optimal_k": valid_thresholds[thresh_key].std() if len(valid_thresholds) > 0 else 0,
                    "achievement_rate": len(valid_thresholds) / len(threshold_df),
                    "min_optimal_k": valid_thresholds[thresh_key].min() if len(valid_thresholds) > 0 else float("inf"),
                    "max_optimal_k": valid_thresholds[thresh_key].max() if len(valid_thresholds) > 0 else float("inf"),
                }

            # Complexity correlations
            for threshold in self.performance_thresholds:
                thresh_key = f"optimal_context_{int(threshold * 100)}"
                valid_data = threshold_df[threshold_df[thresh_key] != float("inf")]
                if len(valid_data) > 3:
                    complexity_corr = self._safe_correlation(valid_data, "complexity_score", thresh_key)
                    optimal_results["complexity_correlations"][f"threshold_{threshold}"] = complexity_corr

        print(f"Analyzed optimal context sizes for {len(optimal_results['model_thresholds'])} models")
        return optimal_results

    def fit_scaling_laws(self, data: pd.DataFrame) -> dict[str, t.Any]:
        """Fit different scaling functions to context-performance curves."""
        scaling_results = {"model_fits": [], "law_comparisons": {}, "parameter_distributions": {}}

        # Group by model for curve fitting
        model_groups = data.groupby(["model_id", "config_L", "config_m", "n_train", "checkpoint_step"])

        for (model_id, config_L, config_m, n_train, checkpoint_step), group in model_groups:
            context_accuracies = group.groupby("context_size")["accuracy"].mean().sort_index()
            if len(context_accuracies) < 4:  # Need sufficient points for fitting
                continue

            context_sizes = context_accuracies.index.values
            accuracies = context_accuracies.values

            # Fit different scaling laws
            model_fit = self._fit_scaling_laws_single_model(
                context_sizes, accuracies, model_id, config_L, config_m, n_train
            )
            if model_fit:
                scaling_results["model_fits"].append(model_fit)

        # Analyze law comparisons
        if scaling_results["model_fits"]:
            fits_df = pd.DataFrame(scaling_results["model_fits"])

            # Compare model performance
            law_counts = {}
            for law in self.scaling_laws:
                law_counts[f"{law}_wins"] = (fits_df["best_model"] == law).sum()
                law_counts[f"mean_{law}_r2"] = fits_df[f"{law}_r2"].mean()

            scaling_results["law_comparisons"] = law_counts

            # Parameter distributions
            param_distributions = {}
            for law in self.scaling_laws:
                for param in ["a", "b"]:
                    col_name = f"{law}_{param}"
                    if col_name in fits_df.columns:
                        param_distributions[col_name] = fits_df[col_name].describe().to_dict()

            scaling_results["parameter_distributions"] = param_distributions

        print(f"Fitted scaling laws for {len(scaling_results['model_fits'])} models")
        return scaling_results

    def analyze_complexity_scaling(self, data: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze how complexity affects context requirements."""
        complexity_results = {"complexity_effects": {}, "configuration_analysis": {}, "training_size_effects": {}}

        # Create comprehensive metrics DataFrame
        model_metrics = []
        model_groups = data.groupby(["model_id", "config_L", "config_m", "n_train", "checkpoint_step"])

        for (model_id, config_L, config_m, n_train, checkpoint_step), group in model_groups:
            context_accuracies = group.groupby("context_size")["accuracy"].mean().sort_index()
            if len(context_accuracies) < 3:
                continue

            complexity_score = config_L * config_m

            # Compute various scaling metrics
            optimal_50 = self._find_optimal_context_size(context_accuracies, 0.5)
            optimal_70 = self._find_optimal_context_size(context_accuracies, 0.7)
            max_accuracy = context_accuracies.max()

            # Scaling efficiency: accuracy per unit context
            max_context = max(context_accuracies.index)
            scaling_efficiency = max_accuracy / max_context if max_context > 0 else 0

            model_metrics.append(
                {
                    "model_id": model_id,
                    "config_L": config_L,
                    "config_m": config_m,
                    "n_train": n_train,
                    "complexity_score": complexity_score,
                    "optimal_context_50": optimal_50,
                    "optimal_context_70": optimal_70,
                    "max_accuracy": max_accuracy,
                    "scaling_efficiency": scaling_efficiency,
                }
            )

        if not model_metrics:
            return complexity_results

        metrics_df = pd.DataFrame(model_metrics)

        # Analyze complexity effects
        complexity_results["complexity_effects"] = {
            "complexity_vs_optimal_50": self._safe_correlation(metrics_df, "complexity_score", "optimal_context_50"),
            "complexity_vs_optimal_70": self._safe_correlation(metrics_df, "complexity_score", "optimal_context_70"),
            "complexity_vs_max_accuracy": self._safe_correlation(metrics_df, "complexity_score", "max_accuracy"),
            "complexity_vs_efficiency": self._safe_correlation(metrics_df, "complexity_score", "scaling_efficiency"),
        }

        # Configuration dimension analysis
        complexity_results["configuration_analysis"] = {
            "L_effects": self._analyze_dimension_effects(metrics_df, "config_L"),
            "m_effects": self._analyze_dimension_effects(metrics_df, "config_m"),
            "interaction_effects": self._analyze_interaction_effects(metrics_df),
        }

        # Training size effects
        complexity_results["training_size_effects"] = {
            "training_vs_optimal_50": self._safe_correlation(metrics_df, "n_train", "optimal_context_50"),
            "training_vs_efficiency": self._safe_correlation(metrics_df, "n_train", "scaling_efficiency"),
            "training_size_scaling": self._analyze_training_size_scaling(metrics_df),
        }

        return complexity_results

    def _find_optimal_context_size(self, context_accuracies: pd.Series, threshold: float) -> float:
        """Find minimum context size achieving performance threshold."""
        for context_size, accuracy in context_accuracies.items():
            if accuracy >= threshold:
                return float(context_size)
        return float("inf")  # Threshold not achieved

    def _find_saturation_point(self, context_accuracies: pd.Series) -> float:
        """Find context size where performance saturates (diminishing returns)."""
        if len(context_accuracies) < 3:
            return float("inf")

        # Compute improvements between consecutive context sizes
        improvements = context_accuracies.diff().dropna()

        # Find where improvement drops below 5% of max improvement
        if len(improvements) > 0 and improvements.max() > 0:
            max_improvement = improvements.max()
            saturation_threshold = 0.05 * max_improvement

            for context_size, improvement in improvements.items():
                if improvement < saturation_threshold:
                    return float(context_size)

        return float(max(context_accuracies.index))  # No clear saturation

    def _fit_scaling_laws_single_model(
        self,
        context_sizes: np.ndarray,
        accuracies: np.ndarray,
        model_id: str,
        config_L: int,
        config_m: int,
        n_train: int,
    ) -> dict[str, t.Any] | None:
        """Fit scaling laws for a single model."""
        try:
            fits = {}

            # 1. Exponential saturation: acc = a * (1 - exp(-b*k))
            if "exponential" in self.scaling_laws:
                try:

                    def exponential_func(k, a, b):
                        return a * (1 - np.exp(-b * k))

                    # Use reasonable bounds
                    bounds = ([0, 0], [1.5, 10])
                    popt_exp, _ = curve_fit(exponential_func, context_sizes, accuracies, bounds=bounds, maxfev=1000)
                    pred_exp = exponential_func(context_sizes, *popt_exp)
                    r2_exp = r2_score(accuracies, pred_exp)
                    fits["exponential"] = {"a": popt_exp[0], "b": popt_exp[1], "r2": r2_exp}
                except:
                    fits["exponential"] = {"a": 0, "b": 0, "r2": -np.inf}

            # 2. Power law: acc = a * k^b
            if "power" in self.scaling_laws:
                try:

                    def power_func(k, a, b):
                        return a * np.power(k, b)

                    # Use log-transform for stability
                    log_k = np.log(context_sizes)
                    log_acc = np.log(np.maximum(accuracies, 1e-10))  # Avoid log(0)
                    slope, intercept, r_value, _, _ = stats.linregress(log_k, log_acc)

                    a_power = np.exp(intercept)
                    b_power = slope
                    pred_power = power_func(context_sizes, a_power, b_power)
                    r2_power = r2_score(accuracies, pred_power)

                    fits["power"] = {"a": a_power, "b": b_power, "r2": r2_power}
                except:
                    fits["power"] = {"a": 0, "b": 0, "r2": -np.inf}

            # 3. Logarithmic: acc = a * log(k) + b
            if "logarithmic" in self.scaling_laws:
                try:
                    log_context = np.log(context_sizes)
                    slope, intercept, r_value, _, _ = stats.linregress(log_context, accuracies)
                    pred_log = slope * log_context + intercept
                    r2_log = r2_score(accuracies, pred_log)

                    fits["logarithmic"] = {"a": slope, "b": intercept, "r2": r2_log}
                except:
                    fits["logarithmic"] = {"a": 0, "b": 0, "r2": -np.inf}

            # Determine best model
            valid_fits = {k: v for k, v in fits.items() if v["r2"] > -np.inf}
            if not valid_fits:
                return None

            best_model = max(valid_fits.keys(), key=lambda k: valid_fits[k]["r2"])
            best_r2 = valid_fits[best_model]["r2"]

            result = {
                "model_id": model_id,
                "config_L": config_L,
                "config_m": config_m,
                "n_train": n_train,
                "complexity_score": config_L * config_m,
                "best_model": best_model,
                "best_r2": best_r2,
            }

            # Add parameters for all scaling laws
            for law in self.scaling_laws:
                if law in fits:
                    result[f"{law}_a"] = fits[law]["a"]
                    result[f"{law}_b"] = fits[law]["b"]
                    result[f"{law}_r2"] = fits[law]["r2"]
                else:
                    result[f"{law}_a"] = 0
                    result[f"{law}_b"] = 0
                    result[f"{law}_r2"] = -np.inf

            return result

        except Exception as e:
            warnings.warn(f"Failed to fit scaling laws for {model_id}: {e}")
            return None

    def _safe_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Compute correlation safely, handling infinite values."""
        valid_data = df[(df[col1] != float("inf")) & (df[col2] != float("inf"))]
        if len(valid_data) < 3:
            return 0.0

        # Check for variance
        if valid_data[col1].std() == 0 or valid_data[col2].std() == 0:
            return 0.0

        corr = valid_data[col1].corr(valid_data[col2])
        return corr if not pd.isna(corr) else 0.0

    def _analyze_dimension_effects(self, metrics_df: pd.DataFrame, dimension: str) -> dict[str, float]:
        """Analyze effects of a single configuration dimension."""
        effects = {}
        for metric in ["optimal_context_50", "optimal_context_70", "max_accuracy", "scaling_efficiency"]:
            correlation = self._safe_correlation(metrics_df, dimension, metric)
            effects[f"{dimension}_vs_{metric}"] = correlation
        return effects

    def _analyze_interaction_effects(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Analyze L×m interaction effects."""
        interactions = {}
        # Create interaction term
        metrics_df["L_m_interaction"] = metrics_df["config_L"] * metrics_df["config_m"]

        for metric in ["optimal_context_50", "max_accuracy", "scaling_efficiency"]:
            correlation = self._safe_correlation(metrics_df, "L_m_interaction", metric)
            interactions[f"L_m_interaction_vs_{metric}"] = correlation

        return interactions

    def _analyze_training_size_scaling(self, metrics_df: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze how training size affects scaling patterns."""
        scaling_by_size = {}
        for n_train in sorted(metrics_df["n_train"].unique()):
            size_data = metrics_df[metrics_df["n_train"] == n_train]
            if len(size_data) > 2:
                scaling_by_size[n_train] = {
                    "mean_optimal_50": size_data["optimal_context_50"].replace(float("inf"), np.nan).mean(),
                    "mean_max_accuracy": size_data["max_accuracy"].mean(),
                    "mean_efficiency": size_data["scaling_efficiency"].mean(),
                    "count": len(size_data),
                }
        return scaling_by_size
