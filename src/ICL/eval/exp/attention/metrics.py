"""Metrics for attention pattern analysis (Experiment 4)."""

import typing as t
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AttentionMetrics:
    """Metrics for attention pattern analysis."""

    model_id: str
    config_L: int
    config_m: int
    n_train: int
    layer_idx: int
    head_idx: int
    context_size: int

    # Attention pattern characteristics
    attention_entropy: float
    attention_concentration: float
    last_token_attention: float
    diagonal_attention: float

    # Head type classifications
    is_induction_head: bool
    is_copying_head: bool
    is_previous_token_head: bool

    # Pattern scores
    induction_score: float
    copying_score: float
    previous_token_score: float

    # Context scaling properties
    attention_scaling_slope: float
    attention_stability: float

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "model_id": self.model_id,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "n_train": self.n_train,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "context_size": self.context_size,
            "attention_entropy": self.attention_entropy,
            "attention_concentration": self.attention_concentration,
            "last_token_attention": self.last_token_attention,
            "diagonal_attention": self.diagonal_attention,
            "is_induction_head": self.is_induction_head,
            "is_copying_head": self.is_copying_head,
            "is_previous_token_head": self.is_previous_token_head,
            "induction_score": self.induction_score,
            "copying_score": self.copying_score,
            "previous_token_score": self.previous_token_score,
            "attention_scaling_slope": self.attention_scaling_slope,
            "attention_stability": self.attention_stability,
        }


class AttentionMetricsCalculator:
    """Calculator for attention pattern metrics."""

    def __init__(
        self, induction_threshold: float = 0.3, copying_threshold: float = 0.3, previous_token_threshold: float = 0.3
    ):
        """Initialize calculator with classification thresholds."""
        self.induction_threshold = induction_threshold
        self.copying_threshold = copying_threshold
        self.previous_token_threshold = previous_token_threshold

    def analyze_attention_matrix(self, attention_matrix: np.ndarray) -> dict[str, float]:
        """Analyze a single attention matrix for pattern characteristics."""
        if attention_matrix.size == 0:
            return self._get_empty_metrics()

        # Ensure matrix is 2D
        if attention_matrix.ndim != 2:
            warnings.warn(f"Expected 2D attention matrix, got {attention_matrix.ndim}D")
            return self._get_empty_metrics()

        seq_len = attention_matrix.shape[0]
        if seq_len == 0:
            return self._get_empty_metrics()

        try:
            metrics = {}

            # Basic attention statistics
            metrics["attention_entropy"] = self._compute_attention_entropy(attention_matrix)
            metrics["attention_concentration"] = self._compute_attention_concentration(attention_matrix)
            metrics["last_token_attention"] = self._compute_last_token_attention(attention_matrix)
            metrics["diagonal_attention"] = self._compute_diagonal_attention(attention_matrix)

            # Pattern detection scores
            metrics["induction_score"] = self._compute_induction_score(attention_matrix)
            metrics["copying_score"] = self._compute_copying_score(attention_matrix)
            metrics["previous_token_score"] = self._compute_previous_token_score(attention_matrix)

            # Head type classification
            metrics["is_induction_head"] = metrics["induction_score"] > self.induction_threshold
            metrics["is_copying_head"] = metrics["copying_score"] > self.copying_threshold
            metrics["is_previous_token_head"] = metrics["previous_token_score"] > self.previous_token_threshold

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to analyze attention matrix: {e}")
            return self._get_empty_metrics()

    def _get_empty_metrics(self) -> dict[str, float]:
        """Return empty metrics for failed analysis."""
        return {
            "attention_entropy": 0.0,
            "attention_concentration": 0.0,
            "last_token_attention": 0.0,
            "diagonal_attention": 0.0,
            "induction_score": 0.0,
            "copying_score": 0.0,
            "previous_token_score": 0.0,
            "is_induction_head": False,
            "is_copying_head": False,
            "is_previous_token_head": False,
        }

    def _compute_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Compute average attention entropy across positions."""
        entropies = []
        for i in range(attention_matrix.shape[0]):
            attention_dist = attention_matrix[i, : i + 1]  # Only attend to previous tokens
            if len(attention_dist) > 0 and attention_dist.sum() > 0:
                # Normalize to probability distribution
                attention_dist = attention_dist / attention_dist.sum()
                # Compute entropy
                entropy = -np.sum(attention_dist * np.log(attention_dist + 1e-12))
                entropies.append(entropy)

        return np.mean(entropies) if entropies else 0.0

    def _compute_attention_concentration(self, attention_matrix: np.ndarray) -> float:
        """Compute how concentrated attention is (max attention weight)."""
        max_attentions = []
        for i in range(attention_matrix.shape[0]):
            attention_dist = attention_matrix[i, : i + 1]  # Only attend to previous tokens
            if len(attention_dist) > 0:
                max_attentions.append(np.max(attention_dist))

        return np.mean(max_attentions) if max_attentions else 0.0

    def _compute_last_token_attention(self, attention_matrix: np.ndarray) -> float:
        """Compute average attention to the last token in context."""
        if attention_matrix.shape[0] < 2:
            return 0.0

        # For each query position, check attention to the immediately previous token
        last_token_attentions = []
        for i in range(1, attention_matrix.shape[0]):
            last_token_attention = attention_matrix[i, i - 1]
            last_token_attentions.append(last_token_attention)

        return np.mean(last_token_attentions) if last_token_attentions else 0.0

    def _compute_diagonal_attention(self, attention_matrix: np.ndarray) -> float:
        """Compute attention to positions with fixed relative distance."""
        if attention_matrix.shape[0] < 3:
            return 0.0

        # Compute attention to positions 1 step back (previous token pattern)
        diagonal_attentions = []
        for i in range(1, attention_matrix.shape[0]):
            if i - 1 >= 0:
                diagonal_attentions.append(attention_matrix[i, i - 1])

        return np.mean(diagonal_attentions) if diagonal_attentions else 0.0

    def _compute_induction_score(self, attention_matrix: np.ndarray) -> float:
        """Compute induction head score (attention to positions that precede repeated patterns)."""
        if attention_matrix.shape[0] < 4:
            return 0.0

        # Simple induction pattern: high attention to positions that could complete patterns
        # Look for attention to positions that are 2+ steps back (allowing for pattern completion)
        induction_scores = []

        for i in range(2, attention_matrix.shape[0]):
            # Check attention to positions that could be part of an AB...AB pattern
            distant_attention = attention_matrix[i, : max(1, i - 2)].sum()
            immediate_attention = attention_matrix[i, max(0, i - 2) : i].sum()

            if immediate_attention > 0:
                induction_score = distant_attention / (distant_attention + immediate_attention + 1e-12)
                induction_scores.append(induction_score)

        return np.mean(induction_scores) if induction_scores else 0.0

    def _compute_copying_score(self, attention_matrix: np.ndarray) -> float:
        """Compute copying head score (attention uniformly distributed over context)."""
        if attention_matrix.shape[0] < 2:
            return 0.0

        # Copying heads attend somewhat uniformly to context tokens
        copying_scores = []

        for i in range(1, attention_matrix.shape[0]):
            attention_dist = attention_matrix[i, :i]
            if len(attention_dist) > 1:
                # Compute how uniform the distribution is (inverse of entropy)
                attention_dist = attention_dist / (attention_dist.sum() + 1e-12)
                entropy = -np.sum(attention_dist * np.log(attention_dist + 1e-12))
                max_entropy = np.log(len(attention_dist))

                # Normalized entropy (1 = uniform, 0 = concentrated)
                uniformity = entropy / max_entropy if max_entropy > 0 else 0
                copying_scores.append(uniformity)

        return np.mean(copying_scores) if copying_scores else 0.0

    def _compute_previous_token_score(self, attention_matrix: np.ndarray) -> float:
        """Compute previous token head score (high attention to immediately previous token)."""
        if attention_matrix.shape[0] < 2:
            return 0.0

        # Previous token heads focus on the immediately preceding token
        previous_scores = []

        for i in range(1, attention_matrix.shape[0]):
            total_attention = attention_matrix[i, :i].sum()
            previous_attention = attention_matrix[i, i - 1]

            if total_attention > 0:
                previous_score = previous_attention / total_attention
                previous_scores.append(previous_score)

        return np.mean(previous_scores) if previous_scores else 0.0

    def compute_attention_scaling(self, attention_data: dict[str, t.Any]) -> dict[str, float]:
        """Compute how attention patterns scale with context size."""
        # Group attention data by context size
        context_metrics = {}

        for key, attention_info in attention_data.items():
            metadata = attention_info.get("metadata", {})
            context_size = metadata.get("context_size", 0)

            if context_size not in context_metrics:
                context_metrics[context_size] = []

            # Analyze this attention matrix
            attention_matrix = attention_info.get("attention_matrix")
            if attention_matrix is not None:
                metrics = self.analyze_attention_matrix(attention_matrix)
                context_metrics[context_size].append(metrics)

        # Compute scaling statistics
        scaling_results = {}

        if len(context_metrics) >= 2:
            context_sizes = sorted(context_metrics.keys())

            # Compute scaling slope for attention entropy
            entropies = []
            for k in context_sizes:
                k_entropies = [m["attention_entropy"] for m in context_metrics[k]]
                if k_entropies:
                    entropies.append(np.mean(k_entropies))
                else:
                    entropies.append(0.0)

            if len(entropies) >= 2 and np.std(context_sizes) > 0:
                try:
                    slope, _, _, _, _ = stats.linregress(context_sizes, entropies)
                    scaling_results["attention_scaling_slope"] = slope
                except:
                    scaling_results["attention_scaling_slope"] = 0.0
            else:
                scaling_results["attention_scaling_slope"] = 0.0

            # Compute attention stability (consistency across context sizes)
            all_entropies = [
                e for k_metrics in context_metrics.values() for m in k_metrics for e in [m["attention_entropy"]]
            ]
            scaling_results["attention_stability"] = 1.0 - (np.std(all_entropies) if all_entropies else 0.0)
        else:
            scaling_results["attention_scaling_slope"] = 0.0
            scaling_results["attention_stability"] = 0.0

        return scaling_results

    def analyze_head_types_by_layer(self, attention_metrics: list[AttentionMetrics]) -> dict[str, t.Any]:
        """Analyze distribution of head types across layers."""
        if not attention_metrics:
            return {}

        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame([m.to_dict() for m in attention_metrics])

        layer_analysis = {}

        for layer_idx in sorted(metrics_df["layer_idx"].unique()):
            layer_data = metrics_df[metrics_df["layer_idx"] == layer_idx]

            layer_analysis[f"layer_{layer_idx}"] = {
                "total_heads": len(layer_data),
                "induction_heads": layer_data["is_induction_head"].sum(),
                "copying_heads": layer_data["is_copying_head"].sum(),
                "previous_token_heads": layer_data["is_previous_token_head"].sum(),
                "mean_attention_entropy": layer_data["attention_entropy"].mean(),
                "mean_attention_concentration": layer_data["attention_concentration"].mean(),
                "mean_induction_score": layer_data["induction_score"].mean(),
                "mean_copying_score": layer_data["copying_score"].mean(),
            }

        return layer_analysis

    def analyze_attention_patterns_by_config(self, attention_metrics: list[AttentionMetrics]) -> dict[str, t.Any]:
        """Analyze attention patterns by model configuration."""
        if not attention_metrics:
            return {}

        metrics_df = pd.DataFrame([m.to_dict() for m in attention_metrics])

        config_analysis = {}

        for (config_L, config_m), config_data in metrics_df.groupby(["config_L", "config_m"]):
            config_key = f"L{config_L}_m{config_m}"

            config_analysis[config_key] = {
                "total_attention_heads": len(config_data),
                "induction_head_rate": config_data["is_induction_head"].mean(),
                "copying_head_rate": config_data["is_copying_head"].mean(),
                "previous_token_head_rate": config_data["is_previous_token_head"].mean(),
                "mean_attention_entropy": config_data["attention_entropy"].mean(),
                "mean_induction_score": config_data["induction_score"].mean(),
                "attention_entropy_std": config_data["attention_entropy"].std(),
                "complexity_score": config_L * config_m,
            }

        return config_analysis

    def compute_attention_context_scaling(self, attention_metrics: list[AttentionMetrics]) -> dict[str, t.Any]:
        """Analyze how attention patterns change with context size."""
        if not attention_metrics:
            return {}

        metrics_df = pd.DataFrame([m.to_dict() for m in attention_metrics])

        scaling_analysis = {}

        # Group by model and head
        head_groups = metrics_df.groupby(["model_id", "layer_idx", "head_idx"])

        scaling_slopes = []
        stability_scores = []

        for (model_id, layer_idx, head_idx), head_data in head_groups:
            if len(head_data) >= 3:  # Need multiple context sizes
                context_sizes = head_data["context_size"].values
                entropies = head_data["attention_entropy"].values

                if np.std(context_sizes) > 0:
                    try:
                        slope, _, _, _, _ = stats.linregress(context_sizes, entropies)
                        scaling_slopes.append(slope)

                        # Stability: inverse of entropy variance
                        stability = 1.0 / (1.0 + np.var(entropies))
                        stability_scores.append(stability)
                    except:
                        continue

        scaling_analysis = {
            "mean_scaling_slope": np.mean(scaling_slopes) if scaling_slopes else 0.0,
            "mean_stability_score": np.mean(stability_scores) if stability_scores else 0.0,
            "positive_scaling_rate": (np.array(scaling_slopes) > 0).mean() if scaling_slopes else 0.0,
            "heads_analyzed": len(scaling_slopes),
        }

        return scaling_analysis
