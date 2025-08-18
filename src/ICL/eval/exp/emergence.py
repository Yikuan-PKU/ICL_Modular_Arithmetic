"""Experiment 1: ICL Emergence Analysis from Phase 1 Data."""

import typing as t
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


class EmergenceAnalyzer:
    """Analyzes ICL emergence patterns from Phase 1 evaluation data."""

    def __init__(self, phase1_results_path: Path, output_dir: Path):
        """Initialize emergence analyzer with Phase 1 results."""
        self.phase1_results_path = phase1_results_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data: pd.DataFrame | None = None
        self.filtered_data: pd.DataFrame | None = None
        self.emergence_metrics: list[EmergenceMetrics] = []

    def load_existing_results(self) -> pd.DataFrame:
        """Load Phase 1 evaluation results."""
        if not self.phase1_results_path.exists():
            raise FileNotFoundError(f"Phase 1 results not found: {self.phase1_results_path}")

        self.data = pd.read_parquet(self.phase1_results_path)
        print(f"Loaded {len(self.data)} evaluation records from Phase 1")

        return self.data

    def filter_for_emergence(self) -> pd.DataFrame:
        """Filter data for emergence analysis (within-config only)."""
        if self.data is None:
            raise ValueError("Must load data first")

        # Filter for within-config transfer condition and normal control type
        self.filtered_data = self.data[
            (self.data["transfer_condition"] == "within_config") & (self.data["control_type"] == "normal")
        ].copy()

        print(f"Filtered to {len(self.filtered_data)} within-config normal sequences")
        return self.filtered_data

    def compute_emergence_metrics(self) -> list[EmergenceMetrics]:
        """Compute emergence metrics for each model."""
        if self.filtered_data is None:
            raise ValueError("Must filter data first")

        self.emergence_metrics = []

        # Group by model characteristics
        model_groups = self.filtered_data.groupby(["model_id", "config_L", "config_m", "n_train", "checkpoint_step"])

        for (model_id, config_L, config_m, n_train, checkpoint_step), group in model_groups:
            metrics = self._compute_single_model_metrics(group, model_id, config_L, config_m, n_train)
            if metrics:
                self.emergence_metrics.append(metrics)

        print(f"Computed emergence metrics for {len(self.emergence_metrics)} models")
        return self.emergence_metrics

    def _compute_single_model_metrics(
        self, model_data: pd.DataFrame, model_id: str, config_L: int, config_m: int, n_train: int
    ) -> EmergenceMetrics | None:
        """Compute emergence metrics for a single model."""
        try:
            # Get accuracy by context size
            context_accuracies = model_data.groupby("context_size")["accuracy"].mean().sort_index()

            if len(context_accuracies) < 3:  # Need sufficient data points
                return None

            # Emergence threshold (first context size > 0.5 accuracy)
            emergence_threshold = self._compute_emergence_threshold(context_accuracies)

            # Max accuracy across all context sizes
            max_accuracy = context_accuracies.max()

            # Context scaling slope using linear regression
            context_sizes = context_accuracies.index.values
            accuracies = context_accuracies.values
            slope, _, _, _, _ = stats.linregress(context_sizes, accuracies)

            # Baseline gap (compare with controls)
            baseline_gap = self._compute_baseline_gap(model_data)

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
                emergence_threshold=emergence_threshold,
                max_accuracy=max_accuracy,
                context_scaling_slope=slope,
                baseline_gap=baseline_gap,
                emergence_step=emergence_step,
                training_size_effect=training_size_effect,
                config_complexity_effect=config_complexity_effect,
            )

        except Exception as e:
            print(f"Failed to compute metrics for {model_id}: {e}")
            return None

    def _compute_emergence_threshold(self, context_accuracies: pd.Series) -> float:
        """Compute emergence threshold (minimum k for >0.5 accuracy)."""
        threshold = 0.5

        for context_size, accuracy in context_accuracies.items():
            if accuracy > threshold:
                return float(context_size)

        # No emergence observed
        return float("inf")

    def _compute_baseline_gap(self, model_data: pd.DataFrame) -> float:
        """Compute gap between normal and control conditions."""
        if self.data is None:
            return 0.0

        # Get control data for same model
        model_id = model_data["model_id"].iloc[0]
        model_controls = self.data[
            (self.data["model_id"] == model_id)
            & (self.data["transfer_condition"] == "within_config")
            & (self.data["control_type"].isin(["shuffled_context", "random_context"]))
        ]

        if len(model_controls) == 0:
            return 0.0

        normal_acc = model_data["accuracy"].mean()
        control_acc = model_controls["accuracy"].mean()

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

    def analyze_emergence_patterns(self) -> dict[str, t.Any]:
        """Analyze cross-model emergence patterns."""
        if not self.emergence_metrics:
            raise ValueError("Must compute emergence metrics first")

        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame(
            [
                {
                    "model_id": m.model_id,
                    "config_L": m.config_L,
                    "config_m": m.config_m,
                    "n_train": m.n_train,
                    "emergence_threshold": m.emergence_threshold,
                    "max_accuracy": m.max_accuracy,
                    "context_scaling_slope": m.context_scaling_slope,
                    "baseline_gap": m.baseline_gap,
                    "emergence_step": m.emergence_step,
                }
                for m in self.emergence_metrics
            ]
        )

        # Analyze patterns
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
        if "config_L" in metrics_df.columns:
            valid_thresholds = metrics_df[metrics_df["emergence_threshold"] != float("inf")]
            if len(valid_thresholds) > 0:
                corr_L = valid_thresholds["config_L"].corr(valid_thresholds["emergence_threshold"])
                config_effects["depth_threshold_correlation"] = corr_L

        # Effect of multiplicity (m) on max accuracy
        if "config_m" in metrics_df.columns:
            corr_m = metrics_df["config_m"].corr(metrics_df["max_accuracy"])
            config_effects["multiplicity_accuracy_correlation"] = corr_m

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

        return training_effects

    def _compute_emergence_statistics(self, metrics_df: pd.DataFrame) -> dict[str, float]:
        """Compute basic emergence statistics."""
        valid_thresholds = metrics_df[metrics_df["emergence_threshold"] != float("inf")]

        stats = {
            "emergence_rate": len(valid_thresholds) / len(metrics_df),
            "mean_emergence_threshold": valid_thresholds["emergence_threshold"].mean()
            if len(valid_thresholds) > 0
            else float("inf"),
            "mean_max_accuracy": metrics_df["max_accuracy"].mean(),
            "mean_baseline_gap": metrics_df["baseline_gap"].mean(),
            "mean_scaling_slope": metrics_df["context_scaling_slope"].mean(),
        }

        return stats

    def _analyze_scaling_patterns(self, metrics_df: pd.DataFrame) -> dict[str, t.Any]:
        """Analyze context scaling patterns."""
        patterns = {
            "positive_slope_rate": (metrics_df["context_scaling_slope"] > 0).mean(),
            "strong_scaling_rate": (metrics_df["context_scaling_slope"] > 0.1).mean(),
            "slope_distribution": metrics_df["context_scaling_slope"].describe().to_dict(),
        }

        return patterns

    def generate_emergence_report(self) -> Path:
        """Generate comprehensive emergence analysis report."""
        report_path = self.output_dir / "emergence_analysis_report.html"

        # Create visualizations
        self._create_emergence_visualizations()

        # Generate HTML report
        html_content = self._generate_html_report()

        with open(report_path, "w") as f:
            f.write(html_content)

        print(f"Generated emergence report: {report_path}")
        return report_path

    def _create_emergence_visualizations(self) -> None:
        """Create emergence analysis visualizations."""
        if not self.emergence_metrics:
            return

        # Emergence threshold distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        thresholds = [m.emergence_threshold for m in self.emergence_metrics if m.emergence_threshold != float("inf")]
        plt.hist(thresholds, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Emergence Threshold (Context Size)")
        plt.ylabel("Number of Models")
        plt.title("Distribution of Emergence Thresholds")

        # Max accuracy vs config complexity
        plt.subplot(2, 2, 2)
        config_complexity = [m.config_L * m.config_m for m in self.emergence_metrics]
        max_accuracies = [m.max_accuracy for m in self.emergence_metrics]
        plt.scatter(config_complexity, max_accuracies, alpha=0.6)
        plt.xlabel("Configuration Complexity (L × m)")
        plt.ylabel("Max Accuracy")
        plt.title("Max Accuracy vs Configuration Complexity")

        # Training size vs emergence threshold
        plt.subplot(2, 2, 3)
        training_sizes = [m.n_train for m in self.emergence_metrics]
        thresholds_for_plot = [
            m.emergence_threshold if m.emergence_threshold != float("inf") else 10 for m in self.emergence_metrics
        ]
        plt.scatter(training_sizes, thresholds_for_plot, alpha=0.6)
        plt.xlabel("Training Size")
        plt.ylabel("Emergence Threshold")
        plt.title("Training Size vs Emergence Threshold")
        plt.yscale("log")

        # Scaling slope distribution
        plt.subplot(2, 2, 4)
        slopes = [m.context_scaling_slope for m in self.emergence_metrics]
        plt.hist(slopes, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Context Scaling Slope")
        plt.ylabel("Number of Models")
        plt.title("Distribution of Context Scaling Slopes")

        plt.tight_layout()
        plt.savefig(self.output_dir / "emergence_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ICL Emergence Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>ICL Emergence Analysis Report</h1>
            <h2>Summary</h2>
            <div class="metric">Total Models Analyzed: {len(self.emergence_metrics)}</div>
            
            <h2>Emergence Visualizations</h2>
            <div class="visualization">
                <img src="emergence_analysis.png" alt="Emergence Analysis Plots" style="max-width: 100%;">
            </div>
            
            <h2>Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>
        </body>
        </html>
        """

    def validate_data_completeness(self) -> dict[str, t.Any]:
        """Validate completeness of Phase 1 data for emergence analysis."""
        if self.data is None or self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        validation = {
            "total_records": len(self.data),
            "within_config_records": len(self.filtered_data),
            "unique_models": self.filtered_data["model_id"].nunique(),
            "context_sizes_available": sorted(self.filtered_data["context_size"].unique()),
            "configs_available": len(self.filtered_data.groupby(["config_L", "config_m"])),
            "missing_context_sizes": [],
            "incomplete_models": [],
        }

        # Check for expected context sizes
        expected_context_sizes = [1, 2, 3, 4, 5, 6, 8]
        available_context_sizes = set(self.filtered_data["context_size"].unique())
        missing = [k for k in expected_context_sizes if k not in available_context_sizes]
        validation["missing_context_sizes"] = missing

        # Check model completeness
        for model_id in self.filtered_data["model_id"].unique():
            model_data = self.filtered_data[self.filtered_data["model_id"] == model_id]
            model_context_sizes = set(model_data["context_size"].unique())
            if not set(expected_context_sizes).issubset(model_context_sizes):
                validation["incomplete_models"].append(model_id)

        return validation

    def save_emergence_metrics(self) -> Path:
        """Save emergence metrics to CSV."""
        if not self.emergence_metrics:
            raise ValueError("No emergence metrics to save")

        metrics_df = pd.DataFrame(
            [
                {
                    "model_id": m.model_id,
                    "config_L": m.config_L,
                    "config_m": m.config_m,
                    "n_train": m.n_train,
                    "emergence_threshold": m.emergence_threshold,
                    "max_accuracy": m.max_accuracy,
                    "context_scaling_slope": m.context_scaling_slope,
                    "baseline_gap": m.baseline_gap,
                    "emergence_step": m.emergence_step,
                    "training_size_effect": m.training_size_effect,
                    "config_complexity_effect": m.config_complexity_effect,
                }
                for m in self.emergence_metrics
            ]
        )

        output_path = self.output_dir / "emergence_metrics.csv"
        metrics_df.to_csv(output_path, index=False)

        print(f"Saved emergence metrics: {output_path}")
        return output_path


def run_emergence_analysis(phase1_results_path: str | Path, output_dir: str | Path) -> dict[str, t.Any]:
    """Run complete emergence analysis pipeline."""
    analyzer = EmergenceAnalyzer(Path(phase1_results_path), Path(output_dir))

    # Load and process data
    analyzer.load_existing_results()
    analyzer.filter_for_emergence()

    # Validate data completeness
    validation = analyzer.validate_data_completeness()
    print(f"Data validation: {validation['within_config_records']} records, {validation['unique_models']} models")

    # Compute metrics and analyze patterns
    analyzer.compute_emergence_metrics()
    patterns = analyzer.analyze_emergence_patterns()

    # Generate outputs
    metrics_path = analyzer.save_emergence_metrics()
    report_path = analyzer.generate_emergence_report()

    return {
        "emergence_metrics_path": metrics_path,
        "report_path": report_path,
        "patterns": patterns,
        "validation": validation,
        "total_models_analyzed": len(analyzer.emergence_metrics),
    }


# Test function
def test_emergence_analysis() -> None:
    """Test emergence analysis with sample data."""
    # This would use actual Phase 1 results path
    phase1_path = Path("raw_evaluations/icl_performance.parquet")
    output_dir = Path("emergence_analysis_results")

    if phase1_path.exists():
        results = run_emergence_analysis(phase1_path, output_dir)
        print(f"Emergence analysis completed: {results['total_models_analyzed']} models")
    else:
        print("Phase 1 results not found - run Phase 1 evaluation first")


if __name__ == "__main__":
    test_emergence_analysis()
