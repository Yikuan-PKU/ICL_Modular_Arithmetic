"""ICL Emergence Analysis (Experiment 1) - Main analyzer."""

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ICL.eval.exp.emergence.metrics import EmergenceMetrics, EmergenceMetricsCalculator
from ICL.eval.exp.shared.exp_config import ExpConfig
from ICL.eval.exp.shared.result_loader import Phase1ResultLoader
from ICL.eval.exp.shared.visualizer import AnalysisVisualizer


class EmergenceAnalyzer:
    """Analyzes ICL emergence patterns from Phase 1 evaluation data."""

    def __init__(self, config: ExpConfig, output_dir: Path):
        """Initialize emergence analyzer."""
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = Phase1ResultLoader(config.collection_results_dir)
        self.visualizer = AnalysisVisualizer(output_dir)
        self.calculator = EmergenceMetricsCalculator(
            emergence_threshold=config.exp1_config.get("emergence_threshold", 0.5)
        )

        # Data storage
        self.data: pd.DataFrame | None = None
        self.filtered_data: pd.DataFrame | None = None
        self.control_data: pd.DataFrame | None = None
        self.emergence_metrics: list[EmergenceMetrics] = []

    def load_and_filter_data(self) -> pd.DataFrame:
        """Load Phase 1 results and filter for emergence analysis."""
        print("Loading Phase 1 evaluation results...")
        self.data = self.loader.load_icl_performance()

        # Validate data completeness
        validation = self.loader.validate_data_completeness(self.data)
        if validation["missing_data_issues"]:
            print("Data completeness warnings:")
            for issue in validation["missing_data_issues"]:
                print(f"  - {issue}")

        # Filter for emergence analysis (within-config normal sequences)
        self.filtered_data = self.loader.filter_for_analysis(self.data, "emergence")

        # Get control data for baseline comparison
        baseline_controls = self.config.exp1_config.get("baseline_controls", ["shuffled_context", "random_context"])
        self.control_data = self.data[
            (self.data["transfer_condition"] == "within_config") & (self.data["control_type"].isin(baseline_controls))
        ].copy()

        print("Filtered data for emergence analysis:")
        print(f"  - Main data: {len(self.filtered_data)} records")
        print(f"  - Control data: {len(self.control_data)} records")
        print(f"  - Unique models: {self.filtered_data['model_id'].nunique()}")

        return self.filtered_data

    def compute_emergence_metrics(self) -> list[EmergenceMetrics]:
        """Compute emergence metrics for each model."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Computing emergence metrics...")
        self.emergence_metrics = []

        # Group by model characteristics
        model_groups = self.filtered_data.groupby(["model_id", "config_L", "config_m", "n_train", "checkpoint_step"])

        for (model_id, config_L, config_m, n_train, checkpoint_step), group in model_groups:
            # Get corresponding control data for this model
            model_control_data = None
            if self.control_data is not None:
                model_control_data = self.control_data[(self.control_data["model_id"] == model_id)]

            # Compute metrics
            metrics = self.calculator.compute_single_model_metrics(group, model_control_data)
            if metrics:
                self.emergence_metrics.append(metrics)

        print(f"Computed emergence metrics for {len(self.emergence_metrics)} models")
        return self.emergence_metrics

    def analyze_emergence_patterns(self) -> dict[str, t.Any]:
        """Analyze cross-model emergence patterns."""
        if not self.emergence_metrics:
            raise ValueError("Must compute emergence metrics first")

        print("Analyzing emergence patterns...")
        patterns = self.calculator.analyze_emergence_patterns(self.emergence_metrics)

        # Print key findings
        stats = patterns.get("emergence_statistics", {})
        print("\nKey emergence findings:")
        print(f"  - Emergence rate: {stats.get('emergence_rate', 0):.1%}")
        print(f"  - Mean emergence threshold: {stats.get('mean_emergence_threshold', float('inf')):.1f}")
        print(f"  - Mean max accuracy: {stats.get('mean_max_accuracy', 0):.3f}")
        print(f"  - Positive scaling rate: {stats.get('positive_slope_rate', 0):.1%}")

        return patterns

    def create_visualizations(self) -> list[Path]:
        """Create emergence analysis visualizations."""
        if not self.emergence_metrics:
            return []

        print("Creating visualizations...")
        visualization_paths = []

        # Convert metrics to DataFrame for plotting
        metrics_df = pd.DataFrame([m.to_dict() for m in self.emergence_metrics])

        # 1. Emergence threshold distribution
        viz_path = self.visualizer.create_distribution_plot(
            metrics_df, "emergence_threshold", "Distribution of Emergence Thresholds", "emergence_threshold_dist"
        )
        visualization_paths.append(viz_path)

        # 2. Max accuracy vs configuration complexity
        metrics_df["complexity"] = metrics_df["config_L"] * metrics_df["config_m"]
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "complexity",
            "max_accuracy",
            "Max Accuracy vs Configuration Complexity",
            "accuracy_vs_complexity",
        )
        visualization_paths.append(viz_path)

        # 3. Training size vs emergence threshold
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "n_train",
            "emergence_threshold",
            "Training Size vs Emergence Threshold",
            "training_vs_emergence",
        )
        visualization_paths.append(viz_path)

        # 4. Context scaling slope distribution
        viz_path = self.visualizer.create_distribution_plot(
            metrics_df, "context_scaling_slope", "Distribution of Context Scaling Slopes", "scaling_slope_dist"
        )
        visualization_paths.append(viz_path)

        # 5. Multi-panel emergence overview
        viz_path = self._create_emergence_overview()
        visualization_paths.append(viz_path)

        return visualization_paths

    def _create_emergence_overview(self) -> Path:
        """Create comprehensive emergence overview visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.emergence_metrics])

        def plot_emergence_threshold_by_config():
            """Plot emergence thresholds by configuration."""
            # Filter finite thresholds
            finite_data = metrics_df[metrics_df["emergence_threshold"] != float("inf")]
            if len(finite_data) > 0:
                plt.scatter(
                    finite_data["config_L"],
                    finite_data["emergence_threshold"],
                    c=finite_data["config_m"],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(label="config_m")
            plt.xlabel("Configuration Depth (L)")
            plt.ylabel("Emergence Threshold")
            plt.title("Emergence Threshold by Configuration")

        def plot_accuracy_vs_baseline_gap():
            """Plot max accuracy vs baseline gap."""
            plt.scatter(metrics_df["baseline_gap"], metrics_df["max_accuracy"], alpha=0.7)
            plt.xlabel("Baseline Gap")
            plt.ylabel("Max Accuracy")
            plt.title("Max Accuracy vs Baseline Gap")

        def plot_scaling_slope_by_training():
            """Plot scaling slopes by training size."""
            training_sizes = sorted(metrics_df["n_train"].unique())
            slope_by_training = [
                metrics_df[metrics_df["n_train"] == size]["context_scaling_slope"].values for size in training_sizes
            ]
            plt.boxplot(slope_by_training, labels=training_sizes)
            plt.xlabel("Training Size")
            plt.ylabel("Context Scaling Slope")
            plt.title("Scaling Slopes by Training Size")
            plt.xticks(rotation=45)

        def plot_emergence_rate_by_config():
            """Plot emergence rate by configuration."""
            config_groups = metrics_df.groupby(["config_L", "config_m"])
            emergence_rates = []
            config_labels = []

            for (L, m), group in config_groups:
                emerged = (group["emergence_threshold"] != float("inf")).mean()
                emergence_rates.append(emerged)
                config_labels.append(f"({L},{m})")

            plt.bar(range(len(emergence_rates)), emergence_rates)
            plt.xticks(range(len(config_labels)), config_labels, rotation=45)
            plt.xlabel("Configuration (L, m)")
            plt.ylabel("Emergence Rate")
            plt.title("Emergence Rate by Configuration")

        plot_functions = [
            plot_emergence_threshold_by_config,
            plot_accuracy_vs_baseline_gap,
            plot_scaling_slope_by_training,
            plot_emergence_rate_by_config,
        ]

        return self.visualizer.create_multi_panel_figure(
            plot_functions, layout=(2, 2), filename="emergence_overview", suptitle="ICL Emergence Analysis Overview"
        )

    def generate_report(self, patterns: dict[str, t.Any], visualization_paths: list[Path]) -> Path:
        """Generate comprehensive emergence analysis report."""
        if not self.config.generate_reports:
            return Path()

        print("Generating emergence analysis report...")

        # Prepare report sections
        sections = []

        # Summary section
        stats = patterns.get("emergence_statistics", {})
        sections.append(
            {
                "title": "Summary",
                "content": f"Analysis of ICL emergence patterns across {len(self.emergence_metrics)} models.",
                "metrics": {
                    "Total Models Analyzed": len(self.emergence_metrics),
                    "Emergence Rate": f"{stats.get('emergence_rate', 0):.1%}",
                    "Mean Emergence Threshold": f"{stats.get('mean_emergence_threshold', float('inf')):.1f}",
                    "Mean Max Accuracy": f"{stats.get('mean_max_accuracy', 0):.3f}",
                    "Positive Scaling Rate": f"{stats.get('positive_slope_rate', 0):.1%}",
                },
            }
        )

        # Configuration effects
        config_effects = patterns.get("config_effects", {})
        sections.append(
            {
                "title": "Configuration Effects",
                "content": "How model configuration affects ICL emergence patterns.",
                "metrics": {
                    "Depth-Threshold Correlation": f"{config_effects.get('depth_threshold_correlation', 0):.3f}",
                    "Multiplicity-Accuracy Correlation": f"{config_effects.get('multiplicity_accuracy_correlation', 0):.3f}",
                    "Complexity-Accuracy Correlation": f"{config_effects.get('complexity_accuracy_correlation', 0):.3f}",
                },
            }
        )

        # Training size effects
        training_effects = patterns.get("training_size_effects", {})
        sections.append(
            {
                "title": "Training Size Effects",
                "content": "How training data size affects emergence characteristics.",
                "metrics": {
                    "Training-Threshold Correlation": f"{training_effects.get('training_size_threshold_correlation', 0):.3f}",
                    "Training-Slope Correlation": f"{training_effects.get('training_size_slope_correlation', 0):.3f}",
                    "Training-Accuracy Correlation": f"{training_effects.get('training_size_accuracy_correlation', 0):.3f}",
                },
            }
        )

        # Add visualizations
        for viz_path in visualization_paths:
            sections.append(
                {
                    "title": viz_path.stem.replace("_", " ").title(),
                    "visualization": viz_path,
                    "viz_title": viz_path.stem.replace("_", " ").title(),
                }
            )

        return self.visualizer.generate_html_report("ICL Emergence Analysis Report", sections, "emergence_report.html")

    def save_metrics(self) -> Path:
        """Save emergence metrics to CSV."""
        if not self.emergence_metrics:
            raise ValueError("No emergence metrics to save")

        metrics_df = pd.DataFrame([m.to_dict() for m in self.emergence_metrics])
        output_path = self.output_dir / "metrics" / "emergence_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_path, index=False)
        print(f"Saved emergence metrics: {output_path}")

        return output_path

    def run_complete_analysis(self) -> dict[str, t.Any]:
        """Run complete emergence analysis pipeline."""
        print("=" * 60)
        print("EXPERIMENT 1: ICL EMERGENCE ANALYSIS")
        print("=" * 60)

        # Load and process data
        self.load_and_filter_data()

        # Compute metrics
        self.compute_emergence_metrics()

        # Analyze patterns
        patterns = self.analyze_emergence_patterns()

        # Create outputs
        visualization_paths = []
        if self.config.create_visualizations:
            visualization_paths = self.create_visualizations()

        report_path = Path()
        if self.config.generate_reports:
            report_path = self.generate_report(patterns, visualization_paths)

        metrics_path = self.save_metrics()

        print("=" * 60)
        print("EMERGENCE ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")

        return {
            "emergence_metrics_path": metrics_path,
            "report_path": report_path,
            "visualization_paths": visualization_paths,
            "patterns": patterns,
            "total_models_analyzed": len(self.emergence_metrics),
            "emergence_rate": patterns.get("emergence_statistics", {}).get("emergence_rate", 0),
        }
