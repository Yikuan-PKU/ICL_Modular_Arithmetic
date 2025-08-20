"""Context Scaling Analysis (Experiment 3) - Main analyzer."""

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ICL.eval.exp.scaling.metrics import ScalingMetrics, ScalingMetricsCalculator
from ICL.eval.exp.shared.exp_config import ExpConfig
from ICL.eval.exp.shared.result_loader import Phase1ResultLoader
from ICL.eval.exp.shared.visualizer import AnalysisVisualizer


class ScalingAnalyzer:
    """Analyzes context length scaling patterns with hierarchical complexity."""

    def __init__(self, config: ExpConfig, output_dir: Path):
        """Initialize scaling analyzer."""
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = Phase1ResultLoader(config.collection_results_dir)
        self.visualizer = AnalysisVisualizer(output_dir)
        self.calculator = ScalingMetricsCalculator(
            performance_thresholds=config.exp3_config.get("performance_thresholds", [0.5, 0.7, 0.9]),
            scaling_laws=config.exp3_config.get("scaling_laws", ["exponential", "power", "logarithmic"]),
        )

        # Data storage
        self.data: pd.DataFrame | None = None
        self.filtered_data: pd.DataFrame | None = None
        self.scaling_metrics: list[ScalingMetrics] = []

    def load_and_filter_data(self) -> pd.DataFrame:
        """Load Phase 1 results and filter for scaling analysis."""
        print("Loading Phase 1 evaluation results...")
        self.data = self.loader.load_icl_performance()

        # Validate data completeness
        validation = self.loader.validate_data_completeness(self.data)
        if validation["missing_data_issues"]:
            print("Data completeness warnings:")
            for issue in validation["missing_data_issues"]:
                print(f"  - {issue}")

        # Filter for scaling analysis (within-config normal sequences only)
        self.filtered_data = self.loader.filter_for_analysis(self.data, "scaling")

        print("Filtered data for scaling analysis:")
        print(f"  - Records: {len(self.filtered_data)}")
        print(f"  - Context sizes: {sorted(self.filtered_data['context_size'].unique())}")
        print(f"  - Unique models: {self.filtered_data['model_id'].nunique()}")
        print(f"  - Configurations: {len(self.filtered_data.groupby(['config_L', 'config_m']))}")

        return self.filtered_data

    def detect_optimal_context_sizes(self) -> dict[str, t.Any]:
        """Detect optimal context sizes for different performance thresholds."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Detecting optimal context sizes...")
        optimal_results = self.calculator.detect_optimal_context_sizes(self.filtered_data)

        # Print key findings
        threshold_stats = optimal_results.get("threshold_statistics", {})
        print("\nOptimal context size findings:")
        for threshold_key, stats in threshold_stats.items():
            threshold = float(threshold_key.split("_")[1])
            achievement_rate = stats.get("achievement_rate", 0)
            mean_optimal = stats.get("mean_optimal_k", float("inf"))
            print(
                f"  - {threshold:.0%} threshold: {achievement_rate:.1%} achievement, "
                f"mean optimal k = {mean_optimal:.1f}"
            )

        return optimal_results

    def fit_scaling_laws(self) -> dict[str, t.Any]:
        """Fit different scaling functions to context-performance curves."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Fitting scaling laws...")
        scaling_results = self.calculator.fit_scaling_laws(self.filtered_data)

        # Print key findings
        law_comparisons = scaling_results.get("law_comparisons", {})
        print("\nScaling law findings:")
        for law in self.calculator.scaling_laws:
            wins = law_comparisons.get(f"{law}_wins", 0)
            mean_r2 = law_comparisons.get(f"mean_{law}_r2", 0)
            print(f"  - {law.title()}: {wins} best fits, mean R² = {mean_r2:.3f}")

        return scaling_results

    def analyze_complexity_scaling(self) -> dict[str, t.Any]:
        """Analyze how complexity affects context requirements."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Analyzing complexity scaling...")
        complexity_results = self.calculator.analyze_complexity_scaling(self.filtered_data)

        # Print key findings
        complexity_effects = complexity_results.get("complexity_effects", {})
        print("\nComplexity scaling findings:")
        print(
            f"  - Complexity vs optimal context (50%): r = {complexity_effects.get('complexity_vs_optimal_50', 0):.3f}"
        )
        print(f"  - Complexity vs max accuracy: r = {complexity_effects.get('complexity_vs_max_accuracy', 0):.3f}")
        print(f"  - Complexity vs efficiency: r = {complexity_effects.get('complexity_vs_efficiency', 0):.3f}")

        return complexity_results

    def compute_scaling_metrics(self) -> list[ScalingMetrics]:
        """Compute comprehensive scaling metrics combining all analyses."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Computing comprehensive scaling metrics...")

        # Get results from individual analyses
        optimal_results = self.detect_optimal_context_sizes()
        scaling_laws = self.fit_scaling_laws()
        complexity_results = self.analyze_complexity_scaling()

        # Combine into comprehensive metrics
        self.scaling_metrics = []

        # Create lookup dictionaries
        optimal_lookup = {row["model_id"]: row for row in optimal_results["model_thresholds"]}
        scaling_lookup = {row["model_id"]: row for row in scaling_laws["model_fits"]}

        # Merge data
        all_model_ids = set(optimal_lookup.keys()) | set(scaling_lookup.keys())

        for model_id in all_model_ids:
            optimal_data = optimal_lookup.get(model_id, {})
            scaling_data = scaling_lookup.get(model_id, {})

            if not optimal_data or not scaling_data:
                continue  # Skip incomplete data

            # Calculate scaling efficiency
            max_acc = optimal_data.get("max_accuracy", 0)
            max_context = max(self.filtered_data["context_size"]) if len(self.filtered_data) > 0 else 1
            scaling_efficiency = max_acc / max_context if max_context > 0 else 0

            metrics = ScalingMetrics(
                model_id=model_id,
                config_L=optimal_data.get("config_L", 0),
                config_m=optimal_data.get("config_m", 0),
                n_train=optimal_data.get("n_train", 0),
                complexity_score=optimal_data.get("complexity_score", 0),
                optimal_context_50=optimal_data.get("optimal_context_50", float("inf")),
                optimal_context_70=optimal_data.get("optimal_context_70", float("inf")),
                optimal_context_90=optimal_data.get("optimal_context_90", float("inf")),
                exponential_a=scaling_data.get("exponential_a", 0),
                exponential_b=scaling_data.get("exponential_b", 0),
                exponential_r2=scaling_data.get("exponential_r2", 0),
                power_a=scaling_data.get("power_a", 0),
                power_b=scaling_data.get("power_b", 0),
                power_r2=scaling_data.get("power_r2", 0),
                log_a=scaling_data.get("log_a", 0),
                log_b=scaling_data.get("log_b", 0),
                log_r2=scaling_data.get("log_r2", 0),
                best_model=scaling_data.get("best_model", "none"),
                best_r2=scaling_data.get("best_r2", 0),
                saturation_point=optimal_data.get("saturation_point", float("inf")),
                max_accuracy=optimal_data.get("max_accuracy", 0),
                scaling_efficiency=scaling_efficiency,
            )

            self.scaling_metrics.append(metrics)

        print(f"Computed comprehensive scaling metrics for {len(self.scaling_metrics)} models")
        return self.scaling_metrics

    def create_visualizations(self) -> list[Path]:
        """Create scaling analysis visualizations."""
        if not self.scaling_metrics:
            return []

        print("Creating visualizations...")
        visualization_paths = []

        # Convert metrics to DataFrame for plotting
        metrics_df = pd.DataFrame([m.to_dict() for m in self.scaling_metrics])

        # 1. Optimal context size vs complexity
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "complexity_score",
            "optimal_context_50",
            "Context Requirements vs Complexity",
            "context_vs_complexity",
        )
        if viz_path:
            visualization_paths.append(viz_path)

        # 2. Scaling law comparison
        viz_path = self._create_scaling_law_comparison()
        if viz_path:
            visualization_paths.append(viz_path)

        # 3. Best model distribution
        viz_path = self._create_best_model_distribution()
        if viz_path:
            visualization_paths.append(viz_path)

        # 4. Scaling efficiency vs complexity
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "complexity_score",
            "scaling_efficiency",
            "Scaling Efficiency vs Complexity",
            "efficiency_vs_complexity",
        )
        if viz_path:
            visualization_paths.append(viz_path)

        # 5. Multi-panel scaling overview
        try:
            viz_path = self._create_scaling_overview()
            if viz_path:
                visualization_paths.append(viz_path)
        except Exception as e:
            print(f"Warning: Failed to create scaling overview: {e}")

        # Filter out any None values
        visualization_paths = [p for p in visualization_paths if p is not None]

        return visualization_paths

    def _create_scaling_law_comparison(self) -> Path:
        """Create scaling law fit comparison visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.scaling_metrics])

        plt.figure(figsize=(12, 6))

        # Box plot comparison of R² scores
        exp_r2s = metrics_df["exponential_r2"].values
        power_r2s = metrics_df["power_r2"].values
        log_r2s = metrics_df["log_r2"].values

        # Filter out invalid R² scores
        exp_r2s = exp_r2s[exp_r2s > -np.inf]
        power_r2s = power_r2s[power_r2s > -np.inf]
        log_r2s = log_r2s[log_r2s > -np.inf]

        data_to_plot = []
        labels = []

        if len(exp_r2s) > 0:
            data_to_plot.append(exp_r2s)
            labels.append("Exponential")

        if len(power_r2s) > 0:
            data_to_plot.append(power_r2s)
            labels.append("Power")

        if len(log_r2s) > 0:
            data_to_plot.append(log_r2s)
            labels.append("Logarithmic")

        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
        else:
            plt.text(
                0.5,
                0.5,
                "No valid R² scores to plot",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )

        plt.ylabel("R² Score")
        plt.title("Scaling Law Fit Comparison")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "scaling_law_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_best_model_distribution(self) -> Path:
        """Create best fitting model distribution."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.scaling_metrics])

        plt.figure(figsize=(10, 6))

        best_models = metrics_df["best_model"].values
        model_counts = {}

        for model in best_models:
            if model != "none":
                model_counts[model] = model_counts.get(model, 0) + 1

        if model_counts:
            plt.bar(model_counts.keys(), model_counts.values())
            plt.ylabel("Number of Models")
            plt.title("Best Fitting Scaling Law Distribution")
            plt.xticks(rotation=45)
        else:
            plt.text(
                0.5,
                0.5,
                "No valid scaling law fits",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )

        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "best_model_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_scaling_overview(self) -> Path:
        """Create comprehensive scaling overview visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.scaling_metrics])

        def plot_optimal_thresholds():
            """Plot optimal context sizes for different thresholds."""
            thresholds = [50, 70, 90]
            threshold_data = []

            for threshold in thresholds:
                col_name = f"optimal_context_{threshold}"
                if col_name in metrics_df.columns:
                    data = metrics_df[metrics_df[col_name] != float("inf")][col_name]
                    threshold_data.append(data.values if len(data) > 0 else [])

            # Filter out empty datasets
            valid_data = [data for data in threshold_data if len(data) > 0]
            valid_labels = [f"{thresholds[i]}%" for i, data in enumerate(threshold_data) if len(data) > 0]

            if valid_data:
                plt.boxplot(valid_data, labels=valid_labels)
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No threshold achievements",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )

            plt.xlabel("Performance Threshold")
            plt.ylabel("Optimal Context Size")
            plt.title("Optimal Context by Threshold")

        def plot_complexity_vs_efficiency():
            """Plot complexity vs scaling efficiency."""
            if len(metrics_df) > 0:
                plt.scatter(metrics_df["complexity_score"], metrics_df["scaling_efficiency"], alpha=0.6)
                plt.xlabel("Configuration Complexity (L×m)")
                plt.ylabel("Scaling Efficiency")
                plt.title("Efficiency vs Complexity")

                # Add trend line if possible
                if len(metrics_df) > 2 and metrics_df["complexity_score"].std() > 0:
                    try:
                        z = np.polyfit(metrics_df["complexity_score"], metrics_df["scaling_efficiency"], 1)
                        p = np.poly1d(z)
                        plt.plot(metrics_df["complexity_score"], p(metrics_df["complexity_score"]), "r--", alpha=0.8)
                    except:
                        pass
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No data available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )

        def plot_saturation_points():
            """Plot distribution of saturation points."""
            saturation_data = metrics_df[metrics_df["saturation_point"] != float("inf")]["saturation_point"]

            if len(saturation_data) > 0:
                plt.hist(saturation_data, bins=min(15, len(saturation_data)), alpha=0.7, edgecolor="black")
                plt.axvline(
                    saturation_data.mean(), color="red", linestyle="--", label=f"Mean: {saturation_data.mean():.1f}"
                )
                plt.legend()
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No saturation points detected",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )

            plt.xlabel("Context Size")
            plt.ylabel("Frequency")
            plt.title("Saturation Point Distribution")

        def plot_r2_by_complexity():
            """Plot best R² by complexity."""
            if len(metrics_df) > 0:
                plt.scatter(
                    metrics_df["complexity_score"],
                    metrics_df["best_r2"],
                    c=metrics_df["optimal_context_50"],
                    cmap="viridis",
                    alpha=0.6,
                )
                plt.colorbar(label="Optimal Context (50%)")
                plt.xlabel("Configuration Complexity (L×m)")
                plt.ylabel("Best R² Score")
                plt.title("Scaling Law Fit Quality vs Complexity")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No data available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )

        plot_functions = [
            plot_optimal_thresholds,
            plot_complexity_vs_efficiency,
            plot_saturation_points,
            plot_r2_by_complexity,
        ]

        return self.visualizer.create_multi_panel_figure(
            plot_functions, layout=(2, 2), filename="scaling_overview", suptitle="ICL Context Scaling Analysis Overview"
        )

    def generate_report(
        self,
        optimal_results: dict[str, t.Any],
        scaling_laws: dict[str, t.Any],
        complexity_results: dict[str, t.Any],
        visualization_paths: list[Path],
    ) -> Path:
        """Generate comprehensive scaling analysis report."""
        if not self.config.generate_reports:
            return Path()

        print("Generating scaling analysis report...")

        # Prepare report sections
        sections = []

        # Summary section
        threshold_stats = optimal_results.get("threshold_statistics", {})
        law_comparisons = scaling_laws.get("law_comparisons", {})

        sections.append(
            {
                "title": "Summary",
                "content": f"Analysis of context scaling patterns across {len(self.scaling_metrics)} models.",
                "metrics": {
                    "Total Models Analyzed": len(self.scaling_metrics),
                    "Scaling Laws Tested": len(self.calculator.scaling_laws),
                    "Performance Thresholds": len(self.calculator.performance_thresholds),
                },
            }
        )

        # Optimal context analysis
        if threshold_stats:
            sections.append(
                {
                    "title": "Optimal Context Analysis",
                    "content": "Context requirements for different performance thresholds.",
                    "metrics": {},
                }
            )

            for threshold_key, stats in threshold_stats.items():
                threshold = float(threshold_key.split("_")[1])
                sections[-1]["metrics"].update(
                    {
                        f"{threshold:.0%} Achievement Rate": f"{stats.get('achievement_rate', 0):.1%}",
                        f"{threshold:.0%} Mean Optimal Context": f"{stats.get('mean_optimal_k', float('inf')):.1f}",
                    }
                )

        # Scaling law comparison
        if law_comparisons:
            sections.append(
                {
                    "title": "Scaling Law Comparison",
                    "content": "Performance of different scaling law fits.",
                    "metrics": {},
                }
            )

            for law in self.calculator.scaling_laws:
                wins = law_comparisons.get(f"{law}_wins", 0)
                mean_r2 = law_comparisons.get(f"mean_{law}_r2", 0)
                sections[-1]["metrics"].update(
                    {f"{law.title()} Best Fits": wins, f"{law.title()} Mean R²": f"{mean_r2:.3f}"}
                )

        # Complexity effects
        complexity_effects = complexity_results.get("complexity_effects", {})
        if complexity_effects:
            sections.append(
                {
                    "title": "Complexity Effects",
                    "content": "How model complexity affects scaling patterns.",
                    "metrics": {
                        "Complexity-Optimal Context Correlation": f"{complexity_effects.get('complexity_vs_optimal_50', 0):.3f}",
                        "Complexity-Max Accuracy Correlation": f"{complexity_effects.get('complexity_vs_max_accuracy', 0):.3f}",
                        "Complexity-Efficiency Correlation": f"{complexity_effects.get('complexity_vs_efficiency', 0):.3f}",
                    },
                }
            )

        # Add visualizations
        for viz_path in visualization_paths:
            if viz_path is not None and viz_path.exists():
                sections.append(
                    {
                        "title": viz_path.stem.replace("_", " ").title(),
                        "visualization": viz_path,
                        "viz_title": viz_path.stem.replace("_", " ").title(),
                    }
                )

        return self.visualizer.generate_html_report(
            "ICL Context Scaling Analysis Report", sections, "scaling_report.html"
        )

    def save_metrics(self) -> Path:
        """Save scaling metrics to CSV."""
        if not self.scaling_metrics:
            raise ValueError("No scaling metrics to save")

        metrics_df = pd.DataFrame([m.to_dict() for m in self.scaling_metrics])
        output_path = self.output_dir / "metrics" / "scaling_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_path, index=False)
        print(f"Saved scaling metrics: {output_path}")

        return output_path

    def run_complete_analysis(self) -> dict[str, t.Any]:
        """Run complete scaling analysis pipeline."""
        print("=" * 60)
        print("EXPERIMENT 3: CONTEXT SCALING ANALYSIS")
        print("=" * 60)

        # Load and process data
        self.load_and_filter_data()

        # Run individual analyses
        optimal_results = self.detect_optimal_context_sizes()
        scaling_laws = self.fit_scaling_laws()
        complexity_results = self.analyze_complexity_scaling()

        # Compute comprehensive metrics
        self.compute_scaling_metrics()

        # Create outputs
        visualization_paths = []
        if self.config.create_visualizations:
            visualization_paths = self.create_visualizations()

        report_path = Path()
        if self.config.generate_reports:
            report_path = self.generate_report(optimal_results, scaling_laws, complexity_results, visualization_paths)

        metrics_path = self.save_metrics()

        print("=" * 60)
        print("SCALING ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")

        return {
            "scaling_metrics_path": metrics_path,
            "report_path": report_path,
            "visualization_paths": visualization_paths,
            "optimal_results": optimal_results,
            "scaling_laws": scaling_laws,
            "complexity_results": complexity_results,
            "total_models_analyzed": len(self.scaling_metrics),
        }
