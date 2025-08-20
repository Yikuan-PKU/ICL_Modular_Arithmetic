"""Transfer Learning Analysis (Experiment 2) - Main analyzer."""

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ICL.eval.exp.shared.exp_config import ExpConfig
from ICL.eval.exp.shared.result_loader import Phase1ResultLoader
from ICL.eval.exp.shared.visualizer import AnalysisVisualizer
from ICL.eval.exp.transfer.metrics import TransferMetrics, TransferMetricsCalculator


class TransferAnalyzer:
    """Analyzes ICL transfer learning patterns from Phase 1 evaluation data."""

    def __init__(self, config: ExpConfig, output_dir: Path):
        """Initialize transfer analyzer."""
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = Phase1ResultLoader(config.collection_results_dir)
        self.visualizer = AnalysisVisualizer(output_dir)
        self.calculator = TransferMetricsCalculator(success_threshold=config.exp2_config.get("success_threshold", 0.5))

        # Data storage
        self.data: pd.DataFrame | None = None
        self.filtered_data: pd.DataFrame | None = None
        self.transfer_pairs: pd.DataFrame | None = None
        self.transfer_metrics: list[TransferMetrics] = []
        self.transfer_matrix: dict[str, pd.DataFrame] = {}

    def load_and_filter_data(self) -> pd.DataFrame:
        """Load Phase 1 results and filter for transfer analysis."""
        print("Loading Phase 1 evaluation results...")
        self.data = self.loader.load_icl_performance()

        # Validate data completeness
        validation = self.loader.validate_data_completeness(self.data)
        if validation["missing_data_issues"]:
            print("Data completeness warnings:")
            for issue in validation["missing_data_issues"]:
                print(f"  - {issue}")

        # Filter for transfer analysis (all conditions, normal sequences only)
        self.filtered_data = self.loader.filter_for_analysis(self.data, "transfer")

        print("Filtered data for transfer analysis:")
        print(f"  - Records: {len(self.filtered_data)}")
        print(f"  - Transfer conditions: {sorted(self.filtered_data['transfer_condition'].unique())}")
        print(f"  - Unique models: {self.filtered_data['model_id'].nunique()}")

        return self.filtered_data

    def create_transfer_pairs(self) -> pd.DataFrame:
        """Create within-config vs transfer performance pairs."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        print("Creating transfer pairs...")
        self.transfer_pairs = self.calculator.create_transfer_pairs(self.filtered_data)

        return self.transfer_pairs

    def compute_transfer_metrics(self) -> list[TransferMetrics]:
        """Compute transfer degradation metrics for each transfer pair."""
        if self.transfer_pairs is None:
            raise ValueError("Must create transfer pairs first")

        print("Computing transfer metrics...")
        self.transfer_metrics = self.calculator.compute_transfer_metrics(self.transfer_pairs, self.filtered_data)

        return self.transfer_metrics

    def analyze_transfer_patterns(self) -> dict[str, t.Any]:
        """Analyze cross-model transfer patterns."""
        if not self.transfer_metrics:
            raise ValueError("Must compute transfer metrics first")

        print("Analyzing transfer patterns...")
        patterns = self.calculator.analyze_transfer_patterns(self.transfer_metrics)

        # Print key findings
        type_analysis = patterns.get("transfer_type_analysis", {})
        degradation = patterns.get("degradation_patterns", {})
        success = patterns.get("success_rates", {})

        print("\nKey transfer findings:")
        print(f"  - Overall success rate: {success.get('overall_success_rate', 0):.1%}")
        print(f"  - Mean degradation: {degradation.get('overall_mean_degradation', 0):.3f}")
        print(f"  - Improvement rate: {degradation.get('improvement_rate', 0):.1%}")

        for transfer_type, analysis in type_analysis.items():
            print(
                f"  - {transfer_type.title()} transfer: {analysis.get('success_rate', 0):.1%} success, "
                f"{analysis.get('mean_degradation', 0):.3f} degradation"
            )

        return patterns

    def create_transfer_matrix(self) -> dict[str, pd.DataFrame]:
        """Create train→test configuration transfer performance matrix."""
        if not self.transfer_metrics:
            raise ValueError("Must compute transfer metrics first")

        print("Creating transfer matrix...")
        self.transfer_matrix = self.calculator.create_transfer_matrix(self.transfer_metrics)

        return self.transfer_matrix

    def create_visualizations(self) -> list[Path]:
        """Create transfer analysis visualizations."""
        if not self.transfer_metrics:
            return []

        print("Creating visualizations...")
        visualization_paths = []

        # Convert metrics to DataFrame for plotting
        metrics_df = pd.DataFrame([m.to_dict() for m in self.transfer_metrics])

        # 1. Transfer degradation by type
        viz_path = self.visualizer.create_comparison_boxplot(
            metrics_df, "transfer_type", "transfer_degradation", "Transfer Degradation by Type", "degradation_by_type"
        )
        visualization_paths.append(viz_path)

        # 2. Transfer vs within-config accuracy scatter
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "within_config_accuracy",
            "transfer_accuracy",
            "Transfer vs Within-Config Accuracy",
            "transfer_vs_within",
        )
        visualization_paths.append(viz_path)

        # 3. Degradation vs depth increase
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "depth_increase",
            "transfer_degradation",
            "Transfer Degradation vs Depth Increase",
            "degradation_vs_depth",
        )
        visualization_paths.append(viz_path)

        # 4. Degradation vs synonym increase
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df,
            "synonym_increase",
            "transfer_degradation",
            "Transfer Degradation vs Synonym Increase",
            "degradation_vs_synonym",
        )
        visualization_paths.append(viz_path)

        # 5. Transfer matrix heatmaps
        if self.transfer_matrix:
            viz_path = self._create_transfer_matrix_visualization()
            visualization_paths.append(viz_path)

        # 6. Multi-panel transfer overview
        viz_path = self._create_transfer_overview()
        visualization_paths.append(viz_path)

        return visualization_paths

    def _create_transfer_matrix_visualization(self) -> Path:
        """Create transfer matrix visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy matrix
        plt.sca(axes[0])
        sns.heatmap(
            self.transfer_matrix["accuracy"],
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Transfer Accuracy"},
        )
        plt.title("Transfer Accuracy Matrix")
        plt.xlabel("Test Configuration")
        plt.ylabel("Train Configuration")

        # Degradation matrix
        plt.sca(axes[1])
        sns.heatmap(
            self.transfer_matrix["degradation"],
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            cbar_kws={"label": "Transfer Degradation"},
        )
        plt.title("Transfer Degradation Matrix")
        plt.xlabel("Test Configuration")
        plt.ylabel("Train Configuration")

        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "transfer_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_transfer_overview(self) -> Path:
        """Create comprehensive transfer overview visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.transfer_metrics])

    def _create_transfer_overview(self) -> Path:
        """Create comprehensive transfer overview visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.transfer_metrics])

        def plot_degradation_by_type():
            """Plot transfer degradation by type."""
            type_data = []
            type_labels = []
            for transfer_type in ["depth", "synonym", "full"]:
                type_metrics = metrics_df[metrics_df["transfer_type"] == transfer_type]
                if len(type_metrics) > 0:
                    type_data.append(type_metrics["transfer_degradation"].values)
                    type_labels.append(transfer_type.title())

            if type_data:
                plt.boxplot(type_data, labels=type_labels)
            plt.ylabel("Transfer Degradation")
            plt.title("Transfer Degradation by Type")
            plt.grid(True, alpha=0.3)

        def plot_success_rate_by_training():
            """Plot success rate by training size."""
            training_sizes = sorted(metrics_df["n_train"].unique())
            success_rates = []
            for size in training_sizes:
                size_data = metrics_df[metrics_df["n_train"] == size]
                success_rate = size_data["transfer_success"].mean()
                success_rates.append(success_rate)

            plt.plot(training_sizes, success_rates, "o-", linewidth=2, markersize=6)
            plt.xlabel("Training Size")
            plt.ylabel("Transfer Success Rate")
            plt.title("Success Rate vs Training Size")
            plt.grid(True, alpha=0.3)
            if training_sizes:
                plt.xscale("log")

        def plot_transfer_vs_within_scatter():
            """Plot transfer vs within-config accuracy."""
            plt.scatter(
                metrics_df["within_config_accuracy"],
                metrics_df["transfer_accuracy"],
                c=metrics_df["depth_increase"],
                cmap="viridis",
                alpha=0.6,
            )
            plt.colorbar(label="Depth Increase")
            plt.plot([0, 1], [0, 1], "r--", alpha=0.8, label="Perfect Transfer")
            plt.xlabel("Within-Config Accuracy")
            plt.ylabel("Transfer Accuracy")
            plt.title("Transfer vs Within-Config Performance")
            plt.legend()
            plt.grid(True, alpha=0.3)

        def plot_degradation_vs_complexity_change():
            """Plot degradation vs complexity change."""
            metrics_df["complexity_change"] = (
                metrics_df["test_config_L"] * metrics_df["test_config_m"]
                - metrics_df["train_config_L"] * metrics_df["train_config_m"]
            )
            plt.scatter(metrics_df["complexity_change"], metrics_df["transfer_degradation"], alpha=0.6)
            plt.xlabel("Complexity Change (ΔL×m)")
            plt.ylabel("Transfer Degradation")
            plt.title("Degradation vs Complexity Change")
            plt.grid(True, alpha=0.3)

            # Add trend line
            if len(metrics_df) > 2:
                z = np.polyfit(metrics_df["complexity_change"], metrics_df["transfer_degradation"], 1)
                p = np.poly1d(z)
                plt.plot(metrics_df["complexity_change"], p(metrics_df["complexity_change"]), "r--", alpha=0.8)

        plot_functions = [
            plot_degradation_by_type,
            plot_success_rate_by_training,
            plot_transfer_vs_within_scatter,
            plot_degradation_vs_complexity_change,
        ]

        return self.visualizer.create_multi_panel_figure(
            plot_functions,
            layout=(2, 2),
            filename="transfer_overview",
            suptitle="ICL Transfer Learning Analysis Overview",
        )

    def generate_report(self, patterns: dict[str, t.Any], visualization_paths: list[Path]) -> Path:
        """Generate comprehensive transfer analysis report."""
        if not self.config.generate_reports:
            return Path()

        print("Generating transfer analysis report...")

        # Prepare report sections
        sections = []

        # Summary section
        success_rates = patterns.get("success_rates", {})
        degradation = patterns.get("degradation_patterns", {})
        sections.append(
            {
                "title": "Summary",
                "content": f"Analysis of ICL transfer learning patterns across {len(self.transfer_metrics)} transfer pairs.",
                "metrics": {
                    "Total Transfer Pairs": len(self.transfer_metrics),
                    "Overall Success Rate": f"{success_rates.get('overall_success_rate', 0):.1%}",
                    "Mean Degradation": f"{degradation.get('overall_mean_degradation', 0):.3f}",
                    "Improvement Rate": f"{degradation.get('improvement_rate', 0):.1%}",
                    "Severe Degradation Rate": f"{degradation.get('severe_degradation_rate', 0):.1%}",
                },
            }
        )

        # Transfer type analysis
        type_analysis = patterns.get("transfer_type_analysis", {})
        if type_analysis:
            sections.append(
                {
                    "title": "Transfer Type Analysis",
                    "content": "Performance breakdown by transfer type (depth, synonym, full).",
                    "metrics": {},
                }
            )

            for transfer_type, analysis in type_analysis.items():
                sections[-1]["metrics"].update(
                    {
                        f"{transfer_type.title()} Success Rate": f"{analysis.get('success_rate', 0):.1%}",
                        f"{transfer_type.title()} Mean Degradation": f"{analysis.get('mean_degradation', 0):.3f}",
                        f"{transfer_type.title()} Sample Count": analysis.get("count", 0),
                    }
                )

        # Configuration effects
        config_effects = patterns.get("config_effects", {})
        sections.append(
            {
                "title": "Configuration Effects",
                "content": "How configuration differences affect transfer performance.",
                "metrics": {
                    "Depth Increase-Degradation Correlation": f"{config_effects.get('depth_increase_degradation_correlation', 0):.3f}",
                    "Synonym Increase-Degradation Correlation": f"{config_effects.get('synonym_increase_degradation_correlation', 0):.3f}",
                    "Complexity Change-Degradation Correlation": f"{config_effects.get('complexity_change_degradation_correlation', 0):.3f}",
                },
            }
        )

        # Success rate patterns
        success_by_training = success_rates.get("success_by_training_size", {})
        if success_by_training:
            sections.append(
                {
                    "title": "Success Rate by Training Size",
                    "content": "How training data size affects transfer success.",
                    "metrics": {f"Training Size {k}": f"{v:.1%}" for k, v in success_by_training.items()},
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

        return self.visualizer.generate_html_report(
            "ICL Transfer Learning Analysis Report", sections, "transfer_report.html"
        )

    def save_metrics(self) -> Path:
        """Save transfer metrics to CSV."""
        if not self.transfer_metrics:
            raise ValueError("No transfer metrics to save")

        metrics_df = pd.DataFrame([m.to_dict() for m in self.transfer_metrics])
        output_path = self.output_dir / "metrics" / "transfer_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_path, index=False)
        print(f"Saved transfer metrics: {output_path}")

        return output_path

    def save_transfer_matrix(self) -> Path:
        """Save transfer matrix data."""
        if not self.transfer_matrix:
            return Path()

        # Save accuracy matrix
        accuracy_path = self.output_dir / "metrics" / "transfer_accuracy_matrix.csv"
        self.transfer_matrix["accuracy"].to_csv(accuracy_path)

        # Save degradation matrix
        degradation_path = self.output_dir / "metrics" / "transfer_degradation_matrix.csv"
        self.transfer_matrix["degradation"].to_csv(degradation_path)

        # Save raw aggregated data
        raw_path = self.output_dir / "metrics" / "transfer_matrix_raw.csv"
        self.transfer_matrix["raw_data"].to_csv(raw_path, index=False)

        print(f"Saved transfer matrices: {accuracy_path.parent}")
        return accuracy_path

    def validate_data_completeness(self) -> dict[str, t.Any]:
        """Validate completeness of Phase 1 data for transfer analysis."""
        if self.filtered_data is None:
            raise ValueError("Must load and filter data first")

        validation = {
            "total_records": len(self.filtered_data),
            "transfer_conditions_available": sorted(self.filtered_data["transfer_condition"].unique()),
            "within_config_records": len(
                self.filtered_data[self.filtered_data["transfer_condition"] == "within_config"]
            ),
            "transfer_records": len(self.filtered_data[self.filtered_data["transfer_condition"] != "within_config"]),
            "unique_models": self.filtered_data["model_id"].nunique(),
            "config_pairs_available": len(
                self.filtered_data.groupby(["config_L", "config_m", "target_config_L", "target_config_m"])
            ),
            "missing_transfer_conditions": [],
            "incomplete_transfer_models": [],
        }

        # Check for expected transfer conditions
        expected_conditions = ["within_config", "cross_L", "cross_m", "cross_config"]
        available_conditions = set(self.filtered_data["transfer_condition"].unique())
        missing = [c for c in expected_conditions if c not in available_conditions]
        validation["missing_transfer_conditions"] = missing

        # Check model transfer completeness
        for model_id in self.filtered_data["model_id"].unique():
            model_data = self.filtered_data[self.filtered_data["model_id"] == model_id]
            model_conditions = set(model_data["transfer_condition"].unique())
            if "within_config" not in model_conditions:
                validation["incomplete_transfer_models"].append(model_id)

        return validation

    def run_complete_analysis(self) -> dict[str, t.Any]:
        """Run complete transfer analysis pipeline."""
        print("=" * 60)
        print("EXPERIMENT 2: ICL TRANSFER LEARNING ANALYSIS")
        print("=" * 60)

        # Load and process data
        self.load_and_filter_data()

        # Validate data completeness
        validation = self.validate_data_completeness()
        if validation["missing_transfer_conditions"]:
            print(f"Warning: Missing transfer conditions: {validation['missing_transfer_conditions']}")

        # Create transfer pairs and compute metrics
        self.create_transfer_pairs()
        self.compute_transfer_metrics()

        # Analyze patterns
        patterns = self.analyze_transfer_patterns()

        # Create transfer matrix
        self.create_transfer_matrix()

        # Create outputs
        visualization_paths = []
        if self.config.create_visualizations:
            visualization_paths = self.create_visualizations()

        report_path = Path()
        if self.config.generate_reports:
            report_path = self.generate_report(patterns, visualization_paths)

        metrics_path = self.save_metrics()
        matrix_path = self.save_transfer_matrix()

        print("=" * 60)
        print("TRANSFER ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")

        return {
            "transfer_metrics_path": metrics_path,
            "transfer_matrix_path": matrix_path,
            "report_path": report_path,
            "visualization_paths": visualization_paths,
            "patterns": patterns,
            "validation": validation,
            "total_transfer_pairs_analyzed": len(self.transfer_metrics),
            "transfer_matrix": self.transfer_matrix,
        }
