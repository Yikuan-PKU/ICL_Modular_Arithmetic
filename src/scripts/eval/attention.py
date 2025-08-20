"""Attention Pattern Analysis (Experiment 4) - Main analyzer."""

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ICL.eval.exp.attention.metrics import AttentionMetrics, AttentionMetricsCalculator
from ICL.eval.exp.shared.exp_config import ExpConfig
from ICL.eval.exp.shared.result_loader import Phase1ResultLoader
from ICL.eval.exp.shared.visualizer import AnalysisVisualizer


class AttentionAnalyzer:
    """Analyzes attention patterns from Phase 1 evaluation data."""

    def __init__(self, config: ExpConfig, output_dir: Path):
        """Initialize attention analyzer."""
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = Phase1ResultLoader(config.collection_results_dir)
        self.visualizer = AnalysisVisualizer(output_dir)
        self.calculator = AttentionMetricsCalculator(
            induction_threshold=config.exp4_config.get("induction_threshold", 0.3),
            copying_threshold=config.exp4_config.get("copying_threshold", 0.3),
            previous_token_threshold=config.exp4_config.get("previous_token_threshold", 0.3),
        )

        # Data storage
        self.data: pd.DataFrame | None = None
        self.attention_data: dict[str, dict[str, t.Any]] = {}
        self.attention_metrics: list[AttentionMetrics] = []

    def load_and_filter_data(self) -> pd.DataFrame:
        """Load Phase 1 results and attention data."""
        print("Loading Phase 1 evaluation results...")
        self.data = self.loader.load_icl_performance()

        # Validate data completeness
        validation = self.loader.validate_data_completeness(self.data)
        if validation["missing_data_issues"]:
            print("Data completeness warnings:")
            for issue in validation["missing_data_issues"]:
                print(f"  - {issue}")

        print("Loaded ICL performance data:")
        print(f"  - Records: {len(self.data)}")
        print(f"  - Unique models: {self.data['model_id'].nunique()}")

        return self.data

    def load_attention_data(self) -> dict[str, dict[str, t.Any]]:
        """Load attention data for analysis."""
        print("Loading attention data...")

        # Get unique model IDs from the data
        model_ids = self.data["model_id"].unique().tolist() if self.data is not None else []

        # Load attention data
        self.attention_data = self.loader.load_attention_data(model_ids)

        if not self.attention_data:
            raise ValueError("No attention data found. Make sure attention capture was enabled during collection.")

        print("Loaded attention data:")
        print(f"  - Models with attention data: {len(self.attention_data)}")
        total_matrices = sum(len(model_data) for model_data in self.attention_data.values())
        print(f"  - Total attention matrices: {total_matrices}")

        return self.attention_data

    def analyze_attention_patterns(self) -> list[AttentionMetrics]:
        """Analyze attention patterns for all models and heads."""
        if not self.attention_data:
            raise ValueError("Must load attention data first")

        print("Analyzing attention patterns...")
        self.attention_metrics = []

        for model_id, model_attention_data in self.attention_data.items():
            # Get model metadata from ICL performance data
            model_data = self.data[self.data["model_id"] == model_id].iloc[0] if self.data is not None else {}
            config_L = model_data.get("config_L", 0)
            config_m = model_data.get("config_m", 0)
            n_train = model_data.get("n_train", 0)

            # Analyze each attention matrix for this model
            for attention_key, attention_info in model_attention_data.items():
                try:
                    metadata = attention_info.get("metadata", {})
                    attention_matrix = attention_info.get("attention_matrix")

                    if attention_matrix is None:
                        continue

                    # Extract metadata
                    layer_idx = metadata.get("layer_idx", 0)
                    head_idx = metadata.get("head_idx", 0)
                    context_size = metadata.get("context_size", 0)

                    # Analyze attention matrix
                    pattern_metrics = self.calculator.analyze_attention_matrix(attention_matrix)

                    # Compute scaling metrics for this head across context sizes
                    head_attention_data = {
                        k: v
                        for k, v in model_attention_data.items()
                        if v.get("metadata", {}).get("layer_idx") == layer_idx
                        and v.get("metadata", {}).get("head_idx") == head_idx
                    }

                    scaling_metrics = self.calculator.compute_attention_scaling(head_attention_data)

                    # Create comprehensive metrics
                    metrics = AttentionMetrics(
                        model_id=model_id,
                        config_L=config_L,
                        config_m=config_m,
                        n_train=n_train,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        context_size=context_size,
                        attention_entropy=pattern_metrics.get("attention_entropy", 0.0),
                        attention_concentration=pattern_metrics.get("attention_concentration", 0.0),
                        last_token_attention=pattern_metrics.get("last_token_attention", 0.0),
                        diagonal_attention=pattern_metrics.get("diagonal_attention", 0.0),
                        is_induction_head=pattern_metrics.get("is_induction_head", False),
                        is_copying_head=pattern_metrics.get("is_copying_head", False),
                        is_previous_token_head=pattern_metrics.get("is_previous_token_head", False),
                        induction_score=pattern_metrics.get("induction_score", 0.0),
                        copying_score=pattern_metrics.get("copying_score", 0.0),
                        previous_token_score=pattern_metrics.get("previous_token_score", 0.0),
                        attention_scaling_slope=scaling_metrics.get("attention_scaling_slope", 0.0),
                        attention_stability=scaling_metrics.get("attention_stability", 0.0),
                    )

                    self.attention_metrics.append(metrics)

                except Exception as e:
                    print(f"Warning: Failed to analyze attention for {model_id}, {attention_key}: {e}")
                    continue

        print(f"Analyzed attention patterns for {len(self.attention_metrics)} attention heads")
        return self.attention_metrics

    def analyze_head_types(self) -> dict[str, t.Any]:
        """Analyze distribution and characteristics of different head types."""
        if not self.attention_metrics:
            raise ValueError("Must analyze attention patterns first")

        print("Analyzing attention head types...")

        # Overall head type statistics
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        head_type_analysis = {
            "overall_statistics": {
                "total_heads": len(metrics_df),
                "induction_heads": metrics_df["is_induction_head"].sum(),
                "copying_heads": metrics_df["is_copying_head"].sum(),
                "previous_token_heads": metrics_df["is_previous_token_head"].sum(),
                "induction_rate": metrics_df["is_induction_head"].mean(),
                "copying_rate": metrics_df["is_copying_head"].mean(),
                "previous_token_rate": metrics_df["is_previous_token_head"].mean(),
            }
        }

        # Layer-wise analysis
        head_type_analysis["layer_analysis"] = self.calculator.analyze_head_types_by_layer(self.attention_metrics)

        # Configuration analysis
        head_type_analysis["config_analysis"] = self.calculator.analyze_attention_patterns_by_config(
            self.attention_metrics
        )

        # Context scaling analysis
        head_type_analysis["scaling_analysis"] = self.calculator.compute_attention_context_scaling(
            self.attention_metrics
        )

        # Print key findings
        overall = head_type_analysis["overall_statistics"]
        print("\nAttention head type findings:")
        print(f"  - Total heads analyzed: {overall['total_heads']}")
        print(f"  - Induction heads: {overall['induction_heads']} ({overall['induction_rate']:.1%})")
        print(f"  - Copying heads: {overall['copying_heads']} ({overall['copying_rate']:.1%})")
        print(f"  - Previous token heads: {overall['previous_token_heads']} ({overall['previous_token_rate']:.1%})")

        return head_type_analysis

    def create_visualizations(self) -> list[Path]:
        """Create attention analysis visualizations."""
        if not self.attention_metrics:
            return []

        print("Creating visualizations...")
        visualization_paths = []

        # Convert metrics to DataFrame for plotting
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        # 1. Head type distribution by layer
        viz_path = self._create_head_type_by_layer_plot()
        if viz_path:
            visualization_paths.append(viz_path)

        # 2. Attention entropy distribution
        viz_path = self.visualizer.create_distribution_plot(
            metrics_df, "attention_entropy", "Distribution of Attention Entropy", "attention_entropy_dist"
        )
        if viz_path:
            visualization_paths.append(viz_path)

        # 3. Head type scores comparison
        viz_path = self._create_head_type_scores_plot()
        if viz_path:
            visualization_paths.append(viz_path)

        # 4. Attention patterns by configuration
        viz_path = self._create_attention_by_config_plot()
        if viz_path:
            visualization_paths.append(viz_path)

        # 5. Attention scaling analysis
        viz_path = self.visualizer.create_correlation_plot(
            metrics_df, "context_size", "attention_entropy", "Attention Entropy vs Context Size", "attention_scaling"
        )
        if viz_path:
            visualization_paths.append(viz_path)

        # 6. Multi-panel attention overview
        try:
            viz_path = self._create_attention_overview()
            if viz_path:
                visualization_paths.append(viz_path)
        except Exception as e:
            print(f"Warning: Failed to create attention overview: {e}")

        # Filter out any None values
        visualization_paths = [p for p in visualization_paths if p is not None]

        return visualization_paths

    def _create_head_type_by_layer_plot(self) -> Path:
        """Create head type distribution by layer visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        plt.figure(figsize=(12, 6))

        layers = sorted(metrics_df["layer_idx"].unique())

        if len(layers) > 0:
            # Count head types by layer
            layer_data = {}
            for layer in layers:
                layer_metrics = metrics_df[metrics_df["layer_idx"] == layer]
                layer_data[layer] = {
                    "induction": layer_metrics["is_induction_head"].sum(),
                    "copying": layer_metrics["is_copying_head"].sum(),
                    "previous_token": layer_metrics["is_previous_token_head"].sum(),
                    "total": len(layer_metrics),
                }

            # Create stacked bar plot
            induction_counts = [layer_data[l]["induction"] for l in layers]
            copying_counts = [layer_data[l]["copying"] for l in layers]
            previous_counts = [layer_data[l]["previous_token"] for l in layers]

            width = 0.6
            plt.bar(layers, induction_counts, width, label="Induction Heads")
            plt.bar(layers, copying_counts, width, bottom=induction_counts, label="Copying Heads")
            plt.bar(
                layers,
                previous_counts,
                width,
                bottom=np.array(induction_counts) + np.array(copying_counts),
                label="Previous Token Heads",
            )

            plt.xlabel("Layer Index")
            plt.ylabel("Number of Heads")
            plt.title("Attention Head Types by Layer")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "No layer data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )

        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "head_types_by_layer.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_head_type_scores_plot(self) -> Path:
        """Create head type scores comparison visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        plt.figure(figsize=(12, 6))

        if len(metrics_df) > 0:
            # Box plot of different head type scores
            scores_data = [
                metrics_df["induction_score"].values,
                metrics_df["copying_score"].values,
                metrics_df["previous_token_score"].values,
            ]

            plt.boxplot(scores_data, labels=["Induction", "Copying", "Previous Token"])
            plt.ylabel("Pattern Score")
            plt.title("Head Type Pattern Scores")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "No score data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )

        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "head_type_scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_attention_by_config_plot(self) -> Path:
        """Create attention patterns by configuration visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        plt.figure(figsize=(12, 8))

        if len(metrics_df) > 0:
            # Create scatter plot of attention entropy by configuration complexity
            metrics_df["complexity"] = metrics_df["config_L"] * metrics_df["config_m"]

            # Color by head type
            colors = []
            for _, row in metrics_df.iterrows():
                if row["is_induction_head"]:
                    colors.append("red")
                elif row["is_copying_head"]:
                    colors.append("blue")
                elif row["is_previous_token_head"]:
                    colors.append("green")
                else:
                    colors.append("gray")

            plt.scatter(metrics_df["complexity"], metrics_df["attention_entropy"], c=colors, alpha=0.6)

            # Create custom legend
            import matplotlib.patches as mpatches

            red_patch = mpatches.Patch(color="red", label="Induction Heads")
            blue_patch = mpatches.Patch(color="blue", label="Copying Heads")
            green_patch = mpatches.Patch(color="green", label="Previous Token Heads")
            gray_patch = mpatches.Patch(color="gray", label="Other Heads")
            plt.legend(handles=[red_patch, blue_patch, green_patch, gray_patch])

            plt.xlabel("Configuration Complexity (L×m)")
            plt.ylabel("Attention Entropy")
            plt.title("Attention Patterns by Configuration")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "No configuration data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )

        plt.tight_layout()

        output_path = self.visualizer.viz_dir / "attention_by_config.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_attention_overview(self) -> Path:
        """Create comprehensive attention overview visualization."""
        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])

        def plot_entropy_by_layer():
            """Plot attention entropy by layer."""
            if len(metrics_df) > 0:
                layers = sorted(metrics_df["layer_idx"].unique())
                entropy_by_layer = [
                    metrics_df[metrics_df["layer_idx"] == layer]["attention_entropy"].values for layer in layers
                ]

                if entropy_by_layer:
                    plt.boxplot(entropy_by_layer, labels=layers)
                    plt.xlabel("Layer Index")
                    plt.ylabel("Attention Entropy")
                    plt.title("Attention Entropy by Layer")
                else:
                    plt.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, "No data available", ha="center", va="center", transform=plt.gca().transAxes)

        def plot_head_type_distribution():
            """Plot overall head type distribution."""
            if len(metrics_df) > 0:
                induction_count = metrics_df["is_induction_head"].sum()
                copying_count = metrics_df["is_copying_head"].sum()
                previous_count = metrics_df["is_previous_token_head"].sum()
                other_count = len(metrics_df) - induction_count - copying_count - previous_count

                labels = ["Induction", "Copying", "Previous Token", "Other"]
                sizes = [induction_count, copying_count, previous_count, other_count]

                # Filter out zero counts
                labels_filtered = [label for label, size in zip(labels, sizes, strict=False) if size > 0]
                sizes_filtered = [size for size in sizes if size > 0]

                if sizes_filtered:
                    plt.pie(sizes_filtered, labels=labels_filtered, autopct="%1.1f%%")
                    plt.title("Head Type Distribution")
                else:
                    plt.text(
                        0.5, 0.5, "No head types detected", ha="center", va="center", transform=plt.gca().transAxes
                    )
            else:
                plt.text(0.5, 0.5, "No data available", ha="center", va="center", transform=plt.gca().transAxes)

        def plot_attention_concentration():
            """Plot attention concentration distribution."""
            if len(metrics_df) > 0 and "attention_concentration" in metrics_df.columns:
                plt.hist(metrics_df["attention_concentration"], bins=20, alpha=0.7, edgecolor="black")
                plt.axvline(
                    metrics_df["attention_concentration"].mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {metrics_df['attention_concentration'].mean():.3f}",
                )
                plt.xlabel("Attention Concentration")
                plt.ylabel("Frequency")
                plt.title("Attention Concentration Distribution")
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No concentration data", ha="center", va="center", transform=plt.gca().transAxes)

        def plot_scaling_analysis():
            """Plot attention scaling patterns."""
            if len(metrics_df) > 0 and "attention_scaling_slope" in metrics_df.columns:
                plt.scatter(metrics_df["context_size"], metrics_df["attention_entropy"], alpha=0.6)
                plt.xlabel("Context Size")
                plt.ylabel("Attention Entropy")
                plt.title("Attention Scaling Patterns")

                # Add trend line if possible
                if len(metrics_df) > 2:
                    try:
                        z = np.polyfit(metrics_df["context_size"], metrics_df["attention_entropy"], 1)
                        p = np.poly1d(z)
                        plt.plot(metrics_df["context_size"], p(metrics_df["context_size"]), "r--", alpha=0.8)
                    except:
                        pass
            else:
                plt.text(0.5, 0.5, "No scaling data", ha="center", va="center", transform=plt.gca().transAxes)

        plot_functions = [
            plot_entropy_by_layer,
            plot_head_type_distribution,
            plot_attention_concentration,
            plot_scaling_analysis,
        ]

        return self.visualizer.create_multi_panel_figure(
            plot_functions,
            layout=(2, 2),
            filename="attention_overview",
            suptitle="ICL Attention Pattern Analysis Overview",
        )

    def generate_report(self, head_type_analysis: dict[str, t.Any], visualization_paths: list[Path]) -> Path:
        """Generate comprehensive attention analysis report."""
        if not self.config.generate_reports:
            return Path()

        print("Generating attention analysis report...")

        # Prepare report sections
        sections = []

        # Summary section
        overall_stats = head_type_analysis.get("overall_statistics", {})
        sections.append(
            {
                "title": "Summary",
                "content": f"Analysis of attention patterns across {overall_stats.get('total_heads', 0)} attention heads.",
                "metrics": {
                    "Total Attention Heads": overall_stats.get("total_heads", 0),
                    "Induction Heads": f"{overall_stats.get('induction_heads', 0)} ({overall_stats.get('induction_rate', 0):.1%})",
                    "Copying Heads": f"{overall_stats.get('copying_heads', 0)} ({overall_stats.get('copying_rate', 0):.1%})",
                    "Previous Token Heads": f"{overall_stats.get('previous_token_heads', 0)} ({overall_stats.get('previous_token_rate', 0):.1%})",
                },
            }
        )

        # Layer analysis
        layer_analysis = head_type_analysis.get("layer_analysis", {})
        if layer_analysis:
            sections.append(
                {
                    "title": "Layer-wise Analysis",
                    "content": "Distribution of attention head types across model layers.",
                    "metrics": {},
                }
            )

            for layer_key, layer_stats in layer_analysis.items():
                sections[-1]["metrics"].update(
                    {
                        f"{layer_key.title()} Total Heads": layer_stats.get("total_heads", 0),
                        f"{layer_key.title()} Induction Heads": layer_stats.get("induction_heads", 0),
                        f"{layer_key.title()} Mean Entropy": f"{layer_stats.get('mean_attention_entropy', 0):.3f}",
                    }
                )

        # Configuration analysis
        config_analysis = head_type_analysis.get("config_analysis", {})
        if config_analysis:
            sections.append(
                {
                    "title": "Configuration Effects",
                    "content": "How model configuration affects attention patterns.",
                    "metrics": {},
                }
            )

            for config_key, config_stats in config_analysis.items():
                sections[-1]["metrics"].update(
                    {
                        f"{config_key} Induction Rate": f"{config_stats.get('induction_head_rate', 0):.1%}",
                        f"{config_key} Copying Rate": f"{config_stats.get('copying_head_rate', 0):.1%}",
                        f"{config_key} Mean Entropy": f"{config_stats.get('mean_attention_entropy', 0):.3f}",
                    }
                )

        # Scaling analysis
        scaling_analysis = head_type_analysis.get("scaling_analysis", {})
        if scaling_analysis:
            sections.append(
                {
                    "title": "Context Scaling Effects",
                    "content": "How attention patterns change with context size.",
                    "metrics": {
                        "Mean Scaling Slope": f"{scaling_analysis.get('mean_scaling_slope', 0):.4f}",
                        "Mean Stability Score": f"{scaling_analysis.get('mean_stability_score', 0):.3f}",
                        "Positive Scaling Rate": f"{scaling_analysis.get('positive_scaling_rate', 0):.1%}",
                        "Heads Analyzed": scaling_analysis.get("heads_analyzed", 0),
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
            "ICL Attention Pattern Analysis Report", sections, "attention_report.html"
        )

    def save_metrics(self) -> Path:
        """Save attention metrics to CSV."""
        if not self.attention_metrics:
            raise ValueError("No attention metrics to save")

        metrics_df = pd.DataFrame([m.to_dict() for m in self.attention_metrics])
        output_path = self.output_dir / "metrics" / "attention_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_path, index=False)
        print(f"Saved attention metrics: {output_path}")

        return output_path

    def run_complete_analysis(self) -> dict[str, t.Any]:
        """Run complete attention analysis pipeline."""
        print("=" * 60)
        print("EXPERIMENT 4: ATTENTION PATTERN ANALYSIS")
        print("=" * 60)

        # Load data
        self.load_and_filter_data()

        # Load attention data
        try:
            self.load_attention_data()
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Attention analysis requires attention data from the collection phase.")
            print("Please run collection with --capture-attention enabled.")
            return {
                "error": "No attention data available",
                "attention_metrics_path": None,
                "report_path": None,
                "visualization_paths": [],
                "head_type_analysis": {},
            }

        # Analyze attention patterns
        self.analyze_attention_patterns()

        # Analyze head types
        head_type_analysis = self.analyze_head_types()

        # Create outputs
        visualization_paths = []
        if self.config.create_visualizations:
            visualization_paths = self.create_visualizations()

        report_path = Path()
        if self.config.generate_reports:
            report_path = self.generate_report(head_type_analysis, visualization_paths)

        metrics_path = self.save_metrics()

        print("=" * 60)
        print("ATTENTION ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")

        return {
            "attention_metrics_path": metrics_path,
            "report_path": report_path,
            "visualization_paths": visualization_paths,
            "head_type_analysis": head_type_analysis,
            "total_heads_analyzed": len(self.attention_metrics),
        }
