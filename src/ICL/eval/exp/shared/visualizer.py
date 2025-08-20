"""Shared visualization utilities for experiment analyses."""

import typing as t
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class AnalysisVisualizer:
    """Shared visualization utilities for all experiment analyses."""

    def __init__(self, output_dir: Path, style: str = "whitegrid"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style(style)
        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "legend.fontsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    def create_performance_heatmap(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        title: str,
        filename: str,
        figsize: tuple[int, int] = (10, 8),
    ) -> Path:
        """Create a performance heatmap."""
        plt.figure(figsize=figsize)

        # Pivot data for heatmap
        heatmap_data = data.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")

        # Create heatmap
        sns.heatmap(
            heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": value_col.replace("_", " ").title()}
        )

        plt.title(title)
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.tight_layout()

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_html_report(
        self, title: str, sections: list[dict[str, t.Any]], output_filename: str = "report.html"
    ) -> Path:
        """Generate HTML report with embedded visualizations."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 40px; 
                    line-height: 1.6;
                }}
                .header {{ 
                    background-color: #f5f5f5; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 5px;
                }}
                .section {{ 
                    margin: 30px 0; 
                }}
                .metric {{ 
                    background-color: #f9f9f9; 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-left: 4px solid #007acc;
                }}
                .visualization {{ 
                    text-align: center; 
                    margin: 20px 0; 
                }}
                .visualization img {{ 
                    max-width: 100%; 
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
                h3 {{ color: #666; }}
                .timestamp {{ 
                    color: #888; 
                    font-style: italic; 
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="header">
                <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """

        for section in sections:
            html_content += f"""
            <div class="section">
                <h2>{section["title"]}</h2>
            """

            if "content" in section:
                html_content += f"<p>{section['content']}</p>"

            if "metrics" in section:
                for metric_name, metric_value in section["metrics"].items():
                    html_content += f"""
                    <div class="metric">
                        <strong>{metric_name.replace("_", " ").title()}:</strong> {metric_value}
                    </div>
                    """

            if "visualization" in section:
                viz_path = section["visualization"]
                if isinstance(viz_path, Path):
                    viz_filename = viz_path.name
                else:
                    viz_filename = viz_path

                html_content += f"""
                <div class="visualization">
                    <h3>{section.get("viz_title", "Visualization")}</h3>
                    <img src="visualizations/{viz_filename}" alt="{section.get("viz_title", "Visualization")}">
                </div>
                """

            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        report_path = self.output_dir / "reports" / output_filename
        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_scaling_plot(
        self,
        data: pd.DataFrame,
        x_col: str = "context_size",
        y_col: str = "accuracy",
        group_col: str | None = None,
        title: str = "Context Size Scaling",
        filename: str = "scaling_plot",
        figsize: tuple[int, int] = (12, 8),
    ) -> Path:
        """Create a context size scaling plot."""
        plt.figure(figsize=figsize)

        if group_col:
            # Plot with grouping
            for group_value in sorted(data[group_col].unique()):
                group_data = data[data[group_col] == group_value]
                scaling_data = group_data.groupby(x_col)[y_col].mean()

                plt.plot(
                    scaling_data.index,
                    scaling_data.values,
                    marker="o",
                    label=f"{group_col}={group_value}",
                    linewidth=2,
                    markersize=6,
                )
            plt.legend()
        else:
            # Single scaling curve
            scaling_data = data.groupby(x_col)[y_col].mean()
            scaling_std = data.groupby(x_col)[y_col].std()

            plt.errorbar(
                scaling_data.index,
                scaling_data.values,
                yerr=scaling_std.values,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=5,
            )

        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_distribution_plot(
        self,
        data: pd.DataFrame,
        column: str,
        title: str,
        filename: str,
        bins: int = 30,
        figsize: tuple[int, int] = (10, 6),
    ) -> Path:
        """Create a distribution plot."""
        plt.figure(figsize=figsize)

        # Remove infinite values for plotting
        clean_data = data[data[column] != float("inf")][column]

        if len(clean_data) == 0:
            plt.text(
                0.5,
                0.5,
                "No finite data to plot",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )
        else:
            plt.hist(clean_data, bins=bins, alpha=0.7, edgecolor="black", density=True)
            plt.axvline(clean_data.mean(), color="red", linestyle="--", label=f"Mean: {clean_data.mean():.3f}")
            plt.axvline(clean_data.median(), color="orange", linestyle="--", label=f"Median: {clean_data.median():.3f}")
            plt.legend()

        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Density")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_correlation_plot(
        self, data: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str, figsize: tuple[int, int] = (10, 6)
    ) -> Path:
        """Create a correlation scatter plot."""
        plt.figure(figsize=figsize)

        # Filter finite values
        clean_data = data[
            (data[x_col] != float("inf")) & (data[y_col] != float("inf")) & data[x_col].notna() & data[y_col].notna()
        ]

        if len(clean_data) == 0:
            plt.text(
                0.5,
                0.5,
                "No valid data for correlation",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )
        else:
            plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6)

            # Add correlation coefficient
            correlation = clean_data[x_col].corr(clean_data[y_col])
            plt.text(
                0.05,
                0.95,
                f"r = {correlation:.3f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Add trend line
            if not clean_data[x_col].isna().all() and not clean_data[y_col].isna().all():
                z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p = np.poly1d(z)
                plt.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8)

        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_comparison_boxplot(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_col: str,
        title: str,
        filename: str,
        figsize: tuple[int, int] = (12, 6),
    ) -> Path:
        """Create a comparison boxplot."""
        plt.figure(figsize=figsize)

        # Filter finite values
        clean_data = data[data[value_col] != float("inf")]

        if len(clean_data) == 0:
            plt.text(
                0.5,
                0.5,
                "No finite data to plot",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )
        else:
            sns.boxplot(data=clean_data, x=group_col, y=value_col)

            # Add sample sizes
            group_counts = clean_data.groupby(group_col).size()
            ax = plt.gca()
            for i, (group, count) in enumerate(group_counts.items()):
                ax.text(i, ax.get_ylim()[0], f"n={count}", horizontalalignment="center", fontsize=10)

        plt.xlabel(group_col.replace("_", " ").title())
        plt.ylabel(value_col.replace("_", " ").title())
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = self.viz_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_multi_panel_figure(
        self,
        plot_functions: list[t.Callable],
        layout: tuple[int, int],
        filename: str,
        figsize: tuple[int, int] = (16, 12),
        suptitle: str | None = None,
    ) -> Path:
        """Create a multi-panel figure with custom plot functions."""
        fig, axes = plt.subplots(*layout, figsize=figsize)

        if layout[0] * layout[1] == 1:
            axes = [axes]
        elif len(axes.shape) == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Call each plot function with its corresponding axis
        for i, plot_func in enumerate(plot_functions):
            if i < len(axes):
                plt.sca(axes[i])
                plot_func()

        # Hide unused subplots
        for j in range(len(plot_functions), len(axes)):
            axes[j].set_visible(False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=16, y=0.98)

        plt.tight_layout()
