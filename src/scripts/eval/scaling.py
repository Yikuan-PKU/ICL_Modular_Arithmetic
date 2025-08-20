"""Standalone script for Experiment 3: Context Scaling Analysis."""

from datetime import datetime

from ICL.eval.exp.scaling.analyzer import ScalingAnalyzer
from ICL.eval.exp.shared.exp_config import create_exp_config_from_args
from ICL.settings import create_base_parser


def create_exp3_parser():
    """Create argument parser for exp3 scaling analysis."""
    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "Experiment 3: Context Scaling Analysis"

    # Experiment-specific arguments
    exp_group = parser.add_argument_group("Experiment 3 Parameters")
    exp_group.add_argument(
        "--performance-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9],
        help="Performance thresholds for optimal context analysis",
    )
    exp_group.add_argument(
        "--scaling-laws",
        nargs="+",
        choices=["exponential", "power", "logarithmic"],
        default=["exponential", "power", "logarithmic"],
        help="Scaling laws to fit",
    )
    exp_group.add_argument("--min-models", type=int, default=5, help="Minimum models required for analysis")

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--no-complexity-analysis", action="store_true", help="Skip complexity effects analysis"
    )
    analysis_group.add_argument("--no-law-fitting", action="store_true", help="Skip scaling law fitting")

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running analysis")

    return parser


def main() -> int:
    """Main entry point for exp3 scaling analysis."""
    parser = create_exp3_parser()
    args = parser.parse_args()

    # Validate required arguments
    required_args = ["dataset_type", "mixture_type", "total_rules", "seed"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"Required arguments missing: {missing_args}")

    # Create configuration using auto-discovery
    try:
        config = create_exp_config_from_args(args)

        # Override exp3 config with command line args
        if hasattr(args, "performance_thresholds"):
            config.exp3_config["performance_thresholds"] = args.performance_thresholds
        if hasattr(args, "scaling_laws"):
            config.exp3_config["scaling_laws"] = args.scaling_laws
        if hasattr(args, "min_models"):
            config.exp3_config["min_models"] = args.min_models

        # Analysis options
        if hasattr(args, "no_complexity_analysis") and args.no_complexity_analysis:
            config.exp3_config["analyze_complexity_effects"] = False
        if hasattr(args, "no_law_fitting") and args.no_law_fitting:
            config.exp3_config["fit_scaling_laws"] = False

        exp3_output_dir = config.get_exp_output_dir("exp3")

    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        return 1

    # Validation-only mode
    if args.validate_only:
        try:
            config.validate()
            print("✓ Experiment 3 setup validation passed")
            print("\nConfiguration Summary:")
            print("-" * 40)
            print(f"Experiment: {config.dataset_config.to_base_name()}")
            print(f"Collection results: {config.collection_results_dir}")
            print(f"ICL performance data: {config.icl_performance_path}")
            print(f"Output directory: {exp3_output_dir}")
            print(f"Performance thresholds: {config.exp3_config.get('performance_thresholds', [])}")
            print(f"Scaling laws: {config.exp3_config.get('scaling_laws', [])}")
            print(f"Minimum models: {config.exp3_config.get('min_models', 5)}")
            return 0
        except Exception as e:
            print(f"✗ Validation error: {e}")
            return 1

    # Validate collection results exist
    if not config.icl_performance_path.exists():
        print(f"ERROR: Collection results not found: {config.icl_performance_path}")
        print("Please run the collection phase first.")
        return 1

    # Run scaling analysis
    try:
        start_time = datetime.now()
        print("=" * 60)
        print("STARTING EXPERIMENT 3: CONTEXT SCALING ANALYSIS")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Output directory: {exp3_output_dir}")
        print()

        # Initialize and run analyzer
        analyzer = ScalingAnalyzer(config, exp3_output_dir)
        results = analyzer.run_complete_analysis()

        # Print completion summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Duration: {duration}")
        print(f"Models analyzed: {results.get('total_models_analyzed', 'N/A')}")

        # Print key findings
        optimal_results = results.get("optimal_results", {})
        scaling_laws = results.get("scaling_laws", {})

        threshold_stats = optimal_results.get("threshold_statistics", {})
        for threshold_key, stats in threshold_stats.items():
            threshold = float(threshold_key.split("_")[1])
            achievement_rate = stats.get("achievement_rate", 0)
            print(f"{threshold:.0%} threshold achievement: {achievement_rate:.1%}")

        law_comparisons = scaling_laws.get("law_comparisons", {})
        for law in config.exp3_config.get("scaling_laws", []):
            wins = law_comparisons.get(f"{law}_wins", 0)
            print(f"{law.title()} best fits: {wins}")

        print(f"Results: {exp3_output_dir}")
        print(f"Metrics: {results.get('scaling_metrics_path', 'N/A')}")
        if results.get("report_path"):
            print(f"Report: {results.get('report_path', 'N/A')}")

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment 3 interrupted by user")
        return 130

    except Exception as e:
        print(f"\nERROR: Experiment 3 failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Check that collection results exist and are complete")
        print("2. Verify sufficient context size diversity in data")
        print("3. Check for adequate number of models per configuration")
        print("4. Use --validate-only to test configuration")

        return 1


if __name__ == "__main__":
    exit(main())
