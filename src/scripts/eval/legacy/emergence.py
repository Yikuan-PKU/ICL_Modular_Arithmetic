"""Standalone script for Experiment 1: ICL Emergence Analysis."""

from datetime import datetime

from ICL.eval.exp.emergence.analyzer import EmergenceAnalyzer
from ICL.eval.exp.shared.exp_config import create_exp_config_from_args
from ICL.settings import create_base_parser


def create_exp1_parser():
    """Create argument parser for exp1 emergence analysis."""
    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "Experiment 1: ICL Emergence Analysis"

    # Experiment-specific arguments
    exp_group = parser.add_argument_group("Experiment 1 Parameters")
    exp_group.add_argument(
        "--emergence-threshold", type=float, default=0.5, help="Accuracy threshold for emergence detection"
    )
    exp_group.add_argument(
        "--baseline-controls",
        nargs="+",
        default=["shuffled_context", "random_context"],
        help="Control types for baseline comparison",
    )

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running analysis")

    return parser


def main() -> int:
    """Main entry point for exp1 emergence analysis."""
    parser = create_exp1_parser()
    args = parser.parse_args()

    # Validate required arguments
    required_args = ["dataset_type", "mixture_type", "total_rules", "seed"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"Required arguments missing: {missing_args}")

    # Create configuration using auto-discovery
    try:
        config = create_exp_config_from_args(args)

        # Override exp1 config with command line args
        if hasattr(args, "emergence_threshold"):
            config.exp1_config["emergence_threshold"] = args.emergence_threshold
        if hasattr(args, "baseline_controls"):
            config.exp1_config["baseline_controls"] = args.baseline_controls

        exp1_output_dir = config.get_exp_output_dir("exp1")

    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        return 1

    # Validation-only mode
    if args.validate_only:
        try:
            config.validate()
            print("✓ Experiment 1 setup validation passed")
            print("\nConfiguration Summary:")
            print("-" * 40)
            print(f"Experiment: {config.dataset_config.to_base_name()}")
            print(f"Collection results: {config.collection_results_dir}")
            print(f"ICL performance data: {config.icl_performance_path}")
            print(f"Output directory: {exp1_output_dir}")
            print(f"Emergence threshold: {config.exp1_config.get('emergence_threshold', 0.5)}")
            print(f"Baseline controls: {config.exp1_config.get('baseline_controls', [])}")
            return 0
        except Exception as e:
            print(f"✗ Validation error: {e}")
            return 1

    # Validate collection results exist
    if not config.icl_performance_path.exists():
        print(f"ERROR: Collection results not found: {config.icl_performance_path}")
        print("Please run the collection phase first.")
        return 1

    # Run emergence analysis
    try:
        start_time = datetime.now()
        print("=" * 60)
        print("STARTING EXPERIMENT 1: ICL EMERGENCE ANALYSIS")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Output directory: {exp1_output_dir}")
        print()

        # Initialize and run analyzer
        analyzer = EmergenceAnalyzer(config, exp1_output_dir)
        results = analyzer.run_complete_analysis()

        # Print completion summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("EXPERIMENT 1 COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Duration: {duration}")
        print(f"Models analyzed: {results.get('total_models_analyzed', 'N/A')}")
        print(f"Emergence rate: {results.get('emergence_rate', 0):.1%}")
        print(f"Results: {exp1_output_dir}")
        print(f"Metrics: {results.get('emergence_metrics_path', 'N/A')}")
        if results.get("report_path"):
            print(f"Report: {results.get('report_path', 'N/A')}")

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment 1 interrupted by user")
        return 130

    except Exception as e:
        print(f"\nERROR: Experiment 1 failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Check that collection results exist and are complete")
        print("2. Verify sufficient models and context sizes in data")
        print("3. Use --validate-only to test configuration")

        return 1


if __name__ == "__main__":
    exit(main())
