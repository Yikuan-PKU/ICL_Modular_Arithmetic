"""Standalone script for Experiment 2: Transfer Learning Analysis."""

from datetime import datetime

from ICL.eval.exp.shared.exp_config import create_exp_config_from_args
from ICL.eval.exp.transfer.analyzer import TransferAnalyzer
from ICL.settings import create_base_parser


def create_exp2_parser():
    """Create argument parser for exp2 transfer analysis."""
    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "Experiment 2: Transfer Learning Analysis"

    # Experiment-specific arguments
    exp_group = parser.add_argument_group("Experiment 2 Parameters")
    exp_group.add_argument(
        "--success-threshold", type=float, default=0.5, help="Accuracy threshold for transfer success"
    )
    exp_group.add_argument(
        "--transfer-types",
        nargs="+",
        choices=["depth", "synonym", "full"],
        default=["depth", "synonym", "full"],
        help="Transfer types to analyze",
    )
    exp_group.add_argument("--degradation-threshold", type=float, default=0.2, help="Threshold for severe degradation")

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running analysis")

    return parser


def main() -> int:
    """Main entry point for exp2 transfer analysis."""
    parser = create_exp2_parser()
    args = parser.parse_args()

    # Validate required arguments
    required_args = ["dataset_type", "mixture_type", "total_rules", "seed"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"Required arguments missing: {missing_args}")

    # Create configuration using auto-discovery
    try:
        config = create_exp_config_from_args(args)

        # Override exp2 config with command line args
        if hasattr(args, "success_threshold"):
            config.exp2_config["success_threshold"] = args.success_threshold
        if hasattr(args, "transfer_types"):
            config.exp2_config["transfer_types"] = args.transfer_types
        if hasattr(args, "degradation_threshold"):
            config.exp2_config["degradation_threshold"] = args.degradation_threshold

        exp2_output_dir = config.get_exp_output_dir("exp2")

    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        return 1

    # Validation-only mode
    if args.validate_only:
        try:
            config.validate()
            print("✓ Experiment 2 setup validation passed")
            print("\nConfiguration Summary:")
            print("-" * 40)
            print(f"Experiment: {config.dataset_config.to_base_name()}")
            print(f"Collection results: {config.collection_results_dir}")
            print(f"ICL performance data: {config.icl_performance_path}")
            print(f"Output directory: {exp2_output_dir}")
            print(f"Success threshold: {config.exp2_config.get('success_threshold', 0.5)}")
            print(f"Transfer types: {config.exp2_config.get('transfer_types', [])}")
            return 0
        except Exception as e:
            print(f"✗ Validation error: {e}")
            return 1

    # Validate collection results exist
    if not config.icl_performance_path.exists():
        print(f"ERROR: Collection results not found: {config.icl_performance_path}")
        print("Please run the collection phase first.")
        return 1

    # Run transfer analysis
    try:
        start_time = datetime.now()
        print("=" * 60)
        print("STARTING EXPERIMENT 2: TRANSFER LEARNING ANALYSIS")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Output directory: {exp2_output_dir}")
        print()

        # Initialize and run analyzer
        analyzer = TransferAnalyzer(config, exp2_output_dir)
        results = analyzer.run_complete_analysis()

        # Print completion summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Duration: {duration}")
        print(f"Transfer pairs analyzed: {results.get('total_transfer_pairs_analyzed', 'N/A')}")

        # Print key findings
        patterns = results.get("patterns", {})
        success_rates = patterns.get("success_rates", {})
        degradation = patterns.get("degradation_patterns", {})

        print(f"Overall success rate: {success_rates.get('overall_success_rate', 0):.1%}")
        print(f"Mean degradation: {degradation.get('overall_mean_degradation', 0):.3f}")
        print(f"Improvement rate: {degradation.get('improvement_rate', 0):.1%}")

        print(f"Results: {exp2_output_dir}")
        print(f"Metrics: {results.get('transfer_metrics_path', 'N/A')}")
        if results.get("report_path"):
            print(f"Report: {results.get('report_path', 'N/A')}")

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment 2 interrupted by user")
        return 130

    except Exception as e:
        print(f"\nERROR: Experiment 2 failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Check that collection results exist and are complete")
        print("2. Verify transfer conditions are present in data")
        print("3. Check for sufficient within-config and transfer pairs")
        print("4. Use --validate-only to test configuration")

        return 1


if __name__ == "__main__":
    exit(main())
