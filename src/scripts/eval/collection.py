"""Main collection script with training script style and minimal config parsing."""

from datetime import datetime

from ICL.eval.collection.collection_config import (
    CollectionConfig,
    create_collection_config_from_args,
    create_collection_parser,
)
from ICL.eval.collection.coordinator import CollectionCoordinator


def main() -> int:
    """Main entry point for collection phase with training script style."""
    parser = create_collection_parser()
    args = parser.parse_args()

    # Validate required arguments (training script style)
    required_args = ["dataset_type", "num_seeds", "seed", "L", "M", "model_type"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"Required arguments missing: {missing_args}")

    # Handle utility commands FIRST (before config creation)
    if getattr(args, "list_combinations", False):
        # Create config with explicit L,M,model_type for discovery
        config = CollectionConfig.from_args(
            dataset_type=args.dataset_type,
            num_seeds=args.num_seeds,
            seed=args.seed,
            config_L=args.L,
            config_m=args.M,
            model_type=args.model_type,
            batch_mode=True,  # Force batch mode for discovery
            device=getattr(args, "device", "cuda"),
        )

        coordinator = CollectionCoordinator(config)
        combination_info = coordinator.list_available_combinations()

        print("\nAvailable Combinations:")
        print("=" * 60)
        print(f"Dataset: {combination_info['shared_identifier']}")
        print(f"Model Type: {args.model_type}")
        print(f"Total combinations: {combination_info['total_combinations']}")

        return 0

    # Create configuration for normal execution with explicit L,M
    try:
        config = create_collection_config_from_args(args)
    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        return 1

    # Initialize coordinator
    coordinator = CollectionCoordinator(config)

    # Validation-only mode
    if getattr(args, "validate_only", False):
        try:
            if coordinator.validate_setup():
                print("✅ Collection setup validation passed")

                if config.batch_mode:
                    combinations = config.discover_available_combinations()
                    print("\nBatch Mode Summary:")
                    print("-" * 40)
                    print(f"Dataset: {config.get_shared_identifier()}")
                    print(f"Model Type: {config.model_type} (explicit)")
                    print(f"Available combinations: {len(combinations)}")
                    print(f"Using explicit L={config.config_L}, M={config.config_m}")
                    print("Model variants from unified training.yaml:")
                    for variant in config.available_model_variants:
                        print(f"  {variant}")

                    for variant, eval_type in combinations[:3]:  # Show first 3
                        print(f"  - {variant} / {eval_type}")
                    if len(combinations) > 3:
                        print(f"  ... and {len(combinations) - 3} more")
                else:
                    print("\nSingle Mode Summary:")
                    print("-" * 40)
                    print(f"Dataset: {config.get_shared_identifier()}")
                    print(f"Model Type: {config.model_type} (explicit)")
                    print(f"Using explicit L={config.config_L}, M={config.config_m}")
                    print(f"Model variant: {config.model_variant}")
                    print(f"Evaluation type: {config.eval_type}")
                    print(f"Evaluation dataset: {config.eval_dataset_path}")
                    print(f"Model directory: {config.model_base_dir}")
                    print(f"Output directory: {config.output_dir}")
                    print(f"Capture attention: {config.capture_attention}")

                return 0
            print("❌ Collection setup validation failed")
            return 1
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return 1

    # Run collection pipeline
    try:
        start_time = datetime.now()

        if config.batch_mode:
            print("=" * 80)
            print("STARTING ICL BATCH COLLECTION")
            print("=" * 80)
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Dataset: {config.get_shared_identifier()}")
            print(f"Model Type: {config.model_type} (explicit)")
            print(f"Using explicit L={config.config_L}, M={config.config_m}")

            combinations = config.discover_available_combinations()
            print(f"Total combinations to process: {len(combinations)}")

            for variant, eval_type in combinations:
                print(f"  - {variant} / {eval_type}")
            print()
        else:
            print("=" * 80)
            print("STARTING ICL SINGLE COLLECTION")
            print("=" * 80)
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Dataset: {config.get_shared_identifier()}")
            print(f"Model Type: {config.model_type} (explicit)")
            print(f"Using explicit L={config.config_L}, M={config.config_m}")
            print(f"Combination: {config.model_variant} / {config.eval_type}")

            print(f"Output directory: {config.output_dir}")
            print(f"Device: {config.device}")
            print()

        # Run collection
        results = coordinator.run_collection_pipeline()

        # Print completion summary
        end_time = datetime.now()
        duration = end_time - start_time

        if config.batch_mode:
            print("\n" + "=" * 80)
            print("BATCH COLLECTION COMPLETED")
            print("=" * 80)
            print(f"Dataset: {config.get_shared_identifier()}")
            print(f"Duration: {duration}")
            print(f"Total combinations: {results.get('total_combinations', 'N/A')}")
            print(f"Successful: {results.get('success_count', 'N/A')}")
            print(f"Failed: {results.get('failure_count', 'N/A')}")
            print(f"Results: {config.output_dir}")
            print(f"Summary: {config.output_dir}/batch_collection_summary.json")

            # Show failed combinations if any
            failed_combinations = results.get("failed_combinations", [])
            if failed_combinations:
                print("\nFailed combinations:")
                for variant, eval_type, error in failed_combinations:
                    print(f"  ❌ {variant} / {eval_type}: {error}")
        else:
            print("\n" + "=" * 80)
            print("SINGLE COLLECTION COMPLETED")
            print("=" * 80)
            print(f"Combination: {config.model_variant} / {config.eval_type}")
            print(f"Duration: {duration}")
            print(f"Total evaluations: {results.get('total_evaluations', 'N/A'):,}")
            print(f"Total models: {results.get('total_models', 'N/A')}")
            print(f"Results: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
        print("Progress has been saved. Use --resume to continue.")
        return 130


if __name__ == "__main__":
    exit(main())
