"""Updated collection script using experiment-based auto-discovery."""

from datetime import datetime

from ICL.settings import create_base_parser, parse_dataset_config


def create_collection_parser():
    """Create argument parser for collection phase using shared arguments."""
    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "ICL Evaluation Collection Phase - Auto-discovery Mode"

    # Collection-specific arguments
    collection_group = parser.add_argument_group("Collection Parameters")
    collection_group.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    collection_group.add_argument("--batch-size", type=int, help="Batch size for evaluation (overrides config)")
    collection_group.add_argument("--max-sequences", type=int, help="Max sequences per condition (overrides config)")
    collection_group.add_argument("--no-attention", action="store_true", help="Disable attention capture")
    collection_group.add_argument("--no-intermediate", action="store_true", help="Disable intermediate saving")

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running")
    util_group.add_argument("--list-checkpoints", action="store_true", help="List discovered checkpoints")

    return parser


def create_collection_config_from_args(args):
    """Create CollectionConfig from parsed arguments using auto-discovery."""
    from ICL.eval.collection.collection_config import CollectionConfig

    dataset_config = parse_dataset_config(args)
    # Force eval=True for collection
    dataset_config = dataset_config.__class__(
        dataset_type=dataset_config.dataset_type,
        mixture_type=dataset_config.mixture_type,
        total_rules=dataset_config.total_rules,
        seed=dataset_config.seed,
        is_eval=True,
    )

    config = CollectionConfig(
        dataset_config=dataset_config,
        device=getattr(args, "device", "cuda"),
        overwrite=getattr(args, "overwrite", False),
        resume=getattr(args, "resume", False),
    )

    # Apply command line overrides
    if hasattr(args, "batch_size") and args.batch_size:
        config.batch_size = args.batch_size

    if hasattr(args, "max_sequences") and args.max_sequences:
        config.max_sequences_per_condition = args.max_sequences

    if hasattr(args, "no_attention") and args.no_attention:
        config.capture_attention = False

    if hasattr(args, "no_intermediate") and args.no_intermediate:
        config.save_intermediate = False

    return config


def main() -> int:
    """Main entry point for collection phase with auto-discovery."""
    parser = create_collection_parser()
    args = parser.parse_args()

    # Validate required arguments
    required_args = ["dataset_type", "mixture_type", "total_rules", "seed"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(f"Required arguments missing: {missing_args}")

    # Create configuration using auto-discovery
    try:
        config = create_collection_config_from_args(args)
    except Exception as e:
        print(f"ERROR: Failed to create configuration with auto-discovery: {e}")
        return 1

    # Handle utility commands
    if args.list_checkpoints:
        from ICL.eval.collection.coordinator import CollectionCoordinator

        coordinator = CollectionCoordinator(config)

        checkpoint_info = coordinator.list_discovered_checkpoints()

        print("\nDiscovered Checkpoints:")
        print("=" * 50)
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Total checkpoint directories: {checkpoint_info['total_directories']}")

        for i, dir_info in enumerate(checkpoint_info["checkpoints"]):
            print(f"\n{i + 1}. {dir_info['directory']}")
            print(f"   Exists: {dir_info['exists']}")

            if dir_info["exists"] and dir_info["subdirs"]:
                print(f"   Subdirectories: {len(dir_info['subdirs'])}")
                for subdir in dir_info["subdirs"][:5]:  # Show first 5
                    status_icons = []
                    if subdir["has_config"]:
                        status_icons.append("📄")
                    if subdir["has_metadata"]:
                        status_icons.append("📋")
                    if subdir["has_model"]:
                        status_icons.append("🤖")

                    status = " ".join(status_icons) if status_icons else "❌"
                    print(f"     - {subdir['name']} {status}")

                if len(dir_info["subdirs"]) > 5:
                    print(f"     ... and {len(dir_info['subdirs']) - 5} more")

        return 0

    # Initialize coordinator
    from ICL.eval.collection.coordinator import CollectionCoordinator

    coordinator = CollectionCoordinator(config)

    # Validation-only mode
    if args.validate_only:
        try:
            if coordinator.validate_setup():
                print("✓ Collection setup validation passed")
                print("\nAuto-Discovery Summary:")
                print("-" * 40)
                print(f"Experiment: {config.dataset_config.to_base_name()}")
                print(f"Evaluation dataset: {config.eval_dataset_path}")
                print(f"Collection config: {config.config_file_path}")
                print(f"Output directory: {config.output_dir}")
                print(f"Checkpoint directories: {len(config.checkpoint_dirs)}")
                for i, checkpoint_dir in enumerate(config.checkpoint_dirs):
                    print(f"  {i + 1}. {checkpoint_dir}")
                print(f"Target configurations: {len(config.target_configs)}")
                print(f"Context sizes: {config.context_sizes}")
                return 0
            print("✗ Collection setup validation failed")
            return 1
        except Exception as e:
            print(f"✗ Validation error: {e}")
            return 1

    # Run collection pipeline
    try:
        start_time = datetime.now()
        print("=" * 60)
        print("STARTING ICL EVALUATION COLLECTION")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Output directory: {config.output_dir}")
        print(f"Device: {config.device}")
        print()

        # Run collection
        results = coordinator.run_collection_pipeline()

        # Generate summary
        summary_path = coordinator.generate_collection_summary(results)

        # Print completion summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Experiment: {config.dataset_config.to_base_name()}")
        print(f"Duration: {duration}")
        print(f"Total evaluations: {results.get('total_evaluations', 'N/A'):,}")
        print(f"Total models: {results.get('total_models', 'N/A')}")
        print(f"Success rate: {results.get('success_rate', 'N/A'):.1f}%")
        print(f"Results: {config.output_dir}")
        print(f"Summary: {summary_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
        print("Progress has been saved. Use --resume to continue.")
        return 130

    except Exception as e:
        print(f"\nERROR: Collection failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Check that evaluation dataset exists")
        print("2. Verify model checkpoints exist")
        print("3. Ensure collection.yaml exists in experiment config")
        print("4. Use --validate-only to test configuration")

        return 1


if __name__ == "__main__":
    exit(main())
