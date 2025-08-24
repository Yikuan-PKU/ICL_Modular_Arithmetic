#!/usr/bin/env python3
"""Simplified collection script with memory management options."""

import logging

from ICL.eval.collection.collection_config import create_collection_config_from_args, create_collection_parser
from ICL.eval.collection.coordinator import CollectionCoordinator

logger = logging.getLogger(__name__)


def main():
    """Main entry point with memory management options."""
    parser = create_collection_parser()
    args = parser.parse_args()

    # Handle utility commands first (no full config needed)
    if args.list_combinations:
        return handle_list_combinations(args)

    if args.validate_only:
        return handle_validate_only(args)

    # Create full configuration for actual collection
    try:
        config = create_collection_config_from_args(args)
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1

    # Run collection pipeline
    try:
        coordinator = CollectionCoordinator(config)

        if not coordinator.validate_setup():
            print("Setup validation failed")
            return 1

        results = coordinator.run_collection_pipeline()

        # Print results with memory info
        print_collection_results(config, results)
        return 0

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        print(f"Collection failed: {e}")
        return 1


def handle_list_combinations(args):
    """Handle --list-combinations command with validation info."""
    try:
        # Create minimal config for discovery
        config = create_collection_config_from_args(args)
        coordinator = CollectionCoordinator(config)

        combinations_info = coordinator.list_available_combinations()

        print("\nAvailable combinations for:")
        print(f"  Dataset: {args.dataset_type}_{args.num_seeds}_L{args.L}_M{args.M}")
        print(f"  Model type filter: {args.model_type}")
        print(f"  Total combinations: {combinations_info['total_combinations']}")
        print("\nCombinations:")

        for combo in combinations_info["combinations"]:
            status = "✓" if combo["dataset_valid"] else "✗"
            print(f"  {status} {combo['model_variant']} / {combo['eval_type']}")

        return 0

    except Exception as e:
        print(f"Failed to list combinations: {e}")
        return 1


def handle_validate_only(args):
    """Handle --validate-only command."""
    try:
        config = create_collection_config_from_args(args)
        coordinator = CollectionCoordinator(config)

        print("\nValidating configuration:")
        print(f"  Dataset: {config.dataset_type}_{config.num_seeds}_L{config.config_L}_M{config.config_m}")
        print(f"  Model type: {config.model_type}")

        if config.batch_mode:
            print("  Mode: Batch (all combinations)")
        else:
            print(f"  Mode: Single ({config.model_variant} / {config.eval_type})")

        # Show memory settings
        print("\nMemory management:")
        print(f"  Device: {config.device}")
        print(f"  Sequence chunk size: {config.sequence_chunk_size}")
        print(f"  Model offloading: {config.offload_models}")
        print(f"  Attention sampling: {config.attention_sampling_rate}")

        if coordinator.validate_setup():
            print("✓ Configuration is valid")
            return 0
        print("✗ Configuration validation failed")
        return 1

    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


def print_collection_results(config, results):
    """Print results with memory management info."""
    print("\nCollection completed successfully!")
    print(f"Dataset: {config.dataset_type}_{config.num_seeds}_L{config.config_L}_M{config.config_m}")

    if config.batch_mode:
        print("Mode: Batch collection")
        print(f"Completed: {len(results.get('completed', []))}/{results.get('total_combinations', 0)}")

        if results.get("failed"):
            print(f"Failed: {len(results['failed'])}")
            for variant, eval_type, error in results["failed"]:
                print(f"  - {variant}/{eval_type}: {error}")
    else:
        print("Mode: Single collection")
        print(f"Model variant: {config.model_variant}")
        print(f"Evaluation type: {config.eval_type}")
        print(f"Total evaluations: {results.get('total_evaluations', 0)}")

    # Memory management info
    print("\nMemory management:")
    print(f"  Sequence chunk size: {config.sequence_chunk_size}")
    print(f"  Model offloading: {config.offload_models}")

    if config.capture_attention:
        print("  Attention streaming: enabled")
        print(f"  Attention sampling rate: {config.attention_sampling_rate}")
        if config.selective_attention != "all":
            print(f"  Selective attention: {config.selective_attention}")
    else:
        print("  Attention capture: disabled")

    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    exit(main())
