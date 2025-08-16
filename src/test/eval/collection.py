"""Example script to run Phase 1 comprehensive evaluation."""

from datetime import datetime
from pathlib import Path


def main():
    """Run comprehensive evaluation pipeline."""
    # Configuration
    checkpoint_dirs = [
        Path("checkpoints/causal_lm"),
        Path("checkpoints/mlm"),
    ]

    eval_dataset_path = data_dir / "eval/verified_transfer_evaluation_dataset.json"
    output_dir = data_dir / "raw_results"

    # Target configurations to evaluate (L, m pairs)
    target_configs = [
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (4, 4),
    ]

    # Evaluation parameters
    evaluation_params = {
        "context_sizes": [1, 2, 3, 4, 5, 6, 8],
        "diversity_levels": [8, 16, 32, 64, 128],
        "model_types": ["causal_lm", "mlm"],
        "device": "cuda",
        "batch_size": 16,
        "max_sequences_per_condition": 100,
        "capture_attention": True,
        "capture_representations": False,
        "save_intermediate": True,
        "overwrite_existing": False,
    }

    print("=" * 80)
    print("COMPREHENSIVE ICL EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Checkpoint directories: {checkpoint_dirs}")
    print(f"Evaluation dataset: {eval_dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target configurations: {target_configs}")
    print(f"Context sizes: {evaluation_params['context_sizes']}")
    print(f"Diversity levels: {evaluation_params['diversity_levels']}")
    print(f"Device: {evaluation_params['device']}")
    print()

    # Validate paths exist
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            print(f"WARNING: Checkpoint directory not found: {checkpoint_dir}")

    if not eval_dataset_path.exists():
        print(f"ERROR: Evaluation dataset not found: {eval_dataset_path}")
        return

    # Run evaluation
    try:
        start_time = datetime.now()

        results = run_comprehensive_evaluation(
            checkpoint_dirs=checkpoint_dirs,
            eval_dataset_path=eval_dataset_path,
            output_dir=output_dir,
            target_configs=target_configs,
            **evaluation_params,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Duration: {duration}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Total models: {results['total_models']}")
        print(f"Output directory: {results['output_dir']}")
        print()

        print("Next steps:")
        print("1. Run RQ1 emergence analysis: python -m analysis.rq1_emergence_analyzer")
        print("2. Run RQ2 scaling analysis: python -m analysis.rq2_scaling_analyzer")
        print("3. Run RQ3 mechanistic analysis: python -m analysis.rq3_mechanistic_analyzer")
        print("4. Run RQ4 transfer analysis: python -m analysis.rq4_transfer_analyzer")
        print("5. Run RQ5 diversity analysis: python -m analysis.rq5_diversity_analyzer")
        print("6. Run RQ6 comparative analysis: python -m analysis.rq6_comparative_analyzer")

    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        raise


def test_configuration():
    """Test evaluation configuration without running full evaluation."""
    from data_collection.evaluation_coordinator import EvaluationCoordinator, create_evaluation_config

    checkpoint_dirs = [Path("checkpoints/causal_lm")]
    eval_dataset_path = Path("eval_data/verified_transfer_evaluation_dataset.json")
    output_dir = Path("results/test_evaluation")
    target_configs = [(2, 2), (3, 3)]

    config = create_evaluation_config(
        checkpoint_dirs=checkpoint_dirs,
        eval_dataset_path=eval_dataset_path,
        output_dir=output_dir,
        target_configs=target_configs,
        max_sequences_per_condition=10,  # Small number for testing
        capture_attention=False,  # Disable for faster testing
    )

    coordinator = EvaluationCoordinator(config)

    print("Testing evaluation configuration...")
    is_valid = coordinator.validate_setup()

    if is_valid:
        print("✓ Configuration is valid")

        # Test checkpoint discovery
        checkpoint_manager = CheckpointManager(config)
        metadata = checkpoint_manager.discover_checkpoints()

        print(f"✓ Discovered {len(metadata)} checkpoints")

        if metadata:
            print("Sample checkpoint metadata:")
            for i, meta in enumerate(metadata[:3]):
                print(f"  {i + 1}. {meta.model_id} - L{meta.config_L}_m{meta.config_m}_ntrain{meta.n_train}")
    else:
        print("✗ Configuration validation failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive ICL evaluation")
    parser.add_argument("--test", action="store_true", help="Test configuration only")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max-sequences", type=int, default=100, help="Maximum sequences per condition")

    args = parser.parse_args()

    if args.test:
        test_configuration()
    else:
        main()
