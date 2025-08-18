"""Phase 1 comprehensive ICL evaluation script."""

import argparse
from datetime import datetime
from pathlib import Path

from ICL.eval.collection.coordinator import run_comprehensive_evaluation


def main() -> None:
    """Run comprehensive evaluation pipeline."""
    # Configuration
    # root_dir = Path("/Users/jliu/workspace/ICL")
    root_dir = Path("/scratch2/jliu/ICL")
    checkpoint_dirs = [root_dir / "models" / "rhm_clm_training", root_dir / "models" / "rhm_mlm_training"]
    eval_dataset_path = root_dir / "datasets" / "eval" / "verified_transfer_evaluation_dataset.json"
    output_dir = root_dir / "results" / "raw"

    # Target configurations (L, m pairs)
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
        "save_intermediate": True,
    }

    print("=" * 60)
    print("COMPREHENSIVE ICL EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Checkpoint directories: {len(checkpoint_dirs)} directories")
    print(f"Evaluation dataset: {eval_dataset_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Target configurations: {len(target_configs)} configs")
    print(f"Context sizes: {evaluation_params['context_sizes']}")
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
        print(f"Starting evaluation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        results = run_comprehensive_evaluation(
            checkpoint_dirs=checkpoint_dirs,
            eval_dataset_path=eval_dataset_path,
            output_dir=output_dir,
            target_configs=target_configs,
            **evaluation_params,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Total evaluations: {results.get('total_evaluations', 'N/A')}")
        print(f"Total models: {results.get('total_models', 'N/A')}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        raise


def validate_setup() -> None:
    """Validate configuration without running full evaluation."""
    root_dir = Path("/Users/jliu/workspace/ICL")
    checkpoint_dirs = [root_dir / "models" / "rhm_clm_training"]
    eval_dataset_path = root_dir / "datasets" / "eval" / "verified_transfer_evaluation_dataset.json"
    output_dir = Path("results/test_evaluation")

    print("Testing evaluation configuration...")

    # Check paths
    all_valid = True
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            print(f"✓ Found checkpoint directory: {checkpoint_dir}")
        else:
            print(f"✗ Missing checkpoint directory: {checkpoint_dir}")
            all_valid = False

    if eval_dataset_path.exists():
        print(f"✓ Found evaluation dataset: {eval_dataset_path}")
    else:
        print(f"✗ Missing evaluation dataset: {eval_dataset_path}")
        all_valid = False

    if all_valid:
        print("✓ Basic configuration validation passed")
    else:
        print("✗ Configuration validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive ICL evaluation")
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    if args.validate:
        validate_setup()
    else:
        main()
