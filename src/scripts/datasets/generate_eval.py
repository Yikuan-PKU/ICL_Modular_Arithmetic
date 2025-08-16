import argparse
from pathlib import Path

from ICL.datasets.eval import TransferConfig, TransferEvaluationGenerator


def generate_evaluation_dataset():
    """Generate complete transfer evaluation dataset and save to target directory."""

    def parse_args() -> argparse.Namespace:
        """Parse command line arguments for evaluation dataset generation."""
        parser = argparse.ArgumentParser(description="Generate transfer evaluation dataset")
        parser.add_argument(
            "--train_metadata_path",
            type=str,
            default="/Users/jliu/workspace/ICL/datasets/train/raw/metadata.pkl",
            help="Path to training dataset metadata file",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="/Users/jliu/workspace/ICL/datasets/eval",
            help="Output directory for evaluation dataset",
        )
        parser.add_argument(
            "--train_config_idx", type=int, default=0, help="Index of training configuration to use (default: 0)"
        )
        parser.add_argument(
            "--num_rules_per_config", type=int, default=32, help="Number of rule sets per configuration"
        )
        parser.add_argument("--sequences_per_rule", type=int, default=200, help="Number of sequences per rule set")
        parser.add_argument("--max_depth", type=int, default=5, help="Maximum hierarchy depth for transfer testing")
        parser.add_argument("--max_multiplicity", type=int, default=6, help="Maximum multiplicity for transfer testing")
        parser.add_argument(
            "--context_sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Context sizes for ICL evaluation"
        )
        parser.add_argument(
            "--include_controls", action="store_true", default=True, help="Include control sequences for verification"
        )
        parser.add_argument(
            "--save_intermediate", action="store_true", default=True, help="Save intermediate results during generation"
        )
        parser.add_argument("--base_seed", type=int, default=42, help="Base seed for evaluation generation")

        return parser.parse_args()

    # Parse arguments
    args = parse_args()

    # Convert paths
    train_metadata_path = Path(args.train_metadata_path)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("TRANSFER EVALUATION DATASET GENERATION")
    print("=" * 80)
    print(f"Training metadata: {train_metadata_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base seed: {args.base_seed}")
    print()

    # Validate training metadata exists
    if not train_metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found: {train_metadata_path}\nPlease run the training dataset generation first."
        )

    # Initialize generator
    print("Initializing TransferEvaluationGenerator...")
    generator = TransferEvaluationGenerator(train_metadata_path=train_metadata_path, base_seed=args.base_seed)

    # Get training configurations
    available_configs = generator.train_metadata.config_list
    print(f"Available training configurations: {available_configs}")

    if args.train_config_idx >= len(available_configs):
        raise ValueError(
            f"train_config_idx {args.train_config_idx} out of range. Available indices: 0-{len(available_configs) - 1}"
        )

    train_config = available_configs[args.train_config_idx]
    print(f"Using training configuration: {train_config} (index {args.train_config_idx})")

    # Generate test configurations
    train_L, train_m = train_config
    test_configs = []

    # Depth transfer configurations (increase L, keep m same)
    for L in range(train_L + 1, args.max_depth + 1):
        test_configs.append((L, train_m))

    # Synonym transfer configurations (keep L same, increase m)
    for m in range(train_m + 1, args.max_multiplicity + 1):
        test_configs.append((train_L, m))

    # Full transfer configurations (increase both L and m)
    for L in range(train_L + 1, min(train_L + 3, args.max_depth + 1)):  # Limit full transfer
        for m in range(train_m + 1, min(train_m + 3, args.max_multiplicity + 1)):
            test_configs.append((L, m))

    # Remove duplicates and limit total configurations
    test_configs = list(set(test_configs))
    test_configs = test_configs[:10]  # Limit to prevent excessive generation time

    print(f"Test configurations: {test_configs}")
    print()

    # Create transfer configuration
    transfer_config = TransferConfig(
        train_config=train_config,
        test_configs=test_configs,
        num_rules_per_config=args.num_rules_per_config,
        sequences_per_rule=args.sequences_per_rule,
        context_sizes=args.context_sizes,
    )

    # Generate the complete evaluation dataset
    try:
        dataset = generator.generate_complete_evaluation_dataset(
            config=transfer_config,
            output_dir=output_dir,
            include_controls=args.include_controls,
            save_intermediate=args.save_intermediate,
        )

        print("\n" + "=" * 80)
        print("EVALUATION DATASET GENERATION SUCCESSFUL")
        print("=" * 80)

        # Print final statistics
        total_models = 0
        for condition, data in dataset["conditions"].items():
            if isinstance(data, list):
                models_count = len(data)
            elif isinstance(data, dict):
                models_count = sum(len(models) for models in data.values())
            else:
                models_count = 0

            total_models += models_count
            print(f"{condition}: {models_count} models")

        print(f"Total models generated: {total_models}")
        print(f"Dataset saved to: {output_dir}")

        return dataset

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Choose generation mode

    dataset = generate_evaluation_dataset()

    if dataset:
        print("\n✅ Evaluation dataset generation complete!")
    else:
        print("\n❌ Evaluation dataset generation failed!")
