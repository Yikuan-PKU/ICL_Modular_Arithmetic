"""Generate transfer evaluation dataset following training pipeline structure."""

import logging
import pickle
import typing as t
from pathlib import Path

from datasets import Dataset

from ICL.datasets.eval import TransferConfig, TransferEvaluationGenerator
from ICL.settings import (
    PATH,
    DatasetConfig,
    create_base_parser,
    get_experiment_config_name,
    load_experiment_config,
    parse_dataset_config,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_evaluation_parser():
    """Create argument parser for evaluation dataset generation."""
    parser = create_base_parser(require_model_type=False, require_eval_flag=True)
    parser.description = "Generate transfer evaluation dataset for ICL experiments"

    # Evaluation-specific arguments
    eval_group = parser.add_argument_group("Evaluation Generation")
    eval_group.add_argument(
        "--train-metadata-path", type=str, help="Path to training dataset metadata file (auto-detected if not provided)"
    )
    eval_group.add_argument(
        "--train-config-idx", type=int, default=0, help="Index of training configuration to use (default: 0)"
    )
    eval_group.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration without generating data"
    )

    return parser


def load_yaml_config(dataset_config: DatasetConfig) -> dict[str, t.Any]:
    """Load YAML configuration file for evaluation from simplified experiment structure."""
    yaml_config = load_experiment_config("eval_dataset", dataset_config)

    if not yaml_config:
        # Get simplified experiment config directory for better error message
        experiment_name = get_experiment_config_name(
            dataset_config.dataset_type, dataset_config.mixture_type, dataset_config.total_rules, dataset_config.seed
        )
        config_path = PATH.conf_dir / experiment_name / "eval_dataset.yaml"
        raise FileNotFoundError(
            f"Evaluation configuration not found at {config_path}. "
            f"Please ensure eval_dataset.yaml exists in the config directory {experiment_name}/. "
            f"Note: Config directories use simplified naming without dataset_type prefix."
        )

    return yaml_config


def auto_detect_training_metadata(dataset_config: DatasetConfig) -> Path:
    """Auto-detect training metadata path from dataset config."""
    # Create a training dataset config (non-eval version)
    training_config = DatasetConfig(
        dataset_type=dataset_config.dataset_type,
        mixture_type=dataset_config.mixture_type,
        total_rules=dataset_config.total_rules,
        seed=dataset_config.seed,
        is_eval=False,  # Always use training dataset for metadata
    )

    training_paths = training_config.get_dataset_paths()
    metadata_path = training_paths["dataset_dir"] / "metadata.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found: {metadata_path}\n"
            f"Please run training dataset generation first or provide --train-metadata-path"
        )

    return metadata_path


def create_evaluation_dataset_config(yaml_config: dict[str, t.Any], args) -> TransferConfig:
    """Create TransferConfig from YAML configuration and arguments."""
    eval_params = yaml_config["evaluation_params"]
    transfer_config = yaml_config["transfer_config"]
    icl_params = yaml_config["icl_params"]

    return TransferConfig(
        train_config_idx=args.train_config_idx,
        num_rules_per_config=eval_params["num_rules_per_config"],
        sequences_per_rule=eval_params["sequences_per_rule"],
        context_sizes=icl_params["context_sizes"],
        max_depth=transfer_config["max_depth"],
        max_multiplicity=transfer_config["max_multiplicity"],
        depth_steps=transfer_config.get("depth_steps", [1, 2]),
        synonym_steps=transfer_config.get("synonym_steps", [1, 2]),
        full_transfer_limit=transfer_config.get("full_transfer_limit", 10),
        control_types=icl_params.get("control_types", ["normal", "shuffled_context", "random_context"]),
        include_controls=eval_params.get("include_controls", True),
        save_intermediate=eval_params.get("save_intermediate", True),
        base_seed=eval_params.get("base_seed", 42),
    )


def generate_transfer_configurations(config: TransferConfig, train_config: tuple[int, int]) -> list[tuple[int, int]]:
    """Generate test configurations for transfer evaluation."""
    train_L, train_m = train_config
    test_configs = []

    # Depth transfer configurations (increase L, keep m same)
    for step in config.depth_steps:
        if train_L + step <= config.max_depth:
            test_configs.append((train_L + step, train_m))

    # Synonym transfer configurations (keep L same, increase m)
    for step in config.synonym_steps:
        if train_m + step <= config.max_multiplicity:
            test_configs.append((train_L, train_m + step))

    # Full transfer configurations (increase both L and m)
    for L_step in config.depth_steps[:2]:  # Limit full transfer
        for m_step in config.synonym_steps[:2]:
            new_L, new_m = train_L + L_step, train_m + m_step
            if new_L <= config.max_depth and new_m <= config.max_multiplicity:
                test_configs.append((new_L, new_m))

    # Remove duplicates and limit total configurations
    test_configs = list(set(test_configs))
    test_configs = test_configs[: config.full_transfer_limit]

    return test_configs


def save_evaluation_dataset(
    dataset_dict: dict[str, t.Any], metadata: dict[str, t.Any], dataset_config: DatasetConfig, args
) -> None:
    """Save evaluation dataset in HuggingFace format with metadata to eval subdirectory."""
    paths = dataset_config.get_dataset_paths()
    dataset_dir = paths["dataset_dir"]  # This now points to the eval subdirectory

    # Create output directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    if (dataset_dir / "dataset").exists() and not args.overwrite:
        if args.resume:
            logger.info(f"Evaluation dataset already exists at {dataset_dir}, skipping generation")
            return
        raise FileExistsError(
            f"Evaluation dataset already exists at {dataset_dir}. Use --overwrite to replace or --resume to skip."
        )

    # Create HuggingFace Dataset
    logger.info("Creating HuggingFace Dataset...")
    dataset = Dataset.from_dict(dataset_dict)

    # Save HuggingFace dataset
    dataset.save_to_disk(str(dataset_dir / "dataset"))
    logger.info(f"✓ Saved HuggingFace evaluation dataset to {dataset_dir / 'dataset'}")

    # Save metadata
    with (dataset_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"✓ Saved metadata to {dataset_dir / 'metadata.pkl'}")

    # Save human-readable summary
    with (dataset_dir / "dataset_summary.txt").open("w") as f:
        f.write(f"EVALUATION DATASET: {dataset_config.to_name()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Experiment Parameters:\n")
        f.write(f"  Dataset Type: {dataset_config.dataset_type}\n")
        f.write(f"  Mixture Type: {dataset_config.mixture_type}\n")
        f.write(f"  Total Rules: {dataset_config.total_rules}\n")
        f.write(f"  Seed: {dataset_config.seed}\n")
        f.write(f"  Is Eval: {dataset_config.is_eval}\n\n")

        f.write("Directory Structure:\n")
        f.write(f"  Base directory: {paths['base_dir']}\n")
        f.write(f"  Eval directory: {dataset_dir}\n")
        f.write(f"  Config directory: {paths['config_dir']}\n\n")

        f.write("Evaluation Statistics:\n")
        f.write(f"  Total sequences: {metadata['dataset_stats']['total_sequences']:,}\n")
        f.write(f"  Transfer conditions: {len(metadata['transfer_conditions'])}\n")
        f.write(f"  Context sizes: {metadata['icl_params']['context_sizes']}\n")
        f.write(f"  Control types: {len(metadata['icl_params']['control_types'])}\n\n")

        f.write("Transfer Conditions:\n")
        for condition, stats in metadata["condition_stats"].items():
            f.write(f"  {condition}: {stats['total_sequences']} sequences\n")
            if "configs" in stats:
                f.write(f"    Configurations: {stats['configs']}\n")

    logger.info(f"✓ Saved summary to {dataset_dir / 'dataset_summary.txt'}")


def main():
    """Main evaluation dataset generation function."""
    parser = create_evaluation_parser()
    args = parser.parse_args()

    # Ensure --eval flag was provided for evaluation dataset generation
    if not args.eval:
        logger.error("--eval flag is required for evaluation dataset generation")
        return 1

    # Convert to dataset config
    dataset_config = parse_dataset_config(args)
    paths = dataset_config.get_dataset_paths()

    # Auto-detect training metadata if not provided
    if args.train_metadata_path:
        train_metadata_path = Path(args.train_metadata_path)
    else:
        train_metadata_path = auto_detect_training_metadata(dataset_config)

    # Load evaluation configuration from experiment-specific YAML
    try:
        yaml_config = load_yaml_config(dataset_config)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please ensure eval_dataset.yaml exists in the experiment configuration directory.")
        return 1

    # Create evaluation configuration
    eval_config = create_evaluation_dataset_config(yaml_config, args)

    if args.validate_only:
        logger.info(f"Configuration validation successful for {dataset_config.to_name()}")
        logger.info(f"Training metadata: {train_metadata_path}")
        logger.info(f"Evaluation config: {eval_config}")
        logger.info(f"Output directory: {paths['dataset_dir']}")
        return 0

    if args.verbose:
        logger.info(f"Evaluation dataset: {dataset_config.to_name()}")
        logger.info(f"Base directory: {paths['base_dir']}")
        logger.info(f"Eval directory: {paths['dataset_dir']}")
        logger.info(f"Config directory: {paths['config_dir']}")
        logger.info(f"Training metadata: {train_metadata_path}")

    # Initialize generator
    logger.info("Initializing TransferEvaluationGenerator...")
    try:
        generator = TransferEvaluationGenerator(
            train_metadata_path=train_metadata_path, dataset_config=dataset_config, base_seed=eval_config.base_seed
        )

        # Generate evaluation dataset
        dataset_dict, metadata = generator.generate_complete_evaluation_dataset(
            config=eval_config, output_dir=paths["dataset_dir"]
        )

        # Save dataset
        save_evaluation_dataset(dataset_dict, metadata, dataset_config, args)

        logger.info(f"\n{'=' * 60}")
        logger.info("EVALUATION DATASET GENERATION COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(f"Total sequences: {len(dataset_dict['input_ids']):,}")
        logger.info(f"Output directory: {paths['dataset_dir']}")
        logger.info(f"Base directory: {paths['base_dir']}")
        logger.info(f"{'=' * 60}")

        return 0

    except Exception as e:
        logger.error(f"Error during evaluation dataset generation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
