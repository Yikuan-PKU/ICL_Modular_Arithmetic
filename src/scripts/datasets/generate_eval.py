#!/usr/bin/env python3
"""Generate ICL evaluation datasets for all four evaluation types."""

import logging
import typing as t

from ICL.datasets.evaluation.eval_builder import ICLEvaluationBuilder
from ICL.datasets.evaluation.eval_config import create_default_eval_config, load_icl_eval_config_from_yaml
from ICL.datasets.evaluation.eval_file_manager import (
    EvalDirectoryManager,
    check_evaluation_prerequisites,
    load_generation_params_from_config,
)
from ICL.settings import DatasetConfig, create_base_parser, parse_dataset_config, validate_args

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_icl_eval_parser():
    """Create argument parser for ICL evaluation generation."""
    parser = create_base_parser(require_eval_flag=False)
    parser.description = "Generate ICL evaluation datasets for four evaluation types"

    # Evaluation-specific arguments
    eval_group = parser.add_argument_group("ICL Evaluation Generation")
    eval_group.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration and check prerequisites"
    )
    eval_group.add_argument("--config-path", type=str, help="Path to YAML config file (auto-detected if not provided)")
    eval_group.add_argument(
        "--enable-types",
        nargs="+",
        choices=["memorization", "id_generalization", "ood_same_rule", "ood_transfer"],
        help="Evaluation types to generate (default: all enabled in config)",
    )
    eval_group.add_argument(
        "--skip-existing", action="store_true", help="Skip generation if evaluation datasets already exist"
    )

    return parser


def load_eval_config(dataset_config: DatasetConfig) -> t.Any:
    """Load ICL evaluation configuration for the specified L,M configuration."""
    import yaml

    paths = dataset_config.get_config_paths(dataset_config.L, dataset_config.m)
    config_path = paths["config_dir"] / "generate_eval.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        yaml_config = yaml.safe_load(f)

    # Check if eval_config exists in YAML
    if "eval_config" in yaml_config:
        return load_icl_eval_config_from_yaml(yaml_config)
    logger.warning(f"No eval_config found in {config_path}, using default configuration")
    return create_default_eval_config()


def generate_evaluation_for_configuration(
    dataset_config: DatasetConfig,
    enable_types: list[str] | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> bool:
    """Generate ICL evaluation datasets for the specified L,M configuration.

    Returns:
        True if successful, False if skipped or failed

    """
    L, m = dataset_config.L, dataset_config.m
    logger.info(f"Processing L={L}, m={m}")

    # Get paths for this configuration
    paths = dataset_config.get_config_paths(L, m)
    base_dir = paths["dataset_dir"].parent  # Remove /raw to get base

    # Initialize directory manager
    dir_manager = EvalDirectoryManager(base_dir)

    # Check prerequisites
    prereq_results = check_evaluation_prerequisites(base_dir)

    if not prereq_results["ready_for_evaluation"]:
        logger.error(f"  Prerequisites not met for L={L}, m={m}")
        logger.error(f"  Missing: {[k for k, v in prereq_results.items() if not v and k.endswith('_exists')]}")
        return False

    # Check if evaluation already exists
    existing_eval = dir_manager.check_eval_exists()
    if skip_existing and any(existing_eval.values()):
        logger.info(f"  Evaluation already exists for L={L}, m={m}, skipping")
        return True

    # Load configuration
    eval_config = load_eval_config(dataset_config)

    # Override enabled types if specified
    if enable_types:
        # Temporarily modify config
        for eval_type in ["memorization", "id_generalization", "ood_same_rule", "ood_transfer"]:
            attr = getattr(eval_config, eval_type)
            attr.enable = eval_type in enable_types

    enabled_types = eval_config.get_enabled_types()
    logger.info(f"  Enabled types: {enabled_types}")

    if not enabled_types:
        logger.warning(f"  No evaluation types enabled for L={L}, m={m}")
        return False

    # Discover training seeds
    train_seeds = dir_manager.discover_train_seeds()
    logger.info(f"  Training seeds: {train_seeds}")

    # Load generation parameters
    generation_params = load_generation_params_from_config(paths["config_dir"])
    logger.info(f"  Generation params: {generation_params}")

    # Initialize evaluation builder
    builder = ICLEvaluationBuilder(eval_config, train_seeds)

    # Create evaluation directories
    dir_manager.create_eval_directories(enabled_types, overwrite=overwrite)

    # Generate evaluation dataset
    logger.info("  Generating evaluation datasets...")
    evaluation_data = builder.generate_complete_evaluation_dataset(
        source_config=(L, m),
        train_dir=dir_manager.train_dir,
        validation_dir=dir_manager.validation_dir,
        generation_params=generation_params,
        output_dir=dir_manager.eval_dir if eval_config.save_intermediate else None,
    )

    # Save evaluation datasets
    logger.info("  Saving evaluation datasets...")
    dir_manager.save_evaluation_datasets(evaluation_data, eval_config)
    dir_manager.create_summary_files(evaluation_data)

    # Log results
    metadata = evaluation_data.get("metadata", {})
    type_stats = metadata.get("type_statistics", {})
    total_sequences = sum(type_stats.values())

    logger.info(f"  ✓ Successfully generated evaluation for L={L}, m={m}")
    logger.info(f"    Total sequences: {total_sequences}")
    logger.info(f"    Type distribution: {type_stats}")
    logger.info(f"    Output: {dir_manager.eval_dir}")

    return True


def main():
    """Main ICL evaluation generation function."""
    parser = create_icl_eval_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert to dataset config (L,M now come from command line)
    dataset_config = parse_dataset_config(args)

    # Log the configuration
    logger.info(f"Dataset configuration: {dataset_config.to_name()}")
    logger.info(f"Using L={dataset_config.L}, m={dataset_config.m} from command line")

    if args.validate_only:
        logger.info(f"Validation successful for {dataset_config.to_name()}")

        # Check prerequisites
        try:
            paths = dataset_config.get_config_paths(dataset_config.L, dataset_config.m)
            base_dir = paths["dataset_dir"].parent
            prereq_results = check_evaluation_prerequisites(base_dir)

            logger.info(f"Configuration: L={dataset_config.L}, m={dataset_config.m}")

            if prereq_results["ready_for_evaluation"]:
                logger.info("    ✓ Ready for evaluation")

                # Check if config file exists
                try:
                    eval_config = load_eval_config(dataset_config)
                    enabled_types = eval_config.get_enabled_types()
                    logger.info(f"    ✓ Config file found, enabled types: {enabled_types}")
                except FileNotFoundError as e:
                    logger.warning(f"    ⚠ Config file issue: {e}")

            else:
                missing = [k for k, v in prereq_results.items() if not v and k.endswith("_exists")]
                logger.error(f"    ✗ Missing: {missing}")
                return 1

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return 1

        return 0

    if args.verbose:
        logger.info(f"Dataset: {dataset_config.to_name()}")
        logger.info(f"Configuration: L={dataset_config.L}, m={dataset_config.m}")
        if args.enable_types:
            logger.info(f"Enabled types override: {args.enable_types}")
        logger.info(f"Skip existing: {args.skip_existing}")
        logger.info(f"Overwrite: {args.overwrite}")

    # Process the configuration
    logger.info(f"\n{'=' * 60}")
    logger.info("STARTING ICL EVALUATION GENERATION")
    logger.info(f"{'=' * 60}")
    logger.info(f"Processing configuration: L={dataset_config.L}, m={dataset_config.m}")
    logger.info("-" * 40)

    try:
        success = generate_evaluation_for_configuration(
            dataset_config,
            enable_types=args.enable_types,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )

        if success:
            logger.info(f"\n{'=' * 60}")
            logger.info("ICL EVALUATION GENERATION COMPLETE")
            logger.info(f"{'=' * 60}")
            return 0
        logger.error("ICL EVALUATION GENERATION FAILED")
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
