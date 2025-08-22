import dataclasses
import json
import logging
import pickle
import random
import typing as t
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from ICL.train.model import RHMTrainingConfig
from ICL.train.tokenizer import RHMTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_separate_seed_datasets(
    dataset_path: str | Path, subdirectory: str = "train"
) -> tuple[dict[int, Dataset], dict[str, Any]]:
    """Load separate seed datasets from the directory structure.

    Args:
        dataset_path: Path to base dataset directory (e.g., datasets/uniform_5_L4_M2/)
        subdirectory: Subdirectory to load from ("train" or "validation")

    Returns:
        tuple: (dict mapping seed -> Dataset, combined metadata)

    """
    dataset_path = Path(dataset_path)
    # Ensure we're pointing to the correct subdirectory
    if dataset_path.name in ["train", "validation"]:
        # If path already includes subdirectory, use parent
        base_path = dataset_path.parent
        target_path = base_path / subdirectory
    else:
        # Path is base directory, add subdirectory
        target_path = dataset_path / subdirectory

    logger.info(f"Loading separate seed datasets from: {target_path}")

    # Load seed index to discover available seeds
    seed_index_path = target_path / "seed_index.json"
    if not seed_index_path.exists():
        raise FileNotFoundError(f"Seed index not found: {seed_index_path}")

    with seed_index_path.open("r") as f:
        seed_index = json.load(f)

    available_seeds = seed_index["available_seeds"]
    logger.info(f"Found {len(available_seeds)} seed datasets in {subdirectory}: {available_seeds}")

    # Load metadata
    metadata_path = target_path / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    # Load each seed dataset
    seed_datasets = {}
    for seed in available_seeds:
        seed_dataset_path = target_path / f"seed_{seed}" / "dataset"
        if not seed_dataset_path.exists():
            logger.warning(f"Seed dataset not found: {seed_dataset_path}, skipping")
            continue

        try:
            seed_dataset = load_from_disk(str(seed_dataset_path))
            seed_datasets[seed] = seed_dataset
            logger.info(f"  ✓ Loaded seed {seed} ({subdirectory}): {len(seed_dataset)} sequences")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load seed {seed} ({subdirectory}): {e}")
            continue

    if not seed_datasets:
        raise RuntimeError(f"No valid seed datasets found in {target_path}")

    logger.info(f"Successfully loaded {len(seed_datasets)} seed datasets from {subdirectory}")
    return seed_datasets, metadata


def prepare_seed_based_dataset(
    dataset_path: str,
    tokenizer: RHMTokenizer,
    config: "RHMTrainingConfig",
    max_samples_per_seed: int | None = None,
) -> tuple[dict[int, Dataset], dict[int, Dataset], dict[str, Any]]:
    """Prepare seed-based datasets for training with separate train/eval loading.

    Args:
        dataset_path: Base dataset path (e.g., datasets/uniform_10_L4_M2/)
        tokenizer: RHM tokenizer
        config: Training configuration
        max_samples_per_seed: Optional limit on samples per seed

    Returns:
        tuple: (train_seed_datasets, eval_seed_datasets, metadata)

    """
    logger.info("=" * 60)
    logger.info("PREPARING SEED-BASED DATASET")
    logger.info("=" * 60)

    base_dataset_path = Path(dataset_path)

    # Ensure we're working with the base directory
    if base_dataset_path.name in ["train", "validation"]:
        base_dataset_path = base_dataset_path.parent

    logger.info(f"Base dataset path: {base_dataset_path}")

    # Load train and eval datasets separately
    logger.info("Loading training datasets...")
    train_seed_datasets, train_metadata = load_separate_seed_datasets(base_dataset_path, "train")

    logger.info("Loading evaluation datasets...")
    eval_seed_datasets, eval_metadata = load_separate_seed_datasets(base_dataset_path, "validation")

    # Verify that train and eval have the same seeds
    train_seeds = set(train_seed_datasets.keys())
    eval_seeds = set(eval_seed_datasets.keys())

    if train_seeds != eval_seeds:
        logger.warning(f"Train seeds {train_seeds} != Eval seeds {eval_seeds}")
        # Use intersection of seeds
        common_seeds = train_seeds & eval_seeds
        logger.info(f"Using common seeds: {common_seeds}")

        train_seed_datasets = {seed: dataset for seed, dataset in train_seed_datasets.items() if seed in common_seeds}
        eval_seed_datasets = {seed: dataset for seed, dataset in eval_seed_datasets.items() if seed in common_seeds}

    # Limit samples per seed if specified
    if max_samples_per_seed is not None:
        logger.info(f"Limiting to {max_samples_per_seed} samples per seed")

        for seed in train_seed_datasets:
            if len(train_seed_datasets[seed]) > max_samples_per_seed:
                indices = list(range(len(train_seed_datasets[seed])))
                import random

                random.Random(seed).shuffle(indices)
                train_seed_datasets[seed] = train_seed_datasets[seed].select(indices[:max_samples_per_seed])
                logger.info(f"Limited train seed {seed} to {max_samples_per_seed} samples")

        for seed in eval_seed_datasets:
            if len(eval_seed_datasets[seed]) > max_samples_per_seed:
                indices = list(range(len(eval_seed_datasets[seed])))
                import random

                random.Random(seed).shuffle(indices)
                eval_seed_datasets[seed] = eval_seed_datasets[seed].select(indices[:max_samples_per_seed])
                logger.info(f"Limited eval seed {seed} to {max_samples_per_seed} samples")

    # Pack sequences for each seed in both train and eval
    logger.info("Packing training sequences...")
    packed_train_seed_datasets = pack_seed_datasets(train_seed_datasets, tokenizer, config)

    logger.info("Packing evaluation sequences...")
    packed_eval_seed_datasets = pack_seed_datasets(eval_seed_datasets, tokenizer, config)

    # Calculate statistics
    total_train_sequences = sum(len(dataset) for dataset in packed_train_seed_datasets.values())
    total_eval_sequences = sum(len(dataset) for dataset in packed_eval_seed_datasets.values())

    # Log statistics per seed
    for seed in packed_train_seed_datasets.keys():
        train_size = len(packed_train_seed_datasets[seed])
        eval_size = len(packed_eval_seed_datasets[seed])
        logger.info(f"  Seed {seed}: {train_size} train, {eval_size} val")

    # Enhanced metadata combining train and eval information
    metadata = {
        # Original dataset sizes (before packing)
        "original_train_seed_datasets": {seed: len(dataset) for seed, dataset in train_seed_datasets.items()},
        "original_eval_seed_datasets": {seed: len(dataset) for seed, dataset in eval_seed_datasets.items()},
        # Packed dataset sizes
        "packed_train_seed_datasets": {seed: len(dataset) for seed, dataset in packed_train_seed_datasets.items()},
        "packed_eval_seed_datasets": {seed: len(dataset) for seed, dataset in packed_eval_seed_datasets.items()},
        # Final sizes
        "train_seed_datasets": {seed: len(dataset) for seed, dataset in packed_train_seed_datasets.items()},
        "eval_seed_datasets": {seed: len(dataset) for seed, dataset in packed_eval_seed_datasets.items()},
        "total_train_sequences": total_train_sequences,
        "total_eval_sequences": total_eval_sequences,
        "available_seeds": list(packed_train_seed_datasets.keys()),
        "num_seeds": len(packed_train_seed_datasets),
        # Dataset loading method
        "train_split_ratio": None,  # Not applicable since we load separate datasets
        "separate_train_eval": True,
        "packing_enabled": True,  # Always true now
        "shuffling_enabled": config.shuffle_before_packing,
        "shuffle_strategy": config.shuffle_strategy if config.shuffle_before_packing else None,
        "seed_based_loading": True,
        # Metadata from datasets
        "train_dataset_metadata": train_metadata,
        "eval_dataset_metadata": eval_metadata,
        "config": config,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        # Paths
        "base_dataset_path": str(base_dataset_path),
        "train_dataset_path": str(base_dataset_path / "train"),
        "eval_dataset_path": str(base_dataset_path / "validation"),
    }

    logger.info(f"Total train sequences: {total_train_sequences:,}")
    logger.info(f"Total eval sequences: {total_eval_sequences:,}")
    logger.info(f"Available seeds: {list(packed_train_seed_datasets.keys())}")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info("=" * 60)

    return packed_train_seed_datasets, packed_eval_seed_datasets, metadata


def pack_seed_datasets(
    seed_datasets: dict[int, Dataset], tokenizer: RHMTokenizer, config: RHMTrainingConfig
) -> dict[int, Dataset]:
    """Pack sequences separately for each seed dataset.

    Args:
        seed_datasets: Dict mapping seed -> Dataset
        tokenizer: RHM tokenizer
        config: Training configuration

    Returns:
        Dict mapping seed -> packed Dataset

    """
    logger.info("Packing sequences per seed dataset...")

    packed_seed_datasets = {}
    total_original = 0
    total_packed = 0

    for seed, dataset in seed_datasets.items():
        logger.info(f"  Packing seed {seed} ({len(dataset)} sequences)...")

        # Apply cross-configuration shuffling if enabled
        if config.shuffle_before_packing:
            dataset = _shuffle_cross_configuration_single_seed(dataset, config, seed)

        # Pack sequences for this seed
        packed_dataset = _pack_sequences_single_seed(dataset, tokenizer, config, seed)
        packed_seed_datasets[seed] = packed_dataset

        total_original += len(dataset)
        total_packed += len(packed_dataset)

        logger.info(f"    ✓ Seed {seed}: {len(dataset)} → {len(packed_dataset)} packed sequences")

    logger.info(
        f"Packing complete: {total_original} → {total_packed} packed sequences across {len(seed_datasets)} seeds"
    )
    return packed_seed_datasets


def _shuffle_cross_configuration_single_seed(dataset: Dataset, config: "RHMTrainingConfig", seed: int) -> Dataset:
    """Apply cross-configuration shuffling to a single seed dataset."""
    if not config.shuffle_before_packing:
        return dataset

    logger.info(f"    Shuffling seed {seed} with strategy: {config.shuffle_strategy}")

    # Create seed-specific random state for reproducibility
    shuffle_random = random.Random(config.shuffle_seed + seed)

    if config.shuffle_strategy == "global":
        # Simple global shuffle
        indices = list(range(len(dataset)))
        shuffle_random.shuffle(indices)
        shuffled_dataset = dataset.select(indices)
        logger.info(f"    Applied global shuffle to seed {seed}")
        return shuffled_dataset

    if config.shuffle_strategy == "balanced":
        # Ensure balanced representation from each config
        import pandas as pd

        # Convert to pandas for easier grouping
        df = dataset.to_pandas()

        # Group by configuration if config columns exist
        config_columns = [col for col in df.columns if col.startswith("config_")]
        if config_columns:
            # Group by all config columns
            groups = df.groupby(config_columns)

            # Shuffle within each group, then interleave
            shuffled_dfs = []
            for group_key, group in groups:
                group_shuffled = group.sample(frac=1, random_state=config.shuffle_seed + seed).reset_index(drop=True)
                shuffled_dfs.append(group_shuffled)

            # Interleave groups
            max_group_size = max(len(df) for df in shuffled_dfs)
            interleaved_rows = []

            for i in range(max_group_size):
                for df in shuffled_dfs:
                    if i < len(df):
                        interleaved_rows.append(df.iloc[i])

            interleaved_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)
            shuffled_dataset = Dataset.from_pandas(interleaved_df)
            logger.info(f"    Applied balanced shuffle across {len(groups)} configurations for seed {seed}")
            return shuffled_dataset
        # Fallback to global shuffle if no config columns
        logger.warning(f"    No config columns found for balanced shuffle, using global for seed {seed}")
        return _shuffle_cross_configuration_single_seed(
            dataset, dataclasses.replace(config, shuffle_strategy="global"), seed
        )

    logger.warning(f"    Unknown shuffle strategy: {config.shuffle_strategy}, using global for seed {seed}")
    return _shuffle_cross_configuration_single_seed(
        dataset, dataclasses.replace(config, shuffle_strategy="global"), seed
    )


def _pack_sequences_single_seed(
    dataset: Dataset, tokenizer: RHMTokenizer, config: "RHMTrainingConfig", seed: int
) -> Dataset:
    """Pack sequences for a single seed dataset."""
    packed_examples = []
    current_sequence = []
    current_length = 0

    # Reserve space for special tokens
    effective_max_length = config.max_sequence_length - 10

    for example in dataset:
        # Convert RHM sequence to token IDs using tokenizer
        if "input_ids" in example:
            sequence = example["input_ids"]
        else:
            logger.warning(f"No input_ids found in example for seed {seed}, skipping")
            continue

        # Skip empty sequences
        if not sequence:
            continue

        # If adding this sequence would exceed max length, finalize current packed sequence
        if current_length + len(sequence) + 1 > effective_max_length and current_sequence:
            # Add EOS token to end of packed sequence
            current_sequence.append(tokenizer.eos_token_id)
            packed_examples.append(
                {
                    "input_ids": current_sequence,
                    "length": len(current_sequence),
                    "seed": seed,  # Track which seed this packed sequence came from
                }
            )
            current_sequence = []
            current_length = 0

        # Add separator if this isn't the first sequence in the pack
        if current_sequence:
            current_sequence.append(tokenizer.sep_token_id)
            current_length += 1

        # Add the sequence
        current_sequence.extend(sequence)
        current_length += len(sequence)

    # Add the last packed sequence if it exists
    if current_sequence:
        current_sequence.append(tokenizer.eos_token_id)
        packed_examples.append(
            {
                "input_ids": current_sequence,
                "length": len(current_sequence),
                "seed": seed,
            }
        )

    return Dataset.from_list(packed_examples)


def analyze_batching_strategy(train_seed_datasets: dict[int, Dataset], config: RHMTrainingConfig) -> dict[str, t.Any]:
    """Analyze the batching strategy and provide insights.

    Args:
        train_seed_datasets: Dict mapping seed -> training Dataset
        config: Training configuration

    Returns:
        Dict containing batching analysis and recommendations

    """
    analysis = {
        "total_seeds": len(train_seed_datasets),
        "seed_dataset_sizes": {seed: len(dataset) for seed, dataset in train_seed_datasets.items()},
        "batch_size": config.per_device_train_batch_size,
        "seed_balanced_batching": config.seed_balanced_batching,
        "warnings": [],
        "recommendations": [],
    }

    if config.seed_balanced_batching:
        # Analyze seed-balanced batching strategy
        seeds_per_batch = config.seeds_per_batch or len(train_seed_datasets)

        # Ensure seeds_per_batch doesn't exceed available seeds
        if seeds_per_batch > len(train_seed_datasets):
            seeds_per_batch = len(train_seed_datasets)
            analysis["warnings"].append(
                f"seeds_per_batch ({config.seeds_per_batch}) > available seeds ({len(train_seed_datasets)}). "
                f"Using all available seeds ({seeds_per_batch})."
            )

        samples_per_seed_per_batch = config.per_device_train_batch_size // seeds_per_batch
        remainder = config.per_device_train_batch_size % seeds_per_batch

        analysis.update(
            {
                "seeds_per_batch": seeds_per_batch,
                "samples_per_seed_per_batch": samples_per_seed_per_batch,
                "remainder_samples": remainder,
                "batch_composition": f"{samples_per_seed_per_batch} samples per seed"
                + (f" (+1 for {remainder} seeds)" if remainder > 0 else ""),
                "batching_mode": "seed_balanced",
                "seed_sampling_strategy": config.seed_sampling_strategy,
            }
        )

        # Check for potential issues
        if samples_per_seed_per_batch == 0:
            analysis["warnings"].append(
                f"Batch size ({config.per_device_train_batch_size}) < seeds_per_batch ({seeds_per_batch}). "
                f"Some seeds will not appear in every batch."
            )
            analysis["recommendations"].append(
                f"Increase batch_size to at least {seeds_per_batch} or reduce seeds_per_batch."
            )

        if remainder > 0:
            analysis["batch_size_warning"] = (
                f"Batch size ({config.per_device_train_batch_size}) not evenly "
                f"divisible by seeds_per_batch ({seeds_per_batch}). "
                f"Some batches will have uneven seed distribution."
            )
            analysis["recommendations"].append(
                f"Consider using batch_size = {seeds_per_batch * (samples_per_seed_per_batch + 1)} "
                f"for even seed distribution."
            )

        # Calculate expected number of batches and training efficiency
        dataset_sizes = list(analysis["seed_dataset_sizes"].values())
        min_dataset_size = min(dataset_sizes)
        max_dataset_size = max(dataset_sizes)
        avg_dataset_size = sum(dataset_sizes) / len(dataset_sizes)

        if samples_per_seed_per_batch > 0:
            max_batches_per_seed = min_dataset_size // samples_per_seed_per_batch
            estimated_total_batches = max_batches_per_seed * len(train_seed_datasets)

            analysis.update(
                {
                    "min_seed_dataset_size": min_dataset_size,
                    "max_seed_dataset_size": max_dataset_size,
                    "avg_seed_dataset_size": avg_dataset_size,
                    "max_batches_per_seed": max_batches_per_seed,
                    "estimated_total_batches": estimated_total_batches,
                    "estimated_samples_used": estimated_total_batches * config.per_device_train_batch_size,
                }
            )

            # Dataset balance analysis
            size_ratio = max_dataset_size / min_dataset_size if min_dataset_size > 0 else float("inf")
            analysis["dataset_balance_ratio"] = size_ratio

            if size_ratio > 2.0:
                analysis["warnings"].append(
                    f"Seed datasets are significantly unbalanced (ratio: {size_ratio:.2f}). "
                    f"Largest: {max_dataset_size}, Smallest: {min_dataset_size}. "
                    f"Some seeds may be underutilized in training."
                )
                analysis["recommendations"].append(
                    "Consider using max_samples_per_seed to balance dataset sizes, "
                    "or investigate why seed generation produced uneven datasets."
                )

            # Efficiency analysis
            total_available_samples = sum(dataset_sizes)
            utilization_rate = (
                analysis["estimated_samples_used"] / total_available_samples if total_available_samples > 0 else 0
            )
            analysis["sample_utilization_rate"] = utilization_rate

            if utilization_rate < 0.5:
                analysis["warnings"].append(
                    f"Low sample utilization rate: {utilization_rate:.1%}. "
                    f"Using {analysis['estimated_samples_used']:,} out of {total_available_samples:,} available samples."
                )
                analysis["recommendations"].append(
                    "Consider increasing batch_size or reducing dataset size imbalance to improve efficiency."
                )

        else:
            # samples_per_seed_per_batch is 0
            analysis.update(
                {
                    "max_batches_per_seed": 0,
                    "estimated_total_batches": 0,
                    "training_feasible": False,
                }
            )

        # Strategy-specific analysis
        if config.seed_sampling_strategy == "balanced":
            analysis["strategy_notes"] = (
                "Balanced strategy ensures equal representation from all seeds. "
                "Good for stable training with consistent seed exposure."
            )
        elif config.seed_sampling_strategy == "random":
            analysis["strategy_notes"] = (
                "Random strategy provides varied seed exposure across batches. "
                "May lead to more diverse gradient updates but less predictable training."
            )
        elif config.seed_sampling_strategy == "weighted":
            analysis["strategy_notes"] = (
                "Weighted strategy allows curriculum learning based on seed difficulty. "
                "Requires careful weight tuning for optimal results."
            )

    else:
        # No seed balancing - standard batching
        total_samples = sum(len(dataset) for dataset in train_seed_datasets.values())
        estimated_batches = (
            total_samples // config.per_device_train_batch_size if config.per_device_train_batch_size > 0 else 0
        )

        analysis.update(
            {
                "total_samples": total_samples,
                "estimated_total_batches": estimated_batches,
                "batching_mode": "standard_shuffling",
                "sample_utilization_rate": 1.0,  # Uses all samples eventually
            }
        )

        analysis["strategy_notes"] = (
            "Standard shuffling mode combines all seed datasets and shuffles globally. "
            "No guarantee of seed diversity within individual batches."
        )

        if len(train_seed_datasets) > 1:
            analysis["recommendations"].append(
                "Consider enabling seed_balanced_batching to ensure seed diversity within batches."
            )

    # Cross-configuration shuffling analysis
    if hasattr(config, "shuffle_before_packing") and config.shuffle_before_packing:
        analysis["cross_config_shuffling"] = {
            "enabled": True,
            "strategy": config.shuffle_strategy,
            "effect": "Sequences from different (L,M) configurations are mixed within packed sequences.",
        }

        if config.shuffle_strategy == "global":
            analysis["cross_config_shuffling"]["description"] = (
                "Global shuffling mixes all configurations randomly. Good for learning mixed hierarchical complexity."
            )
        elif config.shuffle_strategy == "balanced":
            analysis["cross_config_shuffling"]["description"] = (
                "Balanced shuffling ensures systematic representation of all configurations. "
                "Provides controlled exposure to different complexity levels."
            )
    else:
        analysis["cross_config_shuffling"] = {
            "enabled": False,
            "effect": "Packed sequences maintain homogeneous (L,M) configuration.",
            "description": "Clean hierarchical learning with consistent complexity per packed sequence.",
        }

    # Performance estimates
    if "estimated_total_batches" in analysis and analysis["estimated_total_batches"] > 0:
        batches_per_epoch = analysis["estimated_total_batches"]
        analysis["performance_estimates"] = {
            "batches_per_epoch": batches_per_epoch,
            "steps_per_epoch": batches_per_epoch // config.gradient_accumulation_steps,
            "epochs_for_1000_steps": max(1, 1000 // (batches_per_epoch // config.gradient_accumulation_steps)),
        }

    # Final recommendations
    if not analysis["warnings"]:
        analysis["overall_assessment"] = "✓ Batching strategy looks good!"
    else:
        analysis["overall_assessment"] = f"⚠️ {len(analysis['warnings'])} potential issues detected."

    return analysis


def validate_dataset_structure(dataset_path: str | Path) -> dict[str, t.Any]:
    """Validate that the dataset has the expected seed-based structure with train/eval subdirectories.

    Args:
        dataset_path: Path to base dataset directory (e.g., datasets/uniform_10_L4_M2/)

    Returns:
        dict: Comprehensive validation results and discovered information

    """
    dataset_path = Path(dataset_path)

    # Ensure we're working with the base directory
    if dataset_path.name in ["train", "validation"]:
        dataset_path = dataset_path.parent

    validation_result = {
        "valid": False,
        "dataset_path": str(dataset_path),
        "train_seeds_found": [],
        "eval_seeds_found": [],
        "train_metadata_exists": False,
        "eval_metadata_exists": False,
        "train_seed_index_exists": False,
        "eval_seed_index_exists": False,
        "L": None,
        "m": None,
        "dataset_type": None,
        "num_seeds": None,
        "errors": [],
        "warnings": [],
        "train_seed_details": {},
        "eval_seed_details": {},
        "file_structure": {},
    }

    # Check if dataset path exists
    if not dataset_path.exists():
        validation_result["errors"].append(f"Dataset path does not exist: {dataset_path}")
        return validation_result

    if not dataset_path.is_dir():
        validation_result["errors"].append(f"Dataset path is not a directory: {dataset_path}")
        return validation_result

    # Check for train and eval subdirectories
    train_dir = dataset_path / "train"
    eval_dir = dataset_path / "validation"

    if not train_dir.exists():
        validation_result["errors"].append(f"Train directory not found: {train_dir}")

    if not eval_dir.exists():
        validation_result["errors"].append(f"Eval directory not found: {eval_dir}")

    if not train_dir.exists() and not eval_dir.exists():
        validation_result["errors"].append("Neither train nor eval directories found")
        return validation_result

    # Extract L,M and other parameters from path
    try:
        L, m = extract_L_M_from_dataset_path(dataset_path)
        validation_result["L"] = L
        validation_result["m"] = m

        # Extract additional parameters from path
        # Expected pattern: datasets/uniform_10_L4_M2/
        path_parts = dataset_path.parts

        # Find the part that contains the configuration
        config_part = dataset_path.name

        if "_L" in config_part and "_M" in config_part:
            # Extract dataset_type and num_seeds
            import re

            pattern = r"(\w+)_(\d+)_L\d+_M\d+"
            match = re.match(pattern, config_part)
            if match:
                validation_result["dataset_type"] = match.group(1)
                validation_result["num_seeds"] = int(match.group(2))

    except ValueError as e:
        validation_result["errors"].append(f"Could not extract L,M from path: {e}")
        return validation_result

    # Validate train directory if it exists
    if train_dir.exists():
        train_validation = _validate_subdirectory_structure(train_dir, "train")
        validation_result["train_seeds_found"] = train_validation["seeds_found"]
        validation_result["train_metadata_exists"] = train_validation["metadata_exists"]
        validation_result["train_seed_index_exists"] = train_validation["seed_index_exists"]
        validation_result["train_seed_details"] = train_validation["seed_details"]
        validation_result["errors"].extend(train_validation["errors"])
        validation_result["warnings"].extend(train_validation["warnings"])

    # Validate eval directory if it exists
    if eval_dir.exists():
        eval_validation = _validate_subdirectory_structure(eval_dir, "validation")
        validation_result["eval_seeds_found"] = eval_validation["seeds_found"]
        validation_result["eval_metadata_exists"] = eval_validation["metadata_exists"]
        validation_result["eval_seed_index_exists"] = eval_validation["seed_index_exists"]
        validation_result["eval_seed_details"] = eval_validation["seed_details"]
        validation_result["errors"].extend(eval_validation["errors"])
        validation_result["warnings"].extend(eval_validation["warnings"])

    # Cross-validate seeds between train and eval
    if validation_result["train_seeds_found"] and validation_result["eval_seeds_found"]:
        train_seeds = set(validation_result["train_seeds_found"])
        eval_seeds = set(validation_result["eval_seeds_found"])

        if train_seeds != eval_seeds:
            validation_result["warnings"].append(f"Train seeds {train_seeds} != Eval seeds {eval_seeds}")

            missing_in_eval = train_seeds - eval_seeds
            missing_in_train = eval_seeds - train_seeds

            if missing_in_eval:
                validation_result["warnings"].append(f"Seeds missing in eval: {missing_in_eval}")
            if missing_in_train:
                validation_result["warnings"].append(f"Seeds missing in train: {missing_in_train}")

    # Additional file structure analysis
    try:
        all_items = list(dataset_path.iterdir())
        validation_result["file_structure"]["total_items"] = len(all_items)
        validation_result["file_structure"]["directories"] = [item.name for item in all_items if item.is_dir()]
        validation_result["file_structure"]["files"] = [item.name for item in all_items if item.is_file()]

        # Check for expected structure
        expected_dirs = {"train", "validation"}
        actual_dirs = {item.name for item in all_items if item.is_dir()}

        missing_dirs = expected_dirs - actual_dirs
        if missing_dirs:
            validation_result["warnings"].append(f"Missing expected directories: {missing_dirs}")

        unexpected_items = actual_dirs - expected_dirs
        if unexpected_items:
            validation_result["warnings"].append(f"Unexpected directories: {unexpected_items}")

    except Exception as e:
        validation_result["warnings"].append(f"Could not analyze file structure: {e}")

    # Final validation
    has_valid_train = (
        train_dir.exists()
        and validation_result["train_seed_index_exists"]
        and validation_result["train_metadata_exists"]
        and len(validation_result["train_seeds_found"]) > 0
    )

    has_valid_eval = (
        eval_dir.exists()
        and validation_result["eval_seed_index_exists"]
        and validation_result["eval_metadata_exists"]
        and len(validation_result["eval_seeds_found"]) > 0
    )

    validation_result["valid"] = has_valid_train and has_valid_eval and len(validation_result["errors"]) == 0

    # Generate summary
    if validation_result["valid"]:
        train_count = len(validation_result["train_seeds_found"])
        eval_count = len(validation_result["eval_seeds_found"])
        validation_result["summary"] = (
            f"✓ Valid dataset: {train_count} train seeds, {eval_count} eval seeds, "
            f"L={L}, m={m}, type={validation_result.get('dataset_type', 'unknown')}"
        )
    else:
        error_count = len(validation_result["errors"])
        warning_count = len(validation_result["warnings"])
        validation_result["summary"] = f"✗ Invalid dataset: {error_count} errors, {warning_count} warnings"

    return validation_result


def _validate_subdirectory_structure(subdir_path: Path, subdir_name: str) -> dict[str, t.Any]:
    """Validate structure of a train or eval subdirectory.

    Args:
        subdir_path: Path to train or eval subdirectory
        subdir_name: Name of subdirectory ("train" or "validation")

    Returns:
        dict: Validation results for this subdirectory

    """
    result = {
        "seeds_found": [],
        "metadata_exists": False,
        "seed_index_exists": False,
        "seed_details": {},
        "errors": [],
        "warnings": [],
    }

    # Check for seed_index.json
    seed_index_path = subdir_path / "seed_index.json"
    if seed_index_path.exists():
        result["seed_index_exists"] = True

        try:
            import json

            with seed_index_path.open("r") as f:
                seed_index = json.load(f)

            required_keys = ["available_seeds", "num_seeds", "dataset_paths"]
            missing_keys = [key for key in required_keys if key not in seed_index]
            if missing_keys:
                result["warnings"].append(f"{subdir_name}/seed_index.json missing keys: {missing_keys}")

            result["seeds_found"] = seed_index.get("available_seeds", [])

        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON in {subdir_name}/seed_index.json: {e}")
        except Exception as e:
            result["errors"].append(f"Could not read {subdir_name}/seed_index.json: {e}")
    else:
        result["errors"].append(f"{subdir_name}/seed_index.json not found")

    # Check for metadata.pkl
    metadata_path = subdir_path / "metadata.pkl"
    if metadata_path.exists():
        result["metadata_exists"] = True

        try:
            import pickle

            with metadata_path.open("rb") as f:
                metadata = pickle.load(f)
            # Could add more metadata validation here if needed
        except Exception as e:
            result["errors"].append(f"Could not read {subdir_name}/metadata.pkl: {e}")
    else:
        result["errors"].append(f"{subdir_name}/metadata.pkl not found")

    # Check individual seed datasets
    for seed in result["seeds_found"]:
        seed_dir = subdir_path / f"seed_{seed}"
        seed_dataset_path = seed_dir / "dataset"

        seed_info = {
            "seed": seed,
            "directory_exists": seed_dir.exists(),
            "dataset_exists": seed_dataset_path.exists(),
            "dataset_path": str(seed_dataset_path),
        }

        if seed_dir.exists() and seed_dataset_path.exists():
            try:
                from datasets import load_from_disk

                dataset = load_from_disk(str(seed_dataset_path))

                seed_info.update(
                    {
                        "dataset_valid": True,
                        "num_sequences": len(dataset),
                        "columns": dataset.column_names,
                    }
                )

                # Validate expected columns
                expected_columns = ["input_ids", "length"]
                missing_columns = [col for col in expected_columns if col not in dataset.column_names]
                if missing_columns:
                    result["warnings"].append(f"{subdir_name} seed {seed} missing columns: {missing_columns}")
                    seed_info["missing_columns"] = missing_columns

            except Exception as e:
                result["errors"].append(f"{subdir_name} seed {seed} dataset cannot be loaded: {e}")
                seed_info.update(
                    {
                        "dataset_valid": False,
                        "load_error": str(e),
                    }
                )
        else:
            if not seed_dir.exists():
                result["errors"].append(f"{subdir_name} seed directory not found: {seed_dir}")
            if not seed_dataset_path.exists():
                result["errors"].append(f"{subdir_name} seed dataset not found: {seed_dataset_path}")

        result["seed_details"][seed] = seed_info

    return result


def extract_L_M_from_dataset_path(dataset_path: str | Path) -> tuple[int, int]:
    """Extract L and M values from dataset directory path.

    Args:
        dataset_path: Path like datasets/uniform_10_L4_M2/train/

    Returns:
        tuple: (L, m) values

    Raises:
        ValueError: If L,M cannot be extracted from path

    """
    import re

    dataset_path = Path(dataset_path)

    # Look for pattern in the dataset directory path
    # Example: datasets/uniform_10_L4_M2/train/ -> L=4, M=2
    pattern = r".*_L(\d+)_M(\d+)"

    # Check different parts of the path
    path_str = str(dataset_path)

    # Try the full path first
    match = re.search(pattern, path_str)
    if match:
        L = int(match.group(1))
        m = int(match.group(2))
        return L, m

    # Try individual path components
    for part in dataset_path.parts:
        match = re.search(pattern, part)
        if match:
            L = int(match.group(1))
            m = int(match.group(2))
            return L, m

    # Try parent directory if current is 'train'
    if dataset_path.name == "train":
        parent_match = re.search(pattern, dataset_path.parent.name)
        if parent_match:
            L = int(parent_match.group(1))
            m = int(parent_match.group(2))
            return L, m

    raise ValueError(
        f"Could not extract L,M from dataset path: {dataset_path}. "
        f"Expected pattern: *_L{{L}}_M{{m}} in path components."
    )
