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


def load_separate_seed_datasets(dataset_path: str | Path) -> tuple[dict[int, Dataset], dict[str, Any]]:
    """Load separate seed datasets from the new directory structure.

    Args:
        dataset_path: Path to dataset directory (e.g., datasets/uniform_5_L4_M2/train/)

    Returns:
        tuple: (dict mapping seed -> Dataset, combined metadata)

    """
    dataset_path = Path(dataset_path)
    logger.info(f"Loading separate seed datasets from: {dataset_path}")

    # Load seed index to discover available seeds
    seed_index_path = dataset_path / "seed_index.json"
    if not seed_index_path.exists():
        raise FileNotFoundError(f"Seed index not found: {seed_index_path}")

    with seed_index_path.open("r") as f:
        seed_index = json.load(f)

    available_seeds = seed_index["available_seeds"]
    logger.info(f"Found {len(available_seeds)} seed datasets: {available_seeds}")

    # Load metadata
    metadata_path = dataset_path / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    # Load each seed dataset
    seed_datasets = {}
    for seed in available_seeds:
        seed_dataset_path = dataset_path / f"seed_{seed}" / "dataset"
        if not seed_dataset_path.exists():
            logger.warning(f"Seed dataset not found: {seed_dataset_path}, skipping")
            continue

        try:
            seed_dataset = load_from_disk(str(seed_dataset_path))
            seed_datasets[seed] = seed_dataset
            logger.info(f"  ✓ Loaded seed {seed}: {len(seed_dataset)} sequences")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load seed {seed}: {e}")
            continue

    if not seed_datasets:
        raise RuntimeError(f"No valid seed datasets found in {dataset_path}")

    logger.info(f"Successfully loaded {len(seed_datasets)} seed datasets")
    return seed_datasets, metadata


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


def prepare_seed_based_dataset(
    dataset_path: str,
    tokenizer: RHMTokenizer,
    config: "RHMTrainingConfig",
    train_split_ratio: float = 0.8,
    max_samples_per_seed: int | None = None,
) -> tuple[dict[int, Dataset], dict[int, Dataset], dict[str, Any]]:
    """Prepare seed-based datasets for training with separate seed handling.

    Returns:
        tuple: (train_seed_datasets, eval_seed_datasets, metadata)

    """
    logger.info("=" * 60)
    logger.info("PREPARING SEED-BASED DATASET")
    logger.info("=" * 60)

    # Load separate seed datasets
    seed_datasets, dataset_metadata = load_separate_seed_datasets(dataset_path)

    # Limit samples per seed if specified
    if max_samples_per_seed is not None:
        for seed in seed_datasets:
            if len(seed_datasets[seed]) > max_samples_per_seed:
                indices = list(range(len(seed_datasets[seed])))
                import random

                random.Random(seed).shuffle(indices)
                seed_datasets[seed] = seed_datasets[seed].select(indices[:max_samples_per_seed])
                logger.info(f"Limited seed {seed} to {max_samples_per_seed} samples")

    # Pack sequences for each seed
    packed_seed_datasets = pack_seed_datasets(seed_datasets, tokenizer, config)

    # Split each seed dataset into train/eval
    train_seed_datasets = {}
    eval_seed_datasets = {}

    total_train_sequences = 0
    total_eval_sequences = 0

    for seed, packed_dataset in packed_seed_datasets.items():
        split_idx = int(len(packed_dataset) * train_split_ratio)

        train_seed_datasets[seed] = packed_dataset.select(range(split_idx))
        eval_seed_datasets[seed] = packed_dataset.select(range(split_idx, len(packed_dataset)))

        total_train_sequences += len(train_seed_datasets[seed])
        total_eval_sequences += len(eval_seed_datasets[seed])

        logger.info(f"  Seed {seed}: {len(train_seed_datasets[seed])} train, {len(eval_seed_datasets[seed])} eval")

    # Enhanced metadata
    metadata = {
        "original_seed_datasets": {seed: len(dataset) for seed, dataset in seed_datasets.items()},
        "packed_seed_datasets": {seed: len(dataset) for seed, dataset in packed_seed_datasets.items()},
        "train_seed_datasets": {seed: len(dataset) for seed, dataset in train_seed_datasets.items()},
        "eval_seed_datasets": {seed: len(dataset) for seed, dataset in eval_seed_datasets.items()},
        "total_train_sequences": total_train_sequences,
        "total_eval_sequences": total_eval_sequences,
        "available_seeds": list(seed_datasets.keys()),
        "num_seeds": len(seed_datasets),
        "train_split_ratio": train_split_ratio,
        "packing_enabled": True,  # Always true now
        "shuffling_enabled": config.shuffle_before_packing,
        "shuffle_strategy": config.shuffle_strategy if config.shuffle_before_packing else None,
        "seed_based_loading": True,
        "dataset_metadata": dataset_metadata,
        "config": config,
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }

    logger.info(f"Total train sequences: {total_train_sequences:,}")
    logger.info(f"Total eval sequences: {total_eval_sequences:,}")
    logger.info(f"Available seeds: {list(seed_datasets.keys())}")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info("=" * 60)

    return train_seed_datasets, eval_seed_datasets, metadata


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
    """Validate that the dataset has the expected seed-based structure.

    Args:
        dataset_path: Path to dataset directory (e.g., datasets/uniform_10_L4_M2/train/)

    Returns:
        dict: Comprehensive validation results and discovered information

    """
    dataset_path = Path(dataset_path)

    validation_result = {
        "valid": False,
        "dataset_path": str(dataset_path),
        "seed_datasets_found": [],
        "metadata_exists": False,
        "seed_index_exists": False,
        "L": None,
        "m": None,
        "dataset_type": None,
        "num_seeds": None,
        "errors": [],
        "warnings": [],
        "seed_dataset_details": {},
        "file_structure": {},
    }

    # Check if dataset path exists
    if not dataset_path.exists():
        validation_result["errors"].append(f"Dataset path does not exist: {dataset_path}")
        return validation_result

    if not dataset_path.is_dir():
        validation_result["errors"].append(f"Dataset path is not a directory: {dataset_path}")
        return validation_result

    # Extract L,M and other parameters from path
    try:
        L, m = extract_L_M_from_dataset_path(dataset_path)
        validation_result["L"] = L
        validation_result["m"] = m

        # Extract additional parameters from path
        # Expected pattern: datasets/uniform_10_L4_M2/train/
        path_parts = dataset_path.parts

        # Find the part that contains the configuration
        config_part = None
        for part in path_parts:
            if "_L" in part and "_M" in part:
                config_part = part
                break

        if config_part:
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

    # Check for seed_index.json
    seed_index_path = dataset_path / "seed_index.json"
    if seed_index_path.exists():
        validation_result["seed_index_exists"] = True
        validation_result["file_structure"]["seed_index.json"] = "found"

        try:
            import json

            with seed_index_path.open("r") as f:
                seed_index = json.load(f)

            # Validate seed index structure
            required_keys = ["available_seeds", "num_seeds", "dataset_paths"]
            missing_keys = [key for key in required_keys if key not in seed_index]
            if missing_keys:
                validation_result["warnings"].append(f"seed_index.json missing keys: {missing_keys}")

            validation_result["seed_datasets_found"] = seed_index.get("available_seeds", [])

            # Validate seed index consistency
            if "num_seeds" in seed_index:
                expected_num_seeds = seed_index["num_seeds"]
                actual_num_seeds = len(validation_result["seed_datasets_found"])
                if expected_num_seeds != actual_num_seeds:
                    validation_result["warnings"].append(
                        f"seed_index.json reports {expected_num_seeds} seeds but lists {actual_num_seeds} seeds"
                    )

        except json.JSONDecodeError as e:
            validation_result["errors"].append(f"Invalid JSON in seed_index.json: {e}")
        except Exception as e:
            validation_result["errors"].append(f"Could not read seed_index.json: {e}")
    else:
        validation_result["errors"].append(f"seed_index.json not found at {seed_index_path}")
        validation_result["file_structure"]["seed_index.json"] = "missing"

    # Check for metadata.pkl
    metadata_path = dataset_path / "metadata.pkl"
    if metadata_path.exists():
        validation_result["metadata_exists"] = True
        validation_result["file_structure"]["metadata.pkl"] = "found"

        try:
            import pickle

            with metadata_path.open("rb") as f:
                metadata = pickle.load(f)

            # Validate metadata structure and extract useful information
            if isinstance(metadata, dict):
                # Check for expected metadata fields
                expected_fields = ["config_params", "seed_info", "dataset_stats"]
                found_fields = [field for field in expected_fields if field in metadata]

                if found_fields:
                    validation_result["metadata_summary"] = {
                        "found_fields": found_fields,
                        "config_params": metadata.get("config_params", {}),
                        "seed_info": metadata.get("seed_info", {}),
                    }

                # Cross-validate with extracted parameters
                if "config_params" in metadata:
                    config_params = metadata["config_params"]
                    if "L" in config_params and config_params["L"] != L:
                        validation_result["warnings"].append(
                            f"L mismatch: path suggests L={L}, metadata has L={config_params['L']}"
                        )
                    if "m" in config_params and config_params["m"] != m:
                        validation_result["warnings"].append(
                            f"m mismatch: path suggests m={m}, metadata has m={config_params['m']}"
                        )

                # Validate seed information
                if "seed_info" in metadata:
                    seed_info = metadata["seed_info"]
                    if "random_seeds" in seed_info:
                        metadata_seeds = seed_info["random_seeds"]
                        if set(metadata_seeds) != set(validation_result["seed_datasets_found"]):
                            validation_result["warnings"].append(
                                f"Seed mismatch: metadata seeds {metadata_seeds} != index seeds {validation_result['seed_datasets_found']}"
                            )

            else:
                validation_result["warnings"].append("metadata.pkl does not contain a dictionary")

        except Exception as e:
            validation_result["errors"].append(f"Could not read metadata.pkl: {e}")
    else:
        validation_result["errors"].append(f"metadata.pkl not found at {metadata_path}")
        validation_result["file_structure"]["metadata.pkl"] = "missing"

    # Check for actual seed datasets
    found_seeds = []
    seed_details = {}

    for seed in validation_result["seed_datasets_found"]:
        seed_dir = dataset_path / f"seed_{seed}"
        seed_dataset_path = seed_dir / "dataset"

        seed_info = {
            "seed": seed,
            "directory_exists": seed_dir.exists(),
            "dataset_exists": seed_dataset_path.exists(),
            "dataset_path": str(seed_dataset_path),
        }

        if seed_dir.exists():
            if seed_dataset_path.exists():
                try:
                    # Try to load and validate the dataset
                    from datasets import load_from_disk

                    dataset = load_from_disk(str(seed_dataset_path))

                    seed_info.update(
                        {
                            "dataset_valid": True,
                            "num_sequences": len(dataset),
                            "columns": dataset.column_names,
                            "features": dataset.features,
                        }
                    )

                    # Validate expected columns
                    expected_columns = ["input_ids", "length"]
                    missing_columns = [col for col in expected_columns if col not in dataset.column_names]
                    if missing_columns:
                        validation_result["warnings"].append(f"Seed {seed} dataset missing columns: {missing_columns}")
                        seed_info["missing_columns"] = missing_columns

                    # Sample validation
                    if len(dataset) > 0:
                        sample = dataset[0]
                        if "input_ids" in sample:
                            input_ids = sample["input_ids"]
                            if isinstance(input_ids, list) and len(input_ids) > 0:
                                seed_info["sample_length"] = len(input_ids)
                                seed_info["sample_valid"] = True
                            else:
                                validation_result["warnings"].append(f"Seed {seed} has invalid input_ids format")
                                seed_info["sample_valid"] = False

                    found_seeds.append(seed)

                except Exception as e:
                    validation_result["errors"].append(f"Seed {seed} dataset exists but cannot be loaded: {e}")
                    seed_info.update(
                        {
                            "dataset_valid": False,
                            "load_error": str(e),
                        }
                    )
            else:
                validation_result["errors"].append(
                    f"Seed dataset directory exists but dataset not found: {seed_dataset_path}"
                )
        else:
            validation_result["errors"].append(f"Seed directory not found: {seed_dir}")

        seed_details[seed] = seed_info

    validation_result["seed_dataset_details"] = seed_details
    validation_result["seed_datasets_found"] = found_seeds

    # Additional file structure analysis
    try:
        # List all items in the dataset directory
        all_items = list(dataset_path.iterdir())
        validation_result["file_structure"]["total_items"] = len(all_items)
        validation_result["file_structure"]["directories"] = [item.name for item in all_items if item.is_dir()]
        validation_result["file_structure"]["files"] = [item.name for item in all_items if item.is_file()]

        # Check for unexpected files/directories
        expected_items = {"seed_index.json", "metadata.pkl", "dataset_summary.txt"} | {
            f"seed_{seed}" for seed in validation_result["seed_datasets_found"]
        }

        actual_items = {item.name for item in all_items}
        unexpected_items = actual_items - expected_items

        if unexpected_items:
            validation_result["warnings"].append(f"Unexpected items in dataset directory: {unexpected_items}")
            validation_result["file_structure"]["unexpected_items"] = list(unexpected_items)

    except Exception as e:
        validation_result["warnings"].append(f"Could not analyze file structure: {e}")

    # Final validation
    validation_result["valid"] = (
        len(found_seeds) > 0
        and validation_result["seed_index_exists"]
        and validation_result["metadata_exists"]
        and len(validation_result["errors"]) == 0
    )

    # Generate summary
    if validation_result["valid"]:
        validation_result["summary"] = (
            f"✓ Valid dataset: {len(found_seeds)} seeds, L={L}, m={m}, "
            f"type={validation_result.get('dataset_type', 'unknown')}"
        )
    else:
        error_count = len(validation_result["errors"])
        warning_count = len(validation_result["warnings"])
        validation_result["summary"] = f"✗ Invalid dataset: {error_count} errors, {warning_count} warnings"

    return validation_result


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
