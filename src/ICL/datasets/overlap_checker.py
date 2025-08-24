import logging
import warnings
from typing import Any

from datasets import Dataset

from ICL.datasets.RHM import RandomHierarchyModel
from ICL.datasets.utils import create_rule_probabilities

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GlobalSequenceTracker:
    """Tracks sequences across all seeds to prevent duplicates."""

    def __init__(self):
        self.global_sequences: set[str] = set()
        self.sequences_by_seed: dict[int, set[str]] = {}
        self.collision_count = 0

    def sequence_to_key(self, sequence: list[int]) -> str:
        """Convert sequence to string key for efficient set operations."""
        return ",".join(map(str, sequence))

    def add_sequences(self, seed: int, sequences: list[list[int]]) -> tuple[list[list[int]], int]:
        """Add sequences for a seed, returning only unique ones and collision count."""
        if seed not in self.sequences_by_seed:
            self.sequences_by_seed[seed] = set()

        unique_sequences = []
        local_collisions = 0

        for seq in sequences:
            seq_key = self.sequence_to_key(seq)

            if seq_key not in self.global_sequences:
                self.global_sequences.add(seq_key)
                self.sequences_by_seed[seed].add(seq_key)
                unique_sequences.append(seq)
            else:
                local_collisions += 1
                self.collision_count += 1

        return unique_sequences, local_collisions

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the global sequence tracker."""
        return {
            "total_unique_sequences": len(self.global_sequences),
            "total_collisions": self.collision_count,
            "sequences_per_seed": {seed: len(seqs) for seed, seqs in self.sequences_by_seed.items()},
            "seeds_processed": len(self.sequences_by_seed),
        }


def generate_unique_rhm_batch(
    rhm_params: dict[str, Any],
    L: int,
    m: int,
    seed: int,
    target_size: int,
    global_tracker: GlobalSequenceTracker,
    distribution_config: dict[str, Any],
    dedup_config: dict[str, Any],
) -> tuple[list[list[int]], int]:
    """Generate a batch of unique sequences for a specific seed."""
    max_attempts = dedup_config.get("max_attempts", 10)
    batch_multiplier = dedup_config.get("batch_multiplier", 2.0)

    batch_size = min(int(target_size * batch_multiplier), 10000)
    unique_sequences = []
    attempts = 0
    total_generated = 0

    while len(unique_sequences) < target_size and attempts < max_attempts:
        attempts += 1
        current_seed = seed + attempts * 1000  # Modify seed for each attempt

        logger.debug(f"    Attempt {attempts}: generating {batch_size} sequences with seed {current_seed}")

        # Create RHM with current parameters
        rhm_kwargs = {
            "num_features": rhm_params["vocab_size"],
            "num_classes": rhm_params["num_classes"],
            "num_synonyms": m,
            "tuple_size": rhm_params["tuple_size"],
            "num_layers": L,
            "seed_rules": seed,  # Keep rules consistent
            "seed_sample": current_seed,  # Vary sampling seed
            "train_size": batch_size,
            "replacement": rhm_params.get("replacement", True),
            "input_format": rhm_params.get("input_format", "long"),
            "whitening": rhm_params.get("whitening", 0),
        }

        # Create probability distribution if needed
        probability = None
        if distribution_config["type"] != "uniform":
            temp_rhm = RandomHierarchyModel(**{**rhm_kwargs, "train_size": 1})

            probability = create_rule_probabilities(
                temp_rhm.rules,
                distribution_config["type"],
                **{k: v for k, v in distribution_config.items() if k != "type"},
            )

            rhm_kwargs["probability"] = probability
            rhm_kwargs["replacement"] = False  # Required for custom probabilities

        # Generate with final parameters
        rhm = RandomHierarchyModel(**rhm_kwargs)

        # Extract sequences
        batch_sequences = (
            rhm.features.tolist() if hasattr(rhm.features, "tolist") else [list(seq) for seq in rhm.features]
        )
        total_generated += len(batch_sequences)

        # Add to global tracker and get unique ones
        new_unique, collisions = global_tracker.add_sequences(seed, batch_sequences)
        unique_sequences.extend(new_unique)

        logger.debug(f"    Generated {len(batch_sequences)}, found {len(new_unique)} unique, {collisions} collisions")

        # Dynamically adjust batch size based on collision rate
        collision_rate = collisions / len(batch_sequences) if batch_sequences else 0
        if collision_rate > 0.5:  # More than 50% collisions
            batch_size = min(int(batch_size * 1.5), 50000)

    if len(unique_sequences) < target_size:
        logger.warning(
            f"    Only generated {len(unique_sequences)}/{target_size} unique sequences for seed {seed} after {attempts} attempts"
        )

    return unique_sequences[:target_size], total_generated


def generate_rhm_dataset_with_deduplication(
    L: int, m: int, random_seeds: list[int], yaml_config: dict[str, Any]
) -> tuple[dict[int, Dataset], dict[str, Any]]:
    """Generate RHM dataset with global deduplication across seeds."""
    rhm_params = yaml_config["rhm_params"]
    distribution_config = yaml_config["distribution"]
    dedup_config = yaml_config["deduplication"]

    logger.info(f"Generating deduplicated dataset for L={L}, m={m} with {random_seeds} seeds")
    logger.info(f"Seeds: {random_seeds}")
    logger.info(
        f"Deduplication config: max_attempts={dedup_config['max_attempts']}, batch_multiplier={dedup_config['batch_multiplier']}"
    )

    # Initialize global tracker and storage
    global_tracker = GlobalSequenceTracker()
    seed_datasets = {}
    config_stats = {}
    rules_by_seed = {}

    target_sequences_per_seed = rhm_params["samples_per_config"]

    # Generate data for each seed with deduplication
    for seed_idx, random_seed in enumerate(random_seeds):
        logger.info(f"  Generating unique data for seed {random_seed} ({seed_idx + 1}/{len(random_seeds)})")

        try:
            # Generate unique sequences for this seed
            unique_sequences, total_generated = generate_unique_rhm_batch(
                rhm_params,
                L,
                m,
                random_seed,
                target_sequences_per_seed,
                global_tracker,
                distribution_config,
                dedup_config,
            )

            if not unique_sequences:
                logger.error(f"    ✗ Failed to generate any unique sequences for seed {random_seed}")
                continue

            # Generate proper labels
            label_rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                seed_rules=random_seed,
                seed_sample=random_seed + 1,
                train_size=len(unique_sequences),
                replacement=rhm_params.get("replacement", True),
                input_format=rhm_params.get("input_format", "long"),
            )

            labels = label_rhm.labels[: len(unique_sequences)].tolist()

            # Create dataset for this seed
            seed_dataset_dict = {
                "input_ids": unique_sequences,
                "length": [len(seq) for seq in unique_sequences],
            }

            if hasattr(label_rhm, "labels") and label_rhm.labels is not None:
                seed_dataset_dict["labels"] = labels

            seed_datasets[random_seed] = Dataset.from_dict(seed_dataset_dict)

            # Calculate statistics
            seq_lengths = [len(seq) for seq in unique_sequences]
            config_stats[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "num_sequences": len(unique_sequences),
                "min_length": min(seq_lengths) if seq_lengths else 0,
                "max_length": max(seq_lengths) if seq_lengths else 0,
                "avg_length": sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
                "total_tokens": sum(seq_lengths),
                "distribution_type": distribution_config["type"],
                "total_generated": total_generated,
                "efficiency": len(unique_sequences) / total_generated if total_generated > 0 else 0,
            }

            # Store rules
            sample_rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                seed_rules=random_seed,
                seed_sample=random_seed + 1,
                train_size=1,
                replacement=True,
                input_format="long",
            )

            probability = None
            if distribution_config["type"] != "uniform":
                probability = create_rule_probabilities(
                    sample_rhm.rules,
                    distribution_config["type"],
                    **{k: v for k, v in distribution_config.items() if k != "type"},
                )

            rules_by_seed[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "rules_dict": sample_rhm.rules,
                "probability_dict": probability,
                "distribution_type": distribution_config["type"],
            }

            logger.info(
                f"    ✓ Generated {len(unique_sequences)} unique sequences (efficiency: {config_stats[random_seed]['efficiency']:.2%})"
            )

        except Exception as e:
            logger.error(f"    ✗ Error generating data with seed {random_seed}: {e}")
            continue

    # Get final statistics from global tracker
    tracker_stats = global_tracker.get_stats()

    # Calculate combined statistics
    total_sequences = sum(s["num_sequences"] for s in config_stats.values())

    # Create comprehensive metadata
    metadata = {
        "config_params": {
            "L": L,
            "m": m,
            "generation_params": rhm_params,
            "distribution_config": distribution_config,
            "deduplication_config": dedup_config,
        },
        "seed_info": {
            "random_seeds": random_seeds,
            "successful_seeds": list(config_stats.keys()),
            "num_seeds": len(random_seeds),
            "successful_count": len(config_stats),
        },
        "config_stats": config_stats,
        "rules_by_seed": rules_by_seed,
        "dataset_stats": {
            "total_sequences": total_sequences,
            "min_seq_length": min(s["min_length"] for s in config_stats.values()) if config_stats else 0,
            "max_seq_length": max(s["max_length"] for s in config_stats.values()) if config_stats else 0,
            "avg_seq_length": sum(s["avg_length"] * s["num_sequences"] for s in config_stats.values()) / total_sequences
            if total_sequences > 0
            else 0,
            "total_tokens": sum(s["total_tokens"] for s in config_stats.values()),
            "vocab_range": f"1-{rhm_params['vocab_size']} (0 reserved for special tokens)",
        },
        "deduplication_stats": {
            "enabled": True,
            "global_unique_sequences": tracker_stats["total_unique_sequences"],
            "total_collisions_prevented": tracker_stats["total_collisions"],
            "sequences_per_seed": tracker_stats["sequences_per_seed"],
            "average_efficiency": sum(s["efficiency"] for s in config_stats.values()) / len(config_stats)
            if config_stats
            else 0,
        },
    }

    logger.info("  ✓ Global deduplication complete:")
    logger.info(f"    - Total unique sequences: {tracker_stats['total_unique_sequences']}")
    logger.info(f"    - Collisions prevented: {tracker_stats['total_collisions']}")
    logger.info(f"    - Average generation efficiency: {metadata['deduplication_stats']['average_efficiency']:.2%}")

    return seed_datasets, metadata


def generate_rhm_dataset_original(
    L: int, m: int, random_seeds: list[int], yaml_config: dict[str, Any]
) -> tuple[dict[int, Dataset], dict[str, Any]]:
    """Original generation method without deduplication - RENAMED from generate_rhm_dataset_for_config."""
    # Extract configuration sections
    rhm_params = yaml_config["rhm_params"]
    distribution_config = yaml_config["distribution"]

    logger.info(f"Generating dataset (ORIGINAL METHOD) for L={L}, m={m} with {len(random_seeds)} seeds")
    logger.info(f"Seeds: {random_seeds}")

    # Initialize storage for datasets per seed and metadata
    seed_datasets = {}
    config_stats = {}
    rules_by_seed = {}

    total_sequences = 0
    all_sequence_lengths = []

    # Generate data for each random seed separately (ORIGINAL LOGIC)
    for seed_idx, random_seed in enumerate(random_seeds):
        logger.info(f"  Generating data with seed {random_seed} ({seed_idx + 1}/{len(random_seeds)})")

        try:
            # Create initial RHM to get rules structure
            temp_rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                seed_rules=random_seed,  # Use random seed for rules
                seed_sample=random_seed + 1,  # Slightly different seed for sampling
                train_size=1,  # Minimal size to get rules
                replacement=True,
                input_format="long",
            )

            # Create probability distribution for this configuration
            probability = create_rule_probabilities(
                temp_rhm.rules,
                distribution_config["type"],
                **{k: v for k, v in distribution_config.items() if k != "type"},
            )

            # Generate full dataset with proper probabilities
            rhm = RandomHierarchyModel(
                num_features=rhm_params["vocab_size"],
                num_classes=rhm_params["num_classes"],
                num_synonyms=m,
                tuple_size=rhm_params["tuple_size"],
                num_layers=L,
                probability=probability,
                seed_rules=random_seed,
                seed_sample=random_seed + 1,
                train_size=rhm_params["samples_per_config"],
                test_size=0,
                replacement=False,  # Required for custom probabilities
                input_format="long",
            )

            # Extract generated data
            sequences = rhm.features
            labels = rhm.labels
            rules = rhm.rules

            # Convert to lists for HuggingFace compatibility
            sequences_list = sequences.tolist() if hasattr(sequences, "tolist") else [list(seq) for seq in sequences]

            # Create individual dataset for this seed
            seed_dataset_dict = {
                "input_ids": sequences_list,
                "length": [len(seq) for seq in sequences_list],
            }

            seed_datasets[random_seed] = Dataset.from_dict(seed_dataset_dict)

            # Calculate statistics for this seed
            seq_lengths = [len(seq) for seq in sequences_list]
            config_stats[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "num_sequences": len(sequences_list),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "avg_length": sum(seq_lengths) / len(seq_lengths),
                "total_tokens": sum(seq_lengths),
                "distribution_type": distribution_config["type"],
            }

            # Store rules for this seed
            rules_by_seed[random_seed] = {
                "seed": random_seed,
                "L": L,
                "m": m,
                "rules_dict": rules,
                "probability_dict": probability,
                "distribution_type": distribution_config["type"],
            }

            # Update totals
            total_sequences += len(sequences_list)
            all_sequence_lengths.extend(seq_lengths)

            logger.info(f"    ✓ Generated {len(sequences_list)} sequences")
            logger.info(f"    ✓ Length range: {min(seq_lengths)}-{max(seq_lengths)} tokens")

        except Exception as e:
            logger.error(f"    ✗ Error generating data with seed {random_seed}: {e}")
            warnings.warn(f"Skipping seed {random_seed} for L={L}, m={m} due to error: {e}")
            continue

    if total_sequences == 0:
        raise RuntimeError(f"Failed to generate any data for L={L}, m={m}")

    # Create comprehensive metadata for this configuration (ORIGINAL FORMAT)
    metadata = {
        "config_params": {
            "L": L,
            "m": m,
            "generation_params": rhm_params,
            "distribution_config": distribution_config,
        },
        "seed_info": {
            "random_seeds": random_seeds,
            "successful_seeds": [seed for seed in config_stats],
            "num_seeds": len(random_seeds),
            "successful_count": len(config_stats),
        },
        "config_stats": config_stats,
        "rules_by_seed": rules_by_seed,
        "dataset_stats": {
            "total_sequences": total_sequences,
            "min_seq_length": min(all_sequence_lengths),
            "max_seq_length": max(all_sequence_lengths),
            "avg_seq_length": sum(all_sequence_lengths) / len(all_sequence_lengths),
            "total_tokens": sum(all_sequence_lengths),
            "vocab_range": f"1-{rhm_params['vocab_size']} (0 reserved for special tokens)",
        },
    }

    logger.info(f"  ✓ Created {len(seed_datasets)} separate seed datasets with {total_sequences} total sequences")

    return seed_datasets, metadata
