import random
import typing as t
from pathlib import Path

from datasets import Dataset

from ICL.datasets.evaluation.eval_config import ICLEvalConfig
from ICL.datasets.evaluation.generators.base_generator import BaseICLGenerator, ICLSequence


class IDGeneralizationGenerator(BaseICLGenerator):
    """Generates ICL sequences for in-distribution generalization evaluation.

    Uses validation split data (same seeds as training, but different sequences).
    """

    def __init__(self, config: ICLEvalConfig):
        """Initialize ID generalization generator."""
        super().__init__(config, eval_type="id_generalization")
        self.id_gen_config = config.id_generalization

        # Track validation data for sampling
        self.validation_data_by_seed: dict[int, list[tuple[list[int], int]]] = {}
        self.validation_data_loaded = False

    def load_validation_data(self, validation_datasets: dict[int, Dataset]) -> None:
        """Load validation datasets for ID generalization.

        Args:
            validation_datasets: Dictionary mapping seed -> validation Dataset

        """
        self.validation_data_by_seed.clear()

        for seed, dataset in validation_datasets.items():
            data_pairs = self._extract_features_labels_from_dataset(dataset)

            # Apply sampling ratio if configured
            if self.id_gen_config.sampling_ratio < 1.0:
                sample_size = max(
                    self.id_gen_config.min_sequences_per_seed, int(len(data_pairs) * self.id_gen_config.sampling_ratio)
                )
                data_pairs = random.sample(data_pairs, min(sample_size, len(data_pairs)))

            self.validation_data_by_seed[seed] = data_pairs

        self.validation_data_loaded = True

        # Log loading statistics
        total_sequences = sum(len(data) for data in self.validation_data_by_seed.values())
        print(
            f"Loaded validation data for ID generalization: {len(self.validation_data_by_seed)} seeds, "
            f"{total_sequences} total sequences"
        )

    def generate_icl_sequences(
        self, source_config: tuple[int, int], target_config: tuple[int, int] | None = None, **kwargs
    ) -> list[ICLSequence]:
        """Generate ID generalization ICL sequences from validation data.

        Args:
            source_config: Source (L, m) configuration
            target_config: Not used for ID generalization (always None)
            **kwargs: Additional arguments (unused)

        Returns:
            List of ICL sequences from validation data

        """
        if not self.id_gen_config.enable:
            return []

        if not self.validation_data_loaded:
            raise RuntimeError("Validation data not loaded. Call load_validation_data() first.")

        if not self.validation_data_by_seed:
            raise ValueError("No validation data available for ID generalization")

        sequences = []

        # Generate sequences from each seed's validation data
        for seed, validation_data in self.validation_data_by_seed.items():
            if len(validation_data) < max(self.icl_params.context_sizes) + 1:
                print(
                    f"Warning: Seed {seed} has insufficient validation data "
                    f"({len(validation_data)} sequences) for largest context size "
                    f"({max(self.icl_params.context_sizes)})"
                )
                continue

            # Generate sequences for this seed
            seed_sequences = self._generate_sequences_for_context_sizes(
                available_data=validation_data,
                source_config=source_config,
                target_config=None,  # ID generalization doesn't use target config
                source_seeds=[seed],
                appears_in_training=False,  # By definition, validation data not in training
                additional_metadata={
                    "validation_seed": seed,
                    "validation_data_size": len(validation_data),
                    "sampling_ratio": self.id_gen_config.sampling_ratio,
                    "same_rules_as_training": True,  # Same seeds = same rules
                },
            )

            sequences.extend(seed_sequences)

        # Store generated sequences
        self.generated_sequences.extend(sequences)

        # Validate minimum coverage
        self._validate_seed_coverage(sequences)

        print(
            f"Generated {len(sequences)} ID generalization sequences from "
            f"{len(self.validation_data_by_seed)} validation seeds"
        )

        return sequences

    def _validate_seed_coverage(self, sequences: list[ICLSequence]) -> None:
        """Validate that all validation seeds are represented."""
        represented_seeds = set()
        for seq in sequences:
            represented_seeds.update(seq.source_seeds)

        missing_seeds = set(self.validation_data_by_seed.keys()) - represented_seeds

        if missing_seeds:
            print(f"Warning: Some validation seeds not represented in ID generalization: {sorted(missing_seeds)}")

        # Check minimum sequences per seed
        seed_counts = {}
        for seq in sequences:
            for seed in seq.source_seeds:
                seed_counts[seed] = seed_counts.get(seed, 0) + 1

        insufficient_seeds = [
            seed for seed, count in seed_counts.items() if count < self.id_gen_config.min_sequences_per_seed
        ]

        if insufficient_seeds:
            print(
                f"Warning: Some seeds have insufficient ID generalization sequences "
                f"(< {self.id_gen_config.min_sequences_per_seed}): {insufficient_seeds}"
            )

    def get_id_generalization_stats(self) -> dict[str, t.Any]:
        """Get detailed statistics about ID generalization."""
        base_stats = self.get_generation_stats()

        # Add ID generalization-specific stats
        if self.validation_data_loaded:
            validation_stats = {
                "validation_seeds_loaded": len(self.validation_data_by_seed),
                "validation_sequences_per_seed": {
                    seed: len(data) for seed, data in self.validation_data_by_seed.items()
                },
                "total_validation_sequences": sum(len(data) for data in self.validation_data_by_seed.values()),
                "sampling_ratio_applied": self.id_gen_config.sampling_ratio,
                "uses_validation_split": self.id_gen_config.use_validation_split,
            }
            base_stats.update(validation_stats)

        return base_stats

    def compare_with_training_seeds(self, training_seeds: list[int]) -> dict[str, t.Any]:
        """Compare validation seeds with training seeds to verify consistency.

        Args:
            training_seeds: List of seeds used in training

        Returns:
            Comparison statistics

        """
        validation_seeds = list(self.validation_data_by_seed.keys())

        return {
            "training_seeds": sorted(training_seeds),
            "validation_seeds": sorted(validation_seeds),
            "seeds_match": set(training_seeds) == set(validation_seeds),
            "missing_from_validation": sorted(set(training_seeds) - set(validation_seeds)),
            "extra_in_validation": sorted(set(validation_seeds) - set(training_seeds)),
            "common_seeds": sorted(set(training_seeds) & set(validation_seeds)),
            "seed_consistency_ratio": len(set(training_seeds) & set(validation_seeds))
            / len(set(training_seeds) | set(validation_seeds))
            if training_seeds or validation_seeds
            else 0.0,
        }

    def save_validation_sequence_index(self, output_path: Path) -> None:
        """Save index of validation sequences used for ID generalization."""
        import json

        if not self.generated_sequences:
            return

        sequence_index = {
            "id_generalization_sequences": [],
            "validation_data_summary": {
                "seeds": list(self.validation_data_by_seed.keys()),
                "sequences_per_seed": {str(seed): len(data) for seed, data in self.validation_data_by_seed.items()},
                "sampling_ratio": self.id_gen_config.sampling_ratio,
                "use_validation_split": self.id_gen_config.use_validation_split,
            },
        }

        for seq in self.generated_sequences:
            if seq.eval_type == "id_generalization":
                sequence_index["id_generalization_sequences"].append(
                    {
                        "sequence_id": seq.sequence_id,
                        "context_size": seq.context_size,
                        "source_seeds": seq.source_seeds,
                        "context_examples_count": len(seq.context_examples),
                        "query_features_length": len(seq.query_features),
                        "appears_in_training": seq.appears_in_training,
                        "generation_metadata": seq.generation_metadata,
                    }
                )

        with output_path.open("w") as f:
            json.dump(sequence_index, f, indent=2)


def load_validation_datasets_from_directory(validation_dir: Path) -> dict[int, Dataset]:
    """Load validation datasets from validation directory.

    Args:
        validation_dir: Path to validation directory containing seed_*/dataset/ subdirs

    Returns:
        Dictionary mapping seed -> Dataset

    """
    import re

    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

    validation_datasets = {}
    seed_pattern = re.compile(r"seed_(\d+)")

    for item in validation_dir.iterdir():
        if item.is_dir():
            match = seed_pattern.match(item.name)
            if match:
                seed = int(match.group(1))
                dataset_path = item / "dataset"

                if dataset_path.exists():
                    try:
                        dataset = Dataset.load_from_disk(str(dataset_path))
                        validation_datasets[seed] = dataset
                    except Exception as e:
                        print(f"Warning: Failed to load validation dataset for seed {seed}: {e}")

    if not validation_datasets:
        raise ValueError(f"No valid validation datasets found in {validation_dir}")

    return validation_datasets
