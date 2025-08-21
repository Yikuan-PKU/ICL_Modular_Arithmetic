import random
import typing as t
from pathlib import Path

from datasets import Dataset

from ICL.datasets.evaluation.eval_config import ICLEvalConfig
from ICL.datasets.evaluation.generators.base_generator import BaseICLGenerator, ICLSequence


class MemorizationGenerator(BaseICLGenerator):
    """Generates ICL sequences from training data (memorization evaluation)."""

    def __init__(self, config: ICLEvalConfig):
        """Initialize memorization generator."""
        super().__init__(config, eval_type="memorization")
        self.memorization_config = config.memorization

        # Track training data for sampling
        self.training_data_by_seed: dict[int, list[tuple[list[int], int]]] = {}
        self.training_data_loaded = False

    def load_training_data(self, train_datasets: dict[int, Dataset]) -> None:
        """Load training datasets for memorization sampling.

        Args:
            train_datasets: Dictionary mapping seed -> training Dataset

        """
        self.training_data_by_seed.clear()

        for seed, dataset in train_datasets.items():
            data_pairs = self._extract_features_labels_from_dataset(dataset)

            # Apply sampling ratio if configured
            if self.memorization_config.sampling_ratio < 1.0:
                sample_size = max(
                    self.memorization_config.min_sequences_per_seed,
                    int(len(data_pairs) * self.memorization_config.sampling_ratio),
                )
                data_pairs = random.sample(data_pairs, min(sample_size, len(data_pairs)))

            self.training_data_by_seed[seed] = data_pairs

        self.training_data_loaded = True

        # Log loading statistics
        total_sequences = sum(len(data) for data in self.training_data_by_seed.values())
        print(
            f"Loaded training data for memorization: {len(self.training_data_by_seed)} seeds, "
            f"{total_sequences} total sequences"
        )

    def generate_icl_sequences(
        self, source_config: tuple[int, int], target_config: tuple[int, int] | None = None, **kwargs
    ) -> list[ICLSequence]:
        """Generate memorization ICL sequences from training data.

        Args:
            source_config: Source (L, m) configuration
            target_config: Not used for memorization (always None)
            **kwargs: Additional arguments (unused)

        Returns:
            List of ICL sequences from training data

        """
        if not self.memorization_config.enable:
            return []

        if not self.training_data_loaded:
            raise RuntimeError("Training data not loaded. Call load_training_data() first.")

        if not self.training_data_by_seed:
            raise ValueError("No training data available for memorization")

        sequences = []

        # Generate sequences from each seed's training data
        for seed, training_data in self.training_data_by_seed.items():
            if len(training_data) < max(self.icl_params.context_sizes) + 1:
                print(
                    f"Warning: Seed {seed} has insufficient training data "
                    f"({len(training_data)} sequences) for largest context size "
                    f"({max(self.icl_params.context_sizes)})"
                )
                continue

            # Generate sequences for this seed
            seed_sequences = self._generate_sequences_for_context_sizes(
                available_data=training_data,
                source_config=source_config,
                target_config=None,  # Memorization doesn't use target config
                source_seeds=[seed],
                appears_in_training=True,  # By definition, these appear in training
                additional_metadata={
                    "training_seed": seed,
                    "training_data_size": len(training_data),
                    "sampling_ratio": self.memorization_config.sampling_ratio,
                },
            )

            sequences.extend(seed_sequences)

        # Store generated sequences
        self.generated_sequences.extend(sequences)

        # Validate coverage if required
        if self.memorization_config.ensure_coverage:
            self._validate_seed_coverage(sequences)

        print(
            f"Generated {len(sequences)} memorization sequences from {len(self.training_data_by_seed)} training seeds"
        )

        return sequences

    def _validate_seed_coverage(self, sequences: list[ICLSequence]) -> None:
        """Validate that all training seeds are represented in generated sequences."""
        represented_seeds = set()
        for seq in sequences:
            represented_seeds.update(seq.source_seeds)

        missing_seeds = set(self.training_data_by_seed.keys()) - represented_seeds

        if missing_seeds:
            print(f"Warning: Some training seeds not represented in memorization sequences: {sorted(missing_seeds)}")

        # Check minimum sequences per seed
        seed_counts = {}
        for seq in sequences:
            for seed in seq.source_seeds:
                seed_counts[seed] = seed_counts.get(seed, 0) + 1

        insufficient_seeds = [
            seed for seed, count in seed_counts.items() if count < self.memorization_config.min_sequences_per_seed
        ]

        if insufficient_seeds:
            print(
                f"Warning: Some seeds have insufficient memorization sequences "
                f"(< {self.memorization_config.min_sequences_per_seed}): {insufficient_seeds}"
            )

    def get_memorization_stats(self) -> dict[str, t.Any]:
        """Get detailed statistics about memorization generation."""
        base_stats = self.get_generation_stats()

        # Add memorization-specific stats
        if self.training_data_loaded:
            training_stats = {
                "training_seeds_loaded": len(self.training_data_by_seed),
                "training_sequences_per_seed": {seed: len(data) for seed, data in self.training_data_by_seed.items()},
                "total_training_sequences": sum(len(data) for data in self.training_data_by_seed.values()),
                "sampling_ratio_applied": self.memorization_config.sampling_ratio,
            }
            base_stats.update(training_stats)

        return base_stats

    def save_training_sequence_index(self, output_path: Path) -> None:
        """Save index of training sequences used for memorization.

        This creates a mapping that can be used to verify which specific
        training sequences were used in memorization evaluation.
        """
        import json

        if not self.generated_sequences:
            return

        sequence_index = {
            "memorization_sequences": [],
            "training_data_summary": {
                "seeds": list(self.training_data_by_seed.keys()),
                "sequences_per_seed": {str(seed): len(data) for seed, data in self.training_data_by_seed.items()},
                "sampling_ratio": self.memorization_config.sampling_ratio,
            },
        }

        for seq in self.generated_sequences:
            if seq.eval_type == "memorization":
                sequence_index["memorization_sequences"].append(
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


def load_training_datasets_from_directory(train_dir: Path) -> dict[int, Dataset]:
    """Load training datasets from train directory.

    Args:
        train_dir: Path to train directory containing seed_*/dataset/ subdirs

    Returns:
        Dictionary mapping seed -> Dataset

    """
    import re

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    train_datasets = {}
    seed_pattern = re.compile(r"seed_(\d+)")

    for item in train_dir.iterdir():
        if item.is_dir():
            match = seed_pattern.match(item.name)
            if match:
                seed = int(match.group(1))
                dataset_path = item / "dataset"

                if dataset_path.exists():
                    try:
                        dataset = Dataset.load_from_disk(str(dataset_path))
                        train_datasets[seed] = dataset
                    except Exception as e:
                        print(f"Warning: Failed to load training dataset for seed {seed}: {e}")

    if not train_datasets:
        raise ValueError(f"No valid training datasets found in {train_dir}")

    return train_datasets
