import typing as t
from pathlib import Path

from datasets import Dataset

from ICL.datasets.evaluation.eval_config import ICLEvalConfig
from ICL.datasets.evaluation.generators.base_generator import BaseICLGenerator, ICLSequence
from ICL.datasets.RHM import RandomHierarchyModel


class OODTransferGenerator(BaseICLGenerator):
    """Generates ICL sequences for out-of-distribution transfer evaluation.

    Uses different seeds AND different (L,m) configurations from training.
    """

    def __init__(self, config: ICLEvalConfig):
        """Initialize OOD transfer generator."""
        super().__init__(config, eval_type="ood_transfer")
        self.transfer_config = config.ood_transfer

        # Track generated transfer datasets
        self.transfer_datasets: dict[tuple[tuple[int, int], tuple[int, int], int], list[tuple[list[int], int]]] = {}
        # Key format: ((source_L, source_m), (target_L, target_m), seed) -> data_pairs

        self.generation_params: dict[str, t.Any] = {}

    def generate_transfer_configurations(self, source_config: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
        """Generate target configurations for transfer evaluation.

        Args:
            source_config: Source (L, m) configuration from training

        Returns:
            List of (transfer_type, target_config) tuples

        """
        source_L, source_m = source_config
        transfer_configs = []

        # Depth transfer: increase L, keep m same
        if "depth" in self.transfer_config.transfer_types:
            for step in range(1, self.transfer_config.max_transfer_distance + 1):
                target_L = source_L + step
                target_config = (target_L, source_m)
                transfer_configs.append(("depth", target_config))

        # Synonym transfer: keep L same, increase m
        if "synonym" in self.transfer_config.transfer_types:
            for step in range(1, self.transfer_config.max_transfer_distance + 1):
                target_m = source_m + step
                target_config = (source_L, target_m)
                transfer_configs.append(("synonym", target_config))

        # Full transfer: increase both L and m
        if "full" in self.transfer_config.transfer_types:
            for l_step in range(1, self.transfer_config.max_transfer_distance + 1):
                for m_step in range(1, self.transfer_config.max_transfer_distance + 1):
                    target_L = source_L + l_step
                    target_m = source_m + m_step
                    target_config = (target_L, target_m)
                    transfer_configs.append(("full", target_config))

        return transfer_configs

    def generate_icl_sequences(
        self,
        source_config: tuple[int, int],
        target_config: tuple[int, int] | None = None,
        transfer_seeds: list[int] | None = None,
        generation_params: dict[str, t.Any] | None = None,
        **kwargs,
    ) -> list[ICLSequence]:
        """Generate OOD transfer ICL sequences.

        Args:
            source_config: Source (L, m) configuration from training
            target_config: Target (L, m) configuration for transfer
            transfer_seeds: List of seeds to use for transfer generation
            generation_params: RHM generation parameters
            **kwargs: Additional arguments

        Returns:
            List of ICL sequences for transfer evaluation

        """
        if not self.transfer_config.enable:
            return []

        if target_config is None:
            # Generate for all transfer configurations
            return self._generate_all_transfer_sequences(source_config, transfer_seeds, generation_params)

        if transfer_seeds is None:
            raise ValueError("transfer_seeds must be provided for OOD transfer generation")

        if generation_params is None:
            raise ValueError("generation_params must be provided for RHM generation")

        self.generation_params = generation_params
        target_L, target_m = target_config

        # Determine transfer type
        source_L, source_m = source_config
        if target_L > source_L and target_m == source_m:
            transfer_type = "depth"
        elif target_L == source_L and target_m > source_m:
            transfer_type = "synonym"
        elif target_L > source_L and target_m > source_m:
            transfer_type = "full"
        else:
            raise ValueError(f"Invalid transfer from {source_config} to {target_config}")

        # Generate datasets for each transfer seed
        for seed in transfer_seeds:
            dataset_key = (source_config, target_config, seed)
            if dataset_key not in self.transfer_datasets:
                self._generate_transfer_dataset_for_seed(seed, target_L, target_m, generation_params, dataset_key)

        # Generate ICL sequences from transfer datasets
        sequences = []

        for seed in transfer_seeds:
            dataset_key = (source_config, target_config, seed)

            if dataset_key not in self.transfer_datasets:
                print(f"Warning: Failed to generate transfer dataset for seed {seed}")
                continue

            transfer_data = self.transfer_datasets[dataset_key]

            if len(transfer_data) < max(self.icl_params.context_sizes) + 1:
                print(
                    f"Warning: Transfer seed {seed} has insufficient data "
                    f"({len(transfer_data)} sequences) for largest context size"
                )
                continue

            # Generate sequences for this transfer seed
            seed_sequences = self._generate_sequences_for_context_sizes(
                available_data=transfer_data,
                source_config=source_config,
                target_config=target_config,
                source_seeds=[seed],
                appears_in_training=False,  # By definition, transfer data not in training
                additional_metadata={
                    "transfer_seed": seed,
                    "transfer_type": transfer_type,
                    "transfer_data_size": len(transfer_data),
                    "source_L": source_L,
                    "source_m": source_m,
                    "target_L": target_L,
                    "target_m": target_m,
                    "L_increase": target_L - source_L,
                    "m_increase": target_m - source_m,
                    "generation_params": generation_params,
                },
            )

            sequences.extend(seed_sequences)

        # Store generated sequences
        self.generated_sequences.extend(sequences)

        print(
            f"Generated {len(sequences)} OOD transfer sequences "
            f"({transfer_type}) from {source_config} to {target_config} "
            f"using {len(transfer_seeds)} seeds"
        )

        return sequences

    def _generate_all_transfer_sequences(
        self,
        source_config: tuple[int, int],
        transfer_seeds: list[int] | None,
        generation_params: dict[str, t.Any] | None,
    ) -> list[ICLSequence]:
        """Generate sequences for all transfer configurations."""
        if transfer_seeds is None or generation_params is None:
            raise ValueError("transfer_seeds and generation_params required")

        # Get all transfer configurations
        transfer_configs = self.generate_transfer_configurations(source_config)

        all_sequences = []

        for transfer_type, target_config in transfer_configs:
            sequences = self.generate_icl_sequences(
                source_config=source_config,
                target_config=target_config,
                transfer_seeds=transfer_seeds,
                generation_params=generation_params,
            )
            all_sequences.extend(sequences)

        return all_sequences

    def _generate_transfer_dataset_for_seed(
        self,
        seed: int,
        target_L: int,
        target_m: int,
        generation_params: dict[str, t.Any],
        dataset_key: tuple[tuple[int, int], tuple[int, int], int],
    ) -> None:
        """Generate transfer dataset for specific seed and target configuration."""
        try:
            # Create RHM with transfer configuration
            rhm = RandomHierarchyModel(
                num_features=generation_params.get("vocab_size", 32),
                num_classes=generation_params.get("num_classes", 10),
                num_synonyms=target_m,
                tuple_size=generation_params.get("tuple_size", 2),
                num_layers=target_L,
                seed_rules=seed,
                seed_sample=seed + 1,
                train_size=self.transfer_config.sequences_per_seed,
                replacement=True,
                input_format="long",
            )

            # Extract data pairs
            sequences = rhm.features
            labels = rhm.labels

            # Convert to list format
            if hasattr(sequences, "tolist"):
                sequences_list = sequences.tolist()
            else:
                sequences_list = [list(seq) for seq in sequences]

            if hasattr(labels, "tolist"):
                labels_list = labels.tolist()
            else:
                labels_list = list(labels)

            # Create (features, label) pairs
            data_pairs = list(zip(sequences_list, labels_list, strict=True))

            # Store the generated data
            self.transfer_datasets[dataset_key] = data_pairs

            source_config, target_config, _ = dataset_key
            print(
                f"  Generated transfer dataset for seed {seed}: "
                f"{source_config} -> {target_config}, {len(data_pairs)} sequences"
            )

        except Exception as e:
            print(f"  Failed to generate transfer dataset for seed {seed}: {e}")

    def get_transfer_matrix(self) -> dict[str, t.Any]:
        """Get matrix of all transfer experiments conducted."""
        transfer_matrix = {}

        for (source_config, target_config, seed), data in self.transfer_datasets.items():
            source_key = f"L{source_config[0]}_M{source_config[1]}"
            target_key = f"L{target_config[0]}_M{target_config[1]}"

            if source_key not in transfer_matrix:
                transfer_matrix[source_key] = {}

            if target_key not in transfer_matrix[source_key]:
                transfer_matrix[source_key][target_key] = {
                    "seeds": [],
                    "total_sequences": 0,
                    "transfer_type": self._classify_transfer_type(source_config, target_config),
                }

            transfer_matrix[source_key][target_key]["seeds"].append(seed)
            transfer_matrix[source_key][target_key]["total_sequences"] += len(data)

        return transfer_matrix

    def _classify_transfer_type(self, source_config: tuple[int, int], target_config: tuple[int, int]) -> str:
        """Classify the type of transfer between configurations."""
        source_L, source_m = source_config
        target_L, target_m = target_config

        if target_L > source_L and target_m == source_m:
            return "depth"
        if target_L == source_L and target_m > source_m:
            return "synonym"
        if target_L > source_L and target_m > source_m:
            return "full"
        return "unknown"

    def get_ood_transfer_stats(self) -> dict[str, t.Any]:
        """Get detailed statistics about OOD transfer generation."""
        base_stats = self.get_generation_stats()

        # Add transfer-specific stats
        transfer_stats = {
            "transfer_experiments": len(self.transfer_datasets),
            "transfer_matrix": self.get_transfer_matrix(),
            "total_transfer_sequences": sum(len(data) for data in self.transfer_datasets.values()),
            "target_sequences_per_seed": self.transfer_config.sequences_per_seed,
            "transfer_types_enabled": self.transfer_config.transfer_types,
            "max_transfer_distance": self.transfer_config.max_transfer_distance,
            "generation_params_used": self.generation_params,
        }
        base_stats.update(transfer_stats)

        return base_stats

    def save_transfer_datasets(self, output_dir: Path) -> None:
        """Save generated transfer datasets organized by transfer type."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Organize by transfer experiments
        for (source_config, target_config, seed), data_pairs in self.transfer_datasets.items():
            # Create directory structure: from_L*_M*/to_L*_M*/seed_*
            source_dir = f"from_L{source_config[0]}_M{source_config[1]}"
            target_dir = f"to_L{target_config[0]}_M{target_config[1]}"

            transfer_dir = output_dir / source_dir / target_dir / f"seed_{seed}"
            transfer_dir.mkdir(parents=True, exist_ok=True)

            # Convert to HuggingFace Dataset format
            dataset_dict = {
                "input_ids": [pair[0] for pair in data_pairs],
                "labels": [pair[1] for pair in data_pairs],
                "length": [len(pair[0]) for pair in data_pairs],
            }

            dataset = Dataset.from_dict(dataset_dict)
            dataset.save_to_disk(str(transfer_dir / "dataset"))

        # Save transfer matrix and metadata
        import json

        metadata = {
            "ood_transfer_generation": {
                "transfer_matrix": self.get_transfer_matrix(),
                "total_experiments": len(self.transfer_datasets),
                "generation_params": self.generation_params,
                "config_used": {
                    "transfer_types": self.transfer_config.transfer_types,
                    "max_transfer_distance": self.transfer_config.max_transfer_distance,
                    "sequences_per_seed": self.transfer_config.sequences_per_seed,
                },
            }
        }

        with (output_dir / "transfer_matrix.json").open("w") as f:
            json.dump(metadata, f, indent=2)


def generate_transfer_seeds(base_seed: int, count: int, avoid_seeds: list[int]) -> list[int]:
    """Generate seeds for transfer evaluation.

    Args:
        base_seed: Base seed for generation
        count: Number of seeds to generate
        avoid_seeds: Seeds to avoid (training + OOD same rule seeds)

    Returns:
        List of transfer seeds

    """
    avoid_set = set(avoid_seeds)
    transfer_seeds = []

    current_seed = base_seed
    max_attempts = count * 100
    attempts = 0

    while len(transfer_seeds) < count and attempts < max_attempts:
        if current_seed not in avoid_set:
            transfer_seeds.append(current_seed)
            avoid_set.add(current_seed)

        current_seed += 1
        attempts += 1

    if len(transfer_seeds) < count:
        raise RuntimeError(f"Failed to generate {count} transfer seeds. Only generated {len(transfer_seeds)}")

    return sorted(transfer_seeds)
