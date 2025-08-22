import typing as t
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from ICL.datasets.evaluation.eval_config import ICLEvalConfig
from ICL.datasets.evaluation.generators.base_generator import ICLSequence
from ICL.datasets.evaluation.generators.id_generalization import (
    IDGeneralizationGenerator,
    load_validation_datasets_from_directory,
)
from ICL.datasets.evaluation.generators.memorization import MemorizationGenerator, load_training_datasets_from_directory
from ICL.datasets.evaluation.generators.ood_same_rule import OODSameRuleGenerator
from ICL.datasets.evaluation.generators.ood_transfer import OODTransferGenerator
from ICL.datasets.evaluation.seed_manager import EvalSeedManager


class ICLEvaluationBuilder:
    """Orchestrates generation of all four ICL evaluation types."""

    def __init__(self, config: ICLEvalConfig, train_seeds: list[int]):
        """Initialize evaluation builder.

        Args:
            config: ICL evaluation configuration
            train_seeds: Seeds used in training dataset generation

        """
        self.config = config
        self.seed_manager = EvalSeedManager(config, train_seeds)

        # Initialize generators
        self.generators = {
            "memorization": MemorizationGenerator(config),
            "id_generalization": IDGeneralizationGenerator(config),
            "ood_same_rule": OODSameRuleGenerator(config),
            "ood_transfer": OODTransferGenerator(config),
        }

        # Track generated datasets
        self.evaluation_datasets: dict[str, dict[str, t.Any]] = {}
        self.generation_metadata: dict[str, t.Any] = {}

    def generate_complete_evaluation_dataset(
        self,
        source_config: tuple[int, int],
        train_dir: Path,
        validation_dir: Path,
        generation_params: dict[str, t.Any],
        output_dir: Path | None = None,
    ) -> dict[str, t.Any]:
        """Generate complete ICL evaluation dataset with all four types.

        Args:
            source_config: Source (L, m) configuration from training
            train_dir: Directory containing training datasets
            validation_dir: Directory containing validation datasets
            generation_params: RHM generation parameters
            output_dir: Optional output directory for intermediate files

        Returns:
            Dictionary containing all evaluation datasets and metadata

        """
        print("=" * 80)
        print(f"GENERATING ICL EVALUATION DATASET FOR L={source_config[0]}, M={source_config[1]}")
        print("=" * 80)

        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Generate each evaluation type
        enabled_types = self.config.get_enabled_types()
        print(f"Enabled evaluation types: {enabled_types}")
        print(f"Seed allocation: {self.seed_manager.get_allocation_summary()}")
        print()

        all_sequences = []
        type_stats = {}

        # Type 1: Memorization
        if "memorization" in enabled_types:
            print("Generating Type 1: Memorization")
            sequences = self._generate_memorization(source_config, train_dir, output_dir)
            all_sequences.extend(sequences)
            type_stats["memorization"] = len(sequences)
            print(f"✓ Generated {len(sequences)} memorization sequences\n")

        # Type 2: ID Generalization
        if "id_generalization" in enabled_types:
            print("Generating Type 2: In-Distribution Generalization")
            sequences = self._generate_id_generalization(source_config, validation_dir, output_dir)
            all_sequences.extend(sequences)
            type_stats["id_generalization"] = len(sequences)
            print(f"✓ Generated {len(sequences)} ID generalization sequences\n")

        # Type 3: OOD Same Rule
        if "ood_same_rule" in enabled_types:
            print("Generating Type 3: Out-of-Distribution Same Rule")
            sequences = self._generate_ood_same_rule(source_config, generation_params, output_dir)
            all_sequences.extend(sequences)
            type_stats["ood_same_rule"] = len(sequences)
            print(f"✓ Generated {len(sequences)} OOD same rule sequences\n")

        # Type 4: OOD Transfer
        if "ood_transfer" in enabled_types:
            print("Generating Type 4: Out-of-Distribution Transfer")
            sequences = self._generate_ood_transfer(source_config, generation_params, output_dir)
            all_sequences.extend(sequences)
            type_stats["ood_transfer"] = len(sequences)
            print(f"✓ Generated {len(sequences)} OOD transfer sequences\n")

        # Create combined dataset
        combined_dataset = None
        if self.config.create_combined_dataset and all_sequences:
            combined_dataset = self._create_combined_dataset(all_sequences)
            print(f"✓ Created combined dataset with {len(all_sequences)} total sequences")

        # Generate comprehensive metadata
        complete_metadata = self._create_complete_metadata(source_config, type_stats, generation_params)

        # Store results
        self.evaluation_datasets[f"L{source_config[0]}_M{source_config[1]}"] = {
            "individual_types": {
                eval_type: self._sequences_to_dataset([seq for seq in all_sequences if seq.eval_type == eval_type])
                for eval_type in enabled_types
            },
            "combined": combined_dataset,
            "metadata": complete_metadata,
        }

        print("=" * 80)
        print("EVALUATION DATASET GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Type distribution: {type_stats}")
        print("=" * 80)

        return self.evaluation_datasets[f"L{source_config[0]}_M{source_config[1]}"]

    def _generate_memorization(
        self, source_config: tuple[int, int], train_dir: Path, output_dir: Path | None
    ) -> list[ICLSequence]:
        """Generate memorization sequences."""
        generator = self.generators["memorization"]

        # Load training datasets
        train_datasets = load_training_datasets_from_directory(train_dir)
        generator.load_training_data(train_datasets)

        # Generate sequences
        sequences = generator.generate_icl_sequences(source_config)

        # Save intermediate results if requested
        if output_dir and self.config.save_intermediate:
            memorization_dir = output_dir / "memorization"
            memorization_dir.mkdir(parents=True, exist_ok=True)
            generator.save_training_sequence_index(memorization_dir / "sequence_index.json")

            # Save as separate dataset
            if sequences:
                dataset = self._sequences_to_dataset(sequences)
                dataset.save_to_disk(str(memorization_dir / "dataset"))

        return sequences

    def _generate_id_generalization(
        self, source_config: tuple[int, int], validation_dir: Path, output_dir: Path | None
    ) -> list[ICLSequence]:
        """Generate ID generalization sequences."""
        generator = self.generators["id_generalization"]

        # Load validation datasets
        validation_datasets = load_validation_datasets_from_directory(validation_dir)
        generator.load_validation_data(validation_datasets)

        # Generate sequences
        sequences = generator.generate_icl_sequences(source_config)

        # Save intermediate results if requested
        if output_dir and self.config.save_intermediate:
            id_gen_dir = output_dir / "id_generalization"
            id_gen_dir.mkdir(parents=True, exist_ok=True)
            generator.save_validation_sequence_index(id_gen_dir / "sequence_index.json")

            # Save seed comparison
            train_seeds = self.seed_manager.allocation.train_seeds
            comparison = generator.compare_with_training_seeds(train_seeds)

            import json

            with (id_gen_dir / "seed_comparison.json").open("w") as f:
                json.dump(comparison, f, indent=2)

            # Save as separate dataset
            if sequences:
                dataset = self._sequences_to_dataset(sequences)
                dataset.save_to_disk(str(id_gen_dir / "dataset"))

        return sequences

    def _generate_ood_same_rule(
        self, source_config: tuple[int, int], generation_params: dict[str, t.Any], output_dir: Path | None
    ) -> list[ICLSequence]:
        """Generate OOD same rule sequences."""
        generator = self.generators["ood_same_rule"]

        # Get OOD seeds
        ood_seeds = self.seed_manager.get_seeds_for_type("ood_same_rule")

        # Generate sequences
        sequences = generator.generate_icl_sequences(
            source_config=source_config, ood_seeds=ood_seeds, generation_params=generation_params
        )

        # Save intermediate results if requested
        if output_dir and self.config.save_intermediate:
            ood_same_dir = output_dir / "ood_same_rule"
            ood_same_dir.mkdir(parents=True, exist_ok=True)

            # Save generated OOD datasets
            generator.save_ood_datasets(ood_same_dir / "ood_datasets")

            # Save as separate dataset
            if sequences:
                dataset = self._sequences_to_dataset(sequences)
                dataset.save_to_disk(str(ood_same_dir / "dataset"))

        return sequences

    def _generate_ood_transfer(
        self, source_config: tuple[int, int], generation_params: dict[str, t.Any], output_dir: Path | None
    ) -> list[ICLSequence]:
        """Generate OOD transfer sequences."""
        generator = self.generators["ood_transfer"]

        # Get transfer seeds
        transfer_seeds = self.seed_manager.get_seeds_for_type("ood_transfer")

        # Generate sequences for all transfer configurations
        sequences = generator.generate_icl_sequences(
            source_config=source_config,
            target_config=None,  # This will generate all transfer configs
            transfer_seeds=transfer_seeds,
            generation_params=generation_params,
        )

        # Save intermediate results if requested
        if output_dir and self.config.save_intermediate:
            ood_transfer_dir = output_dir / "ood_transfer"
            ood_transfer_dir.mkdir(parents=True, exist_ok=True)

            # Save transfer datasets organized by transfer type
            generator.save_transfer_datasets(ood_transfer_dir / "transfer_datasets")

            # Save as separate dataset
            if sequences:
                dataset = self._sequences_to_dataset(sequences)
                dataset.save_to_disk(str(ood_transfer_dir / "dataset"))

        return sequences

    def _sequences_to_dataset(self, sequences: list[ICLSequence]) -> Dataset:
        """Convert ICL sequences to HuggingFace Dataset."""
        if not sequences:
            return Dataset.from_dict({})

        # Convert all sequences to dictionaries
        dataset_dict = defaultdict(list)

        for seq in sequences:
            seq_dict = seq.to_dict()
            for key, value in seq_dict.items():
                dataset_dict[key].append(value)

        return Dataset.from_dict(dict(dataset_dict))

    def _create_combined_dataset(self, all_sequences: list[ICLSequence]) -> Dataset:
        """Create combined dataset from all sequences."""
        return self._sequences_to_dataset(all_sequences)

    def _create_complete_metadata(
        self, source_config: tuple[int, int], type_stats: dict[str, int], generation_params: dict[str, t.Any]
    ) -> dict[str, t.Any]:
        """Create comprehensive metadata for the evaluation dataset."""
        from datetime import datetime

        # Collect stats from all generators
        generator_stats = {}
        for eval_type, generator in self.generators.items():
            if eval_type in self.config.get_enabled_types():
                if eval_type == "memorization":
                    generator_stats[eval_type] = generator.get_memorization_stats()
                elif eval_type == "id_generalization":
                    generator_stats[eval_type] = generator.get_id_generalization_stats()
                elif eval_type == "ood_same_rule":
                    generator_stats[eval_type] = generator.get_ood_same_rule_stats()
                elif eval_type == "ood_transfer":
                    generator_stats[eval_type] = generator.get_ood_transfer_stats()

        return {
            "generation_info": {
                "source_config": {"L": source_config[0], "m": source_config[1]},
                "generation_params": generation_params,
                "config_used": {
                    "icl_params": self.config.icl_params.__dict__,
                    "enabled_types": self.config.get_enabled_types(),
                    "base_seed": self.config.base_seed,
                },
                "created_at": datetime.now().isoformat(),
            },
            "seed_allocation": self.seed_manager.get_allocation_summary(),
            "type_statistics": type_stats,
            "generator_details": generator_stats,
            "dataset_summary": {
                "total_sequences": sum(type_stats.values()),
                "types_generated": len(type_stats),
                "context_sizes_tested": self.config.icl_params.context_sizes,
                "sequences_per_context_size": self.config.icl_params.sequences_per_context_size,
            },
        }

    def get_evaluation_summary(self) -> dict[str, t.Any]:
        """Get summary of all generated evaluation datasets."""
        summary = {
            "configurations_generated": list(self.evaluation_datasets.keys()),
            "total_configurations": len(self.evaluation_datasets),
            "enabled_types": self.config.get_enabled_types(),
            "seed_allocation": self.seed_manager.get_allocation_summary(),
        }

        # Add per-configuration stats
        config_stats = {}
        for config_name, config_data in self.evaluation_datasets.items():
            metadata = config_data["metadata"]
            config_stats[config_name] = {
                "total_sequences": metadata["dataset_summary"]["total_sequences"],
                "type_distribution": metadata["type_statistics"],
                "source_config": metadata["generation_info"]["source_config"],
            }

        summary["configuration_details"] = config_stats
        return summary

    def save_complete_evaluation_dataset(self, output_dir: Path, config_name: str | None = None) -> None:
        """Save complete evaluation dataset to directory.

        Args:
            output_dir: Directory to save evaluation datasets
            config_name: Specific configuration to save (None for all)

        """
        output_dir.mkdir(parents=True, exist_ok=True)

        configs_to_save = [config_name] if config_name else list(self.evaluation_datasets.keys())

        for config_key in configs_to_save:
            if config_key not in self.evaluation_datasets:
                print(f"Warning: Configuration {config_key} not found")
                continue

            config_data = self.evaluation_datasets[config_key]
            config_dir = output_dir / config_key
            config_dir.mkdir(parents=True, exist_ok=True)

            # Save individual type datasets
            for eval_type, dataset in config_data["individual_types"].items():
                if len(dataset) > 0:  # Only save non-empty datasets
                    type_dir = config_dir / eval_type
                    type_dir.mkdir(parents=True, exist_ok=True)
                    dataset.save_to_disk(str(type_dir / "dataset"))

            # Save combined dataset
            if config_data["combined"] and len(config_data["combined"]) > 0:
                config_data["combined"].save_to_disk(str(config_dir / "combined" / "dataset"))

            # Save metadata
            import json

            with (config_dir / "eval_metadata.json").open("w") as f:
                json.dump(config_data["metadata"], f, indent=2)

        # Save overall summary
        summary = self.get_evaluation_summary()
        with (output_dir / "evaluation_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        # Save seed allocation
        self.seed_manager.save_allocation(output_dir / "seed_allocation.json")
