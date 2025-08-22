import pickle
import shutil
import typing as t
from pathlib import Path

from ICL.datasets.evaluation.eval_config import ICLEvalConfig


class EvalDirectoryManager:
    """Manages directory structure and file operations for ICL evaluation datasets."""

    def __init__(self, base_dataset_dir: Path):
        """Initialize with base dataset directory.

        Args:
            base_dataset_dir: Base directory like datasets/{dataset_type}_{num_seeds}_L{L}_M{m}/

        """
        self.base_dir = base_dataset_dir
        self.eval_dir = self.base_dir / "eval"

        # Individual evaluation type directories
        self.memorization_dir = self.eval_dir / "memorization"
        self.id_gen_dir = self.eval_dir / "id_generalization"
        self.ood_same_rule_dir = self.eval_dir / "ood_same_rule"
        self.ood_transfer_dir = self.eval_dir / "ood_transfer"

        # Required source directories
        self.train_dir = self.base_dir / "train"
        self.validation_dir = self.base_dir / "validation"

    def validate_source_directories(self) -> None:
        """Validate that required source directories exist."""
        missing_dirs = []

        if not self.train_dir.exists():
            missing_dirs.append(str(self.train_dir))

        if not self.validation_dir.exists():
            missing_dirs.append(str(self.validation_dir))

        if missing_dirs:
            raise FileNotFoundError(
                f"Required source directories not found: {missing_dirs}. Please run train/validation splitting first."
            )

        # Check for seed directories
        train_seeds = self._discover_seeds_in_directory(self.train_dir)
        val_seeds = self._discover_seeds_in_directory(self.validation_dir)

        if not train_seeds:
            raise FileNotFoundError(f"No seed directories found in {self.train_dir}")

        if not val_seeds:
            raise FileNotFoundError(f"No seed directories found in {self.validation_dir}")

        print(f"Found training seeds: {sorted(train_seeds)}")
        print(f"Found validation seeds: {sorted(val_seeds)}")

    def _discover_seeds_in_directory(self, directory: Path) -> list[int]:
        """Discover seed numbers from seed_* directories."""
        import re

        if not directory.exists():
            return []

        seed_pattern = re.compile(r"seed_(\d+)")
        seeds = []

        for item in directory.iterdir():
            if item.is_dir():
                match = seed_pattern.match(item.name)
                if match and (item / "dataset").exists():
                    seeds.append(int(match.group(1)))

        return seeds

    def create_eval_directories(self, enabled_types: list[str], overwrite: bool = False) -> None:
        """Create evaluation directories for enabled types.

        Args:
            enabled_types: List of evaluation types to create directories for
            overwrite: Whether to overwrite existing directories

        """
        # Create main eval directory
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Create type-specific directories
        type_dirs = {
            "memorization": self.memorization_dir,
            "id_generalization": self.id_gen_dir,
            "ood_same_rule": self.ood_same_rule_dir,
            "ood_transfer": self.ood_transfer_dir,
        }

        for eval_type in enabled_types:
            if eval_type not in type_dirs:
                continue

            type_dir = type_dirs[eval_type]

            if type_dir.exists() and not overwrite:
                print(f"Directory already exists: {type_dir} (use overwrite=True to replace)")
                continue

            if type_dir.exists() and overwrite:
                shutil.rmtree(type_dir)

            type_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created evaluation directory: {type_dir}")

    def save_evaluation_datasets(self, evaluation_data: dict[str, t.Any], config: ICLEvalConfig) -> None:
        """Save evaluation datasets to appropriate directories.

        Args:
            evaluation_data: Dictionary containing datasets and metadata
            config: ICL evaluation configuration

        """
        enabled_types = config.get_enabled_types()

        # Save individual type datasets
        individual_types = evaluation_data.get("individual_types", {})

        for eval_type in enabled_types:
            if eval_type not in individual_types:
                continue

            dataset = individual_types[eval_type]
            if len(dataset) == 0:
                print(f"Skipping empty dataset for {eval_type}")
                continue

            # Get type directory
            type_dir = getattr(self, f"{eval_type.replace('_', '_')}_dir", None)
            if type_dir is None:
                if eval_type == "id_generalization":
                    type_dir = self.id_gen_dir
                elif eval_type == "ood_same_rule":
                    type_dir = self.ood_same_rule_dir
                elif eval_type == "ood_transfer":
                    type_dir = self.ood_transfer_dir
                else:
                    type_dir = self.eval_dir / eval_type

            type_dir.mkdir(parents=True, exist_ok=True)

            # Save dataset
            dataset.save_to_disk(str(type_dir / "dataset"))
            print(f"Saved {eval_type} dataset: {len(dataset)} sequences")

        # Save combined dataset if it exists
        combined_dataset = evaluation_data.get("combined")
        if combined_dataset and len(combined_dataset) > 0:
            combined_dir = self.eval_dir / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined_dataset.save_to_disk(str(combined_dir / "dataset"))
            print(f"Saved combined dataset: {len(combined_dataset)} sequences")

        # Save metadata
        metadata = evaluation_data.get("metadata", {})
        if metadata:
            with (self.eval_dir / "eval_metadata.pkl").open("wb") as f:
                pickle.dump(metadata, f)
            print("Saved evaluation metadata as pickle")
            print("Saved evaluation metadata")

    def create_summary_files(self, evaluation_data: dict[str, t.Any]) -> None:
        """Create human-readable summary files."""
        metadata = evaluation_data.get("metadata", {})

        # Main summary
        summary_lines = ["ICL EVALUATION DATASET SUMMARY", "=" * 50, ""]

        # Configuration info
        gen_info = metadata.get("generation_info", {})
        source_config = gen_info.get("source_config", {})
        summary_lines.extend(
            [
                f"Source Configuration: L={source_config.get('L', '?')}, m={source_config.get('m', '?')}",
                f"Generated at: {gen_info.get('created_at', 'unknown')}",
                "",
            ]
        )

        # Type statistics
        type_stats = metadata.get("type_statistics", {})
        dataset_summary = metadata.get("dataset_summary", {})

        summary_lines.extend(
            [
                "Evaluation Types Generated:",
                f"  Total sequences: {dataset_summary.get('total_sequences', 0)}",
                f"  Context sizes tested: {dataset_summary.get('context_sizes_tested', [])}",
                "",
            ]
        )

        for eval_type, count in type_stats.items():
            summary_lines.append(f"  {eval_type}: {count} sequences")

        summary_lines.extend(["", "Seed Allocation:"])
        seed_allocation = metadata.get("seed_allocation", {})
        for seed_type, info in seed_allocation.items():
            if isinstance(info, dict) and "count" in info:
                summary_lines.append(f"  {seed_type}: {info['count']} seeds")

        # Write main summary
        with (self.eval_dir / "eval_summary.txt").open("w") as f:
            f.write("\n".join(summary_lines))

        # Create type-specific summaries
        generator_details = metadata.get("generator_details", {})

        for eval_type, details in generator_details.items():
            type_dir = self.eval_dir / eval_type
            if type_dir.exists():
                self._create_type_summary(type_dir, eval_type, details)

    def _create_type_summary(self, type_dir: Path, eval_type: str, details: dict[str, t.Any]) -> None:
        """Create summary file for specific evaluation type."""
        summary_lines = [f"{eval_type.upper().replace('_', ' ')} EVALUATION", "=" * 40, ""]

        # Basic stats
        total_seqs = details.get("total_sequences", 0)
        context_dist = details.get("context_size_distribution", {})

        summary_lines.extend([f"Total sequences: {total_seqs}", f"Context size distribution: {context_dist}", ""])

        # Type-specific information
        if eval_type == "memorization":
            train_info = details.get("training_sequences_per_seed", {})
            summary_lines.extend(
                [
                    "Training Data Used:",
                    f"  Seeds: {len(train_info)}",
                    f"  Sequences per seed: {dict(list(train_info.items())[:3])}{'...' if len(train_info) > 3 else ''}",
                    "",
                ]
            )

        elif eval_type == "ood_transfer":
            transfer_matrix = details.get("transfer_matrix", {})
            summary_lines.extend(
                [
                    "Transfer Experiments:",
                    f"  Transfer types: {details.get('transfer_types_enabled', [])}",
                    f"  Max transfer distance: {details.get('max_transfer_distance', '?')}",
                    f"  Experiments conducted: {len(transfer_matrix)}",
                    "",
                ]
            )

        with (type_dir / "type_summary.txt").open("w") as f:
            f.write("\n".join(summary_lines))

    def get_directory_structure(self) -> dict[str, Path]:
        """Get all relevant directory paths."""
        return {
            "base": self.base_dir,
            "eval": self.eval_dir,
            "train": self.train_dir,
            "validation": self.validation_dir,
            "memorization": self.memorization_dir,
            "id_generalization": self.id_gen_dir,
            "ood_same_rule": self.ood_same_rule_dir,
            "ood_transfer": self.ood_transfer_dir,
        }

    def check_eval_exists(self) -> dict[str, bool]:
        """Check which evaluation types already exist."""
        return {
            "memorization": (self.memorization_dir / "dataset").exists(),
            "id_generalization": (self.id_gen_dir / "dataset").exists(),
            "ood_same_rule": (self.ood_same_rule_dir / "dataset").exists(),
            "ood_transfer": (self.ood_transfer_dir / "dataset").exists(),
            "combined": (self.eval_dir / "combined" / "dataset").exists(),
        }

    def discover_train_seeds(self) -> list[int]:
        """Discover training seeds from train directory."""
        return self._discover_seeds_in_directory(self.train_dir)


def check_evaluation_prerequisites(base_dir: Path) -> dict[str, t.Any]:
    """Check if prerequisites for evaluation generation exist.

    Args:
        base_dir: Base dataset directory

    Returns:
        Dictionary with prerequisite check results

    """
    results = {
        "base_dir_exists": base_dir.exists(),
        "train_dir_exists": (base_dir / "train").exists(),
        "validation_dir_exists": (base_dir / "validation").exists(),
        "split_info_exists": (base_dir / "split_info.json").exists(),
        "train_seeds": [],
        "validation_seeds": [],
        "ready_for_evaluation": False,
    }

    if results["base_dir_exists"]:
        dir_manager = EvalDirectoryManager(base_dir)

        if results["train_dir_exists"]:
            results["train_seeds"] = dir_manager._discover_seeds_in_directory(base_dir / "train")

        if results["validation_dir_exists"]:
            results["validation_seeds"] = dir_manager._discover_seeds_in_directory(base_dir / "validation")

    # Check if ready for evaluation
    results["ready_for_evaluation"] = (
        results["base_dir_exists"]
        and results["train_dir_exists"]
        and results["validation_dir_exists"]
        and len(results["train_seeds"]) > 0
        and len(results["validation_seeds"]) > 0
    )

    return results


def load_generation_params_from_config(config_dir: Path) -> dict[str, t.Any]:
    """Load RHM generation parameters from config directory.

    Args:
        config_dir: Path to configuration directory containing generate_eval.yaml

    Returns:
        Dictionary with RHM generation parameters

    """
    import yaml

    config_file = config_dir / "generate_eval.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with config_file.open("r") as f:
        config_data = yaml.safe_load(f)

    # Extract RHM parameters
    rhm_params = config_data.get("rhm_params", {})
    distribution = config_data.get("distribution", {})

    return {
        "vocab_size": rhm_params.get("vocab_size", 32),
        "num_classes": rhm_params.get("num_classes", 10),
        "tuple_size": rhm_params.get("tuple_size", 2),
        "samples_per_config": rhm_params.get("samples_per_config", 1000),
        "distribution_type": distribution.get("type", "uniform"),
        "zipf_alpha": distribution.get("zipf_alpha", 1.0),
    }
