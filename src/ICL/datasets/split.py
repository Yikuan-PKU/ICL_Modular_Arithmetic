import json
import pickle
import shutil
import typing as t
from pathlib import Path

from datasets import Dataset

from ICL.datasets.utils import SplitMetadata, save_split_metadata


class SplitDirectoryManager:
    """Manages directory structure and file operations for dataset splits."""

    def __init__(self, base_dataset_dir: Path):
        """Initialize with base dataset directory.

        Args:
            base_dataset_dir: Base directory like datasets/{dataset_type}_{num_seeds}_L{L}_M{m}/

        """
        self.base_dir = base_dataset_dir
        self.raw_dir = self.base_dir / "raw"
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "validation"

    def validate_raw_directory(self) -> None:
        """Validate that raw directory exists and contains seed datasets."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {self.raw_dir}")

        seed_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found in {self.raw_dir}")

        # Validate at least one seed has a dataset
        valid_seeds = []
        for seed_dir in seed_dirs:
            if (seed_dir / "dataset").exists():
                valid_seeds.append(seed_dir.name)

        if not valid_seeds:
            raise FileNotFoundError(f"No valid seed datasets found in {self.raw_dir}")

    def create_split_directories(self, overwrite: bool = False) -> None:
        """Create train and validation directories.

        Args:
            overwrite: Whether to overwrite existing directories

        """
        for split_dir in [self.train_dir, self.val_dir]:
            if split_dir.exists():
                if overwrite:
                    shutil.rmtree(split_dir)
                else:
                    raise FileExistsError(
                        f"Split directory already exists: {split_dir}. Use overwrite=True to replace."
                    )

            split_dir.mkdir(parents=True, exist_ok=True)

    def save_seed_datasets(self, datasets: dict[int, Dataset], split_type: str) -> None:
        """Save seed datasets to appropriate split directory.

        Args:
            datasets: Dictionary mapping seed -> Dataset
            split_type: Either "train" or "validation"

        """
        if split_type == "train":
            target_dir = self.train_dir
        elif split_type == "validation":
            target_dir = self.val_dir
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        for seed, dataset in datasets.items():
            seed_dir = target_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = seed_dir / "dataset"
            dataset.save_to_disk(str(dataset_path))

    def save_split_metadata_files(
        self, metadata: SplitMetadata, train_datasets: dict[int, Dataset], val_datasets: dict[int, Dataset]
    ) -> None:
        """Save metadata files for both train and validation splits."""
        # Save main split metadata
        save_split_metadata(metadata, self.base_dir / "split_info.json")

        # Create and save train metadata
        train_metadata = self._create_split_specific_metadata(metadata, train_datasets, "train")
        self._save_pickle_metadata(train_metadata, self.train_dir / "metadata.pkl")

        # Create and save validation metadata
        val_metadata = self._create_split_specific_metadata(metadata, val_datasets, "validation")
        self._save_pickle_metadata(val_metadata, self.val_dir / "metadata.pkl")

        # Save seed indices for easy discovery
        self._save_seed_indices(train_datasets, self.train_dir)
        self._save_seed_indices(val_datasets, self.val_dir)

    def _create_split_specific_metadata(
        self, split_metadata: SplitMetadata, datasets: dict[int, Dataset], split_type: str
    ) -> dict[str, t.Any]:
        """Create metadata specific to a split (train or validation)."""
        # Calculate split-specific statistics
        total_sequences = sum(len(ds) for ds in datasets.values())
        all_lengths = []
        for dataset in datasets.values():
            if "length" in dataset.column_names:
                all_lengths.extend(dataset["length"])
            else:
                # Fallback: calculate lengths from input_ids
                all_lengths.extend([len(seq) for seq in dataset["input_ids"]])

        split_stats = {
            "split_type": split_type,
            "total_sequences": total_sequences,
            "num_seeds": len(datasets),
            "available_seeds": sorted(datasets.keys()),
        }

        if all_lengths:
            split_stats.update(
                {
                    "min_sequence_length": min(all_lengths),
                    "max_sequence_length": max(all_lengths),
                    "avg_sequence_length": sum(all_lengths) / len(all_lengths),
                    "total_tokens": sum(all_lengths),
                }
            )

        # Extract relevant split information
        per_seed_info = {}
        for seed in datasets:
            if seed in split_metadata.per_seed_splits:
                size_key = f"{split_type}_size" if split_type in ["train", "validation"] else "total_size"
                per_seed_info[seed] = split_metadata.per_seed_splits[seed].get(size_key, len(datasets[seed]))

        return {
            "split_info": split_stats,
            "split_config": {
                "validation_ratio": split_metadata.split_config.validation_ratio,
                "split_method": split_metadata.split_config.split_method,
                "split_seed": split_metadata.split_config.split_seed,
            },
            "per_seed_sizes": per_seed_info,
            "created_from_split": True,
        }

    def _save_pickle_metadata(self, metadata: dict[str, t.Any], path: Path) -> None:
        """Save metadata as pickle file."""
        with path.open("wb") as f:
            pickle.dump(metadata, f)

    def _save_seed_indices(self, datasets: dict[int, Dataset], target_dir: Path) -> None:
        """Save seed index file for easy dataset discovery."""
        seed_index = {
            "available_seeds": sorted(datasets.keys()),
            "num_seeds": len(datasets),
            "dataset_paths": {seed: f"seed_{seed}/dataset" for seed in datasets},
        }

        with (target_dir / "seed_index.json").open("w") as f:
            json.dump(seed_index, f, indent=2)

    def create_summary_files(self, metadata: SplitMetadata) -> None:
        """Create human-readable summary files."""
        from ICL.datasets.utils import create_split_summary

        summary_content = create_split_summary(metadata)

        # Save to base directory
        with (self.base_dir / "split_summary.txt").open("w") as f:
            f.write(summary_content)

        # Create brief summaries for each split directory
        train_summary = f"TRAIN SPLIT\n{'-' * 20}\n"
        train_summary += f"Sequences: {metadata.total_stats['total_train_sequences']:,}\n"
        train_summary += f"Seeds: {metadata.total_stats['num_seeds']}\n"

        with (self.train_dir / "split_summary.txt").open("w") as f:
            f.write(train_summary)

        val_summary = f"VALIDATION SPLIT\n{'-' * 20}\n"
        val_summary += f"Sequences: {metadata.total_stats['total_val_sequences']:,}\n"
        val_summary += f"Seeds: {metadata.total_stats['num_seeds']}\n"

        with (self.val_dir / "split_summary.txt").open("w") as f:
            f.write(val_summary)

    def get_directory_structure(self) -> dict[str, Path]:
        """Get all relevant directory paths."""
        return {"base": self.base_dir, "raw": self.raw_dir, "train": self.train_dir, "validation": self.val_dir}


def check_split_exists(base_dataset_dir: Path) -> bool:
    """Check if dataset splits already exist."""
    split_dirs = [base_dataset_dir / "train", base_dataset_dir / "validation"]

    return any(d.exists() and any(d.iterdir()) for d in split_dirs)


def copy_raw_metadata_to_splits(raw_dir: Path, train_dir: Path, val_dir: Path) -> None:
    """Copy relevant files from raw directory to split directories."""
    # Files to copy (if they exist)
    files_to_copy = [
        "dataset_summary.txt",
        # Note: metadata.pkl is created separately for splits
    ]

    for filename in files_to_copy:
        raw_file = raw_dir / filename
        if raw_file.exists():
            # Copy to both train and validation directories
            shutil.copy2(raw_file, train_dir / f"raw_{filename}")
            shutil.copy2(raw_file, val_dir / f"raw_{filename}")
