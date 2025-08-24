import random
import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset

T = t.TypeVar("T")
ConfigTuple = tuple[int, int]  # (L, m)
#############################################
# RHM-related util func
#############################################


def dec2bin(n, bits=None):
    """Convert integers to binary.

    Args:
            n: The numbers to convert (tensor of size [*]).
         bits: The length of the representation.

    Returns:
        A tensor (size [*, bits]) with the binary representations.

    """
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def dec2base(n, b, length=None):
    """Convert integers into a different base.

    Args:
            n: The numbers to convert (tensor of size [*]).
            b: The base (integer).
       length: The length of the representation.

    Returns:
        A tensor (size [*, length]) containing the input numbers in the new base.

    """
    digits = []
    while n.sum():
        digits.append(n % b)
        n = n.div(b, rounding_mode="floor")
    if length:
        assert len(digits) <= length, "Length required is too small to represent input numbers!"
        digits += [torch.zeros(len(n), dtype=int)] * (length - len(digits))
    return torch.stack(digits[::-1]).t()


def base2dec(t, b):
    """Convert tuples of s integers in base b into integers in base b**s.

    Args:
            t: tuples to convert (tesor of size [*,s]).
            b: the base (integer).

    Returns:
        A tensor (size [*]) with the inputs in the new base

    """
    length = t.size(-1)  # Length of the tuples, which gives the number of digits
    # Create powers of b: [b**(length-1), b**(length-2), ..., b**0]
    powers = torch.tensor([b**i for i in reversed(range(length))], dtype=t.dtype, device=t.device)
    # Multiply the tensor by the powers of b and sum along the last dimension
    result = torch.sum(t * powers, dim=-1)

    return result


#############################################
# Training dataset util func
#############################################


def generate_zipf_distribution(n: int, alpha: float = 1.0) -> list[float]:
    """Generate normalized Zipf distribution with n elements."""
    if n <= 0:
        raise ValueError("Number of elements must be positive")

    # Generate Zipf probabilities: 1/k^alpha for k=1,2,...,n
    ranks = np.arange(1, n + 1)
    raw_probs = 1.0 / (ranks**alpha)

    # Normalize to sum to 1
    normalized_probs = raw_probs / raw_probs.sum()

    return normalized_probs.tolist()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def create_rule_probabilities(
    rules: dict[int, torch.Tensor], distribution_type: str, **params
) -> dict[int, torch.Tensor] | None:
    """Create rule sampling probabilities for RHM based on distribution type."""
    if distribution_type == "uniform":
        return None  # RHM uses uniform probabilities by default

    if distribution_type == "zipf":
        alpha = params.get("zipf_alpha", 1.0)
        probability = {}

        for level, rule_tensor in rules.items():
            m = rule_tensor.shape[1]  # Number of synonymic rules at this level
            zipf_probs = generate_zipf_distribution(m, alpha)
            probability[level] = torch.tensor(zipf_probs, dtype=torch.float32)

        return probability

    raise ValueError(f"Unsupported distribution type: {distribution_type}")


#############################################
# Validation dataset util func
#############################################


@dataclass
class SplitConfig:
    """Configuration for train/validation splitting."""

    validation_ratio: float = 0.2
    split_method: str = "sequential"  # "sequential", "random"
    min_sequences_per_split: int = 1
    split_seed: int | None = None  # For reproducible random splits

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")

        if self.split_method not in ["sequential", "random"]:
            raise ValueError("split_method must be 'sequential' or 'random'")

        if self.min_sequences_per_split < 1:
            raise ValueError("min_sequences_per_split must be at least 1")


@dataclass
class SplitMetadata:
    """Metadata for dataset splits."""

    split_config: SplitConfig
    per_seed_splits: dict[int, dict[str, int]] = field(default_factory=dict)
    total_stats: dict[str, int] = field(default_factory=dict)
    created_at: str = ""

    def add_seed_split(self, seed: int, train_size: int, val_size: int) -> None:
        """Add split information for a seed."""
        self.per_seed_splits[seed] = {
            "train_size": train_size,
            "val_size": val_size,
            "total_size": train_size + val_size,
        }

    def calculate_total_stats(self) -> None:
        """Calculate overall split statistics."""
        total_train = sum(split["train_size"] for split in self.per_seed_splits.values())
        total_val = sum(split["val_size"] for split in self.per_seed_splits.values())
        total_sequences = total_train + total_val

        self.total_stats = {
            "total_train_sequences": total_train,
            "total_val_sequences": total_val,
            "total_sequences": total_sequences,
            "actual_val_ratio": total_val / total_sequences if total_sequences > 0 else 0.0,
            "num_seeds": len(self.per_seed_splits),
        }


def load_split_config_from_yaml(yaml_config: dict[str, t.Any]) -> SplitConfig:
    """Extract split configuration from YAML config."""
    split_section = yaml_config.get("split_config", {})

    return SplitConfig(
        validation_ratio=split_section.get("validation_ratio", 0.2),
        split_method=split_section.get("split_method", "sequential"),
        min_sequences_per_split=split_section.get("min_sequences_per_split", 1),
        split_seed=split_section.get("split_seed", None),
    )


def save_split_metadata(metadata: SplitMetadata, output_path: Path) -> None:
    """Save split metadata to JSON file."""
    import json
    from datetime import datetime

    # Update timestamp
    metadata.created_at = datetime.now().isoformat()

    # Convert to dictionary for JSON serialization
    data = {
        "split_config": {
            "validation_ratio": metadata.split_config.validation_ratio,
            "split_method": metadata.split_config.split_method,
            "min_sequences_per_split": metadata.split_config.min_sequences_per_split,
            "split_seed": metadata.split_config.split_seed,
        },
        "per_seed_splits": metadata.per_seed_splits,
        "total_stats": metadata.total_stats,
        "created_at": metadata.created_at,
    }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def load_split_metadata(metadata_path: Path) -> SplitMetadata:
    """Load split metadata from JSON file."""
    import json

    with metadata_path.open("r") as f:
        data = json.load(f)

    split_config = SplitConfig(**data["split_config"])

    metadata = SplitMetadata(
        split_config=split_config,
        per_seed_splits=data["per_seed_splits"],
        total_stats=data["total_stats"],
        created_at=data["created_at"],
    )

    return metadata


class DatasetSplitter:
    """Handles splitting of seed datasets into train/validation."""

    def __init__(self, split_config: SplitConfig):
        """Initialize splitter with configuration."""
        self.config = split_config

        # Set random seed for reproducible splits if specified
        if self.config.split_seed is not None:
            random.seed(self.config.split_seed)

    def split_single_seed_dataset(self, dataset: Dataset, seed: int) -> tuple[Dataset, Dataset]:
        """Split a single seed's dataset into train and validation parts.

        Args:
            dataset: HuggingFace Dataset to split
            seed: Seed identifier for logging/metadata

        Returns:
            Tuple of (train_dataset, validation_dataset)

        Raises:
            ValueError: If dataset is too small to split properly

        """
        total_sequences = len(dataset)

        # Calculate split sizes
        val_size = max(1, int(total_sequences * self.config.validation_ratio))
        train_size = total_sequences - val_size

        # Validate minimum requirements
        if train_size < self.config.min_sequences_per_split:
            raise ValueError(
                f"Seed {seed}: train split ({train_size}) below minimum ({self.config.min_sequences_per_split})"
            )

        if val_size < self.config.min_sequences_per_split:
            raise ValueError(
                f"Seed {seed}: validation split ({val_size}) below minimum ({self.config.min_sequences_per_split})"
            )

        # Perform the actual split
        if self.config.split_method == "sequential":
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, total_sequences))

        elif self.config.split_method == "random":
            # Create random indices for splitting
            indices = list(range(total_sequences))
            random.shuffle(indices)

            train_indices = sorted(indices[:train_size])
            val_indices = sorted(indices[train_size:])

            train_dataset = dataset.select(train_indices)
            val_dataset = dataset.select(val_indices)

        else:
            raise ValueError(f"Unknown split method: {self.config.split_method}")

        return train_dataset, val_dataset

    def split_all_seed_datasets(
        self, seed_datasets: dict[int, Dataset]
    ) -> tuple[dict[int, Dataset], dict[int, Dataset], SplitMetadata]:
        """Split all seed datasets and return train/validation splits with metadata.

        Args:
            seed_datasets: Dictionary mapping seed -> Dataset

        Returns:
            Tuple of (train_datasets, val_datasets, split_metadata)

        """
        train_datasets = {}
        val_datasets = {}
        metadata = SplitMetadata(split_config=self.config)

        successful_splits = 0
        failed_seeds = []

        for seed, dataset in seed_datasets.items():
            try:
                train_ds, val_ds = self.split_single_seed_dataset(dataset, seed)

                train_datasets[seed] = train_ds
                val_datasets[seed] = val_ds

                # Record split metadata
                metadata.add_seed_split(seed, len(train_ds), len(val_ds))
                successful_splits += 1

            except ValueError as e:
                failed_seeds.append((seed, str(e)))
                continue

        if successful_splits == 0:
            raise RuntimeError(f"Failed to split any datasets. Errors: {failed_seeds}")

        if failed_seeds:
            import warnings

            warnings.warn(f"Failed to split {len(failed_seeds)} seeds: {failed_seeds}")

        # Calculate overall statistics
        metadata.calculate_total_stats()

        return train_datasets, val_datasets, metadata


def load_seed_datasets(raw_dataset_dir: Path) -> dict[int, Dataset]:
    """Load all seed datasets from raw directory.

    Args:
        raw_dataset_dir: Path to directory containing seed_*/dataset/ subdirs

    Returns:
        Dictionary mapping seed -> loaded Dataset

    """
    seed_datasets = {}

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dataset_dir}")

    # Find all seed directories
    seed_dirs = [d for d in raw_dataset_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found in {raw_dataset_dir}")

    for seed_dir in seed_dirs:
        # Extract seed number from directory name
        try:
            seed = int(seed_dir.name.replace("seed_", ""))
        except ValueError:
            continue

        dataset_path = seed_dir / "dataset"
        if dataset_path.exists():
            try:
                dataset = Dataset.load_from_disk(str(dataset_path))
                seed_datasets[seed] = dataset
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to load dataset for seed {seed}: {e}")
                continue

    if not seed_datasets:
        raise RuntimeError(f"No valid seed datasets found in {raw_dataset_dir}")

    return seed_datasets


def validate_split_consistency(
    original_datasets: dict[int, Dataset], train_datasets: dict[int, Dataset], val_datasets: dict[int, Dataset]
) -> bool:
    """Validate that splits maintain data integrity.

    Args:
        original_datasets: Original seed datasets
        train_datasets: Train split datasets
        val_datasets: Validation split datasets

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails

    """
    for seed in original_datasets:
        if seed not in train_datasets or seed not in val_datasets:
            raise ValueError(f"Seed {seed} missing from splits")

        original_size = len(original_datasets[seed])
        train_size = len(train_datasets[seed])
        val_size = len(val_datasets[seed])

        if train_size + val_size != original_size:
            raise ValueError(
                f"Seed {seed}: split size mismatch. Original: {original_size}, Train: {train_size}, Val: {val_size}"
            )

    return True


def create_split_summary(metadata: SplitMetadata) -> str:
    """Create human-readable summary of split results."""
    lines = [
        "DATASET SPLIT SUMMARY",
        "=" * 50,
        f"Split method: {metadata.split_config.split_method}",
        f"Target validation ratio: {metadata.split_config.validation_ratio:.1%}",
        f"Actual validation ratio: {metadata.total_stats['actual_val_ratio']:.1%}",
        "",
        "Overall Statistics:",
        f"  Total sequences: {metadata.total_stats['total_sequences']:,}",
        f"  Train sequences: {metadata.total_stats['total_train_sequences']:,}",
        f"  Validation sequences: {metadata.total_stats['total_val_sequences']:,}",
        f"  Number of seeds: {metadata.total_stats['num_seeds']}",
        "",
        "Per-Seed Breakdown:",
    ]

    for seed, split_info in metadata.per_seed_splits.items():
        val_ratio = split_info["val_size"] / split_info["total_size"]
        lines.append(f"  Seed {seed}: {split_info['train_size']} train, {split_info['val_size']} val ({val_ratio:.1%})")

    return "\n".join(lines)
