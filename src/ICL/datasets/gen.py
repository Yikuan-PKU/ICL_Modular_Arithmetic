import json
import pickle
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


class UnifiedRHMDataset:
    """Unified interface for RHM datasets that can be used across different training paradigms.
    Loads raw RHM data and provides task-agnostic access with rich metadata.
    """

    def __init__(self, dataset_path: str):
        """Initialize unified dataset from saved raw RHM data.

        Args:
            dataset_path: Path to directory containing raw_dataset and metadata.pkl

        """
        self.dataset_path = Path(dataset_path)
        self.dataset = None
        self.metadata = None

        self._load_dataset()
        self._validate_dataset()
        self._compute_statistics()

    def _load_dataset(self):
        """Load the HuggingFace dataset and metadata"""
        print("Loading RHM dataset...")

        # Load HuggingFace dataset
        dataset_file = self.dataset_path / "raw_dataset"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_file}")

        self.dataset = load_from_disk(str(dataset_file))
        print(f"✓ Loaded dataset with {len(self.dataset)} sequences")

        # Load metadata
        metadata_file = self.dataset_path / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_file}")

        with metadata_file.open("rb") as f:
            self.metadata = pickle.load(f)
        print(f"✓ Loaded metadata for {len(self.metadata['configurations'])} configurations")

    def _validate_dataset(self):
        """Validate dataset structure and consistency"""
        print("Validating dataset structure...")

        # Check required columns
        required_columns = ["input_ids", "task_id", "config_L", "config_m", "length"]
        missing_columns = [col for col in required_columns if col not in self.dataset.column_names]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data consistency
        assert len(self.dataset) > 0, "Dataset is empty"

        # Validate sequence lengths match recorded lengths
        sample_indices = range(min(100, len(self.dataset)))  # Check first 100 samples
        for i in sample_indices:
            recorded_length = self.dataset[i]["length"]
            actual_length = len(self.dataset[i]["input_ids"])
            assert recorded_length == actual_length, (
                f"Length mismatch at index {i}: recorded={recorded_length}, actual={actual_length}"
            )

        # Check task_id consistency
        unique_task_ids = set(self.dataset["task_id"])
        metadata_task_ids = set(config["task_id"] for config in self.metadata["configurations"])
        assert unique_task_ids == metadata_task_ids, (
            f"Task ID mismatch: dataset={unique_task_ids}, metadata={metadata_task_ids}"
        )

        print("✓ Dataset validation passed")

    def _compute_statistics(self):
        """Compute and cache dataset statistics"""
        print("Computing dataset statistics...")

        # Overall statistics
        lengths = self.dataset["length"]
        self.stats = {
            "total_sequences": len(self.dataset),
            "total_tokens": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "vocab_size": self.metadata["generation_params"]["vocab_size"],
            "vocab_range": f"1-{self.metadata['generation_params']['vocab_size']} (0 reserved)",
        }

        # Per-configuration statistics
        self.config_stats = {}
        for config in self.metadata["configurations"]:
            task_id = config["task_id"]
            L = config["L"]
            m = config["m"]

            # Filter sequences for this configuration
            config_mask = [tid == task_id for tid in self.dataset["task_id"]]
            config_lengths = [length for length, mask in zip(lengths, config_mask, strict=False) if mask]
            config_sequences = sum(config_mask)

            self.config_stats[task_id] = {
                "L": L,
                "m": m,
                "num_sequences": config_sequences,
                "min_length": min(config_lengths) if config_lengths else 0,
                "max_length": max(config_lengths) if config_lengths else 0,
                "avg_length": sum(config_lengths) / len(config_lengths) if config_lengths else 0,
                "total_tokens": sum(config_lengths),
                "proportion": config_sequences / len(self.dataset),
            }

        print("✓ Statistics computed")

    def get_dataset(self) -> Dataset:
        """Get the raw HuggingFace dataset"""
        return self.dataset

    def get_metadata(self) -> dict[str, Any]:
        """Get complete metadata"""
        return self.metadata

    def get_statistics(self) -> dict[str, Any]:
        """Get computed statistics"""
        return {"overall": self.stats, "per_config": self.config_stats}

    def get_vocab_info(self) -> dict[str, Any]:
        """Get vocabulary information"""
        return {
            "vocab_size": self.stats["vocab_size"],
            "vocab_range": self.stats["vocab_range"],
            "reserved_tokens": {
                0: "PAD/EOS/SEP",
                "vocab_size + 1": "MASK (for MLM)",
                "vocab_size + 2": "CLS (if needed)",
                "vocab_size + 3": "BOS (if needed)",
            },
            "effective_vocab_size": self.stats["vocab_size"] + 4,  # Including special tokens
        }

    def filter_by_config(
        self, L: int | None = None, m: int | None = None, task_ids: list[int] | None = None
    ) -> Dataset:
        """Filter dataset by hierarchical configuration parameters.

        Args:
            L: Hierarchy depth to filter by
            m: Multiplicity to filter by
            task_ids: Specific task IDs to include

        Returns:
            Filtered HuggingFace dataset

        """
        indices_to_keep = []

        for i in range(len(self.dataset)):
            keep = True

            if L is not None and self.dataset[i]["config_L"] != L:
                keep = False
            if m is not None and self.dataset[i]["config_m"] != m:
                keep = False
            if task_ids is not None and self.dataset[i]["task_id"] not in task_ids:
                keep = False

            if keep:
                indices_to_keep.append(i)

        return self.dataset.select(indices_to_keep)

    def filter_by_length(self, min_length: int | None = None, max_length: int | None = None) -> Dataset:
        """Filter dataset by sequence length.

        Args:
            min_length: Minimum sequence length (inclusive)
            max_length: Maximum sequence length (inclusive)

        Returns:
            Filtered HuggingFace dataset

        """
        indices_to_keep = []

        for i in range(len(self.dataset)):
            length = self.dataset[i]["length"]
            keep = True

            if min_length is not None and length < min_length:
                keep = False
            if max_length is not None and length > max_length:
                keep = False

            if keep:
                indices_to_keep.append(i)

        return self.dataset.select(indices_to_keep)

    def get_config_groups(self) -> dict[tuple[int, int], list[int]]:
        """Group sequence indices by (L, m) configuration.

        Returns:
            Dictionary mapping (L, m) tuples to lists of sequence indices

        """
        config_groups = {}

        for i in range(len(self.dataset)):
            L = self.dataset[i]["config_L"]
            m = self.dataset[i]["config_m"]
            config_key = (L, m)

            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(i)

        return config_groups

    def get_length_groups(self, bucket_size: int = 50) -> dict[str, list[int]]:
        """Group sequence indices by length buckets.

        Args:
            bucket_size: Size of each length bucket

        Returns:
            Dictionary mapping bucket names to lists of sequence indices

        """
        length_groups = {}

        for i in range(len(self.dataset)):
            length = self.dataset[i]["length"]
            bucket_start = (length // bucket_size) * bucket_size
            bucket_end = bucket_start + bucket_size - 1
            bucket_name = f"{bucket_start}-{bucket_end}"

            if bucket_name not in length_groups:
                length_groups[bucket_name] = []
            length_groups[bucket_name].append(i)

        return length_groups

    def print_summary(self):
        """Print a comprehensive summary of the dataset"""
        print("\n" + "=" * 60)
        print("UNIFIED RHM DATASET SUMMARY")
        print("=" * 60)

        print(f"Total sequences: {self.stats['total_sequences']:,}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        print(
            f"Sequence length: {self.stats['min_length']}-{self.stats['max_length']} (avg: {self.stats['avg_length']:.1f})"
        )
        print(f"Vocabulary: {self.stats['vocab_range']}")

        print(f"\nConfigurations ({len(self.config_stats)}):")
        print("-" * 40)
        for task_id, stats in self.config_stats.items():
            print(f"Task {task_id}: L={stats['L']}, m={stats['m']}")
            print(f"  Sequences: {stats['num_sequences']:,} ({stats['proportion']:.1%})")
            print(f"  Length: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.1f})")
            print(f"  Tokens: {stats['total_tokens']:,}")

        print("=" * 60)

    def save_analysis(self, output_path: str):
        """Save dataset analysis to file"""
        output_file = Path(output_path)

        analysis = {
            "statistics": self.get_statistics(),
            "vocab_info": self.get_vocab_info(),
            "config_groups": {str(k): v for k, v in self.get_config_groups().items()},
            "sample_sequences": {
                "first_10_tokens": [seq[:10] for seq in self.dataset["input_ids"][:5]],
                "sequence_lengths": self.dataset["length"][:10],
            },
        }

        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w") as f:
            json.dump(analysis, f, indent=2)

        print(f"✓ Analysis saved to {output_file}")
