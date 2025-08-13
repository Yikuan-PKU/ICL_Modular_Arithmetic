import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from ICL.datasets.gen import UnifiedRHMDataset


class RHMDataLoaderFactory:
    """Unified factory for creating task-specific DataLoaders for RHM datasets."""

    def __init__(self, dataset_path: str, vocab_size: int = 32):
        """Initialize the DataLoader factory.

        Args:
            dataset_path: Path to the unified RHM dataset
            vocab_size: Vocabulary size of the RHM data

        """
        self.dataset_path = Path(dataset_path)
        self.vocab_size = vocab_size

        # Load unified dataset
        print("Loading unified RHM dataset...")

        self.unified_dataset = UnifiedRHMDataset(str(self.dataset_path))
        self.unified_dataset.print_summary()

    def create_dataloader(
        self,
        task_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        # Task-specific parameters
        max_length: int | None = None,
        mask_probability: float = 0.15,
        mask_strategy: str = "random",
        # Batching parameters
        batching_strategy: str = "config_then_length",
        length_bucket_size: int = 50,
        # Filtering parameters
        filter_config_L: int | None = None,
        filter_config_m: int | None = None,
        filter_min_length: int | None = None,
        filter_max_length: int | None = None,
        # Other parameters
        seed: int | None = None,
    ) -> tuple[DataLoader, dict[str, Any]]:
        """Create a task-specific DataLoader.

        Args:
            task_name: 'clm' or 'mlm'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for DataLoader
            max_length: Maximum sequence length
            mask_probability: Probability of masking tokens (MLM only)
            mask_strategy: Masking strategy (MLM only)
            batching_strategy: Dynamic batching strategy
            length_bucket_size: Size of length buckets
            filter_config_L: Filter by hierarchy depth
            filter_config_m: Filter by multiplicity
            filter_min_length: Filter by minimum length
            filter_max_length: Filter by maximum length
            seed: Random seed

        Returns:
            DataLoader and metadata dictionary

        """
        print(f"\n{'=' * 60}")
        print(f"CREATING {task_name.upper()} DATALOADER")
        print(f"{'=' * 60}")

        # Step 1: Filter dataset if needed
        dataset = self.unified_dataset.get_dataset()

        if any([filter_config_L, filter_config_m, filter_min_length, filter_max_length]):
            print("Applying filters...")

            if filter_config_L is not None or filter_config_m is not None:
                dataset = self.unified_dataset.filter_by_config(L=filter_config_L, m=filter_config_m)
                print(f"  Config filter: {len(dataset)} sequences remaining")

            if filter_min_length is not None or filter_max_length is not None:
                dataset = self.unified_dataset.filter_by_length(
                    min_length=filter_min_length, max_length=filter_max_length
                )
                print(f"  Length filter: {len(dataset)} sequences remaining")

        # Step 2: Create task configuration
        task_config = TaskConfig(
            task_name=task_name,
            max_length=max_length,
            mask_probability=mask_probability,
            mask_strategy=mask_strategy,
            pad_token_id=0,
            mask_token_id=self.vocab_size + 1,
            cls_token_id=self.vocab_size + 2,
            sep_token_id=self.vocab_size + 3,
        )

        print("Task configuration:")
        print(f"  Max length: {task_config.max_length}")
        if task_name == "mlm":
            print(f"  Mask probability: {task_config.mask_probability}")
            print(f"  Mask strategy: {task_config.mask_strategy}")

        # Step 3: Process dataset for specific task
        processor = ProcessorFactory.create_processor(task_name, task_config, self.vocab_size)
        processed_dataset = processor.process_dataset(dataset)

        # Step 4: Create dynamic batcher
        batcher = DynamicBatcher(
            strategy=batching_strategy,
            length_bucket_size=length_bucket_size,
            max_batch_size=batch_size,
            shuffle_within_groups=shuffle,
        )

        print(f"Batching strategy: {batching_strategy}")

        # Step 5: Create custom sampler if using dynamic batching
        if batching_strategy != "none":
            sampler = ConfigAwareSampler(dataset=processed_dataset, batcher=batcher, shuffle=shuffle, seed=seed)
            shuffle = False  # Disable DataLoader shuffle when using custom sampler
        else:
            sampler = None

        # Step 6: Create DataLoader
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=processor.get_collate_fn(),
            pin_memory=torch.cuda.is_available(),
        )

        # Step 7: Create metadata
        metadata = {
            "task_name": task_name,
            "task_config": task_config,
            "original_dataset_size": len(self.unified_dataset.get_dataset()),
            "filtered_dataset_size": len(dataset),
            "processed_dataset_size": len(processed_dataset),
            "batch_size": batch_size,
            "batching_strategy": batching_strategy,
            "vocab_info": self.unified_dataset.get_vocab_info(),
            "sampler_stats": sampler.get_statistics() if sampler else None,
            "filters_applied": {
                "config_L": filter_config_L,
                "config_m": filter_config_m,
                "min_length": filter_min_length,
                "max_length": filter_max_length,
            },
        }

        print("\nDataLoader created successfully!")
        print(f"  Original dataset: {metadata['original_dataset_size']:,} sequences")
        print(f"  After filtering: {metadata['filtered_dataset_size']:,} sequences")
        print(f"  After processing: {metadata['processed_dataset_size']:,} sequences")
        print(f"  Batches per epoch: {len(dataloader):,}")

        if sampler:
            stats = sampler.get_statistics()
            print(f"  Config coherence: {stats['config_coherence_ratio']:.1%}")
            print(f"  Avg length variance: {stats['avg_length_variance']:.2f}")

        print(f"{'=' * 60}")

        return dataloader, metadata


###########################################
# dynamic batching
###########################################


class DynamicBatcher:
    """Handles dynamic batching strategies for RHM datasets.
    Supports configuration-aware and length-aware batching.
    """

    def __init__(
        self,
        strategy: str = "config_then_length",  # 'config_only', 'length_only', 'config_then_length'
        length_bucket_size: int = 50,
        max_batch_size: int = 32,
        shuffle_within_groups: bool = True,
    ):
        """Initialize dynamic batcher.

        Args:
            strategy: Batching strategy to use
            length_bucket_size: Size of length buckets
            max_batch_size: Maximum batch size
            shuffle_within_groups: Whether to shuffle within groups

        """
        self.strategy = strategy
        self.length_bucket_size = length_bucket_size
        self.max_batch_size = max_batch_size
        self.shuffle_within_groups = shuffle_within_groups

    def create_batches(self, dataset: Dataset, seed: int | None = None) -> list[list[int]]:
        """Create batches according to the specified strategy.

        Args:
            dataset: Dataset to batch
            seed: Random seed for shuffling

        Returns:
            List of batches, where each batch is a list of indices

        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.strategy == "config_only":
            return self._batch_by_config(dataset)
        if self.strategy == "length_only":
            return self._batch_by_length(dataset)
        if self.strategy == "config_then_length":
            return self._batch_by_config_then_length(dataset)
        raise ValueError(f"Unknown batching strategy: {self.strategy}")

    def _batch_by_config(self, dataset: Dataset) -> list[list[int]]:
        """Batch sequences by hierarchical configuration"""
        print("Batching by configuration...")

        # Group by (L, m) configuration
        config_groups = defaultdict(list)
        for i in range(len(dataset)):
            config_key = (dataset[i]["config_L"], dataset[i]["config_m"])
            config_groups[config_key].append(i)

        # Create batches within each configuration group
        all_batches = []
        for config_key, indices in config_groups.items():
            L, m = config_key
            print(f"  Config L={L}, m={m}: {len(indices)} sequences")

            if self.shuffle_within_groups:
                indices = np.random.permutation(indices).tolist()

            # Split into batches
            config_batches = [indices[i : i + self.max_batch_size] for i in range(0, len(indices), self.max_batch_size)]
            all_batches.extend(config_batches)

        print(f"  Created {len(all_batches)} batches")
        return all_batches

    def _batch_by_length(self, dataset: Dataset) -> list[list[int]]:
        """Batch sequences by length buckets"""
        print(f"Batching by length (bucket size: {self.length_bucket_size})...")

        # Group by length buckets
        length_groups = defaultdict(list)
        for i in range(len(dataset)):
            length = dataset[i]["length"]
            bucket = (length // self.length_bucket_size) * self.length_bucket_size
            bucket_key = f"{bucket}-{bucket + self.length_bucket_size - 1}"
            length_groups[bucket_key].append(i)

        # Create batches within each length group
        all_batches = []
        for bucket_key, indices in length_groups.items():
            print(f"  Length bucket {bucket_key}: {len(indices)} sequences")

            if self.shuffle_within_groups:
                indices = np.random.permutation(indices).tolist()

            # Split into batches
            length_batches = [indices[i : i + self.max_batch_size] for i in range(0, len(indices), self.max_batch_size)]
            all_batches.extend(length_batches)

        print(f"  Created {len(all_batches)} batches")
        return all_batches

    def _batch_by_config_then_length(self, dataset: Dataset) -> list[list[int]]:
        """Batch by configuration first, then by length within each config"""
        print("Batching by config then length...")

        # Group by (L, m) configuration first
        config_groups = defaultdict(list)
        for i in range(len(dataset)):
            config_key = (dataset[i]["config_L"], dataset[i]["config_m"])
            config_groups[config_key].append(i)

        all_batches = []
        for config_key, config_indices in config_groups.items():
            L, m = config_key
            print(f"  Config L={L}, m={m}: {len(config_indices)} sequences")

            # Within this config, group by length
            length_groups = defaultdict(list)
            for idx in config_indices:
                length = dataset[idx]["length"]
                bucket = (length // self.length_bucket_size) * self.length_bucket_size
                bucket_key = f"{bucket}-{bucket + self.length_bucket_size - 1}"
                length_groups[bucket_key].append(idx)

            # Create batches within each length group
            for bucket_key, indices in length_groups.items():
                if self.shuffle_within_groups:
                    indices = np.random.permutation(indices).tolist()

                # Split into batches
                config_length_batches = [
                    indices[i : i + self.max_batch_size] for i in range(0, len(indices), self.max_batch_size)
                ]
                all_batches.extend(config_length_batches)

                print(
                    f"    Length bucket {bucket_key}: {len(indices)} sequences -> {len(config_length_batches)} batches"
                )

        print(f"  Total batches created: {len(all_batches)}")
        return all_batches

    def get_batch_statistics(self, dataset: Dataset, batches: list[list[int]]) -> dict[str, Any]:
        """Get statistics about the created batches"""
        batch_sizes = [len(batch) for batch in batches]
        batch_length_variances = []

        for batch_indices in batches:
            lengths = [dataset[i]["length"] for i in batch_indices]
            if len(lengths) > 1:
                batch_length_variances.append(np.var(lengths))
            else:
                batch_length_variances.append(0.0)

        # Configuration coherence: how many batches contain only one config
        config_coherent_batches = 0
        for batch_indices in batches:
            configs = set((dataset[i]["config_L"], dataset[i]["config_m"]) for i in batch_indices)
            if len(configs) == 1:
                config_coherent_batches += 1

        return {
            "total_batches": len(batches),
            "avg_batch_size": np.mean(batch_sizes),
            "min_batch_size": min(batch_sizes),
            "max_batch_size": max(batch_sizes),
            "avg_length_variance": np.mean(batch_length_variances),
            "config_coherent_batches": config_coherent_batches,
            "config_coherence_ratio": config_coherent_batches / len(batches) if batches else 0.0,
        }


class ConfigAwareSampler(torch.utils.data.Sampler):
    """Custom sampler that respects hierarchical configuration grouping."""

    def __init__(self, dataset: Dataset, batcher: DynamicBatcher, shuffle: bool = True, seed: int | None = None):
        """Initialize configuration-aware sampler.

        Args:
            dataset: Dataset to sample from
            batcher: Dynamic batcher instance
            shuffle: Whether to shuffle batch order
            seed: Random seed

        """
        self.dataset = dataset
        self.batcher = batcher
        self.shuffle = shuffle
        self.seed = seed

        # Create batches
        self.batches = self.batcher.create_batches(dataset, seed)

        # Get statistics
        self.stats = self.batcher.get_batch_statistics(dataset, self.batches)

        print(f"Sampler created with {len(self.batches)} batches")
        print(f"Configuration coherence: {self.stats['config_coherence_ratio']:.2%}")
        print(f"Average length variance: {self.stats['avg_length_variance']:.2f}")

    def __iter__(self):
        """Iterate over batch indices"""
        batch_order = list(range(len(self.batches)))

        if self.shuffle:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            batch_order = torch.randperm(len(self.batches)).tolist()

        for batch_idx in batch_order:
            for sample_idx in self.batches[batch_idx]:
                yield sample_idx

    def __len__(self):
        """Return total number of samples"""
        return len(self.dataset)

    def get_statistics(self) -> dict[str, Any]:
        """Get sampler statistics"""
        return self.stats


###########################################
# dynamic batching
###########################################


@dataclass
class TaskConfig:
    """Configuration for different language modeling tasks"""

    task_name: str
    max_length: int | None = None
    pad_token_id: int = 0
    mask_token_id: int = 33  # vocab_size + 1
    cls_token_id: int = 34  # vocab_size + 2
    sep_token_id: int = 35  # vocab_size + 3

    # Task-specific parameters
    mask_probability: float = 0.15
    mask_strategy: str = "random"  # 'random', 'hierarchical', 'level_specific'
    causal_mask: bool = True

    def __post_init__(self):
        if self.max_length is None:
            self.max_length = 512 if self.task_name == "mlm" else 2048


class BaseTaskProcessor(ABC):
    """Base class for task-specific processors"""

    def __init__(self, config: TaskConfig, vocab_size: int = 32):
        self.config = config
        self.vocab_size = vocab_size
        self.effective_vocab_size = vocab_size + 4  # Including special tokens

    @abstractmethod
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for the specific task"""

    @abstractmethod
    def get_collate_fn(self):
        """Get collate function for DataLoader"""

    def _validate_input(self, dataset: Dataset):
        """Validate input dataset has required columns"""
        required_columns = ["input_ids", "task_id", "config_L", "config_m", "length"]
        missing = [col for col in required_columns if col not in dataset.column_names]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class CLMProcessor(BaseTaskProcessor):
    """Processor for Causal Language Modeling"""

    def __init__(self, config: TaskConfig, vocab_size: int = 32):
        super().__init__(config, vocab_size)
        self.config.causal_mask = True

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for causal language modeling.
        Creates input_ids and labels with shifting.
        """
        self._validate_input(dataset)

        print("Processing dataset for Causal Language Modeling...")
        print(f"Original sequences: {len(dataset)}")

        processed_data = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "task_id": [],
            "config_L": [],
            "config_m": [],
            "length": [],
        }

        for example in dataset:
            sequence = example["input_ids"]

            # Skip sequences that are too short
            if len(sequence) < 2:
                continue

            # Truncate if too long
            if self.config.max_length and len(sequence) > self.config.max_length:
                sequence = sequence[: self.config.max_length]

            # Create input and labels (shifted by 1)
            input_ids = sequence[:-1]  # All but last token
            labels = sequence[1:]  # All but first token

            # Create attention mask (all 1s for now, padding handled in collate_fn)
            attention_mask = [1] * len(input_ids)

            processed_data["input_ids"].append(input_ids)
            processed_data["labels"].append(labels)
            processed_data["attention_mask"].append(attention_mask)
            processed_data["task_id"].append(example["task_id"])
            processed_data["config_L"].append(example["config_L"])
            processed_data["config_m"].append(example["config_m"])
            processed_data["length"].append(len(input_ids))

        print(f"Processed sequences: {len(processed_data['input_ids'])}")
        print(f"Average length: {sum(processed_data['length']) / len(processed_data['length']):.1f}")

        return Dataset.from_dict(processed_data)

    def get_collate_fn(self):
        """Get collate function for CLM"""

        def collate_fn(batch):
            # Extract data
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            labels = [torch.tensor(item["labels"]) for item in batch]

            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence

            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.config.pad_token_id)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignored in loss

            # Create attention mask
            attention_mask = (input_ids_padded != self.config.pad_token_id).long()

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask,
                "labels": labels_padded,
                # "task_ids": [item["task_id"] for item in batch],
                # "config_L": [item["config_L"] for item in batch],
                # "config_m": [item["config_m"] for item in batch],
            }

        return collate_fn


class MLMProcessor(BaseTaskProcessor):
    """Processor for Masked Language Modeling"""

    def __init__(self, config: TaskConfig, vocab_size: int = 32):
        super().__init__(config, vocab_size)
        self.config.causal_mask = False

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for masked language modeling.
        Applies masking strategy to create input_ids and labels.
        """
        self._validate_input(dataset)

        print("Processing dataset for Masked Language Modeling...")
        print(f"Masking strategy: {self.config.mask_strategy}")
        print(f"Mask probability: {self.config.mask_probability}")
        print(f"Original sequences: {len(dataset)}")

        processed_data = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "task_id": [],
            "config_L": [],
            "config_m": [],
            "length": [],
        }

        for example in dataset:
            sequence = example["input_ids"]

            # Skip sequences that are too short
            if len(sequence) < 2:
                continue

            # Truncate if too long
            if self.config.max_length and len(sequence) > self.config.max_length:
                sequence = sequence[: self.config.max_length]

            # Apply masking
            input_ids, labels = self._apply_masking(sequence, example["config_L"], example["config_m"])

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            processed_data["input_ids"].append(input_ids)
            processed_data["labels"].append(labels)
            processed_data["attention_mask"].append(attention_mask)
            processed_data["task_id"].append(example["task_id"])
            processed_data["config_L"].append(example["config_L"])
            processed_data["config_m"].append(example["config_m"])
            processed_data["length"].append(len(input_ids))

        print(f"Processed sequences: {len(processed_data['input_ids'])}")
        print(f"Average length: {sum(processed_data['length']) / len(processed_data['length']):.1f}")

        return Dataset.from_dict(processed_data)

    def _apply_masking(self, sequence: list[int], config_L: int, config_m: int) -> tuple[list[int], list[int]]:
        """Apply masking strategy to sequence"""
        if self.config.mask_strategy == "random":
            return self._random_masking(sequence)
        if self.config.mask_strategy == "hierarchical":
            return self._hierarchical_masking(sequence, config_L, config_m)
        if self.config.mask_strategy == "level_specific":
            return self._level_specific_masking(sequence, config_L, config_m)
        raise ValueError(f"Unknown masking strategy: {self.config.mask_strategy}")

    def _random_masking(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        """Standard BERT-style random masking"""
        input_ids = sequence.copy()
        labels = [-100] * len(sequence)  # -100 = ignore in loss

        for i in range(len(sequence)):
            if random.random() < self.config.mask_probability:
                labels[i] = sequence[i]  # Store original token

                # 80% mask, 10% random, 10% keep
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.config.mask_token_id
                elif rand < 0.9:
                    input_ids[i] = random.randint(1, self.vocab_size)
                # else: keep original

        return input_ids, labels

    def _hierarchical_masking(self, sequence: list[int], config_L: int, config_m: int) -> tuple[list[int], list[int]]:
        """Hierarchical-aware masking that respects the underlying structure.
        This is a simplified version - in practice, you'd use the actual rules.
        """
        # For now, implement as block masking to simulate hierarchical structure
        input_ids = sequence.copy()
        labels = [-100] * len(sequence)

        # Calculate approximate block size based on hierarchy
        block_size = max(2, len(sequence) // (config_L * config_m))

        i = 0
        while i < len(sequence):
            if random.random() < self.config.mask_probability:
                # Mask entire block
                block_end = min(i + block_size, len(sequence))
                for j in range(i, block_end):
                    labels[j] = sequence[j]
                    input_ids[j] = self.config.mask_token_id
                i = block_end
            else:
                i += 1

        return input_ids, labels

    def _level_specific_masking(self, sequence: list[int], config_L: int, config_m: int) -> tuple[list[int], list[int]]:
        """Focus masking on specific positions that correspond to hierarchy levels.
        This is a simplified version.
        """
        input_ids = sequence.copy()
        labels = [-100] * len(sequence)

        # Focus on positions that are multiples of tuple_size (typically 2)
        tuple_size = 2  # This should come from metadata
        step = tuple_size**config_L

        for i in range(0, len(sequence), step):
            if random.random() < self.config.mask_probability:
                labels[i] = sequence[i]
                input_ids[i] = self.config.mask_token_id

        return input_ids, labels

    def get_collate_fn(self):
        """Get collate function for MLM"""

        def collate_fn(batch):
            # Extract data
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            labels = [torch.tensor(item["labels"]) for item in batch]

            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence

            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.config.pad_token_id)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

            # Create attention mask (bidirectional for MLM)
            attention_mask = (input_ids_padded != self.config.pad_token_id).long()

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask,
                "labels": labels_padded,
                # "task_ids": [item["task_id"] for item in batch],
                # "config_L": [item["config_L"] for item in batch],
                # "config_m": [item["config_m"] for item in batch],
            }

        return collate_fn


class ProcessorFactory:
    """Factory for creating task processors"""

    @staticmethod
    def create_processor(task_name: str, config: TaskConfig, vocab_size: int = 32) -> BaseTaskProcessor:
        """Create processor for specified task"""
        if task_name.lower() == "clm":
            return CLMProcessor(config, vocab_size)
        if task_name.lower() == "mlm":
            return MLMProcessor(config, vocab_size)
        raise ValueError(f"Unknown task: {task_name}. Supported: ['clm', 'mlm']")
