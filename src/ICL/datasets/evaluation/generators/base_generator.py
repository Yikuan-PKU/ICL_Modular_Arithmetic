import random
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset

from ICL.datasets.evaluation.eval_config import ICLEvalConfig


@dataclass
class ICLSequence:
    """Standard ICL evaluation sequence."""

    # ICL structure
    context_examples: list[tuple[list[int], int]]  # (features, label) pairs
    query_features: list[int]
    query_label: int
    context_size: int
    sequence_id: int

    # Evaluation metadata
    eval_type: str  # "memorization", "id_generalization", "ood_same_rule", "ood_transfer"
    source_config: tuple[int, int]  # (L, m)
    target_config: tuple[int, int] | None = None  # For transfer only
    source_seeds: list[int] = field(default_factory=list)

    # Training relationship
    appears_in_training: bool = False
    training_overlap_ratio: float = 0.0

    # Additional metadata
    generation_metadata: dict[str, t.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for dataset storage."""
        # Flatten context examples for HuggingFace compatibility
        context_features = [ex[0] for ex in self.context_examples]
        context_labels = [ex[1] for ex in self.context_examples]

        return {
            "context_features": context_features,
            "context_labels": context_labels,
            "query_features": self.query_features,
            "query_label": self.query_label,
            "context_size": self.context_size,
            "sequence_id": self.sequence_id,
            "eval_type": self.eval_type,
            "source_config_L": self.source_config[0],
            "source_config_m": self.source_config[1],
            "target_config_L": self.target_config[0] if self.target_config else None,
            "target_config_m": self.target_config[1] if self.target_config else None,
            "source_seeds": self.source_seeds,
            "appears_in_training": self.appears_in_training,
            "training_overlap_ratio": self.training_overlap_ratio,
            "generation_metadata": self.generation_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "ICLSequence":
        """Create from dictionary."""
        context_examples = list(zip(data["context_features"], data["context_labels"], strict=True))

        target_config = None
        if data.get("target_config_L") is not None and data.get("target_config_m") is not None:
            target_config = (data["target_config_L"], data["target_config_m"])

        return cls(
            context_examples=context_examples,
            query_features=data["query_features"],
            query_label=data["query_label"],
            context_size=data["context_size"],
            sequence_id=data["sequence_id"],
            eval_type=data["eval_type"],
            source_config=(data["source_config_L"], data["source_config_m"]),
            target_config=target_config,
            source_seeds=data.get("source_seeds", []),
            appears_in_training=data.get("appears_in_training", False),
            training_overlap_ratio=data.get("training_overlap_ratio", 0.0),
            generation_metadata=data.get("generation_metadata", {}),
        )


class BaseICLGenerator(ABC):
    """Base class for ICL sequence generators."""

    def __init__(self, config: ICLEvalConfig, eval_type: str):
        """Initialize base generator.

        Args:
            config: ICL evaluation configuration
            eval_type: Type of evaluation ("memorization", "id_generalization", etc.)

        """
        self.config = config
        self.eval_type = eval_type
        self.icl_params = config.icl_params

        # Set random seed for reproducible generation
        random.seed(config.base_seed)

        # Track generated sequences
        self.sequence_counter = 0
        self.generated_sequences: list[ICLSequence] = []

    @abstractmethod
    def generate_icl_sequences(
        self, source_config: tuple[int, int], target_config: tuple[int, int] | None = None, **kwargs
    ) -> list[ICLSequence]:
        """Generate ICL sequences for this evaluation type.

        Args:
            source_config: Source (L, m) configuration
            target_config: Target (L, m) configuration (for transfer only)
            **kwargs: Additional generator-specific arguments

        Returns:
            List of generated ICL sequences

        """

    def _create_icl_sequence_from_data(
        self,
        available_data: list[tuple[list[int], int]],  # (features, label) pairs
        context_size: int,
        source_config: tuple[int, int],
        target_config: tuple[int, int] | None = None,
        source_seeds: list[int] | None = None,
        appears_in_training: bool = False,
        additional_metadata: dict[str, t.Any] | None = None,
    ) -> ICLSequence | None:
        """Create single ICL sequence from available data.

        Args:
            available_data: Pool of (features, label) pairs to sample from
            context_size: Number of context examples
            source_config: Source (L, m) configuration
            target_config: Target (L, m) configuration (for transfer)
            source_seeds: Seeds used to generate the data
            appears_in_training: Whether this sequence appears in training
            additional_metadata: Extra metadata to include

        Returns:
            ICL sequence or None if insufficient data

        """
        if len(available_data) < context_size + 1:
            return None

        # Sample context + query (without replacement)
        sampled_indices = random.sample(range(len(available_data)), context_size + 1)

        # Split into context and query
        context_examples = [available_data[i] for i in sampled_indices[:context_size]]
        query_features, query_label = available_data[sampled_indices[-1]]

        # Create metadata
        metadata = additional_metadata or {}
        metadata.update(
            {
                "sampled_indices": sampled_indices,
                "data_pool_size": len(available_data),
                "generation_method": self.__class__.__name__,
            }
        )

        sequence = ICLSequence(
            context_examples=context_examples,
            query_features=query_features,
            query_label=query_label,
            context_size=context_size,
            sequence_id=self.sequence_counter,
            eval_type=self.eval_type,
            source_config=source_config,
            target_config=target_config,
            source_seeds=source_seeds or [],
            appears_in_training=appears_in_training,
            training_overlap_ratio=1.0 if appears_in_training else 0.0,
            generation_metadata=metadata,
        )

        self.sequence_counter += 1
        return sequence

    def _extract_features_labels_from_dataset(self, dataset: Dataset) -> list[tuple[list[int], int]]:
        """Extract (features, label) pairs from HuggingFace dataset.

        Args:
            dataset: HuggingFace Dataset with 'input_ids' and optionally labels

        Returns:
            List of (features, label) tuples

        """
        data_pairs = []

        for i in range(len(dataset)):
            features = dataset[i]["input_ids"]

            # For RHM datasets, we need to extract the label
            # Labels are typically embedded in the sequence or available separately
            if "labels" in dataset.column_names:
                label = dataset[i]["labels"]
            elif "query_label" in dataset.column_names:
                label = dataset[i]["query_label"]
            else:
                # For RHM, the label might be encoded in the sequence
                # This is a fallback - actual implementation depends on RHM format
                label = features[-1] if features else 0

            data_pairs.append((features, label))

        return data_pairs

    def _generate_sequences_for_context_sizes(
        self,
        available_data: list[tuple[list[int], int]],
        source_config: tuple[int, int],
        target_config: tuple[int, int] | None = None,
        source_seeds: list[int] | None = None,
        appears_in_training: bool = False,
        additional_metadata: dict[str, t.Any] | None = None,
    ) -> list[ICLSequence]:
        """Generate ICL sequences for all configured context sizes.

        Args:
            available_data: Pool of data to sample from
            source_config: Source (L, m) configuration
            target_config: Target (L, m) configuration
            source_seeds: Seeds used to generate the data
            appears_in_training: Whether sequences appear in training
            additional_metadata: Extra metadata

        Returns:
            List of generated ICL sequences

        """
        sequences = []

        for context_size in self.icl_params.context_sizes:
            # Generate multiple sequences for each context size
            for _ in range(self.icl_params.sequences_per_context_size):
                sequence = self._create_icl_sequence_from_data(
                    available_data=available_data,
                    context_size=context_size,
                    source_config=source_config,
                    target_config=target_config,
                    source_seeds=source_seeds,
                    appears_in_training=appears_in_training,
                    additional_metadata=additional_metadata,
                )

                if sequence is not None:
                    sequences.append(sequence)

        return sequences

    def get_generation_stats(self) -> dict[str, t.Any]:
        """Get statistics about generated sequences."""
        if not self.generated_sequences:
            return {"total_sequences": 0}

        # Count by context size
        context_size_counts = {}
        for seq in self.generated_sequences:
            size = seq.context_size
            context_size_counts[size] = context_size_counts.get(size, 0) + 1

        # Count by configuration
        config_counts = {}
        for seq in self.generated_sequences:
            config = seq.source_config
            config_counts[config] = config_counts.get(config, 0) + 1

        return {
            "total_sequences": len(self.generated_sequences),
            "eval_type": self.eval_type,
            "context_size_distribution": context_size_counts,
            "config_distribution": config_counts,
            "training_overlap_count": sum(1 for seq in self.generated_sequences if seq.appears_in_training),
            "unique_source_seeds": len(set(seed for seq in self.generated_sequences for seed in seq.source_seeds)),
        }

    def reset_generator(self) -> None:
        """Reset generator state for fresh generation."""
        self.sequence_counter = 0
        self.generated_sequences.clear()
        random.seed(self.config.base_seed)
