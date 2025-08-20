import copy
import random
import typing as t
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from ICL.datasets.RHM import RandomHierarchyModel
from ICL.datasets.utils import load_training_metadata
from ICL.settings import DatasetConfig

T = t.TypeVar("T")
ConfigTuple = tuple[int, int]  # (L, m)


@dataclass
class TransferConfig:
    """Configuration for transfer learning evaluation."""

    # Core configuration
    train_config_idx: int = 0

    # Generation parameters (loaded from YAML)
    num_rules_per_config: int = 32
    sequences_per_rule: int = 200
    base_seed: int = 42

    # Transfer testing parameters
    max_depth: int = 5
    max_multiplicity: int = 6
    depth_steps: list[int] = None
    synonym_steps: list[int] = None
    full_transfer_limit: int = 10

    # ICL parameters
    context_sizes: list[int] = None
    control_types: list[str] = None

    # Pipeline control
    include_controls: bool = True
    save_intermediate: bool = True

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.context_sizes is None:
            self.context_sizes = [1, 2, 3, 4, 5]

        if self.control_types is None:
            self.control_types = ["normal", "shuffled_context", "random_context"]

        if self.depth_steps is None:
            self.depth_steps = [1, 2]

        if self.synonym_steps is None:
            self.synonym_steps = [1, 2]

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_rules_per_config <= 0:
            raise ValueError("num_rules_per_config must be positive")

        if self.sequences_per_rule <= 0:
            raise ValueError("sequences_per_rule must be positive")

        if not self.context_sizes or min(self.context_sizes) < 1:
            raise ValueError("context_sizes must contain positive integers")

        if self.max_depth < 2:
            raise ValueError("max_depth must be at least 2")

        if self.max_multiplicity < 2:
            raise ValueError("max_multiplicity must be at least 2")

        valid_control_types = {"normal", "shuffled_context", "random_context"}
        if not all(ct in valid_control_types for ct in self.control_types):
            raise ValueError(f"control_types must be subset of {valid_control_types}")


@dataclass
class EvaluationSequence:
    """Single evaluation sequence with hierarchical annotations."""

    # Basic ICL structure
    context_features: list[list[int]]
    context_labels: list[int]
    query_features: list[int]
    query_label: int
    context_size: int
    sequence_id: int

    # Multi-token testing annotations
    step_labels: list[int]  # Next token at each position
    hierarchy_path: list[int]  # Rule indices at each level
    critical_positions: list[int]  # Key decision points

    # Transfer metadata
    config: ConfigTuple
    transfer_condition: str
    control_type: str = "normal"

    # Rule metadata reference
    rule_metadata: dict[str, t.Any] = None

    def __post_init__(self):
        """Initialize rule metadata if not provided."""
        if self.rule_metadata is None:
            self.rule_metadata = {}

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for dataset storage."""
        return {
            "context_features": self.context_features,
            "context_labels": self.context_labels,
            "query_features": self.query_features,
            "query_label": self.query_label,
            "context_size": self.context_size,
            "sequence_id": self.sequence_id,
            "step_labels": self.step_labels,
            "hierarchy_path": self.hierarchy_path,
            "critical_positions": self.critical_positions,
            "config_L": self.config[0],
            "config_m": self.config[1],
            "transfer_condition": self.transfer_condition,
            "control_type": self.control_type,
            "rule_metadata": self.rule_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "EvaluationSequence":
        """Create from dictionary."""
        return cls(
            context_features=data["context_features"],
            context_labels=data["context_labels"],
            query_features=data["query_features"],
            query_label=data["query_label"],
            context_size=data["context_size"],
            sequence_id=data["sequence_id"],
            step_labels=data["step_labels"],
            hierarchy_path=data["hierarchy_path"],
            critical_positions=data["critical_positions"],
            config=(data["config_L"], data["config_m"]),
            transfer_condition=data["transfer_condition"],
            control_type=data.get("control_type", "normal"),
            rule_metadata=data.get("rule_metadata", {}),
        )


class TransferEvaluationGenerator:
    """Generates transfer evaluation datasets with hierarchical annotations."""

    def __init__(self, train_metadata_path: Path | str, dataset_config: DatasetConfig, base_seed: int = 42):
        """Initialize with training dataset metadata and evaluation config.

        Args:
            train_metadata_path: Path to training dataset metadata.pkl
            dataset_config: Evaluation dataset configuration
            base_seed: Base seed for evaluation generation

        """
        self.base_seed = base_seed
        self.dataset_config = dataset_config
        self.train_metadata = load_training_metadata(train_metadata_path)
        self.eval_seed_start = 10000  # Large gap from training seeds
        self._setup_seed_isolation()

    def _setup_seed_isolation(self):
        """Setup seed ranges to ensure no overlap with training."""
        # Find maximum seed used in training
        max_train_seed = 0
        for seeds_list in self.train_metadata.used_seeds.values():
            max_train_seed = max(max_train_seed, max(seeds_list))

        # Ensure large gap between training and evaluation seeds
        self.eval_seed_start = max(10000, max_train_seed + 1000)

        print(f"Training seeds: 0-{max_train_seed}")
        print(f"Evaluation seeds start: {self.eval_seed_start}")

    def _verify_rule_independence(self, eval_rules: dict, train_config: ConfigTuple) -> bool:
        """Verify that evaluation rules are independent from training rules."""
        if train_config not in self.train_metadata.used_seeds:
            return True

        train_task_ids = self.train_metadata.used_seeds[train_config]

        for train_task_id in train_task_ids:
            if train_task_id in self.train_metadata.rules:
                train_rules = self.train_metadata.rules[train_task_id]["rules_dict"]

                for level in eval_rules:
                    if level in train_rules:
                        if torch.equal(eval_rules[level], train_rules[level]):
                            return False

        return True

    def _generate_verified_model(
        self, L: int, m: int, seed: int, sequences_per_rule: int, max_attempts: int = 10
    ) -> RandomHierarchyModel | None:
        """Generate a model with verified independence from training."""
        config = (L, m)

        for attempt in range(max_attempts):
            current_seed = seed + attempt * 1000

            try:
                model = RandomHierarchyModel(
                    num_features=self.train_metadata.generation_params.get("vocab_size", 32),
                    num_classes=self.train_metadata.generation_params.get("num_classes", 10),
                    num_synonyms=m,
                    tuple_size=self.train_metadata.generation_params.get("tuple_size", 2),
                    num_layers=L,
                    seed_rules=current_seed,
                    seed_sample=current_seed + 50000,
                    train_size=sequences_per_rule,
                    input_format="long",
                    replacement=True,
                )

                # Verify independence
                if not self._verify_rule_independence(model.rules, config):
                    continue

                return model

            except Exception as e:
                warnings.warn(f"Failed to generate model for {config}, attempt {attempt}: {e}")
                continue

        raise RuntimeError(f"Failed to generate verified model for config {config} after {max_attempts} attempts")

    def _extract_hierarchical_annotations(
        self, features: torch.Tensor, model: RandomHierarchyModel
    ) -> tuple[list[int], list[int], list[int]]:
        """Extract hierarchical annotations from a sequence."""
        # Convert features to list
        if hasattr(features, "tolist"):
            feature_list = features.tolist()
        else:
            feature_list = list(features)

        # For now, create simplified annotations
        # In a full implementation, this would trace through the RHM generation process

        # Step labels (next token at each position)
        step_labels = feature_list[1:] if len(feature_list) > 1 else []

        # Hierarchy path (simplified - would need actual rule tracing)
        hierarchy_path = list(range(len(feature_list)))

        # Critical positions (simplified heuristic)
        critical_positions = [0, len(feature_list) - 1] if len(feature_list) > 1 else [0]

        return step_labels, hierarchy_path, critical_positions

    def _create_icl_sequence_with_annotations(
        self,
        context_features: list[torch.Tensor],
        context_labels: list[int],
        query_features: torch.Tensor,
        query_label: int,
        model: RandomHierarchyModel,
        sequence_id: int,
        context_size: int,
        config: ConfigTuple,
        transfer_condition: str,
        control_type: str = "normal",
    ) -> dict[str, t.Any]:
        """Create ICL sequence with hierarchical annotations."""
        # Convert tensors to lists
        context_features_list = []
        for cf in context_features:
            if hasattr(cf, "tolist"):
                context_features_list.append(cf.tolist())
            else:
                context_features_list.append(list(cf))

        if hasattr(query_features, "tolist"):
            query_features_list = query_features.tolist()
        else:
            query_features_list = list(query_features)

        # Extract hierarchical annotations for query
        step_labels, hierarchy_path, critical_positions = self._extract_hierarchical_annotations(query_features, model)

        # Create rule metadata
        rule_metadata = {
            "L": config[0],
            "m": config[1],
            "vocab_size": model.num_features,
            "num_classes": model.num_classes,
            "rule_references": f"config_L{config[0]}_m{config[1]}",
        }

        return {
            "context_features": context_features_list,
            "context_labels": [int(cl) if hasattr(cl, "item") else cl for cl in context_labels],
            "query_features": query_features_list,
            "query_label": int(query_label) if hasattr(query_label, "item") else query_label,
            "context_size": context_size,
            "sequence_id": sequence_id,
            "step_labels": step_labels,
            "hierarchy_path": hierarchy_path,
            "critical_positions": critical_positions,
            "config_L": config[0],
            "config_m": config[1],
            "transfer_condition": transfer_condition,
            "control_type": control_type,
            "rule_metadata": rule_metadata,
        }

    def _create_icl_sequences_for_model(
        self, model: RandomHierarchyModel, context_sizes: list[int], config: ConfigTuple, transfer_condition: str
    ) -> list[dict[str, t.Any]]:
        """Create ICL sequences with annotations for a single model."""
        sequences = []

        for k in context_sizes:
            sequences_per_k = min(len(model) // (len(context_sizes) * (k + 1)), 50)

            for start_idx in range(0, sequences_per_k * (k + 1), k + 1):
                if start_idx + k >= len(model):
                    break

                # Get context examples
                context_features = []
                context_labels = []

                for i in range(k):
                    features, label = model[start_idx + i]
                    context_features.append(features)
                    context_labels.append(label)

                # Get query
                query_features, query_label = model[start_idx + k]

                # Create sequence with annotations
                sequence = self._create_icl_sequence_with_annotations(
                    context_features=context_features,
                    context_labels=context_labels,
                    query_features=query_features,
                    query_label=query_label,
                    model=model,
                    sequence_id=len(sequences),
                    context_size=k,
                    config=config,
                    transfer_condition=transfer_condition,
                    control_type="normal",
                )

                sequences.append(sequence)

        return sequences

    def _create_control_sequences(
        self, base_sequences: list[dict[str, t.Any]], control_types: list[str]
    ) -> list[dict[str, t.Any]]:
        """Create control sequences from base sequences."""
        control_sequences = []

        for base_seq in base_sequences:
            for control_type in control_types:
                if control_type == "normal":
                    continue  # Already have normal sequences

                control_seq = copy.deepcopy(base_seq)
                control_seq["control_type"] = control_type

                if control_type == "shuffled_context":
                    # Shuffle context order
                    context_pairs = list(
                        zip(control_seq["context_features"], control_seq["context_labels"], strict=False)
                    )
                    random.shuffle(context_pairs)
                    control_seq["context_features"] = [pair[0] for pair in context_pairs]
                    control_seq["context_labels"] = [pair[1] for pair in context_pairs]

                elif control_type == "random_context":
                    # Replace with random context (simplified)
                    if len(base_sequences) > control_seq["context_size"]:
                        random_contexts = random.sample(base_sequences, control_seq["context_size"])
                        control_seq["context_features"] = [ctx["query_features"] for ctx in random_contexts]
                        control_seq["context_labels"] = [ctx["query_label"] for ctx in random_contexts]

                control_sequences.append(control_seq)

        return control_sequences

    def generate_complete_evaluation_dataset(
        self, config, output_dir: Path
    ) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
        """Generate complete evaluation dataset with hierarchical annotations."""
        print("=" * 80)
        print("GENERATING ENHANCED TRANSFER EVALUATION DATASET")
        print("=" * 80)
        print(f"Training configurations: {len(self.train_metadata.config_list)}")
        print(f"Evaluation seed start: {self.eval_seed_start}")
        print()

        # Get training configuration
        available_configs = self.train_metadata.config_list
        if config.train_config_idx >= len(available_configs):
            raise ValueError(f"train_config_idx {config.train_config_idx} out of range")

        train_config = available_configs[config.train_config_idx]
        print(f"Using training configuration: {train_config}")

        # Generate test configurations
        test_configs = self._generate_test_configurations(config, train_config)
        print(f"Test configurations: {test_configs}")

        # Store all sequences
        all_sequences = []
        condition_stats = {}

        # Generate sequences for each transfer condition
        conditions = [
            ("within_config", [train_config]),
            ("depth_transfer", [tc for tc in test_configs if tc[1] == train_config[1] and tc[0] > train_config[0]]),
            ("synonym_transfer", [tc for tc in test_configs if tc[0] == train_config[0] and tc[1] > train_config[1]]),
            ("full_transfer", [tc for tc in test_configs if tc[0] > train_config[0] and tc[1] > train_config[1]]),
        ]

        base_seed_offset = 0

        for condition_name, condition_configs in conditions:
            print(f"\nGenerating {condition_name}...")
            condition_sequences = []

            for test_config in condition_configs:
                config_sequences = []

                for model_idx in range(config.num_rules_per_config):
                    seed = self.eval_seed_start + base_seed_offset + model_idx

                    try:
                        # Generate model
                        model = self._generate_verified_model(
                            L=test_config[0], m=test_config[1], seed=seed, sequences_per_rule=config.sequences_per_rule
                        )

                        # Generate sequences
                        model_sequences = self._create_icl_sequences_for_model(
                            model=model,
                            context_sizes=config.context_sizes,
                            config=test_config,
                            transfer_condition=condition_name,
                        )

                        # Generate control sequences
                        if config.include_controls:
                            control_sequences = self._create_control_sequences(model_sequences, config.control_types)
                            model_sequences.extend(control_sequences)

                        config_sequences.extend(model_sequences)

                    except Exception as e:
                        warnings.warn(f"Failed to generate model for {test_config}: {e}")
                        continue

                condition_sequences.extend(config_sequences)
                base_seed_offset += 10000  # Separate seed ranges for each config

            all_sequences.extend(condition_sequences)
            condition_stats[condition_name] = {
                "total_sequences": len(condition_sequences),
                "configs": condition_configs if condition_configs else [train_config],
            }

            print(f"  Generated {len(condition_sequences)} sequences for {condition_name}")

        # Convert to HuggingFace dataset format
        dataset_dict = self._convert_to_dataset_dict(all_sequences)

        # Create metadata
        metadata = {
            "dataset_config": {
                "dataset_type": self.dataset_config.dataset_type,
                "mixture_type": self.dataset_config.mixture_type,
                "total_rules": self.dataset_config.total_rules,
                "seed": self.dataset_config.seed,
            },
            "generation_params": {
                "base_seed": config.base_seed,
                "eval_seed_start": self.eval_seed_start,
                "num_rules_per_config": config.num_rules_per_config,
                "sequences_per_rule": config.sequences_per_rule,
            },
            "transfer_conditions": [name for name, _ in conditions],
            "train_config": train_config,
            "test_configs": test_configs,
            "icl_params": {"context_sizes": config.context_sizes, "control_types": config.control_types},
            "condition_stats": condition_stats,
            "dataset_stats": {
                "total_sequences": len(all_sequences),
                "unique_configs": len(set((seq["config_L"], seq["config_m"]) for seq in all_sequences)),
                "transfer_conditions": len(conditions),
                "control_types": len(config.control_types),
            },
            "hierarchical_annotations": {
                "step_labels_included": True,
                "hierarchy_path_included": True,
                "critical_positions_included": True,
                "rule_metadata_included": True,
            },
        }

        print(f"\nGenerated {len(all_sequences)} total sequences")
        return dataset_dict, metadata

    def _generate_test_configurations(self, config, train_config: ConfigTuple) -> list[ConfigTuple]:
        """Generate test configurations for transfer evaluation."""
        train_L, train_m = train_config
        test_configs = []

        # Depth transfer configurations
        for step in config.depth_steps:
            if train_L + step <= config.max_depth:
                test_configs.append((train_L + step, train_m))

        # Synonym transfer configurations
        for step in config.synonym_steps:
            if train_m + step <= config.max_multiplicity:
                test_configs.append((train_L, train_m + step))

        # Full transfer configurations
        for L_step in config.depth_steps[:2]:
            for m_step in config.synonym_steps[:2]:
                new_L, new_m = train_L + L_step, train_m + m_step
                if new_L <= config.max_depth and new_m <= config.max_multiplicity:
                    test_configs.append((new_L, new_m))

        # Remove duplicates and limit
        test_configs = list(set(test_configs))
        test_configs = test_configs[: config.full_transfer_limit]

        return test_configs

    def _convert_to_dataset_dict(self, sequences: list[dict[str, t.Any]]) -> dict[str, t.Any]:
        """Convert sequences to HuggingFace dataset dictionary format."""
        if not sequences:
            return {}

        # Initialize lists for each field
        dataset_dict = defaultdict(list)

        for seq in sequences:
            # Flatten context features and labels for HuggingFace format
            flattened_context = []
            for cf, cl in zip(seq["context_features"], seq["context_labels"], strict=False):
                flattened_context.extend(cf)
                flattened_context.append(cl)

            # Create input_ids by combining context and query
            input_ids = flattened_context + seq["query_features"]

            # Add to dataset dict
            dataset_dict["input_ids"].append(input_ids)
            dataset_dict["context_features"].append(seq["context_features"])
            dataset_dict["context_labels"].append(seq["context_labels"])
            dataset_dict["query_features"].append(seq["query_features"])
            dataset_dict["query_label"].append(seq["query_label"])
            dataset_dict["context_size"].append(seq["context_size"])
            dataset_dict["sequence_id"].append(seq["sequence_id"])
            dataset_dict["step_labels"].append(seq["step_labels"])
            dataset_dict["hierarchy_path"].append(seq["hierarchy_path"])
            dataset_dict["critical_positions"].append(seq["critical_positions"])
            dataset_dict["config_L"].append(seq["config_L"])
            dataset_dict["config_m"].append(seq["config_m"])
            dataset_dict["transfer_condition"].append(seq["transfer_condition"])
            dataset_dict["control_type"].append(seq["control_type"])
            dataset_dict["rule_metadata"].append(seq["rule_metadata"])

        return dict(dataset_dict)
