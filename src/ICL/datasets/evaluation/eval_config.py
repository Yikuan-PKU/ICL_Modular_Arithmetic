import typing as t
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ICLParams:
    """Parameters for ICL sequence construction."""

    context_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    sequences_per_context_size: int = 100
    max_sequence_length: int | None = None
    min_sequence_length: int | None = None

    def __post_init__(self):
        """Validate ICL parameters."""
        if not self.context_sizes or min(self.context_sizes) < 1:
            raise ValueError("context_sizes must contain positive integers")

        if self.sequences_per_context_size <= 0:
            raise ValueError("sequences_per_context_size must be positive")

    # def __post_init__(self):
    #     """Initialize paths and load configuration using explicit L,M,model_type."""
    #     # Validate that L,M,model_type are explicitly provided
    #     if self.config_L is None or self.config_m is None:
    #         raise ValueError("config_L and config_m must be explicitly provided - no auto-discovery")

    #     if not self.model_type:
    #         raise ValueError("model_type must be explicitly provided from bash script")

    #     # Validate L,M values
    #     if self.config_L < 1:
    #         raise ValueError(f"config_L must be >= 1, got {self.config_L}")
    #     if self.config_m < 1:
    #         raise ValueError(f"config_m must be >= 1, got {self.config_m}")

    #     # Validate model_type
    #     if self.model_type not in ["clm", "mlm"]:
    #         raise ValueError(f"model_type must be 'clm' or 'mlm', got '{self.model_type}'")

    #     print(f"DEBUG __post_init__: After validation, model_type: '{self.model_type}'")

    #     # Auto-generate model_variant if not provided
    #     if not self.model_variant and not self.batch_mode:
    #         print("DEBUG __post_init__: Generating model_variant from training.yaml")
    #         self.model_variant = self._generate_model_variant_from_training_yaml()

    #     # Discover paths using explicit L,M
    #     if not self.eval_dataset_path:
    #         print("DEBUG __post_init__: Discovering paths")
    #         self._discover_paths()

    #     # Load collection configuration
    #     if not self.available_evaluation_types:
    #         print("DEBUG __post_init__: Loading collection config")
    #         self._load_collection_config()

    #     print(f"DEBUG __post_init__: Finished, model_type: '{self.model_type}'")


@dataclass
class MemorizationConfig:
    """Configuration for memorization evaluation."""

    enable: bool = True
    sampling_ratio: float = 0.5  # Sample 50% of training sequences
    ensure_coverage: bool = True  # Ensure all seeds represented
    min_sequences_per_seed: int = 10

    def __post_init__(self):
        """Validate memorization configuration."""
        if not 0.0 < self.sampling_ratio <= 1.0:
            raise ValueError("sampling_ratio must be between 0 and 1")

        if self.min_sequences_per_seed < 1:
            raise ValueError("min_sequences_per_seed must be at least 1")


@dataclass
class IDGeneralizationConfig:
    """Configuration for in-distribution generalization evaluation."""

    enable: bool = True
    use_validation_split: bool = True
    sampling_ratio: float = 1.0  # Use all validation sequences
    min_sequences_per_seed: int = 10

    def __post_init__(self):
        """Validate ID generalization configuration."""
        if not 0.0 < self.sampling_ratio <= 1.0:
            raise ValueError("sampling_ratio must be between 0 and 1")


@dataclass
class OODSameRuleConfig:
    """Configuration for OOD same rule evaluation."""

    enable: bool = True
    num_ood_seeds: int = 3
    ood_seed_offset: int = 100000
    sequences_per_seed: int = 200
    min_sequences_per_seed: int = 50

    def __post_init__(self):
        """Validate OOD same rule configuration."""
        if self.num_ood_seeds < 1:
            raise ValueError("num_ood_seeds must be at least 1")

        if self.sequences_per_seed < self.min_sequences_per_seed:
            raise ValueError("sequences_per_seed must be >= min_sequences_per_seed")


@dataclass
class OODTransferConfig:
    """Configuration for OOD transfer evaluation."""

    enable: bool = True
    transfer_types: list[str] = field(default_factory=lambda: ["depth", "synonym", "full"])
    max_transfer_distance: int = 2
    num_transfer_seeds: int = 2
    sequences_per_seed: int = 200
    min_sequences_per_seed: int = 50
    transfer_seed_offset: int = 200000

    def __post_init__(self):
        """Validate OOD transfer configuration."""
        valid_types = {"depth", "synonym", "full"}
        if not all(t in valid_types for t in self.transfer_types):
            raise ValueError(f"transfer_types must be subset of {valid_types}")

        if self.max_transfer_distance < 1:
            raise ValueError("max_transfer_distance must be at least 1")

        if self.num_transfer_seeds < 1:
            raise ValueError("num_transfer_seeds must be at least 1")


@dataclass
class ICLEvalConfig:
    """Complete configuration for ICL evaluation dataset generation."""

    # Core ICL parameters
    icl_params: ICLParams = field(default_factory=ICLParams)

    # Four evaluation type configurations
    memorization: MemorizationConfig = field(default_factory=MemorizationConfig)
    id_generalization: IDGeneralizationConfig = field(default_factory=IDGeneralizationConfig)
    ood_same_rule: OODSameRuleConfig = field(default_factory=OODSameRuleConfig)
    ood_transfer: OODTransferConfig = field(default_factory=OODTransferConfig)

    # Global settings
    base_seed: int = 42
    create_combined_dataset: bool = True
    save_intermediate: bool = True

    def validate(self) -> None:
        """Validate entire configuration."""
        # Individual configs validate themselves in __post_init__

        # Check that at least one evaluation type is enabled
        enabled_types = [
            self.memorization.enable,
            self.id_generalization.enable,
            self.ood_same_rule.enable,
            self.ood_transfer.enable,
        ]

        if not any(enabled_types):
            raise ValueError("At least one evaluation type must be enabled")

    def get_enabled_types(self) -> list[str]:
        """Get list of enabled evaluation types."""
        enabled = []
        if self.memorization.enable:
            enabled.append("memorization")
        if self.id_generalization.enable:
            enabled.append("id_generalization")
        if self.ood_same_rule.enable:
            enabled.append("ood_same_rule")
        if self.ood_transfer.enable:
            enabled.append("ood_transfer")
        return enabled


def load_icl_eval_config_from_yaml(yaml_config: dict[str, t.Any]) -> ICLEvalConfig:
    """Load ICL evaluation configuration from YAML."""
    eval_section = yaml_config.get("eval_config", {})

    # Extract subsections
    icl_params_dict = eval_section.get("icl_params", {})
    memorization_dict = eval_section.get("memorization", {})
    id_gen_dict = eval_section.get("id_generalization", {})
    ood_same_dict = eval_section.get("ood_same_rule", {})
    ood_transfer_dict = eval_section.get("ood_transfer", {})

    # Create configuration objects
    icl_params = ICLParams(**icl_params_dict)
    memorization = MemorizationConfig(**memorization_dict)
    id_generalization = IDGeneralizationConfig(**id_gen_dict)
    ood_same_rule = OODSameRuleConfig(**ood_same_dict)
    ood_transfer = OODTransferConfig(**ood_transfer_dict)

    # Global settings
    base_seed = eval_section.get("base_seed", 42)
    create_combined = eval_section.get("create_combined_dataset", True)
    save_intermediate = eval_section.get("save_intermediate", True)

    config = ICLEvalConfig(
        icl_params=icl_params,
        memorization=memorization,
        id_generalization=id_generalization,
        ood_same_rule=ood_same_rule,
        ood_transfer=ood_transfer,
        base_seed=base_seed,
        create_combined_dataset=create_combined,
        save_intermediate=save_intermediate,
    )

    config.validate()
    return config


def save_icl_eval_config_to_json(config: ICLEvalConfig, output_path: Path) -> None:
    """Save ICL evaluation configuration to JSON file."""
    import json
    from datetime import datetime

    def dataclass_to_dict(obj):
        """Convert dataclass to dictionary."""
        if hasattr(obj, "__dataclass_fields__"):
            return {field: dataclass_to_dict(getattr(obj, field)) for field in obj.__dataclass_fields__}
        if isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        return obj

    config_dict = {
        "config": dataclass_to_dict(config),
        "created_at": datetime.now().isoformat(),
        "enabled_types": config.get_enabled_types(),
    }

    with output_path.open("w") as f:
        json.dump(config_dict, f, indent=2)


def create_default_eval_config() -> ICLEvalConfig:
    """Create a default ICL evaluation configuration."""
    return ICLEvalConfig(
        icl_params=ICLParams(context_sizes=[1, 2, 3, 4, 5], sequences_per_context_size=100),
        memorization=MemorizationConfig(enable=True, sampling_ratio=0.5),
        id_generalization=IDGeneralizationConfig(enable=True, use_validation_split=True),
        ood_same_rule=OODSameRuleConfig(enable=True, num_ood_seeds=3),
        ood_transfer=OODTransferConfig(enable=True, transfer_types=["depth", "synonym"], max_transfer_distance=2),
    )
