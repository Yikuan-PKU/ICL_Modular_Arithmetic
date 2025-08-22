"""Collection configuration with minimal model config parsing."""

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ICL.eval.collection.data_schema import (
    EvalType,
    generate_model_variant_name,
    load_minimal_model_config,
)
from ICL.settings import PATH


@dataclass
class CollectionConfig:
    """Collection configuration using training script style arguments."""

    # Core experiment identification (from shared args)
    dataset_type: str
    num_seeds: int
    seed: int
    config_L: int | None = None  # Auto-discovered
    config_m: int | None = None  # Auto-discovered

    # Collection mode
    model_variant: str | None = None  # Auto-discovered or specified
    eval_type: EvalType | None = None  # Specified or batch mode
    batch_mode: bool = False

    # Auto-discovered paths
    eval_dataset_path: Path | None = None
    model_base_dir: Path | None = None
    output_dir: Path | None = None
    config_file_path: Path | None = None

    # Collection parameters (loaded from collection.yaml)
    available_evaluation_types: list[EvalType] = field(default_factory=list)
    available_model_variants: list[str] = field(default_factory=list)
    capture_attention: bool = True

    # Evaluation parameters
    context_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])
    control_types: list[str] = field(default_factory=lambda: ["normal", "shuffled_context", "random_context"])

    # Execution parameters
    device: str = "cuda"
    batch_size: int = 32
    max_sequences_per_condition: int = 200

    # Data collection options
    save_intermediate: bool = True
    intermediate_save_frequency: int = 5

    # Memory management
    clear_cache_frequency: int = 10
    max_memory_usage_gb: float = 12.0

    # Error handling
    max_model_failures: int = 5
    retry_failed_models: bool = True
    failure_retry_delay: int = 60

    # Pipeline control
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self):
        """Auto-discover L,M and load configuration."""
        if self.config_L is None or self.config_m is None:
            self._auto_discover_L_M()
        if not self.eval_dataset_path:
            self._discover_paths()
        if not self.available_evaluation_types:
            self._load_collection_config()

    def _auto_discover_L_M(self) -> None:
        """Auto-discover L and M from available dataset directories."""
        # Look for dataset directories matching pattern
        base_pattern = f"{self.dataset_type}_{self.num_seeds}"
        dataset_root = PATH.dataset_root

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        import re

        pattern = re.compile(rf"^{re.escape(base_pattern)}_L(\d+)_M(\d+)$")

        found_configs = []
        for item in dataset_root.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    L = int(match.group(1))
                    M = int(match.group(2))
                    # Verify that eval directory exists
                    eval_dir = item / "eval"
                    if eval_dir.exists():
                        found_configs.append((L, M))

        if not found_configs:
            raise FileNotFoundError(f"No dataset configurations found for {base_pattern}")

        if len(found_configs) > 1:
            print(f"Warning: Multiple L,M configurations found: {found_configs}")
            print(f"Using the first one: {found_configs[0]}")

        self.config_L, self.config_m = found_configs[0]
        print(f"Auto-discovered L={self.config_L}, M={self.config_m}")

    def get_shared_identifier(self) -> str:
        """Generate shared identifier."""
        return f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"

    def _discover_paths(self) -> None:
        """Auto-discover all required paths."""
        shared_id = self.get_shared_identifier()

        # Base paths
        dataset_base = PATH.dataset_root / shared_id
        model_base = PATH.model_dir / shared_id
        config_base = PATH.conf_dir / shared_id

        # Collection config file path
        self.config_file_path = config_base / "collection.yaml"

        if not self.batch_mode and self.eval_type and self.model_variant:
            # Single combination mode
            self.eval_dataset_path = dataset_base / "eval" / self.eval_type / "dataset"
            self.model_base_dir = model_base / self.model_variant

            # Output directory: results/{shared_id}/{model_variant}/{eval_type}/collection
            self.output_dir = PATH.result_dir / shared_id / self.model_variant / self.eval_type / "collection"
        else:
            # Batch mode or discovery mode - use base paths
            self.model_base_dir = model_base
            self.output_dir = PATH.result_dir / shared_id

    def _load_collection_config(self) -> None:
        """Load collection configuration from YAML file."""
        try:
            if self.config_file_path and self.config_file_path.exists():
                with open(self.config_file_path) as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
                print(f"Warning: No collection.yaml found at {self.config_file_path}")

            # Load base config
            base_config = config_data.get("base_config", {})
            self.available_evaluation_types = base_config.get(
                "evaluation_types", ["memorization", "id_generalization", "ood_same_rule", "ood_transfer"]
            )

            # Auto-discover model variants from config files (minimal parsing)
            self.available_model_variants = self._auto_discover_model_variants()

            # Load collection config
            collection_config = config_data.get("collection_config", {})
            self.capture_attention = collection_config.get("capture_attention", self.capture_attention)

            # Evaluation parameters
            eval_params = collection_config.get("evaluation_parameters", {})
            self.context_sizes = eval_params.get("context_sizes", self.context_sizes)
            self.control_types = eval_params.get("control_types", self.control_types)

            # Execution parameters
            exec_params = collection_config.get("execution_parameters", {})
            self.batch_size = exec_params.get("batch_size", self.batch_size)
            self.max_sequences_per_condition = exec_params.get(
                "max_sequences_per_condition", self.max_sequences_per_condition
            )
            self.device = exec_params.get("device", self.device)
            self.save_intermediate = exec_params.get("save_intermediate", self.save_intermediate)
            self.intermediate_save_frequency = exec_params.get(
                "intermediate_save_frequency", self.intermediate_save_frequency
            )

            # Memory management
            memory_config = collection_config.get("memory_management", {})
            self.clear_cache_frequency = memory_config.get("clear_cache_frequency", self.clear_cache_frequency)
            self.max_memory_usage_gb = memory_config.get("max_memory_usage_gb", self.max_memory_usage_gb)

            # Error handling
            error_config = collection_config.get("error_handling", {})
            self.max_model_failures = error_config.get("max_model_failures", self.max_model_failures)
            self.retry_failed_models = error_config.get("retry_failed_models", self.retry_failed_models)
            self.failure_retry_delay = error_config.get("failure_retry_delay", self.failure_retry_delay)

        except Exception as e:
            print(f"Error loading collection config: {e}")
            print("Using default configuration values")

    def _auto_discover_model_variants(self) -> list[str]:
        """Auto-discover model variants from existing model config files using minimal parsing."""
        shared_id = self.get_shared_identifier()
        config_dir = PATH.conf_dir / shared_id

        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            return []

        model_variants = []

        # Scan for model config files (excluding collection.yaml)
        for config_file in config_dir.glob("*.yaml"):
            if config_file.name != "collection.yaml":
                try:
                    # Use minimal config loading - only parse 3 essential fields
                    model_config = load_minimal_model_config(config_file)
                    variant_name = generate_model_variant_name(model_config)
                    model_variants.append(variant_name)

                    print(f"Auto-discovered model variant: {variant_name}")
                    print(f"  From: {config_file.name}")
                    print(
                        f"  Config: task_name={model_config['task_name']}, "
                        f"shuffle={model_config['shuffle_before_packing']}, "
                        f"seed_balanced={model_config['seed_balanced_batching']}"
                    )

                except Exception as e:
                    print(f"Warning: Failed to process config file {config_file}: {e}")

        return model_variants

    def discover_available_combinations(self) -> list[tuple[str, EvalType]]:
        """Discover all available (model_variant, eval_type) combinations."""
        from ICL.settings import PATH

        combinations = []
        shared_id = self.get_shared_identifier()

        # Use auto-discovered model variants
        model_variants = self.available_model_variants

        # Discover evaluation types
        eval_base = PATH.dataset_root / shared_id / "eval"
        if eval_base.exists():
            discovered_eval_types = [d.name for d in eval_base.iterdir() if d.is_dir()]
            if self.available_evaluation_types:
                # Filter by config
                eval_types = [t for t in discovered_eval_types if t in self.available_evaluation_types]
            else:
                eval_types = discovered_eval_types
        else:
            eval_types = []

        # Create all combinations and validate they exist
        model_base = PATH.model_dir / shared_id
        for variant in model_variants:
            for eval_type in eval_types:
                eval_dataset_path = eval_base / eval_type / "dataset"
                model_variant_path = model_base / variant

                if eval_dataset_path.exists() and model_variant_path.exists():
                    combinations.append((variant, eval_type))

        return combinations

    def create_single_config(self, model_variant: str, eval_type: EvalType) -> "CollectionConfig":
        """Create a single-combination config from this batch config."""
        return CollectionConfig(
            dataset_type=self.dataset_type,
            num_seeds=self.num_seeds,
            seed=self.seed,
            config_L=self.config_L,
            config_m=self.config_m,
            model_variant=model_variant,
            eval_type=eval_type,
            capture_attention=self.capture_attention,
            context_sizes=self.context_sizes,
            control_types=self.control_types,
            device=self.device,
            batch_size=self.batch_size,
            max_sequences_per_condition=self.max_sequences_per_condition,
            save_intermediate=self.save_intermediate,
            intermediate_save_frequency=self.intermediate_save_frequency,
            clear_cache_frequency=self.clear_cache_frequency,
            max_memory_usage_gb=self.max_memory_usage_gb,
            max_model_failures=self.max_model_failures,
            retry_failed_models=self.retry_failed_models,
            failure_retry_delay=self.failure_retry_delay,
            resume=self.resume,
            overwrite=self.overwrite,
            batch_mode=False,
        )

    def validate(self) -> None:
        """Validate configuration and paths."""
        if self.batch_mode:
            # Validate base paths for batch mode
            from ICL.settings import PATH

            shared_id = self.get_shared_identifier()
            base_dataset = PATH.dataset_root / shared_id
            base_model = PATH.model_dir / shared_id

            if not base_dataset.exists():
                raise FileNotFoundError(f"Dataset directory not found: {base_dataset}")

            if not base_model.exists():
                raise FileNotFoundError(f"Model directory not found: {base_model}")

        else:
            # Validate specific paths for single mode
            if not self.eval_dataset_path or not self.eval_dataset_path.exists():
                raise FileNotFoundError(f"Evaluation dataset not found: {self.eval_dataset_path}")

            if not self.model_base_dir or not self.model_base_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_base_dir}")

        # Validate parameters
        if not self.context_sizes or any(k <= 0 for k in self.context_sizes):
            raise ValueError("Context sizes must be positive integers")

    def create_output_structure(self) -> None:
        """Create output directory structure."""
        if not self.output_dir:
            raise ValueError("Output directory not set")

        if self.output_dir.exists() and not self.overwrite and not self.resume:
            raise FileExistsError(f"Output directory exists: {self.output_dir}. Use --overwrite or --resume")

        # Create directory structure
        directories = [
            self.output_dir,
            self.output_dir / "raw_evaluations",
            self.output_dir / "metadata",
            self.output_dir / "logs",
        ]

        if self.capture_attention:
            directories.append(self.output_dir / "raw_evaluations" / "attention_data")

        if self.save_intermediate:
            directories.append(self.output_dir / "intermediate")

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_resume_path(self) -> Path:
        """Get path for resume checkpoint."""
        return self.output_dir / "metadata" / "resume_checkpoint.json"

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_type": self.dataset_type,
            "num_seeds": self.num_seeds,
            "seed": self.seed,
            "config_L": self.config_L,
            "config_m": self.config_m,
            "model_variant": self.model_variant,
            "eval_type": self.eval_type,
            "paths": {
                "eval_dataset_path": str(self.eval_dataset_path) if self.eval_dataset_path else None,
                "model_base_dir": str(self.model_base_dir) if self.model_base_dir else None,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "config_file_path": str(self.config_file_path) if self.config_file_path else None,
            },
            "available_evaluation_types": self.available_evaluation_types,
            "available_model_variants": self.available_model_variants,
            "capture_attention": self.capture_attention,
            "context_sizes": self.context_sizes,
            "control_types": self.control_types,
            "execution_parameters": {
                "device": self.device,
                "batch_size": self.batch_size,
                "max_sequences_per_condition": self.max_sequences_per_condition,
                "save_intermediate": self.save_intermediate,
                "intermediate_save_frequency": self.intermediate_save_frequency,
            },
            "memory_management": {
                "clear_cache_frequency": self.clear_cache_frequency,
                "max_memory_usage_gb": self.max_memory_usage_gb,
            },
            "error_handling": {
                "max_model_failures": self.max_model_failures,
                "retry_failed_models": self.retry_failed_models,
                "failure_retry_delay": self.failure_retry_delay,
            },
            "batch_mode": self.batch_mode,
        }

    @classmethod
    def from_args(
        cls,
        dataset_type: str,
        num_seeds: int,
        seed: int,
        model_variant: str | None = None,
        eval_type: EvalType | None = None,
        device: str = "cuda",
        overwrite: bool = False,
        resume: bool = False,
        batch_mode: bool = False,
        **kwargs,
    ) -> "CollectionConfig":
        """Create CollectionConfig from command line arguments."""
        config = cls(
            dataset_type=dataset_type,
            num_seeds=num_seeds,
            seed=seed,
            model_variant=model_variant,
            eval_type=eval_type,
            device=device,
            overwrite=overwrite,
            resume=resume,
            batch_mode=batch_mode,
        )

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


def create_collection_parser() -> t.Any:
    """Create argument parser for collection phase (training script style)."""
    import argparse

    parser = argparse.ArgumentParser(description="ICL Evaluation Collection Phase")

    # Core identification (training script style)
    core_group = parser.add_argument_group("Core Experiment Parameters")
    core_group.add_argument("--dataset-type", required=True, choices=["uniform", "zipf"], help="Dataset type")
    core_group.add_argument("--num-seeds", type=int, required=True, help="Number of seeds")
    core_group.add_argument("--seed", type=int, required=True, help="Random seed")

    # Single vs batch mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--batch-mode", action="store_true", help="Run all available combinations")

    single_group = parser.add_argument_group("Single Combination Mode")
    single_group.add_argument("--model-variant", help="Specific model variant (auto-discovered from configs)")
    single_group.add_argument(
        "--eval-type",
        choices=["memorization", "id_generalization", "ood_same_rule", "ood_transfer"],
        help="Specific evaluation type",
    )

    # Collection-specific arguments
    collection_group = parser.add_argument_group("Collection Parameters")
    collection_group.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    collection_group.add_argument("--batch-size", type=int, help="Batch size for evaluation (overrides config)")
    collection_group.add_argument("--max-sequences", type=int, help="Max sequences per condition (overrides config)")
    collection_group.add_argument("--no-attention", action="store_true", help="Disable attention capture")
    collection_group.add_argument("--no-intermediate", action="store_true", help="Disable intermediate saving")

    # Pipeline control
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    control_group.add_argument("--resume", action="store_true", help="Resume from existing partial results")
    control_group.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running")
    util_group.add_argument("--list-combinations", action="store_true", help="List available combinations")

    return parser


def create_collection_config_from_args(args) -> CollectionConfig:
    """Create CollectionConfig from parsed arguments."""
    # Skip validation for utility commands
    is_utility_command = getattr(args, "list_combinations", False) or getattr(args, "validate_only", False)

    # Validate argument combinations (except for utility commands)
    if not is_utility_command and not args.batch_mode and (not args.model_variant or not args.eval_type):
        raise ValueError("Must specify --model-variant and --eval-type, or use --batch-mode")

    config = CollectionConfig.from_args(
        dataset_type=args.dataset_type,
        num_seeds=args.num_seeds,
        seed=args.seed,
        model_variant=getattr(args, "model_variant", None),
        eval_type=getattr(args, "eval_type", None),
        device=getattr(args, "device", "cuda"),
        overwrite=getattr(args, "overwrite", False),
        resume=getattr(args, "resume", False),
        batch_mode=getattr(args, "batch_mode", False),
    )

    # Apply command line overrides
    if hasattr(args, "batch_size") and args.batch_size:
        config.batch_size = args.batch_size

    if hasattr(args, "max_sequences") and args.max_sequences:
        config.max_sequences_per_condition = args.max_sequences

    if hasattr(args, "no_attention") and args.no_attention:
        config.capture_attention = False

    if hasattr(args, "no_intermediate") and args.no_intermediate:
        config.save_intermediate = False

    return config
