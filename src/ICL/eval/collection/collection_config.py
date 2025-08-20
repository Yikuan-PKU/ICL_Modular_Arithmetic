"""Revised collection configuration using experiment-based auto-discovery."""

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ICL.settings import DatasetConfig


@dataclass
class CollectionConfig:
    """Simplified collection configuration using auto-discovery."""

    # Core experiment identification
    dataset_config: DatasetConfig

    # Auto-discovered paths
    eval_dataset_path: Path = None
    checkpoint_dirs: list[Path] = field(default_factory=list)
    output_dir: Path = None
    config_file_path: Path = None

    # Collection parameters (loaded from collection.yaml)
    target_configs: list[tuple[int, int]] = field(default_factory=list)
    diversity_levels: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    model_types: list[str] = field(default_factory=lambda: ["causal_lm", "mlm"])

    # Evaluation parameters
    context_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])
    transfer_conditions: list[str] = field(
        default_factory=lambda: ["within_config", "cross_L", "cross_m", "cross_config"]
    )
    control_types: list[str] = field(default_factory=lambda: ["normal", "shuffled_context", "random_context"])

    # Execution parameters
    device: str = "cuda"
    batch_size: int = 32
    max_sequences_per_condition: int = 200

    # Data collection options
    capture_attention: bool = True
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
        """Auto-discover paths and load configuration."""
        if not self.eval_dataset_path:
            self._discover_paths()
        if not self.target_configs:
            self._load_collection_config()

    def _discover_paths(self) -> None:
        """Auto-discover all required paths using settings infrastructure."""
        # Get dataset paths
        dataset_paths = self.dataset_config.get_dataset_paths()

        # Look for evaluation dataset in multiple formats
        dataset_dir = dataset_paths["dataset_dir"]

        # Check for HuggingFace dataset format
        hf_dataset_path = dataset_dir / "dataset"
        if hf_dataset_path.exists() and hf_dataset_path.is_dir():
            self.eval_dataset_path = hf_dataset_path

        # Check for JSON format
        elif (dataset_dir / "evaluation_dataset.json").exists():
            self.eval_dataset_path = dataset_dir / "evaluation_dataset.json"

        # Check for alternative JSON names
        elif (dataset_dir / "dataset.json").exists():
            self.eval_dataset_path = dataset_dir / "dataset.json"

        # Check in parent directory as fallback
        elif (dataset_dir.parent / "evaluation_dataset.json").exists():
            self.eval_dataset_path = dataset_dir.parent / "evaluation_dataset.json"

        else:
            # Set path anyway for better error messages
            self.eval_dataset_path = dataset_dir / "evaluation_dataset.json"

        # Collection config file path
        self.config_file_path = dataset_paths["config_dir"] / "collection.yaml"

        # Auto-discover checkpoint directories
        self._discover_checkpoint_dirs()

        # Output directory
        base_name = self.dataset_config.to_base_name()
        from ICL.settings import PATH

        self.output_dir = PATH.result_dir / f"{base_name}_collection"

    def _discover_checkpoint_dirs(self) -> None:
        """Auto-discover model checkpoint directories for this experiment."""
        from ICL.settings import PATH

        base_name = self.dataset_config.to_base_name()
        model_base_dir = PATH.model_dir / base_name

        # Look for model type subdirectories
        checkpoint_dirs = []
        for model_type in ["causal_lm", "mlm"]:
            model_dir = model_base_dir / model_type
            if model_dir.exists():
                checkpoint_dirs.append(model_dir)

        # Also check for direct model directories (alternative structure)
        if model_base_dir.exists() and any(model_base_dir.iterdir()):
            # Check if there are checkpoint-like subdirectories directly
            for subdir in model_base_dir.iterdir():
                if subdir.is_dir() and (
                    "checkpoint" in subdir.name.lower()
                    or subdir.name.startswith("step")
                    or (subdir / "config.json").exists()
                ):
                    checkpoint_dirs.append(model_base_dir)
                    break

        self.checkpoint_dirs = checkpoint_dirs

    def _load_collection_config(self) -> None:
        """Load collection configuration from experiment config folder."""
        try:
            # Direct YAML loading since load_experiment_config doesn't support "collection" yet
            if self.config_file_path and self.config_file_path.exists():
                with open(self.config_file_path) as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            if not config_data:
                print(f"Warning: No collection.yaml found at {self.config_file_path}")
                print("Using default configuration values")
                # Set some reasonable defaults
                self.target_configs = [(2, 2), (2, 3), (3, 2), (3, 3)]
                return

            # Load base config
            base_config = config_data.get("base_config", {})
            self.target_configs = [tuple(config) for config in base_config.get("target_configs", [])]
            self.diversity_levels = base_config.get("diversity_levels", self.diversity_levels)
            self.model_types = base_config.get("model_types", self.model_types)

            # Load collection config
            collection_config = config_data.get("collection_config", {})

            # Evaluation parameters
            eval_params = collection_config.get("evaluation_parameters", {})
            self.context_sizes = eval_params.get("context_sizes", self.context_sizes)
            self.transfer_conditions = eval_params.get("transfer_conditions", self.transfer_conditions)
            self.control_types = eval_params.get("control_types", self.control_types)

            # Execution parameters
            exec_params = collection_config.get("execution_parameters", {})
            self.batch_size = exec_params.get("batch_size", self.batch_size)
            self.max_sequences_per_condition = exec_params.get(
                "max_sequences_per_condition", self.max_sequences_per_condition
            )
            self.capture_attention = exec_params.get("capture_attention", self.capture_attention)
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

    def validate(self) -> None:
        """Validate configuration and paths."""
        # Check evaluation dataset exists
        if not self.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.eval_dataset_path}")

        # Check if it's a HuggingFace dataset directory
        if self.eval_dataset_path.is_dir():
            # Check for HuggingFace dataset files
            required_hf_files = ["dataset_info.json"]
            if not any((self.eval_dataset_path / f).exists() for f in required_hf_files):
                print(f"Warning: Directory exists but may not be a valid HuggingFace dataset: {self.eval_dataset_path}")

        # Check at least one checkpoint directory exists
        if not self.checkpoint_dirs:
            raise FileNotFoundError(f"No model checkpoints found for experiment {self.dataset_config.to_base_name()}")

        # Validate checkpoint directories exist
        missing_dirs = [d for d in self.checkpoint_dirs if not d.exists()]
        if missing_dirs:
            raise FileNotFoundError(f"Checkpoint directories not found: {missing_dirs}")

        # Validate parameters
        if not self.target_configs:
            raise ValueError("No target configurations specified in collection.yaml")

        if not self.context_sizes or any(k <= 0 for k in self.context_sizes):
            raise ValueError("Context sizes must be positive integers")

    def create_output_structure(self) -> None:
        """Create output directory structure."""
        if self.output_dir.exists() and not self.overwrite and not self.resume:
            raise FileExistsError(f"Output directory exists: {self.output_dir}. Use --overwrite or --resume")

        # Create directory structure
        directories = [
            self.output_dir,
            self.output_dir / "raw_evaluations",
            self.output_dir / "raw_evaluations" / "attention_data",
            self.output_dir / "intermediate",
            self.output_dir / "metadata",
            self.output_dir / "logs",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_resume_path(self) -> Path:
        """Get path for resume checkpoint."""
        return self.output_dir / "metadata" / "resume_checkpoint.json"

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_config": {
                "dataset_type": self.dataset_config.dataset_type,
                "mixture_type": self.dataset_config.mixture_type,
                "total_rules": self.dataset_config.total_rules,
                "seed": self.dataset_config.seed,
                "is_eval": self.dataset_config.is_eval,
            },
            "paths": {
                "eval_dataset_path": str(self.eval_dataset_path),
                "checkpoint_dirs": [str(d) for d in self.checkpoint_dirs],
                "output_dir": str(self.output_dir),
                "config_file_path": str(self.config_file_path),
            },
            "target_configs": self.target_configs,
            "diversity_levels": self.diversity_levels,
            "model_types": self.model_types,
            "context_sizes": self.context_sizes,
            "transfer_conditions": self.transfer_conditions,
            "control_types": self.control_types,
            "execution_parameters": {
                "device": self.device,
                "batch_size": self.batch_size,
                "max_sequences_per_condition": self.max_sequences_per_condition,
                "capture_attention": self.capture_attention,
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
        }

    @classmethod
    def from_experiment_args(
        cls,
        dataset_type: str,
        mixture_type: str,
        total_rules: int,
        seed: int,
        device: str = "cuda",
        overwrite: bool = False,
        resume: bool = False,
        **kwargs,
    ) -> "CollectionConfig":
        """Create CollectionConfig from experiment identification arguments."""
        dataset_config = DatasetConfig(
            dataset_type=dataset_type,
            mixture_type=mixture_type,
            total_rules=total_rules,
            seed=seed,
            is_eval=True,  # Collection always uses eval dataset
        )

        config = cls(dataset_config=dataset_config, device=device, overwrite=overwrite, resume=resume)

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


def create_collection_parser() -> t.Any:
    """Create argument parser for collection phase using shared arguments."""
    from ICL.settings import create_base_parser

    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "ICL Evaluation Collection Phase - Auto-discovery Mode"

    # Collection-specific arguments
    collection_group = parser.add_argument_group("Collection Parameters")
    collection_group.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    collection_group.add_argument("--batch-size", type=int, help="Batch size for evaluation (overrides config)")
    collection_group.add_argument("--max-sequences", type=int, help="Max sequences per condition (overrides config)")
    collection_group.add_argument("--no-attention", action="store_true", help="Disable attention capture")
    collection_group.add_argument("--no-intermediate", action="store_true", help="Disable intermediate saving")

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument("--validate-only", action="store_true", help="Validate setup without running")
    util_group.add_argument("--list-checkpoints", action="store_true", help="List discovered checkpoints")

    return parser


def create_collection_config_from_args(args) -> CollectionConfig:
    """Create CollectionConfig from parsed arguments using auto-discovery."""
    config = CollectionConfig.from_experiment_args(
        dataset_type=args.dataset_type,
        mixture_type=args.mixture_type,
        total_rules=args.total_rules,
        seed=args.seed,
        device=getattr(args, "device", "cuda"),
        overwrite=getattr(args, "overwrite", False),
        resume=getattr(args, "resume", False),
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
