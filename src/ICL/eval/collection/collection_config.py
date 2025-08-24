"""Simplified collection configuration with memory management."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ICL.eval.collection.data_schema import validate_eval_dataset
from ICL.settings import PATH


def detect_best_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        # Check if CUDA is actually usable
        try:
            torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA GPU detected: {device_name}")
            return "cuda"
        except Exception as e:
            print(f"CUDA available but not usable: {e}")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) detected")
        return "mps"

    print("Using CPU (no GPU detected)")
    return "cpu"


@dataclass
class CollectionConfig:
    """Simplified collection configuration with memory management."""

    # Core parameters from bash script
    dataset_type: str
    num_seeds: int
    seed: int
    config_L: int
    config_m: int
    model_type: str

    # Mode selection
    model_variant: str | None = None
    eval_type: str | None = None
    batch_mode: bool = False

    # Memory management parameters
    device: str = None  # Will be auto-detected in __post_init__
    batch_size: int = 32
    max_sequences_per_condition: int = 200
    sequence_chunk_size: int = 50  # Process sequences in chunks
    offload_models: bool = False  # Move models to CPU between evaluations

    # Attention capture parameters
    capture_attention: bool = True
    selective_attention: str | None = None  # "all", "last", "first", "middle", "custom"
    attention_layers: list[int] | None = None  # Specific layers to capture
    attention_heads: list[int] | None = None  # Specific heads to capture
    attention_sampling_rate: float = 1.0  # Fraction of sequences to capture attention for

    # Basic paths and execution parameters
    context_sizes: list[int] = None
    control_types: list[str] = None
    eval_dataset_path: Path | None = None
    model_base_dir: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self):
        """Simple initialization with auto device detection and validation."""
        # Auto-detect device if not specified
        if self.device is None:
            self.device = detect_best_device()

        # Set defaults first
        if self.context_sizes is None:
            self.context_sizes = [1, 2, 3, 4, 5]
        if self.control_types is None:
            self.control_types = ["normal", "shuffled_context", "random_context"]

        # Generate model variant if not provided
        if not self.model_variant:
            self.model_variant = f"{self.model_type}_noshuffle_seedbalanced"

        # Set up paths and validate
        self._setup_paths()
        self._load_execution_parameters_from_yaml()

        # Enable model offloading for batch mode by default
        if self.batch_mode and not hasattr(self, "_offload_set"):
            self.offload_models = True

    def _setup_paths(self):
        """Simple path setup with evaluation type validation."""
        shared_id = f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"

        if not self.batch_mode:
            # Single mode paths
            self.eval_dataset_path = PATH.dataset_root / shared_id / "eval" / self.eval_type / "dataset"
            self.model_base_dir = PATH.model_dir / shared_id / self.model_variant
            self.output_dir = PATH.result_dir / shared_id / self.model_variant / self.eval_type / "collection"

            # Validate eval dataset matches eval_type
            if self.eval_type and self.eval_dataset_path:
                if not validate_eval_dataset(self.eval_dataset_path, self.eval_type):
                    raise ValueError(f"Dataset at {self.eval_dataset_path} does not match eval_type {self.eval_type}")
        else:
            # Batch mode paths
            self.model_base_dir = PATH.model_dir / shared_id
            self.output_dir = PATH.result_dir / shared_id

    def _load_execution_parameters_from_yaml(self):
        """Load execution parameters from YAML file."""
        try:
            import yaml

            shared_id = f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"
            config_path = PATH.conf_dir / shared_id / "collection.yaml"

            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f) or {}

                # Load collection config
                collection_config = config_data.get("collection_config", {})

                # Load evaluation parameters
                eval_params = collection_config.get("evaluation_parameters", {})
                if "context_sizes" in eval_params:
                    self.context_sizes = eval_params["context_sizes"]
                if "control_types" in eval_params:
                    self.control_types = eval_params["control_types"]

                # Load execution parameters (only if not explicitly set)
                exec_params = collection_config.get("execution_parameters", {})
                if "batch_size" in exec_params and self.batch_size == 32:
                    self.batch_size = exec_params["batch_size"]
                if "max_sequences_per_condition" in exec_params and self.max_sequences_per_condition == 200:
                    self.max_sequences_per_condition = exec_params["max_sequences_per_condition"]
                if "capture_attention" in exec_params and self.capture_attention == True:
                    self.capture_attention = exec_params["capture_attention"]

                print(f"Loaded execution parameters from {config_path}")
            else:
                print(f"No collection.yaml found at {config_path}, using defaults")

        except Exception as e:
            print(f"Warning: Failed to load YAML config: {e}, using defaults")

    def discover_combinations(self) -> list[tuple[str, str]]:
        """Simple combination discovery with validation for batch mode."""
        if not self.batch_mode:
            return [(self.model_variant, self.eval_type)]

        shared_id = f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"

        # Find model variants
        model_base = PATH.model_dir / shared_id
        model_variants = [d.name for d in model_base.iterdir() if d.is_dir()]

        # Find eval types
        eval_base = PATH.dataset_root / shared_id / "eval"
        eval_types = [d.name for d in eval_base.iterdir() if d.is_dir()]

        # Create combinations with validation
        combinations = []
        for variant in model_variants:
            for eval_type in eval_types:
                model_path = model_base / variant
                eval_dataset_path = eval_base / eval_type / "dataset"

                if model_path.exists() and eval_dataset_path.exists():
                    # Validate eval dataset matches eval type
                    if validate_eval_dataset(eval_dataset_path, eval_type):
                        combinations.append((variant, eval_type))
                    else:
                        print(f"Warning: Skipping {variant}/{eval_type} - dataset validation failed")

        return combinations

    def create_single_config(self, model_variant: str, eval_type: str) -> "CollectionConfig":
        """Create single config from batch config."""
        return CollectionConfig(
            dataset_type=self.dataset_type,
            num_seeds=self.num_seeds,
            seed=self.seed,
            config_L=self.config_L,
            config_m=self.config_m,
            model_type=self.model_type,
            model_variant=model_variant,
            eval_type=eval_type,
            device=self.device,
            batch_size=self.batch_size,
            max_sequences_per_condition=self.max_sequences_per_condition,
            sequence_chunk_size=self.sequence_chunk_size,
            offload_models=self.offload_models,
            capture_attention=self.capture_attention,
            selective_attention=self.selective_attention,
            attention_layers=self.attention_layers.copy() if self.attention_layers else None,
            attention_heads=self.attention_heads.copy() if self.attention_heads else None,
            attention_sampling_rate=self.attention_sampling_rate,
            context_sizes=self.context_sizes.copy(),
            control_types=self.control_types.copy(),
            batch_mode=False,
        )

    @classmethod
    def from_args(
        cls,
        dataset_type: str,
        num_seeds: int,
        seed: int,
        config_L: int,
        config_m: int,
        model_type: str,
        model_variant: str | None = None,
        eval_type: str | None = None,
        device: str = "cuda",
        batch_mode: bool = False,
        **kwargs,
    ) -> "CollectionConfig":
        """Create CollectionConfig from arguments."""
        config = cls(
            dataset_type=dataset_type,
            num_seeds=num_seeds,
            seed=seed,
            config_L=config_L,
            config_m=config_m,
            model_type=model_type,
            model_variant=model_variant,
            eval_type=eval_type,
            device=device,
            batch_mode=batch_mode,
        )

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def create_output_structure(self):
        """Create basic output directories."""
        if not self.output_dir:
            return

        directories = [
            self.output_dir,
            self.output_dir / "raw_evaluations",
            self.output_dir / "logs",
        ]

        if self.capture_attention:
            directories.append(self.output_dir / "raw_evaluations" / "attention_data")

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def create_collection_parser():
    """Simplified argument parser with memory management options."""
    import argparse

    parser = argparse.ArgumentParser(description="ICL Evaluation Collection Phase")

    # Required arguments
    parser.add_argument("--dataset-type", choices=["uniform", "zipf"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--model-type", choices=["clm", "mlm"], required=True)

    # Mode selection
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--model-variant")
    parser.add_argument("--eval-type", choices=["memorization", "id_generalization", "ood_same_rule", "ood_transfer"])

    # Memory management parameters
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "mps"], help="Device to use (auto-detected if not specified)"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--sequence-chunk-size", type=int, default=50, help="Process sequences in chunks of this size")
    parser.add_argument("--offload-models", action="store_true", help="Move models to CPU between evaluations")

    # Attention capture parameters
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument(
        "--selective-attention",
        choices=["all", "last", "first", "middle"],
        default="all",
        help="Which layers to capture attention from",
    )
    parser.add_argument("--attention-layers", type=int, nargs="+", help="Specific layer indices to capture")
    parser.add_argument("--attention-heads", type=int, nargs="+", help="Specific head indices to capture")
    parser.add_argument(
        "--attention-sampling-rate",
        type=float,
        default=1.0,
        help="Fraction of sequences to capture attention for (0.0-1.0)",
    )

    # Utility commands
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-combinations", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser


def create_collection_config_from_args(args) -> CollectionConfig:
    """Create config from parsed arguments."""
    config = CollectionConfig(
        dataset_type=args.dataset_type,
        num_seeds=args.num_seeds,
        seed=args.seed,
        config_L=args.L,
        config_m=args.M,
        model_type=args.model_type,
        model_variant=getattr(args, "model_variant", None),
        eval_type=getattr(args, "eval_type", None),
        device=getattr(args, "device", None),  # None triggers auto-detection
        batch_mode=getattr(args, "batch_mode", False),
    )

    # Apply memory management overrides
    if hasattr(args, "batch_size") and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, "max_sequences") and args.max_sequences:
        config.max_sequences_per_condition = args.max_sequences
    if hasattr(args, "sequence_chunk_size") and args.sequence_chunk_size:
        config.sequence_chunk_size = args.sequence_chunk_size
    if hasattr(args, "offload_models") and args.offload_models:
        config.offload_models = args.offload_models

    # Apply attention overrides
    if hasattr(args, "no_attention") and args.no_attention:
        config.capture_attention = False
    if hasattr(args, "selective_attention") and args.selective_attention:
        config.selective_attention = args.selective_attention
    if hasattr(args, "attention_layers") and args.attention_layers:
        config.attention_layers = args.attention_layers
    if hasattr(args, "attention_heads") and args.attention_heads:
        config.attention_heads = args.attention_heads
    if hasattr(args, "attention_sampling_rate") and args.attention_sampling_rate is not None:
        config.attention_sampling_rate = args.attention_sampling_rate

    return config
