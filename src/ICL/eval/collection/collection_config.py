"""Simplified collection configuration."""

from dataclasses import dataclass
from pathlib import Path

from ICL.settings import PATH


@dataclass
class CollectionConfig:
    """Simplified collection configuration."""

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

    # Simple execution parameters
    device: str = "cuda"
    batch_size: int = 32
    max_sequences_per_condition: int = 200
    capture_attention: bool = True
    context_sizes: list[int] = None
    control_types: list[str] = None

    # Basic paths (auto-generated)
    eval_dataset_path: Path | None = None
    model_base_dir: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self):
        """Simple initialization."""
        # Set defaults
        if self.context_sizes is None:
            self.context_sizes = [1, 2, 3, 4, 5]
        if self.control_types is None:
            self.control_types = ["normal", "shuffled_context", "random_context"]

        # Generate model variant if not provided
        if not self.model_variant:
            self.model_variant = f"{self.model_type}_noshuffle_seedbalanced"

        # Set up paths
        self._setup_paths()

    def _setup_paths(self):
        """Simple path setup."""
        shared_id = f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"

        if not self.batch_mode:
            # Single mode paths
            self.eval_dataset_path = PATH.dataset_root / shared_id / "eval" / self.eval_type / "dataset"
            self.model_base_dir = PATH.model_dir / shared_id / self.model_variant
            self.output_dir = PATH.result_dir / shared_id / self.model_variant / self.eval_type / "collection"
        else:
            # Batch mode paths
            self.model_base_dir = PATH.model_dir / shared_id
            self.output_dir = PATH.result_dir / shared_id

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

    def discover_combinations(self) -> list[tuple[str, str]]:
        """Simple combination discovery for batch mode."""
        if not self.batch_mode:
            return [(self.model_variant, self.eval_type)]

        shared_id = f"{self.dataset_type}_{self.num_seeds}_L{self.config_L}_M{self.config_m}"

        # Find model variants
        model_base = PATH.model_dir / shared_id
        model_variants = [d.name for d in model_base.iterdir() if d.is_dir()]

        # Find eval types
        eval_base = PATH.dataset_root / shared_id / "eval"
        eval_types = [d.name for d in eval_base.iterdir() if d.is_dir()]

        # Create combinations
        combinations = []
        for variant in model_variants:
            for eval_type in eval_types:
                if (model_base / variant).exists() and (eval_base / eval_type / "dataset").exists():
                    combinations.append((variant, eval_type))

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
            capture_attention=self.capture_attention,
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


def create_collection_parser():
    """Simplified argument parser."""
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

    # Execution parameters
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--no-attention", action="store_true")

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
        device=getattr(args, "device", "cuda"),
        batch_mode=getattr(args, "batch_mode", False),
    )

    # Apply overrides
    if hasattr(args, "batch_size") and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, "max_sequences") and args.max_sequences:
        config.max_sequences_per_condition = args.max_sequences
    if hasattr(args, "no_attention") and args.no_attention:
        config.capture_attention = False

    return config
