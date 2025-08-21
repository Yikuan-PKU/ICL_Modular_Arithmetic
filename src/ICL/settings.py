import argparse
import dataclasses as _dataclasses
import os as _os
import socket as _socket
import typing as t
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _Path

import yaml

#################################
# Path settings


@_dataclasses.dataclass
class _MyPathSettings:
    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    KNOWN_HOSTS: tuple[str, ...] = (*COML_SERVERS, "mbp-de-jliu.home")

    def __post_init__(self) -> None:
        if "DATA_DIR" not in _os.environ:
            hostname = _socket.gethostname()
            if hostname in self.COML_SERVERS:
                self.DATA_DIR = _Path("/scratch2/jliu/ICL")
            elif hostname == "mbp-de-jliu.home":
                # Default for your MacBook (adjust if you want another location)
                self.DATA_DIR = _Path.home() / "local_data"
            elif hostname == "PC-20211018VJML":
                self.DATA_DIR = _Path("./src/data")
            else:
                # fallback for unknown hosts
                self.DATA_DIR = _Path("/lustre1/qi_pkuhpc/ykzhang/ICL_RHM")

        if not self.DATA_DIR.is_dir():
            _warnings.warn(
                f"Provided DATA_DIR: {self.DATA_DIR} does not exist.\nSet $DATA_DIR or check hostname defaults.",
                stacklevel=1,
            )

    @property
    def dataset_root(self) -> _Path:
        self._assert_dir(self.DATA_DIR / "datasets")
        return self.DATA_DIR / "datasets"

    @property
    def script_dir(self) -> _Path:
        return self.DATA_DIR / "ICL_Modular_Arithmetic"

    @property
    def model_dir(self) -> _Path:
        return self.DATA_DIR / "models"

    @property
    def result_dir(self) -> _Path:
        return self.DATA_DIR / "results"

    @property
    def conf_dir(self) -> _Path:
        """Flattened configuration directory."""
        return self.DATA_DIR / "ICL_Modular_Arithmetic" / "experiments" / "conf"

    def _assert_dir(self, dir_location: _Path) -> None:
        if not dir_location.is_dir():
            _warnings.warn(
                f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
                stacklevel=1,
            )


PATH = _MyPathSettings()


#################################
# Configuration loading utilities


def get_experiment_config_name(dataset_type: str, mixture_type: str, total_rules: int, seed: int) -> str:
    """Generate experiment configuration directory name (without dataset_type prefix)."""
    return f"{mixture_type}_{total_rules}_{seed}"


def load_experiment_config(
    config_type: str, dataset_config: "DatasetConfig", L: int | None = None, m: int | None = None
) -> dict[str, t.Any]:
    """Load configuration from experiment-specific config structure.

    Args:
        config_type: Type of config to load
        dataset_config: Dataset configuration
        L: Hierarchy depth (required for per-config loading)
        m: Multiplicity (required for per-config loading)

    """
    valid_types = ["train_dataset", "eval_dataset", "clm", "mlm", "collection"]
    if config_type not in valid_types:
        raise ValueError(f"Invalid config type: {config_type}. Must be one of {valid_types}")

    if L is None or m is None:
        raise ValueError("L and m parameters are required for configuration loading")

    # Build config directory path: conf/{dataset_type}_{num_seeds}_L{L}_M{m}/
    config_dir_name = f"{dataset_config.dataset_type}_{dataset_config.num_seeds}_L{L}_M{m}"
    config_path = PATH.conf_dir / config_dir_name / f"{config_type}.yaml"

    if not config_path.exists():
        _warnings.warn(f"Configuration file not found: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
            return config_data
    except yaml.YAMLError as e:
        _warnings.warn(f"Error parsing YAML config {config_path}: {e}")
        return {}
    except Exception as e:
        _warnings.warn(f"Unexpected error loading config {config_path}: {e}")
        return {}


def get_dataset_subdir(is_eval: bool) -> str:
    """Get dataset subdirectory name based on dataset type."""
    return "eval" if is_eval else "train"


#################################
# Updated experiment configuration


@dataclass(frozen=True)
class DatasetConfig:
    """Model-agnostic dataset identification with simplified structure."""

    dataset_type: str  # uniform, zipf
    seed: int  # RNG seed for generating random seeds
    num_seeds: int  # Number of random seeds to generate
    is_eval: bool = False

    def to_name(self) -> str:
        """Generate model-agnostic dataset name (without L,M since those vary per config)."""
        base_name = f"{self.dataset_type}_{self.num_seeds}"
        return f"{base_name}_eval" if self.is_eval else base_name

    def to_base_name(self) -> str:
        """Generate base dataset name without eval suffix (without L,M since those vary per config)."""
        return f"{self.dataset_type}_{self.num_seeds}"

    def get_base_paths(self) -> dict[str, Path]:
        """Generate base dataset paths (without L,M specification)."""
        return {
            "base_dataset_dir": PATH.dataset_root,
            # Note: config_dir is per (L,M) configuration, not shared
        }

    def get_config_paths(self, L: int, m: int) -> dict[str, Path]:
        """Generate dataset-specific paths for a given (L,M) configuration."""
        subdir = get_dataset_subdir(self.is_eval)

        # Main directory name includes dataset_type, num_seeds, L, and M
        config_dir_name = f"{self.dataset_type}_{self.num_seeds}_L{L}_M{m}"

        # Config directory follows same pattern
        config_yaml_dir = f"{self.dataset_type}_{self.num_seeds}_L{L}_M{m}"

        return {
            "dataset_dir": PATH.dataset_root / config_dir_name / subdir,
            "config_base_dir": PATH.dataset_root / config_dir_name,
            "config_dir": PATH.conf_dir / config_yaml_dir,
            "base_dataset_dir": PATH.dataset_root / config_dir_name,
        }

    def get_all_config_dirs(self, L_M_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Path]]:
        """Generate paths for all (L,M) configurations."""
        return {(L, m): self.get_config_paths(L, m) for L, m in L_M_pairs}


@dataclass(frozen=True)
class ModelConfig:
    """Model-specific training identification."""

    dataset_config: DatasetConfig
    model_type: str

    def to_name(self) -> str:
        """Generate full model name."""
        return f"{self.dataset_config.to_base_name()}_{self.model_type}"

    def get_model_paths(self) -> dict[str, Path]:
        """Generate model-specific paths."""
        base_paths = self.dataset_config.get_base_paths()
        train_subdir = get_dataset_subdir(False)  # Models always reference train data

        return {
            "model_dir": PATH.model_dir / self.dataset_config.to_base_name() / self.model_type,
            "config_dir": base_paths["config_dir"],
            "base_dataset_dir": base_paths["base_dataset_dir"],
        }

    def get_eval_paths(self, eval_suffix: str = "") -> dict[str, Path]:
        """Generate evaluation paths."""
        base_name = self.to_name()
        result_name = f"{base_name}__{eval_suffix}" if eval_suffix else base_name

        paths = self.get_model_paths()
        paths["results_dir"] = PATH.result_dir / result_name
        return paths


# Backward compatibility
@dataclass(frozen=True)
class ExperimentConfig:
    """Legacy experiment config for backward compatibility."""

    dataset_type: str
    model_type: str
    mixture_type: str
    total_rules: int
    seed: int

    def to_name(self) -> str:
        """Generate the canonical experiment name."""
        return f"{self.dataset_type}_{self.model_type}_{self.mixture_type}_{self.total_rules}_{self.seed}"

    def get_paths(self) -> dict[str, Path]:
        """Auto-generate all required paths for this experiment."""
        # Convert to new structure for backward compatibility
        base_name = f"{self.dataset_type}_{self.mixture_type}_{self.total_rules}_{self.seed}"
        experiment_config_name = get_experiment_config_name(
            self.dataset_type, self.mixture_type, self.total_rules, self.seed
        )

        return {
            "dataset_dir": PATH.dataset_root / base_name / "train",
            "model_dir": PATH.model_dir / base_name / self.model_type,
            "config_dir": PATH.conf_dir / experiment_config_name,
            "results_dir": PATH.result_dir / base_name / self.model_type,
        }


#################################
# Updated argument parser


def create_base_parser(require_eval_flag: bool = False) -> argparse.ArgumentParser:
    """Create minimal shared parser for dataset generation."""
    parser = argparse.ArgumentParser(add_help=False)

    # Dataset identification
    dataset_group = parser.add_argument_group("Dataset Identification")
    dataset_group.add_argument(
        "--dataset-type", choices=["uniform", "zipf"], required=True, help="Probability distribution for rule sampling"
    )
    dataset_group.add_argument(
        "--seed", type=int, required=True, help="RNG seed for generating random seeds (for reproducibility)"
    )
    dataset_group.add_argument(
        "--num-seeds", type=int, default=1, help="Number of random seeds to generate (default: 1)"
    )

    # Add eval flag if required
    if require_eval_flag:
        dataset_group.add_argument(
            "--eval", action="store_true", help="Generate evaluation dataset (uses eval subdirectory)"
        )

    # Pipeline control (optional for all scripts)
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    control_group.add_argument("--resume", action="store_true", help="Skip if outputs already exist")
    control_group.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def parse_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Convert parsed arguments to DatasetConfig."""
    is_eval = getattr(args, "eval", False)
    return DatasetConfig(
        dataset_type=args.dataset_type,
        seed=args.seed,
        num_seeds=args.num_seeds,
        is_eval=is_eval,
    )


def parse_model_config(args: argparse.Namespace) -> ModelConfig:
    """Convert parsed arguments to ModelConfig."""
    dataset_config = parse_dataset_config(args)
    return ModelConfig(
        dataset_config=dataset_config,
        model_type=args.model_type,
    )


def parse_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Convert parsed command line arguments to ExperimentConfig (legacy)."""
    return ExperimentConfig(
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        mixture_type=getattr(args, "mixture_type", "unknown"),
        total_rules=getattr(args, "total_rules", 0),
        seed=args.seed,
    )


def validate_args(args: argparse.Namespace) -> bool:
    """Validate argument combinations and constraints."""
    # Check for conflicting flags
    if args.overwrite and args.resume:
        raise ValueError("Cannot specify both --overwrite and --resume")

    # Validate num_seeds
    if args.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")

    return True
