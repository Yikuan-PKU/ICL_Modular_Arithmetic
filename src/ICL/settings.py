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
                self.DATA_DIR = _Path("data/")

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


def load_experiment_config(config_type: str, dataset_config: "DatasetConfig") -> dict[str, t.Any]:
    """Load configuration from experiment-specific config structure."""
    valid_types = ["train_dataset", "eval_dataset", "clm", "mlm", "collection"]
    if config_type not in valid_types:
        raise ValueError(f"Invalid config type: {config_type}. Must be one of {valid_types}")

    # Create experiment config directory name (without dataset_type)
    experiment_name = get_experiment_config_name(
        dataset_config.dataset_type, dataset_config.mixture_type, dataset_config.total_rules, dataset_config.seed
    )

    config_path = PATH.conf_dir / experiment_name / f"{config_type}.yaml"

    if not config_path.exists():
        _warnings.warn(f"Configuration file not found: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}

            # Add dataset_type to config data for disambiguation if needed
            if config_data and "dataset_type" not in config_data:
                config_data["_dataset_type"] = dataset_config.dataset_type

            return config_data
    except yaml.YAMLError as e:
        _warnings.warn(f"Error parsing YAML config {config_path}: {e}")
        return {}
    except Exception as e:
        _warnings.warn(f"Unexpected error loading config {config_path}: {e}")
        return {}


def load_global_config(config_type: str) -> dict[str, t.Any]:
    """Load configuration from flattened config structure (deprecated - use load_experiment_config)."""
    _warnings.warn(
        "load_global_config is deprecated. Use load_experiment_config with dataset_config parameter.",
        DeprecationWarning,
        stacklevel=2,
    )
    valid_types = ["train_dataset", "eval_dataset", "clm", "mlm"]
    if config_type not in valid_types:
        raise ValueError(f"Invalid config type: {config_type}. Must be one of {valid_types}")

    config_path = PATH.conf_dir / f"{config_type}.yaml"

    if not config_path.exists():
        _warnings.warn(f"Configuration file not found: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
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
# Shared exp args


@dataclass(frozen=True)
class DatasetConfig:
    """Model-agnostic dataset identification."""

    dataset_type: str
    mixture_type: str
    total_rules: int
    seed: int
    is_eval: bool = False  # New field to distinguish train/eval datasets

    def to_name(self) -> str:
        """Generate model-agnostic dataset name."""
        base_name = f"{self.dataset_type}_{self.mixture_type}_{self.total_rules}_{self.seed}"
        return f"{base_name}_eval" if self.is_eval else base_name

    def to_base_name(self) -> str:
        """Generate base dataset name without eval suffix."""
        return f"{self.dataset_type}_{self.mixture_type}_{self.total_rules}_{self.seed}"

    def get_dataset_paths(self) -> dict[str, Path]:
        """Generate dataset-specific paths with train/eval subdirectories."""
        base_name = self.to_base_name()
        subdir = get_dataset_subdir(self.is_eval)

        # Create simplified experiment config directory name (without dataset_type)
        experiment_config_name = get_experiment_config_name(
            self.dataset_type, self.mixture_type, self.total_rules, self.seed
        )

        return {
            "dataset_dir": PATH.dataset_root / base_name / subdir,
            "base_dir": PATH.dataset_root / base_name,
            "config_dir": PATH.conf_dir / experiment_config_name,  # Simplified config location
        }


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
        base_name = self.dataset_config.to_base_name()
        train_subdir = get_dataset_subdir(False)  # Models always reference train data

        # Create simplified experiment config directory name (without dataset_type)
        experiment_config_name = get_experiment_config_name(
            self.dataset_config.dataset_type,
            self.dataset_config.mixture_type,
            self.dataset_config.total_rules,
            self.dataset_config.seed,
        )

        return {
            "dataset_dir": PATH.dataset_root / base_name / train_subdir,
            "model_dir": PATH.model_dir / base_name / self.model_type,
            "config_dir": PATH.conf_dir / experiment_config_name,  # Simplified config location
            "base_dataset_dir": PATH.dataset_root / base_name,
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
# Shared arg parser


def create_base_parser(require_model_type: bool = True, require_eval_flag: bool = False) -> argparse.ArgumentParser:
    """Create minimal shared parser for experiment identification."""
    parser = argparse.ArgumentParser(add_help=False)

    # Experiment identification
    exp_group = parser.add_argument_group("Experiment Identification")
    exp_group.add_argument(
        "--dataset-type", choices=["uniform", "zipf"], required=True, help="Probability distribution for rule sampling"
    )
    exp_group.add_argument(
        "--model-type",
        choices=["clm", "mlm"],
        required=require_model_type,
        help="Model type: causal (clm) or masked (mlm) language model",
    )
    exp_group.add_argument(
        "--mixture-type",
        choices=["allmix", "depthmix_L2", "depthmix_L3", "depthmix_L4", "complexmix_low", "complexmix_high"],
        required=True,
        help="Training mixture strategy (handled in dataloader)",
    )
    exp_group.add_argument(
        "--total-rules",
        type=int,
        choices=[144, 288, 432, 576],
        required=True,
        help="Total rules parameter (stored in metadata)",
    )
    exp_group.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")

    # Add eval flag if required
    if require_eval_flag:
        exp_group.add_argument(
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
        mixture_type=args.mixture_type,
        total_rules=args.total_rules,
        seed=args.seed,
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
        mixture_type=args.mixture_type,
        total_rules=args.total_rules,
        seed=args.seed,
    )


def validate_args(args: argparse.Namespace) -> bool:
    """Validate argument combinations and constraints."""
    # Check for conflicting flags
    if args.overwrite and args.resume:
        raise ValueError("Cannot specify both --overwrite and --resume")

    return True
