import argparse
import dataclasses as _dataclasses
import logging
import os as _os
import re
import socket as _socket
import typing as t
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _Path

logger = logging.getLogger(__name__)
import yaml

logger = logging.getLogger(__name__)
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


def load_experiment_config(config_type: str, dataset_config: "DatasetConfig", L: int, m: int) -> dict[str, t.Any]:
    """Load configuration from correct path structure.

    Args:
        config_type: Type of config to load ("clm", "mlm", etc.)
        dataset_config: Dataset configuration
        L: Hierarchy depth (required)
        m: Multiplicity (required)

    """
    valid_types = ["train_dataset", "eval_dataset", "clm", "mlm", "collection"]
    if config_type not in valid_types:
        raise ValueError(f"Invalid config type: {config_type}. Must be one of {valid_types}")

    # Build config directory path: conf/{dataset_type}_{num_seeds}_L{L}_M{m}/
    config_dir_name = f"{dataset_config.dataset_type}_{dataset_config.num_seeds}_L{L}_M{m}"
    config_path = PATH.conf_dir / config_dir_name / f"{config_type}.yaml"

    logger.info(f"Loading config from: {config_path}")

    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
            logger.info(f"✓ Successfully loaded {config_type} configuration")
            return config_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        return {}


def get_dataset_subdir(is_eval: bool) -> str:
    """Get dataset subdirectory name based on dataset type."""
    return "eval" if is_eval else "train"


#################################
# Updated experiment configuration


@dataclass(frozen=True)
class DatasetConfig:
    """Model-agnostic dataset identification with L,M auto-discovery."""

    dataset_type: str  # uniform, zipf
    seed: int  # RNG seed for generating random seeds
    num_seeds: int  # Number of random seeds to generate
    is_eval: bool = False
    L: int | None = None  # Will be auto-discovered if None
    m: int | None = None  # Will be auto-discovered if None

    def __post_init__(self):
        """Auto-discover L,M if not provided."""
        if self.L is None or self.m is None:
            # Use object.__setattr__ because this is a frozen dataclass
            L, m = get_default_L_M_config(self.dataset_type, self.num_seeds)
            if self.L is None:
                object.__setattr__(self, "L", L)
            if self.m is None:
                object.__setattr__(self, "m", m)

            logger.info(f"Auto-discovered dataset configuration: L={self.L}, m={self.m}")

    def to_name(self) -> str:
        """Generate full dataset name including L,M."""
        base_name = f"{self.dataset_type}_{self.num_seeds}_L{self.L}_M{self.m}"
        return f"{base_name}_eval" if self.is_eval else base_name

    def to_base_name(self) -> str:
        """Generate base dataset name without eval suffix."""
        return f"{self.dataset_type}_{self.num_seeds}_L{self.L}_M{self.m}"

    def get_paths(self) -> dict[str, Path]:
        """Generate complete dataset paths with L,M included."""
        subdir = get_dataset_subdir(self.is_eval)
        base_name = self.to_base_name()

        return {
            "dataset_dir": PATH.dataset_root / base_name / subdir,
            "config_base_dir": PATH.dataset_root / base_name,
            "config_dir": PATH.conf_dir / base_name,
            "base_dataset_dir": PATH.dataset_root / base_name,
        }

    def validate_config(self) -> bool:
        """Validate that this configuration exists in the filesystem."""
        return validate_L_M_config(self.dataset_type, self.num_seeds, self.L, self.m)

    def get_available_configs(self) -> list[tuple[int, int]]:
        """Get all available (L,M) configurations for this dataset type."""
        return discover_available_L_M_configs(self.dataset_type, self.num_seeds)

    @classmethod
    def create_with_L_M(cls, dataset_type: str, seed: int, num_seeds: int, L: int, m: int, is_eval: bool = False):
        """Create DatasetConfig with explicit L,M values."""
        return cls(dataset_type=dataset_type, seed=seed, num_seeds=num_seeds, L=L, m=m, is_eval=is_eval)

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
    """Model-specific training identification with automatic L,M discovery."""

    dataset_config: DatasetConfig
    model_type: str

    def to_name(self) -> str:
        """Generate full model name including L,M."""
        return f"{self.dataset_config.to_base_name()}_{self.model_type}"

    def get_model_paths(self) -> dict[str, Path]:
        """Generate model-specific paths with correct L,M included."""
        # Get dataset paths (these now include L,M)
        dataset_paths = self.dataset_config.get_paths()

        # Model directory includes the full dataset name with L,M
        model_base_name = self.dataset_config.to_base_name()  # e.g., "uniform_10_L4_M2"

        return {
            "model_dir": PATH.model_dir / model_base_name,
            "dataset_dir": dataset_paths["dataset_dir"],  # e.g., "datasets/uniform_10_L4_M2/train"
            "config_dir": dataset_paths["config_dir"],  # e.g., "conf/uniform_10_L4_M2"
            "base_dataset_dir": dataset_paths["base_dataset_dir"],  # e.g., "datasets/uniform_10_L4_M2"
        }

    def get_eval_paths(self, eval_suffix: str = "") -> dict[str, Path]:
        """Generate evaluation paths."""
        base_name = self.to_name()
        result_name = f"{base_name}__{eval_suffix}" if eval_suffix else base_name

        paths = self.get_model_paths()
        # Since model_dir is now the base path, results should also follow the enhanced naming
        paths["results_dir"] = PATH.result_dir / result_name
        return paths

    def validate_config(self) -> bool:
        """Validate that the dataset configuration exists."""
        return self.dataset_config.validate_config()

    def get_available_dataset_configs(self) -> list[tuple[int, int]]:
        """Get all available (L,M) configurations for this dataset."""
        return self.dataset_config.get_available_configs()

    @classmethod
    def create_with_L_M(cls, dataset_type: str, model_type: str, seed: int, num_seeds: int, L: int, m: int):
        """Create ModelConfig with explicit L,M values."""
        dataset_config = DatasetConfig.create_with_L_M(
            dataset_type=dataset_type, seed=seed, num_seeds=num_seeds, L=L, m=m, is_eval=False
        )
        return cls(dataset_config=dataset_config, model_type=model_type)


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
# Argument parser


def create_base_parser(
    require_eval_flag: bool = False, require_model_type: bool = False, allow_L_M_override: bool = False
) -> argparse.ArgumentParser:
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

    # Add model type if required
    if require_model_type:
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model-type", choices=["clm", "mlm"], required=True, help="Type of model to train (CLM or MLM)"
        )

    # Add L,M override options if requested
    if allow_L_M_override:
        hierarchy_group = parser.add_argument_group("Hierarchy Override (Optional)")
        hierarchy_group.add_argument(
            "--L", type=int, help="Override hierarchy depth L (will auto-discover if not specified)"
        )
        hierarchy_group.add_argument(
            "--M", type=int, help="Override multiplicity M (will auto-discover if not specified)"
        )

    # Pipeline control (optional for all scripts)
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    control_group.add_argument("--resume", action="store_true", help="Skip if outputs already exist")
    control_group.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def parse_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Convert parsed arguments to DatasetConfig with L,M auto-discovery."""
    is_eval = getattr(args, "eval", False)
    L = getattr(args, "L", None)
    m = getattr(args, "M", None)

    return DatasetConfig(
        dataset_type=args.dataset_type,
        seed=args.seed,
        num_seeds=args.num_seeds,
        is_eval=is_eval,
        L=L,
        m=m,
    )


def parse_model_config(args: argparse.Namespace) -> ModelConfig:
    """Convert parsed arguments to ModelConfig with L,M auto-discovery."""
    dataset_config = parse_dataset_config(args)
    return ModelConfig(
        dataset_config=dataset_config,
        model_type=args.model_type,
    )


def validate_args(args: argparse.Namespace) -> bool:
    """Validate argument combinations and constraints."""
    # Check for conflicting flags
    if args.overwrite and args.resume:
        raise ValueError("Cannot specify both --overwrite and --resume")

    # Validate num_seeds
    if args.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")

    # Validate L,M if provided
    L = getattr(args, "L", None)
    M = getattr(args, "M", None)

    if (L is None) != (M is None):
        raise ValueError("Both --L and --M must be specified together, or neither")

    if L is not None and L < 1:
        raise ValueError("L must be at least 1")

    if M is not None and M < 1:
        raise ValueError("M must be at least 1")

    return True


#################################
# Helper func in parsing l and m


def discover_available_L_M_configs(dataset_type: str, num_seeds: int) -> list[tuple[int, int]]:
    """Discover available (L,M) configurations for a dataset.

    Args:
        dataset_type: Type of dataset (uniform, zipf)
        num_seeds: Number of seeds

    Returns:
        List of (L, M) tuples found in the filesystem

    """
    base_pattern = f"{dataset_type}_{num_seeds}"
    configs = []

    # Look for directories matching pattern: {dataset_type}_{num_seeds}_L*_M*
    dataset_root = PATH.dataset_root
    if not dataset_root.exists():
        logger.warning(f"Dataset root does not exist: {dataset_root}")
        return []

    pattern = re.compile(rf"^{re.escape(base_pattern)}_L(\d+)_M(\d+)$")

    for item in dataset_root.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                L = int(match.group(1))
                M = int(match.group(2))

                # Verify that train subdirectory exists
                train_dir = item / "train"
                if train_dir.exists():
                    configs.append((L, M))
                    logger.debug(f"Found dataset config: L={L}, M={M} at {item}")
                else:
                    logger.warning(f"Found directory {item.name} but no train subdirectory")

    # Sort by L, then M for consistent ordering
    configs.sort()

    if configs:
        logger.info(f"Discovered {len(configs)} dataset configurations for {base_pattern}: {configs}")
    else:
        logger.warning(f"No dataset configurations found for {base_pattern}")

    return configs


def get_default_L_M_config(dataset_type: str, num_seeds: int) -> tuple[int, int]:
    """Get the default (L,M) configuration for a dataset.

    This will use the first available configuration, or fall back to (4,2) if none found.

    Args:
        dataset_type: Type of dataset
        num_seeds: Number of seeds

    Returns:
        (L, M) tuple

    """
    available_configs = discover_available_L_M_configs(dataset_type, num_seeds)

    if available_configs:
        L, M = available_configs[0]
        logger.info(f"Using default L={L}, M={M} configuration for {dataset_type}_{num_seeds}")
        return L, M
    # Fallback default
    logger.warning(f"No configurations found for {dataset_type}_{num_seeds}, using fallback L=4, M=2")
    return 4, 2


def validate_L_M_config(dataset_type: str, num_seeds: int, L: int, M: int) -> bool:
    """Validate that a specific (L,M) configuration exists.

    Args:
        dataset_type: Type of dataset
        num_seeds: Number of seeds
        L: Hierarchy depth
        M: Multiplicity

    Returns:
        True if configuration exists, False otherwise

    """
    config_name = f"{dataset_type}_{num_seeds}_L{L}_M{M}"
    dataset_dir = PATH.dataset_root / config_name
    train_dir = dataset_dir / "train"

    exists = train_dir.exists()
    if not exists:
        logger.error(f"Dataset configuration {config_name} not found at {train_dir}")

    return exists
