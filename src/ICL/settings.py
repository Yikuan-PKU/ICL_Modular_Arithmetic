import argparse
import dataclasses as _dataclasses
import os as _os
import socket as _socket
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _Path

#################################
# Path settings


@_dataclasses.dataclass
class _MyPathSettings:
    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    KNOWN_HOSTS: tuple[str, ...] = (*COML_SERVERS, "mbp-de-jliu.home")
    # print hostname

    def __post_init__(self) -> None:
        if "DATA_DIR" not in _os.environ:
            hostname = _socket.gethostname()
            print(hostname)
            if hostname in self.COML_SERVERS:
                self.DATA_DIR = _Path("/scratch2/jliu/ICL")
            elif hostname == "mbp-de-jliu.home":
                # Default for your MacBook (adjust if you want another location)
                self.DATA_DIR = _Path.home() / "local_data"
            elif hostname == "PC-20211018VJML":
                self.DATA_DIR = _Path("./src/data")
            else:
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
    def train_dir(self) -> _Path:
        return self.DATA_DIR / "datasets" / "train"

    @property
    def conf_dir(self) -> _Path:
        return self.DATA_DIR / "ICL_Modular_Arithmetic" / "experiments" / "conf"

    def _assert_dir(self, dir_location: _Path) -> None:
        if not dir_location.is_dir():
            _warnings.warn(
                f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
                stacklevel=1,
            )


PATH = _MyPathSettings()


#################################
# Shared exp args


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment identifier that generates all paths consistently."""

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
        name = self.to_name()
        return {
            "dataset_dir": PATH.dataset_root / name,
            "model_dir": PATH.model_dir / name,
            "config_dir": PATH.conf_dir / name,
            "results_dir": PATH.result_dir / name,
        }


#################################
# Shared arg parser


def create_base_parser() -> argparse.ArgumentParser:
    """Create minimal shared parser for experiment identification only."""
    parser = argparse.ArgumentParser(add_help=False)

    # Experiment identification (required for all scripts)
    exp_group = parser.add_argument_group("Experiment Identification")
    exp_group.add_argument(
        "--dataset-type", choices=["uniform", "zipf"], required=True, help="Probability distribution for rule sampling"
    )
    exp_group.add_argument(
        "--model-type",
        choices=["clm", "mlm"],
        required=True,
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

    # Pipeline control (optional for all scripts)
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    control_group.add_argument("--resume", action="store_true", help="Skip if outputs already exist")
    control_group.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def parse_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Convert parsed command line arguments to ExperimentConfig."""
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

    # Add any other validation logic here
    return True
