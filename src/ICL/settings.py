import dataclasses as _dataclasses
import os as _os
import socket as _socket
import warnings as _warnings
from pathlib import Path as _Path


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
        return self.DATA_DIR / "target_neuron_ablation"

    @property
    def model_dir(self) -> _Path:
        return self.DATA_DIR / "models"

    @property
    def result_dir(self) -> _Path:
        return self.DATA_DIR / "results"

    @property
    def train_dir(self) -> _Path:
        return self.DATA_DIR / "datasets" / "train"

    def _assert_dir(self, dir_location: _Path) -> None:
        if not dir_location.is_dir():
            _warnings.warn(
                f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
                stacklevel=1,
            )


PATH = _MyPathSettings()
