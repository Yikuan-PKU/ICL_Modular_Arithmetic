"""Shared configuration base for both collection and analysis phases."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BaseConfig:
    """Base configuration shared between collection and analysis phases."""

    # Core paths
    checkpoint_base_dirs: list[Path]
    eval_dataset_path: Path
    output_dir: Path

    # Target configurations to evaluate
    target_configs: list[tuple[int, int]] = field(default_factory=list)
    diversity_levels: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    model_types: list[str] = field(default_factory=lambda: ["causal_lm", "mlm"])

    # Device configuration
    device: str = "cuda"

    def __post_init__(self):
        """Validate base configuration."""
        # Convert string paths to Path objects
        self.checkpoint_base_dirs = [Path(p) for p in self.checkpoint_base_dirs]
        self.eval_dataset_path = Path(self.eval_dataset_path)
        self.output_dir = Path(self.output_dir)

    def validate_paths(self) -> bool:
        """Validate that required paths exist."""
        # Check if evaluation dataset exists
        if not self.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.eval_dataset_path}")

        # Check if checkpoint directories exist
        missing_dirs = []
        for checkpoint_dir in self.checkpoint_base_dirs:
            if not checkpoint_dir.exists():
                missing_dirs.append(checkpoint_dir)

        if missing_dirs:
            raise FileNotFoundError(f"Missing checkpoint directories: {missing_dirs}")

        return True

    @classmethod
    def from_args_and_yaml(cls, args, yaml_path: Path | None = None) -> "BaseConfig":
        """Create configuration from command line args and optional YAML."""
        # Load YAML config if provided
        yaml_config = {}
        if yaml_path and yaml_path.exists():
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)

        # Priority: command line args > YAML > defaults
        config_dict = {}

        # Core paths from args
        config_dict.update(
            {
                "checkpoint_base_dirs": getattr(args, "checkpoint_dirs", []),
                "eval_dataset_path": getattr(args, "eval_dataset_path", ""),
                "output_dir": getattr(args, "output_dir", ""),
                "device": getattr(args, "device", "cuda"),
            }
        )

        # Override with YAML values if present
        if "base_config" in yaml_config:
            base_yaml = yaml_config["base_config"]
            for key in ["target_configs", "diversity_levels", "model_types"]:
                if key in base_yaml:
                    config_dict[key] = base_yaml[key]

        # Override with command line args if present
        if hasattr(args, "target_configs") and args.target_configs:
            config_dict["target_configs"] = args.target_configs

        return cls(**config_dict)


def create_base_parser(require_yaml: bool = False):
    """Create base argument parser with shared arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="ICL Evaluation Pipeline")

    # Core paths
    parser.add_argument("--checkpoint-dirs", nargs="+", required=True, help="Directories containing model checkpoints")
    parser.add_argument("--eval-dataset-path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")

    # Configuration
    if require_yaml:
        parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    else:
        parser.add_argument("--config", type=str, help="Path to YAML configuration file (optional)")

    # Device
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation")

    # Common flags
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial results")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration without running")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser
