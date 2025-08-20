"""Experiment configuration with auto-discovery for analysis phase."""

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ICL.settings import DatasetConfig


@dataclass
class ExpConfig:
    """Auto-discovery configuration for experiment analysis phase."""

    # Core experiment identification (same as collection)
    dataset_config: DatasetConfig

    # Auto-discovered paths
    collection_results_dir: Path = None
    icl_performance_path: Path = None
    model_registry_path: Path = None
    attention_data_dir: Path = None
    output_base_dir: Path = None
    config_file_path: Path = None

    # Experiment parameters (loaded from exp.yaml)
    enabled_experiments: list[str] = field(default_factory=list)
    exp1_config: dict = field(default_factory=dict)
    exp2_config: dict = field(default_factory=dict)
    exp3_config: dict = field(default_factory=dict)
    exp4_config: dict = field(default_factory=dict)

    # Output control
    generate_reports: bool = True
    save_intermediate: bool = True
    create_visualizations: bool = True
    overwrite: bool = False

    def __post_init__(self):
        """Auto-discover collection results and load exp config."""
        if not self.collection_results_dir:
            self._discover_collection_results()
        if not self.enabled_experiments:
            self._load_exp_config()

    def _discover_collection_results(self) -> None:
        """Auto-discover collection results directory."""
        from ICL.settings import PATH

        base_name = self.dataset_config.to_base_name()

        # Collection results are nested: uniform_allmix_576_42/collection/
        exp_base_dir = PATH.result_dir / base_name
        collection_dir = exp_base_dir / "collection"

        if not collection_dir.exists():
            raise FileNotFoundError(f"Collection results not found: {collection_dir}")

        self.collection_results_dir = collection_dir
        self.icl_performance_path = collection_dir / "raw_evaluations" / "icl_performance.parquet"
        self.model_registry_path = collection_dir / "metadata" / "model_registry.parquet"
        self.attention_data_dir = collection_dir / "raw_evaluations" / "attention_data"

        # Experiment output base directory: uniform_allmix_576_42/exp/
        self.output_base_dir = exp_base_dir / "exp"

        # Config file path - use the simplified experiment config name (without dataset_type)
        experiment_config_name = (
            f"{self.dataset_config.mixture_type}_{self.dataset_config.total_rules}_{self.dataset_config.seed}"
        )
        self.config_file_path = PATH.conf_dir / experiment_config_name / "exp.yaml"

    def get_exp_output_dir(self, exp_name: str) -> Path:
        """Get output directory for specific experiment."""
        return self.output_base_dir / exp_name

    def _load_exp_config(self) -> None:
        """Load experiment configuration from exp.yaml."""
        try:
            if self.config_file_path and self.config_file_path.exists():
                with open(self.config_file_path) as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                print(f"Warning: No exp.yaml found at {self.config_file_path}")
                print("Using default configuration values")
                config_data = {}

            # Load base config
            base_config = config_data.get("base_config", {})

            # Load experiment configs
            exp_config = config_data.get("experiment_config", {})
            self.exp1_config = exp_config.get(
                "exp1_emergence",
                {
                    "enabled": True,
                    "emergence_threshold": 0.5,
                    "baseline_controls": ["shuffled_context", "random_context"],
                },
            )
            self.exp2_config = exp_config.get(
                "exp2_transfer",
                {"enabled": True, "transfer_types": ["depth", "synonym", "full"], "success_threshold": 0.5},
            )
            self.exp3_config = exp_config.get(
                "exp3_scaling",
                {
                    "enabled": True,
                    "scaling_laws": ["exponential", "power", "logarithmic"],
                    "performance_thresholds": [0.5, 0.7, 0.9],
                },
            )
            self.exp4_config = exp_config.get("exp4_attention", {"enabled": False})

            # Load output config
            output_config = config_data.get("output_config", {})
            self.generate_reports = output_config.get("generate_reports", True)
            self.save_intermediate = output_config.get("save_intermediate", True)
            self.create_visualizations = output_config.get("create_visualizations", True)

            # Determine enabled experiments
            self.enabled_experiments = [
                exp_name
                for exp_name, exp_data in {
                    "exp1": self.exp1_config,
                    "exp2": self.exp2_config,
                    "exp3": self.exp3_config,
                    "exp4": self.exp4_config,
                }.items()
                if exp_data.get("enabled", False)
            ]

        except Exception as e:
            print(f"Error loading exp config: {e}")
            print("Using default configuration values")

    def validate(self) -> None:
        """Validate configuration and paths."""
        # Check collection results exist
        if not self.icl_performance_path.exists():
            raise FileNotFoundError(f"ICL performance data not found: {self.icl_performance_path}")

        if not self.model_registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self.model_registry_path}")

    def create_output_structure(self, exp_name: str) -> None:
        """Create output directory structure for specific experiment."""
        exp_output_dir = self.get_exp_output_dir(exp_name)

        if exp_output_dir.exists() and not self.overwrite:
            print(f"Warning: Output directory exists: {exp_output_dir}")
            print("Use --overwrite to replace existing results")

        # Create directory structure
        directories = [
            exp_output_dir,
            exp_output_dir / "metrics",
            exp_output_dir / "visualizations",
            exp_output_dir / "reports",
            exp_output_dir / "intermediate",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_experiment_args(
        cls,
        dataset_type: str,
        mixture_type: str,
        total_rules: int,
        seed: int,
        overwrite: bool = False,
        **kwargs,
    ) -> "ExpConfig":
        """Create ExpConfig from experiment identification arguments."""
        dataset_config = DatasetConfig(
            dataset_type=dataset_type,
            mixture_type=mixture_type,
            total_rules=total_rules,
            seed=seed,
            is_eval=True,  # Experiments always use eval dataset
        )

        config = cls(dataset_config=dataset_config, overwrite=overwrite)

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


def create_exp_parser() -> t.Any:
    """Create argument parser for experiment phase using shared arguments."""
    from ICL.settings import create_base_parser

    parser = create_base_parser(require_model_type=False, require_eval_flag=False)
    parser.description = "ICL Experiment Analysis Phase"

    # Experiment control arguments
    exp_group = parser.add_argument_group("Experiment Control")
    exp_group.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Specific experiments to run (default: all enabled)",
    )

    # Output control arguments
    output_group = parser.add_argument_group("Output Control")
    output_group.add_argument("--no-reports", action="store_true", help="Skip HTML report generation")
    output_group.add_argument("--no-visualizations", action="store_true", help="Skip visualization creation")
    output_group.add_argument("--no-intermediate", action="store_true", help="Skip intermediate result saving")

    return parser


def create_exp_config_from_args(args) -> ExpConfig:
    """Create ExpConfig from parsed arguments using auto-discovery."""
    config = ExpConfig.from_experiment_args(
        dataset_type=args.dataset_type,
        mixture_type=args.mixture_type,
        total_rules=args.total_rules,
        seed=args.seed,
        overwrite=getattr(args, "overwrite", False),
    )

    # Apply command line overrides
    if hasattr(args, "no_reports") and args.no_reports:
        config.generate_reports = False

    if hasattr(args, "no_visualizations") and args.no_visualizations:
        config.create_visualizations = False

    if hasattr(args, "no_intermediate") and args.no_intermediate:
        config.save_intermediate = False

    return config
