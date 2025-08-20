"""Revised collection coordinator using experiment-based auto-discovery."""

import json
import logging
import typing as t
from datetime import datetime
from pathlib import Path

from ICL.eval.collection.collection_config import CollectionConfig
from ICL.eval.collection.evaluator import CollectionEvaluator

logger = logging.getLogger(__name__)


class CollectionCoordinator:
    """Collection coordinator using experiment-based auto-discovery."""

    def __init__(self, config: CollectionConfig):
        """Initialize coordinator with auto-discovered configuration."""
        self.config = config
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging for the collection phase."""
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Collection coordinator initialized. Logs: {log_file}")

    def validate_setup(self) -> bool:
        """Validate auto-discovered setup."""
        self.logger.info("Validating collection setup using auto-discovery...")

        try:
            # Validate configuration
            self.config.validate()

            # Log discovered paths
            self._log_discovery_summary()

            # Validate evaluation dataset format
            self._validate_evaluation_dataset()

            self.logger.info("Setup validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False

    def _log_discovery_summary(self) -> None:
        """Log summary of auto-discovered paths and configuration."""
        self.logger.info("\nAuto-Discovery Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Experiment: {self.config.dataset_config.to_base_name()}")
        self.logger.info(f"Evaluation dataset: {self.config.eval_dataset_path}")
        self.logger.info(f"Checkpoint directories: {len(self.config.checkpoint_dirs)}")
        for i, checkpoint_dir in enumerate(self.config.checkpoint_dirs):
            self.logger.info(f"  {i + 1}. {checkpoint_dir}")
        self.logger.info(f"Collection config: {self.config.config_file_path}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Target configurations: {len(self.config.target_configs)}")
        self.logger.info(f"Context sizes: {self.config.context_sizes}")
        self.logger.info(f"Transfer conditions: {self.config.transfer_conditions}")

    def list_discovered_checkpoints(self) -> dict[str, t.Any]:
        """List all discovered model checkpoints."""
        checkpoint_info = {"total_directories": len(self.config.checkpoint_dirs), "checkpoints": []}

        for checkpoint_dir in self.config.checkpoint_dirs:
            dir_info = {
                "directory": str(checkpoint_dir),
                "exists": checkpoint_dir.exists(),
                "subdirs": [],
                "metadata_files": [],
            }

            if checkpoint_dir.exists():
                # Find checkpoint subdirectories
                for subdir in checkpoint_dir.iterdir():
                    if subdir.is_dir():
                        subdir_info = {
                            "name": subdir.name,
                            "path": str(subdir),
                            "has_config": (subdir / "config.json").exists(),
                            "has_metadata": (subdir / "metadata.json").exists(),
                            "has_model": any(
                                (subdir / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "model.pt"]
                            ),
                        }
                        dir_info["subdirs"].append(subdir_info)

                # Find metadata files
                metadata_files = list(checkpoint_dir.glob("*/metadata.json"))
                dir_info["metadata_files"] = [str(f) for f in metadata_files]

            checkpoint_info["checkpoints"].append(dir_info)

        return checkpoint_info

    def _validate_evaluation_dataset(self) -> None:
        """Validate the evaluation dataset format."""
        if not self.config.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.config.eval_dataset_path}")

        # Check if it's HuggingFace dataset format
        if self.config.eval_dataset_path.is_dir():
            # HuggingFace format
            required_files = ["dataset_info.json"]
            missing_files = [f for f in required_files if not (self.config.eval_dataset_path / f).exists()]
            if missing_files:
                self.logger.warning(f"HuggingFace dataset may be incomplete, missing: {missing_files}")

        # Check if it's JSON format
        elif self.config.eval_dataset_path.suffix == ".json":
            with open(self.config.eval_dataset_path) as f:
                dataset = json.load(f)

            # Check required top-level keys
            required_keys = ["metadata", "conditions"]
            for key in required_keys:
                if key not in dataset:
                    raise ValueError(f"Missing required key in evaluation dataset: {key}")

        else:
            raise ValueError(f"Unknown evaluation dataset format: {self.config.eval_dataset_path}")

        self.logger.info("Evaluation dataset format validation passed")

    def run_collection_pipeline(self) -> dict[str, t.Any]:
        """Run the collection pipeline using auto-discovered configuration."""
        self.logger.info("Starting ICL evaluation collection pipeline with auto-discovery")

        try:
            # Check if we need to convert HuggingFace dataset to JSON format
            eval_dataset_path = self.config.eval_dataset_path

            if self.config.eval_dataset_path.is_dir():
                self.logger.info("Converting HuggingFace dataset to JSON format for evaluator...")
                eval_dataset_path = self._convert_hf_dataset_to_json()
                self.logger.info(f"Using JSON dataset: {eval_dataset_path}")
            else:
                self.logger.info(f"Using existing JSON dataset: {eval_dataset_path}")

            # Create legacy config with correct path
            legacy_config = self._create_legacy_config_with_path(eval_dataset_path)

            # Create enhanced evaluator with converted config
            evaluator = CollectionEvaluator(legacy_config)

            # Run comprehensive collection
            results = evaluator.run_comprehensive_collection()

            self.logger.info("Collection pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Collection pipeline failed: {e}")
            raise

    def _create_legacy_config_with_path(self, eval_dataset_path):
        """Convert new CollectionConfig to legacy format with specific dataset path."""
        # Create output structure here since we have the method
        self.config.create_output_structure()

        # Import the legacy EvaluationConfig
        from ICL.eval.collection.data_schema import EvaluationConfig

        # Create legacy config with the correct eval dataset path
        legacy_config = EvaluationConfig(
            checkpoint_base_dirs=self.config.checkpoint_dirs,
            eval_dataset_path=eval_dataset_path,  # Use the provided path
            output_dir=self.config.output_dir,
            context_sizes=self.config.context_sizes,
            transfer_conditions=self.config.transfer_conditions,
            control_types=self.config.control_types,
            target_configs=self.config.target_configs,
            diversity_levels=self.config.diversity_levels,
            model_types=self.config.model_types,
            device=self.config.device,
            batch_size=self.config.batch_size,
            max_sequences_per_condition=self.config.max_sequences_per_condition,
            capture_attention=self.config.capture_attention,
            save_intermediate=self.config.save_intermediate,
            overwrite_existing=self.config.overwrite,
        )

        # Add missing attributes that the evaluator expects
        legacy_config.create_output_structure = lambda: None  # No-op since we already created it
        legacy_config.resume = self.config.resume  # Add missing resume attribute
        legacy_config.intermediate_save_frequency = getattr(self.config, "intermediate_save_frequency", 5)
        legacy_config.clear_cache_frequency = getattr(self.config, "clear_cache_frequency", 10)
        legacy_config.max_memory_usage_gb = getattr(self.config, "max_memory_usage_gb", 12.0)
        legacy_config.max_model_failures = getattr(self.config, "max_model_failures", 5)
        legacy_config.retry_failed_models = getattr(self.config, "retry_failed_models", True)
        legacy_config.failure_retry_delay = getattr(self.config, "failure_retry_delay", 60)

        return legacy_config

    def _convert_hf_dataset_to_json(self) -> Path:
        """Convert HuggingFace dataset to JSON format expected by evaluator."""
        import json

        from datasets import load_from_disk

        # Load HuggingFace dataset
        hf_dataset = load_from_disk(str(self.config.eval_dataset_path))

        # Create JSON output path
        json_path = self.config.eval_dataset_path.parent / "evaluation_dataset.json"

        if json_path.exists():
            self.logger.info(f"JSON dataset already exists: {json_path}")
            return json_path

        self.logger.info(f"Converting HuggingFace dataset to JSON: {json_path}")

        # Convert to the expected JSON format
        # Group sequences by transfer condition and other criteria
        sequences_by_condition = {}

        for i, example in enumerate(hf_dataset):
            transfer_condition = example.get("transfer_condition", "unknown")
            control_type = example.get("control_type", "normal")
            config_L = example.get("config_L", 2)
            config_m = example.get("config_m", 2)

            if transfer_condition not in sequences_by_condition:
                sequences_by_condition[transfer_condition] = []

            # Convert to expected sequence format
            sequence = {
                "context_features": example.get("context_features", []),
                "context_labels": example.get("context_labels", []),
                "query_features": example.get("query_features", []),
                "query_label": example.get("query_label", 0),
                "context_size": example.get("context_size", 1),
                "sequence_id": i,
                "config": [config_L, config_m],
                "transfer_condition": transfer_condition,
                "control_type": control_type,
            }

            sequences_by_condition[transfer_condition].append(sequence)

        # Create the expected JSON structure
        json_data = {
            "metadata": {
                "dataset_type": self.config.dataset_config.dataset_type,
                "mixture_type": self.config.dataset_config.mixture_type,
                "total_rules": self.config.dataset_config.total_rules,
                "seed": self.config.dataset_config.seed,
                "total_sequences": len(hf_dataset),
                "conversion_timestamp": datetime.now().isoformat(),
            },
            "conditions": {},
        }

        # Organize conditions in the expected format
        for condition, sequences in sequences_by_condition.items():
            if condition == "within_config":
                # Group by config
                config_groups = {}
                for seq in sequences:
                    config_key = tuple(seq["config"])
                    if config_key not in config_groups:
                        config_groups[config_key] = []
                    config_groups[config_key].append(seq)

                json_data["conditions"][condition] = []
                for config, config_sequences in config_groups.items():
                    # Group by context size
                    sequences_by_k = {}
                    for seq in config_sequences:
                        k = seq["context_size"]
                        if k not in sequences_by_k:
                            sequences_by_k[k] = []
                        sequences_by_k[k].append(seq)

                    json_data["conditions"][condition].append({"config": list(config), "sequences": sequences_by_k})

            else:
                # For transfer conditions, group differently
                config_groups = {}
                for seq in sequences:
                    config_key = tuple(seq["config"])
                    if config_key not in config_groups:
                        config_groups[config_key] = []
                    config_groups[config_key].append(seq)

                json_data["conditions"][condition] = {}
                for config, config_sequences in config_groups.items():
                    config_str = f"{config[0]}_{config[1]}"

                    # Group by context size
                    sequences_by_k = {}
                    for seq in config_sequences:
                        k = seq["context_size"]
                        if k not in sequences_by_k:
                            sequences_by_k[k] = []
                        sequences_by_k[k].append(seq)

                    json_data["conditions"][condition][config_str] = [
                        {"config": list(config), "sequences": sequences_by_k}
                    ]

        # Save JSON file
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        self.logger.info(f"Successfully converted to JSON format: {json_path}")
        return json_path

    def _create_legacy_config(self):
        """Convert new CollectionConfig to legacy format for evaluator."""
        # Create output structure here since we have the method
        self.config.create_output_structure()

        # Import the legacy EvaluationConfig
        from ICL.eval.collection.data_schema import EvaluationConfig

        # Create legacy config
        legacy_config = EvaluationConfig(
            checkpoint_base_dirs=self.config.checkpoint_dirs,
            eval_dataset_path=self.config.eval_dataset_path,
            output_dir=self.config.output_dir,
            context_sizes=self.config.context_sizes,
            transfer_conditions=self.config.transfer_conditions,
            control_types=self.config.control_types,
            target_configs=self.config.target_configs,
            diversity_levels=self.config.diversity_levels,
            model_types=self.config.model_types,
            device=self.config.device,
            batch_size=self.config.batch_size,
            max_sequences_per_condition=self.config.max_sequences_per_condition,
            capture_attention=self.config.capture_attention,
            save_intermediate=self.config.save_intermediate,
            overwrite_existing=self.config.overwrite,
        )

        # Add the missing method to the legacy config
        legacy_config.create_output_structure = lambda: None  # No-op since we already created it

        return legacy_config

    def generate_collection_summary(self, results: dict[str, t.Any]) -> Path:
        """Generate comprehensive summary including auto-discovery info."""
        summary_path = self.config.output_dir / "collection_summary.json"

        summary = {
            "experiment_info": {
                "dataset_type": self.config.dataset_config.dataset_type,
                "mixture_type": self.config.dataset_config.mixture_type,
                "total_rules": self.config.dataset_config.total_rules,
                "seed": self.config.dataset_config.seed,
                "experiment_name": self.config.dataset_config.to_base_name(),
            },
            "auto_discovery": {
                "eval_dataset_path": str(self.config.eval_dataset_path),
                "checkpoint_dirs": [str(d) for d in self.config.checkpoint_dirs],
                "config_file_path": str(self.config.config_file_path),
                "config_file_exists": self.config.config_file_path.exists(),
            },
            "collection_config": self.config.to_dict(),
            "results": results,
            "output_files": {
                "icl_performance": "raw_evaluations/icl_performance.parquet",
                "model_registry": "metadata/model_registry.parquet",
                "attention_data": "raw_evaluations/attention_data/",
                "logs": "logs/",
                "intermediate_results": "intermediate/partial_results.parquet",
            },
            "collection_statistics": {
                "total_evaluations": results.get("total_evaluations", 0),
                "total_models": results.get("total_models", 0),
                "completed_models": results.get("completed_models", 0),
                "failed_models": results.get("failed_models", 0),
                "success_rate": (results.get("completed_models", 0) / max(results.get("total_models", 1), 1) * 100),
            },
            "next_steps": [
                "# Analysis Phase Commands:",
                "# 1. Load collected data:",
                "import pandas as pd",
                f"df = pd.read_parquet('{self.config.output_dir}/raw_evaluations/icl_performance.parquet')",
                f"model_registry = pd.read_parquet('{self.config.output_dir}/metadata/model_registry.parquet')",
                "",
                "# 2. Run analysis phase:",
                f"# python run_analysis.py --dataset-type {self.config.dataset_config.dataset_type} "
                f"--mixture-type {self.config.dataset_config.mixture_type} "
                f"--total-rules {self.config.dataset_config.total_rules} "
                f"--seed {self.config.dataset_config.seed} "
                f"--data-dir {self.config.output_dir}/raw_evaluations/",
            ],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Collection summary saved: {summary_path}")
        return summary_path


def run_collection_phase_auto(
    dataset_type: str, mixture_type: str, total_rules: int, seed: int, device: str = "cuda", **kwargs
) -> dict[str, t.Any]:
    """High-level function to run collection phase with auto-discovery."""
    # Create configuration using auto-discovery
    config = CollectionConfig.from_experiment_args(
        dataset_type=dataset_type,
        mixture_type=mixture_type,
        total_rules=total_rules,
        seed=seed,
        device=device,
        **kwargs,
    )

    # Initialize coordinator
    coordinator = CollectionCoordinator(config)

    # Validate setup
    if not coordinator.validate_setup():
        raise RuntimeError("Collection setup validation failed")

    # Run collection pipeline
    results = coordinator.run_collection_pipeline()

    # Generate summary
    summary_path = coordinator.generate_collection_summary(results)

    print("\nCollection completed successfully!")
    print(f"Experiment: {config.dataset_config.to_base_name()}")
    print(f"Results: {config.output_dir}")
    print(f"Summary: {summary_path}")

    return results
