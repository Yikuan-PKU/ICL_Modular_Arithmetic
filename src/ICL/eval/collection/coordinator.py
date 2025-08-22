"""Collection coordinator with minimal model config integration."""

import json
import logging
import typing as t
from datetime import datetime
from pathlib import Path

from ICL.eval.collection.collection_config import CollectionConfig
from ICL.eval.collection.evaluator import CollectionEvaluator

logger = logging.getLogger(__name__)


class CollectionCoordinator:
    """Collection coordinator with minimal model config integration."""

    def __init__(self, config: CollectionConfig):
        """Initialize coordinator with configuration."""
        self.config = config
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging for the collection phase."""
        log_dir = self.config.output_dir / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.batch_mode:
            log_file = log_dir / f"batch_collection_{timestamp}.log"
        else:
            log_file = log_dir / f"collection_{self.config.model_variant}_{self.config.eval_type}_{timestamp}.log"

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
        """Validate setup for batch or single mode."""
        self.logger.info("Validating collection setup...")

        try:
            if self.config.batch_mode:
                return self._validate_batch_setup()
            return self._validate_single_setup()

        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False

    def _validate_batch_setup(self) -> bool:
        """Validate setup for batch mode."""
        # Discover available combinations
        combinations = self.config.discover_available_combinations()

        if not combinations:
            self.logger.error("No valid (model_variant, eval_type) combinations found")
            return False

        self.logger.info(f"Found {len(combinations)} valid combinations:")
        for variant, eval_type in combinations:
            self.logger.info(f"  - {variant} / {eval_type}")

        return True

    def _validate_single_setup(self) -> bool:
        """Validate setup for single combination mode."""
        # Validate configuration
        self.config.validate()

        # Log single combination info
        self.logger.info("\nSingle Combination Setup:")
        self.logger.info("-" * 40)
        self.logger.info(f"Dataset: {self.config.get_shared_identifier()}")
        self.logger.info(f"Model variant: {self.config.model_variant}")
        self.logger.info(f"Evaluation type: {self.config.eval_type}")
        self.logger.info(f"Evaluation dataset: {self.config.eval_dataset_path}")
        self.logger.info(f"Model directory: {self.config.model_base_dir}")
        self.logger.info(f"Output directory: {self.config.output_dir}")

        # Validate HuggingFace dataset format
        self._validate_evaluation_dataset()

        return True

    def _validate_evaluation_dataset(self) -> None:
        """Validate the evaluation dataset format."""
        if not self.config.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.config.eval_dataset_path}")

        # Check if it's a HuggingFace dataset directory
        if self.config.eval_dataset_path.is_dir():
            # Check for HuggingFace dataset files
            required_hf_files = ["dataset_info.json"]
            missing_files = [f for f in required_hf_files if not (self.config.eval_dataset_path / f).exists()]
            if missing_files:
                self.logger.warning(f"HuggingFace dataset may be incomplete, missing: {missing_files}")
        else:
            raise ValueError(f"Expected HuggingFace dataset directory, got: {self.config.eval_dataset_path}")

        self.logger.info("Evaluation dataset format validation passed")

    def list_available_combinations(self) -> dict[str, t.Any]:
        """List all available (model_variant, eval_type) combinations."""
        combinations = self.config.discover_available_combinations()

        combination_info = {
            "shared_identifier": self.config.get_shared_identifier(),
            "total_combinations": len(combinations),
            "combinations": [],
        }

        for variant, eval_type in combinations:
            # Get additional info for each combination
            if self.config.batch_mode:
                from ICL.settings import PATH

                shared_id = self.config.get_shared_identifier()
                model_dir = PATH.model_dir / shared_id / variant
                eval_dataset_path = PATH.dataset_root / shared_id / "eval" / eval_type / "dataset"
            else:
                model_dir = (
                    self.config.model_base_dir.parent / variant
                    if self.config.batch_mode
                    else self.config.model_base_dir
                )
                eval_dataset_path = (
                    self.config.eval_dataset_path.parent.parent / eval_type / "dataset"
                    if not self.config.batch_mode
                    else None
                )

            # Count checkpoints
            checkpoint_count = 0
            if model_dir.exists():
                checkpoint_dirs = [d for d in model_dir.iterdir() if d.is_dir() and "checkpoint" in d.name.lower()]
                checkpoint_count = len(checkpoint_dirs)

            combo_info = {
                "model_variant": variant,
                "eval_type": eval_type,
                "model_dir": str(model_dir),
                "eval_dataset_path": str(eval_dataset_path),
                "model_dir_exists": model_dir.exists(),
                "eval_dataset_exists": eval_dataset_path.exists() if eval_dataset_path else False,
                "checkpoint_count": checkpoint_count,
            }

            combination_info["combinations"].append(combo_info)

        return combination_info

    def run_collection_pipeline(self) -> dict[str, t.Any]:
        """Run the collection pipeline in batch or single mode."""
        if self.config.batch_mode:
            return self._run_batch_collection()
        return self._run_single_collection()

    def _run_batch_collection(self) -> dict[str, t.Any]:
        """Run collection for all available combinations."""
        self.logger.info("Starting batch collection pipeline")

        combinations = self.config.discover_available_combinations()
        if not combinations:
            raise RuntimeError("No valid combinations found for batch collection")

        batch_results = {
            "shared_identifier": self.config.get_shared_identifier(),
            "start_time": datetime.now(),
            "total_combinations": len(combinations),
            "completed_combinations": [],
            "failed_combinations": [],
            "combination_results": {},
        }

        for i, (variant, eval_type) in enumerate(combinations, 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"BATCH COLLECTION [{i}/{len(combinations)}]")
            self.logger.info(f"Model variant: {variant}")
            self.logger.info(f"Evaluation type: {eval_type}")
            self.logger.info(f"{'=' * 60}")

            try:
                # Create single config for this combination
                single_config = self.config.create_single_config(variant, eval_type)

                # Run single collection
                single_coordinator = CollectionCoordinator(single_config)
                if not single_coordinator.validate_setup():
                    raise RuntimeError(f"Setup validation failed for {variant}/{eval_type}")

                results = single_coordinator._run_single_collection()

                # Record success
                batch_results["completed_combinations"].append((variant, eval_type))
                batch_results["combination_results"][f"{variant}_{eval_type}"] = results

                self.logger.info(f"✓ Completed {variant}/{eval_type}")

            except Exception as e:
                self.logger.error(f"✗ Failed {variant}/{eval_type}: {e}")
                batch_results["failed_combinations"].append((variant, eval_type, str(e)))

                # Continue with next combination
                continue

        # Generate batch summary
        batch_results["end_time"] = datetime.now()
        batch_results["duration"] = batch_results["end_time"] - batch_results["start_time"]
        batch_results["success_count"] = len(batch_results["completed_combinations"])
        batch_results["failure_count"] = len(batch_results["failed_combinations"])

        # Save batch summary
        summary_path = self.config.output_dir / "batch_collection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(batch_results, f, indent=2, default=str)

        self.logger.info(
            f"\nBatch collection completed: {batch_results['success_count']}/{batch_results['total_combinations']} successful"
        )
        self.logger.info(f"Batch summary saved: {summary_path}")

        return batch_results

    def _run_single_collection(self) -> dict[str, t.Any]:
        """Run collection for a single (model_variant, eval_type) combination."""
        self.logger.info("Starting single combination collection pipeline")

        try:
            # Convert to legacy format for evaluator
            legacy_config = self._create_legacy_config()

            # Create and run evaluator
            evaluator = CollectionEvaluator(legacy_config)
            results = evaluator.run_comprehensive_collection()

            self.logger.info("Single collection completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Single collection failed: {e}")
            raise

    def _create_legacy_config(self):
        """Convert CollectionConfig to legacy format for evaluator."""

        # Simple object to hold config data
        class LegacyConfig:
            pass

        legacy_config = LegacyConfig()

        # Core identification
        legacy_config.dataset_type = self.config.dataset_type
        legacy_config.num_seeds = self.config.num_seeds
        legacy_config.seed = self.config.seed
        legacy_config.config_L = self.config.config_L
        legacy_config.config_m = self.config.config_m

        # Model and evaluation
        legacy_config.model_variant = self.config.model_variant
        legacy_config.eval_type = self.config.eval_type

        # Paths
        legacy_config.eval_dataset_path = self.config.eval_dataset_path
        legacy_config.model_base_dir = self.config.model_base_dir
        legacy_config.output_dir = self.config.output_dir

        # Evaluation parameters
        legacy_config.context_sizes = self.config.context_sizes
        legacy_config.control_types = self.config.control_types
        legacy_config.device = self.config.device
        legacy_config.batch_size = self.config.batch_size
        legacy_config.max_sequences_per_condition = self.config.max_sequences_per_condition
        legacy_config.capture_attention = self.config.capture_attention
        legacy_config.save_intermediate = self.config.save_intermediate

        # Methods
        legacy_config.validate = lambda: self.config.validate()
        legacy_config.create_output_structure = lambda: self.config.create_output_structure()
        legacy_config.get_shared_identifier = lambda: self.config.get_shared_identifier()
        legacy_config.get_resume_path = lambda: self.config.get_resume_path()
        legacy_config.to_dict = lambda: self.config.to_dict()

        # Additional attributes that evaluator expects
        legacy_config.resume = self.config.resume
        legacy_config.intermediate_save_frequency = self.config.intermediate_save_frequency
        legacy_config.clear_cache_frequency = self.config.clear_cache_frequency
        legacy_config.max_memory_usage_gb = self.config.max_memory_usage_gb
        legacy_config.max_model_failures = self.config.max_model_failures
        legacy_config.retry_failed_models = self.config.retry_failed_models
        legacy_config.failure_retry_delay = self.config.failure_retry_delay

        return legacy_config

    def generate_collection_summary(self, results: dict[str, t.Any]) -> Path:
        """Generate collection summary for single combination."""
        summary_path = self.config.output_dir / "collection_summary.json"

        summary = {
            "shared_identifier": self.config.get_shared_identifier(),
            "model_variant": self.config.model_variant,
            "eval_type": self.config.eval_type,
            "collection_config": self.config.to_dict(),
            "results": results,
            "output_files": {
                "icl_performance": "raw_evaluations/icl_performance.parquet",
                "model_registry": "metadata/model_registry.parquet",
                "attention_data": "raw_evaluations/attention_data/" if self.config.capture_attention else None,
                "logs": "logs/",
                "intermediate_results": "intermediate/partial_results.parquet"
                if self.config.save_intermediate
                else None,
            },
            "collection_statistics": {
                "total_evaluations": results.get("total_evaluations", 0),
                "total_models": results.get("total_models", 0),
                "completed_models": results.get("completed_models", 0),
                "failed_models": results.get("failed_models", 0),
                "success_rate": (results.get("completed_models", 0) / max(results.get("total_models", 1), 1) * 100),
            },
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Collection summary saved: {summary_path}")
        return summary_path


def run_collection_phase(
    dataset_type: str,
    num_seeds: int,
    seed: int,
    model_variant: str | None = None,
    eval_type: str | None = None,
    batch_mode: bool = False,
    device: str = "cuda",
    **kwargs,
) -> dict[str, t.Any]:
    """High-level function to run collection phase with training script style."""
    # Create configuration
    config = CollectionConfig.from_args(
        dataset_type=dataset_type,
        num_seeds=num_seeds,
        seed=seed,
        model_variant=model_variant,
        eval_type=eval_type,
        batch_mode=batch_mode,
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

    if not batch_mode:
        # Generate summary for single mode
        summary_path = coordinator.generate_collection_summary(results)
        print("\nCollection completed successfully!")
        print(f"Combination: {model_variant}/{eval_type}")
        print(f"Results: {config.output_dir}")
        print(f"Summary: {summary_path}")
    else:
        print("\nBatch collection completed!")
        print(f"Dataset: {config.get_shared_identifier()}")
        print(f"Results: {config.output_dir}")
        print(f"Summary: {config.output_dir}/batch_collection_summary.json")

    return results
