"""Main coordinator for the comprehensive evaluation pipeline."""

import json
import logging
import typing as t
from datetime import datetime
from pathlib import Path

from ICL.eval.collection.data_schema import EvaluationConfig
from ICL.eval.collection.evaluator import ComprehensiveEvaluator


class EvaluationCoordinator:
    """Coordinates the complete evaluation pipeline."""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation coordinator."""
        self.config = config
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging for the evaluation pipeline."""
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Evaluation coordinator initialized. Logs: {log_file}")

    def run_evaluation_pipeline(self) -> dict[str, t.Any]:
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting comprehensive ICL evaluation pipeline")

        try:
            # Initialize evaluator
            evaluator = ComprehensiveEvaluator(self.config)

            # Run comprehensive evaluation
            results = evaluator.run_comprehensive_evaluation()

            self.logger.info("Evaluation pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {e}")
            raise

    def validate_setup(self) -> bool:
        """Validate that all required components are available."""
        self.logger.info("Validating evaluation setup...")

        try:
            # Validate configuration
            self.config.validate()

            # Check if evaluation dataset exists
            if not self.config.eval_dataset_path.exists():
                raise FileNotFoundError(f"Evaluation dataset not found: {self.config.eval_dataset_path}")

            # Check if checkpoint directories exist
            missing_dirs = []
            for checkpoint_dir in self.config.checkpoint_base_dirs:
                if not checkpoint_dir.exists():
                    missing_dirs.append(checkpoint_dir)

            if missing_dirs:
                raise FileNotFoundError(f"Missing checkpoint directories: {missing_dirs}")

            # Validate evaluation dataset format
            self._validate_evaluation_dataset()

            self.logger.info("Setup validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False

    def _validate_evaluation_dataset(self) -> None:
        """Validate the evaluation dataset format."""
        with open(self.config.eval_dataset_path) as f:
            dataset = json.load(f)

        # Check required top-level keys
        required_keys = ["metadata", "conditions"]
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"Missing required key in evaluation dataset: {key}")

        # Check conditions structure
        conditions = dataset["conditions"]
        expected_conditions = ["within_config", "depth_transfer", "synonym_transfer", "full_transfer"]

        missing_conditions = []
        for condition in expected_conditions:
            if condition not in conditions:
                missing_conditions.append(condition)

        if missing_conditions:
            self.logger.warning(f"Missing evaluation conditions: {missing_conditions}")

        # Validate that each condition has proper structure
        for condition_name, condition_data in conditions.items():
            if condition_name == "within_config":
                if not isinstance(condition_data, list):
                    raise ValueError(f"Expected list for {condition_name}, got {type(condition_data)}")
            elif not isinstance(condition_data, dict):
                raise ValueError(f"Expected dict for {condition_name}, got {type(condition_data)}")

        self.logger.info("Evaluation dataset format validation passed")

    def generate_evaluation_summary(self, results: dict[str, t.Any]) -> Path:
        """Generate a comprehensive summary of evaluation results."""
        summary_path = self.config.output_dir / "evaluation_summary.json"

        summary = {
            "evaluation_config": {
                "context_sizes": self.config.context_sizes,
                "transfer_conditions": self.config.transfer_conditions,
                "control_types": self.config.control_types,
                "target_configs": self.config.target_configs,
                "diversity_levels": self.config.diversity_levels,
                "model_types": self.config.model_types,
                "device": self.config.device,
                "capture_attention": self.config.capture_attention,
            },
            "results": results,
            "output_files": {
                "icl_performance": "raw_evaluations/icl_performance.parquet",
                "model_registry": "metadata/model_registry.parquet",
                "aggregated_metrics": "intermediate/aggregated_metrics.parquet",
                "attention_data": "raw_evaluations/attention_data/",
                "logs": "logs/",
            },
            "analysis_commands": [
                "# Load and analyze results:",
                "import pandas as pd",
                "df = pd.read_parquet('raw_evaluations/icl_performance.parquet')",
                "model_registry = pd.read_parquet('metadata/model_registry.parquet')",
                "",
                "# Quick analysis examples:",
                "# emergence_analysis = df.groupby(['config_L', 'config_m', 'n_train'])['accuracy'].mean()",
                "# transfer_analysis = df.groupby('transfer_condition')['accuracy'].mean()",
            ],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Evaluation summary saved: {summary_path}")
        return summary_path


def create_evaluation_config(
    checkpoint_dirs: list[Path],
    eval_dataset_path: Path,
    output_dir: Path,
    target_configs: list[tuple[int, int]],
    **kwargs,
) -> EvaluationConfig:
    """Helper function to create evaluation configuration."""
    config = EvaluationConfig(
        checkpoint_base_dirs=checkpoint_dirs,
        eval_dataset_path=eval_dataset_path,
        output_dir=output_dir,
        target_configs=target_configs,
        **kwargs,
    )

    return config


def run_comprehensive_evaluation(
    checkpoint_dirs: list[str | Path],
    eval_dataset_path: str | Path,
    output_dir: str | Path,
    target_configs: list[tuple[int, int]],
    **kwargs,
) -> dict[str, t.Any]:
    """High-level function to run comprehensive evaluation."""
    # Convert paths
    checkpoint_dirs = [Path(d) for d in checkpoint_dirs]
    eval_dataset_path = Path(eval_dataset_path)
    output_dir = Path(output_dir)

    # Create configuration
    config = create_evaluation_config(
        checkpoint_dirs=checkpoint_dirs,
        eval_dataset_path=eval_dataset_path,
        output_dir=output_dir,
        target_configs=target_configs,
        **kwargs,
    )

    # Initialize coordinator
    coordinator = EvaluationCoordinator(config)

    # Validate setup
    if not coordinator.validate_setup():
        raise RuntimeError("Evaluation setup validation failed")

    # Run evaluation pipeline
    results = coordinator.run_evaluation_pipeline()

    # Generate summary
    summary_path = coordinator.generate_evaluation_summary(results)

    print("\nEvaluation completed successfully!")
    print(f"Results: {output_dir}")
    print(f"Summary: {summary_path}")

    return results
