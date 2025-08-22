"""Simplified collection coordinator."""

import logging

from ICL.eval.collection.evaluator import CollectionEvaluator

logger = logging.getLogger(__name__)


class CollectionCoordinator:
    """Simplified collection coordinator."""

    def __init__(self, config):
        """Simple initialization."""
        self.config = config

    def validate_setup(self) -> bool:
        """Basic setup validation."""
        try:
            if self.config.batch_mode:
                combinations = self.config.discover_combinations()
                if not combinations:
                    logger.error("No valid combinations found")
                    return False
                logger.info(f"Found {len(combinations)} combinations")
            else:
                # Basic path checks
                if not self.config.eval_dataset_path.exists():
                    logger.error(f"Dataset not found: {self.config.eval_dataset_path}")
                    return False
                if not self.config.model_base_dir.exists():
                    logger.error(f"Model dir not found: {self.config.model_base_dir}")
                    return False
                logger.info(f"Single mode: {self.config.model_variant}/{self.config.eval_type}")

            return True

        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False

    def list_available_combinations(self) -> dict:
        """Simple combination listing."""
        combinations = self.config.discover_combinations()

        return {
            "total_combinations": len(combinations),
            "combinations": [{"model_variant": variant, "eval_type": eval_type} for variant, eval_type in combinations],
        }

    def run_collection_pipeline(self) -> dict:
        """Simple collection pipeline."""
        if self.config.batch_mode:
            return self._run_batch_collection()
        return self._run_single_collection()

    def _run_batch_collection(self) -> dict:
        """Simple batch collection."""
        logger.info("Starting batch collection")

        combinations = self.config.discover_combinations()
        batch_results = {
            "total_combinations": len(combinations),
            "completed": [],
            "failed": [],
        }

        for variant, eval_type in combinations:
            logger.info(f"Processing {variant}/{eval_type}")

            try:
                # Create single config
                single_config = self.config.create_single_config(variant, eval_type)

                # Run evaluation
                evaluator = CollectionEvaluator(single_config)
                results = evaluator.run_comprehensive_collection()

                batch_results["completed"].append((variant, eval_type))
                logger.info(f"✓ Completed {variant}/{eval_type}")

            except Exception as e:
                logger.error(f"✗ Failed {variant}/{eval_type}: {e}")
                batch_results["failed"].append((variant, eval_type, str(e)))
                continue

        logger.info(f"Batch completed: {len(batch_results['completed'])}/{len(combinations)} successful")
        return batch_results

    def _run_single_collection(self) -> dict:
        """Simple single collection."""
        logger.info(f"Starting single collection: {self.config.model_variant}/{self.config.eval_type}")

        try:
            evaluator = CollectionEvaluator(self.config)
            results = evaluator.run_comprehensive_collection()
            logger.info("Collection completed successfully")
            return results

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise


def run_collection_phase(
    dataset_type: str,
    num_seeds: int,
    seed: int,
    config_L: int,
    config_m: int,
    model_type: str,
    model_variant: str | None = None,
    eval_type: str | None = None,
    batch_mode: bool = False,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """Simple high-level collection function."""
    from ICL.eval.collection.collection_config import CollectionConfig

    # Create configuration
    config = CollectionConfig(
        dataset_type=dataset_type,
        num_seeds=num_seeds,
        seed=seed,
        config_L=config_L,
        config_m=config_m,
        model_type=model_type,
        model_variant=model_variant,
        eval_type=eval_type,
        batch_mode=batch_mode,
        device=device,
    )

    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Run collection
    coordinator = CollectionCoordinator(config)

    if not coordinator.validate_setup():
        raise RuntimeError("Setup validation failed")

    results = coordinator.run_collection_pipeline()

    # Simple output
    if batch_mode:
        print(f"Batch collection completed: {len(results['completed'])}/{results['total_combinations']} successful")
    else:
        print(f"Single collection completed: {config.model_variant}/{config.eval_type}")

    return results
