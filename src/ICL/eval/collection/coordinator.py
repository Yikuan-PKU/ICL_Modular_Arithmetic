"""Simplified collection coordinator."""

import logging

import torch

from ICL.eval.collection.data_schema import extract_dataset_metadata, validate_eval_dataset
from ICL.eval.collection.evaluator import CollectionEvaluator

logger = logging.getLogger(__name__)


class CollectionCoordinator:
    """Simplified collection coordinator with validation."""

    def __init__(self, config):
        """Simple initialization."""
        self.config = config

    def validate_setup(self) -> bool:
        """Enhanced setup validation with eval_type checking."""
        try:
            if self.config.batch_mode:
                combinations = self.config.discover_combinations()
                if not combinations:
                    logger.error("No valid combinations found")
                    return False

                logger.info(f"Found {len(combinations)} valid combinations")

                # Validate each combination's eval dataset
                shared_id = f"{self.config.dataset_type}_{self.config.num_seeds}_L{self.config.config_L}_M{self.config.config_m}"
                eval_base = self.config.eval_dataset_path.parent.parent if self.config.eval_dataset_path else None

                if not eval_base:
                    eval_base = self.config.model_base_dir.parent / "datasets" / shared_id / "eval"

                for variant, eval_type in combinations:
                    eval_path = eval_base / eval_type / "dataset"
                    if not validate_eval_dataset(eval_path, eval_type):
                        logger.warning(f"Eval dataset validation failed for {variant}/{eval_type}")

            else:
                # Single mode validation
                if not self.config.eval_dataset_path.exists():
                    logger.error(f"Dataset not found: {self.config.eval_dataset_path}")
                    return False
                if not self.config.model_base_dir.exists():
                    logger.error(f"Model dir not found: {self.config.model_base_dir}")
                    return False

                # Validate eval_type consistency
                if not validate_eval_dataset(self.config.eval_dataset_path, self.config.eval_type):
                    logger.error(f"Eval dataset does not match eval_type {self.config.eval_type}")
                    return False

                # Log dataset metadata
                metadata = extract_dataset_metadata(self.config.eval_dataset_path)
                logger.info(
                    f"Dataset: {metadata['num_examples']} examples, hint: {metadata.get('eval_type_hint', 'unknown')}"
                )

                logger.info(f"Single mode: {self.config.model_variant}/{self.config.eval_type}")

            # Log memory management settings
            self._log_memory_settings()

            return True

        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False

    def _log_memory_settings(self):
        """Log current memory management configuration."""
        logger.info("Memory management settings:")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Sequence chunk size: {self.config.sequence_chunk_size}")
        logger.info(f"  Model offloading: {self.config.offload_models}")
        logger.info(f"  Attention capture: {self.config.capture_attention}")
        if self.config.capture_attention:
            logger.info(f"  Selective attention: {self.config.selective_attention}")
            logger.info(f"  Attention sampling rate: {self.config.attention_sampling_rate}")

    def list_available_combinations(self) -> dict:
        """Simple combination listing with validation info."""
        combinations = self.config.discover_combinations()

        # Add validation info
        shared_id = (
            f"{self.config.dataset_type}_{self.config.num_seeds}_L{self.config.config_L}_M{self.config.config_m}"
        )
        eval_base = self.config.model_base_dir.parent / "datasets" / shared_id / "eval"

        validated_combinations = []
        for variant, eval_type in combinations:
            eval_path = eval_base / eval_type / "dataset"
            is_valid = validate_eval_dataset(eval_path, eval_type)
            validated_combinations.append({"model_variant": variant, "eval_type": eval_type, "dataset_valid": is_valid})

        return {
            "total_combinations": len(combinations),
            "combinations": validated_combinations,
        }

    def run_collection_pipeline(self) -> dict:
        """Simple collection pipeline with memory monitoring."""
        if self.config.batch_mode:
            return self._run_batch_collection()
        return self._run_single_collection()

    def _run_batch_collection(self) -> dict:
        """Batch collection with memory monitoring."""
        logger.info("Starting batch collection")

        combinations = self.config.discover_combinations()
        batch_results = {
            "total_combinations": len(combinations),
            "completed": [],
            "failed": [],
        }

        for variant, eval_type in combinations:
            # check whether there is already the file

            logger.info(f"Processing {variant}/{eval_type}")

            # Log memory usage if CUDA available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory before evaluation: {memory_allocated:.2f} GB")

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

                # Force cleanup on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
