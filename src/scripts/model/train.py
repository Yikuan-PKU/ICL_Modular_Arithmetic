import logging
from datetime import datetime
from typing import Any

from ICL import setting
from ICL.datasets.hf import RHMDataLoaderFactory
from ICL.train.model import RHMTrainer, RHMTrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_trainer(
    training_config: RHMTrainingConfig, train_dataloader, eval_dataloader, dataloader_metadata: dict[str, Any]
) -> RHMTrainer:
    """Convenience function to create an RHM trainer.

    Args:
        training_config: Training configuration
        train_dataloader: Training DataLoader
        eval_dataloader: Evaluation DataLoader
        dataloader_metadata: Metadata from DataLoader creation

    Returns:
        RHMTrainer instance

    """
    return RHMTrainer(training_config, train_dataloader, eval_dataloader, dataloader_metadata)


def main_training_example():
    """Example of how to use the training framework"""
    # Initialize factory and create DataLoaders
    factory = RHMDataLoaderFactory(setting.PATH.train / "raw_rhm_data", vocab_size=32)

    # Create train DataLoader
    train_dataloader, train_metadata = factory.create_dataloader(
        task_name="clm", batch_size=16, max_length=1024, batching_strategy="config_then_length", seed=42
    )

    # Create eval DataLoader (subset for faster evaluation)
    eval_dataloader, eval_metadata = factory.create_dataloader(
        task_name="clm",
        batch_size=32,
        max_length=1024,
        batching_strategy="config_then_length",
        filter_max_length=512,  # Smaller sequences for eval
        seed=42,
    )

    # Create training configuration
    training_config = RHMTrainingConfig(
        task_name="clm",
        output_dir=setting.PATH.model / "rhm_clm_training",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        track_hierarchical_metrics=True,
        early_stopping=True,
        early_stopping_patience=3,
        run_name=f"rhm_clm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=42,
    )

    # Create and run trainer
    trainer = create_trainer(
        training_config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        dataloader_metadata=train_metadata,
    )

    # Train model
    results = trainer.train()

    print(f"Training completed! Results saved to {training_config.output_dir}")
    return results


if __name__ == "__main__":
    main_training_example()
