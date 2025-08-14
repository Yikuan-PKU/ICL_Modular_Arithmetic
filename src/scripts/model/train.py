import argparse
import logging
from datetime import datetime

from ICL import settings
from ICL.datasets.hf import RHMDataLoaderFactory
from ICL.train.model import RHMTrainer, RHMTrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models on synthetic dataset.")
    parser.add_argument("--model_type", action="store_true", help="Resume from the existing checkpoint")

    return parser.parse_args()


def main():
    """Example of how to use the training framework"""
    # Initialize factory and create DataLoaders
    args = parse_args()
    factory = RHMDataLoaderFactory(settings.PATH.train_dir / "raw", vocab_size=32)

    # Create train DataLoader
    train_dataloader, train_metadata = factory.create_dataloader(
        task_name="mlm", batch_size=16, max_length=1024, batching_strategy="config_then_length", seed=42
    )

    # Create eval DataLoader (subset for faster evaluation)
    eval_dataloader, eval_metadata = factory.create_dataloader(
        task_name="mlm",
        batch_size=32,
        max_length=1024,
        batching_strategy="config_then_length",
        filter_max_length=512,  # Smaller sequences for eval
        seed=42,
    )

    # Create training configuration
    training_config = RHMTrainingConfig(
        task_name="mlm",
        output_dir=settings.PATH.model_dir / "rhm_mlm_training",
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
        run_name=f"rhm_mlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=42,
    )

    # Create and run trainer
    trainer = RHMTrainer(
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
    main()
