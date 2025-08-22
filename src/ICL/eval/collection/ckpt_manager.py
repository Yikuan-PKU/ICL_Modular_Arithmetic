"""Checkpoint management with explicit L,M values - no auto-discovery."""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from ICL.train.tokenizer import RHMTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Simplified checkpoint manager."""

    def __init__(self, config):
        """Simplified initialization."""
        self.config = config
        self.device = torch.device(config.device)

    def discover_checkpoints(self) -> list[dict]:
        """Simplified checkpoint discovery - returns simple dicts."""
        if not self.config.model_base_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.config.model_base_dir}")

        checkpoint_dirs = self._find_checkpoint_directories(self.config.model_base_dir)
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {self.config.model_base_dir}")

        all_metadata = []
        for checkpoint_path in checkpoint_dirs:
            try:
                checkpoint_step = self._extract_checkpoint_step(checkpoint_path)
                model_id = f"{self.config.model_variant}_step{checkpoint_step}"

                # Simple metadata dict
                metadata = {
                    "model_variant": self.config.model_variant,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": checkpoint_path,
                    "model_id": model_id,
                    "config_L": self.config.config_L,
                    "config_m": self.config.config_m,
                }
                all_metadata.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to process checkpoint {checkpoint_path}: {e}")
                continue

        all_metadata.sort(key=lambda x: x["checkpoint_step"])
        logger.info(f"Discovered {len(all_metadata)} checkpoints")
        return all_metadata

    def _find_checkpoint_directories(self, model_base_dir: Path) -> list[Path]:
        """Find all checkpoint directories within the model variant directory."""
        # Debug: check what's actually in the directory
        logger.info(f"Searching for checkpoints in: {model_base_dir}")
        if model_base_dir.exists():
            all_items = list(model_base_dir.iterdir())
            logger.info(f"Found {len(all_items)} items in directory")
            for item in all_items[:5]:  # Log first 5 items
                logger.info(f"  Item: {item.name} (dir: {item.is_dir()})")

        checkpoint_patterns = [
            "checkpoint-*",  # Standard HuggingFace format
            "step_*",  # Alternative naming
            "step-*",  # Another alternative
        ]

        found_checkpoints = []
        for pattern in checkpoint_patterns:
            matches = list(model_base_dir.glob(pattern))
            logger.info(f"Pattern '{pattern}' found {len(matches)} matches")
            found_checkpoints.extend(matches)

        # Filter to only directories that look like valid checkpoints
        valid_checkpoints = []
        for checkpoint_path in found_checkpoints:
            logger.info(f"Checking checkpoint: {checkpoint_path}")
            if checkpoint_path.is_dir():
                is_valid = self._is_valid_checkpoint(checkpoint_path)
                logger.info(f"  Is valid checkpoint: {is_valid}")
                if is_valid:
                    valid_checkpoints.append(checkpoint_path)

        logger.info(f"Found {len(valid_checkpoints)} valid checkpoints in {model_base_dir}")
        return sorted(valid_checkpoints)

    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """Check if path contains a valid model checkpoint."""
        # Check for essential files
        config_file = checkpoint_path / "config.json"
        if not config_file.exists():
            logger.info(f"  Missing config.json in {checkpoint_path}")
            return False

        # Check for model weights (at least one format)
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",  # This is the correct filename
            "pytorch_model.safetensors",
            "model.pt",
        ]

        for model_file in model_files:
            if (checkpoint_path / model_file).exists():
                logger.info(f"  Found model file: {model_file}")
                return True

        logger.info(f"  No model files found in {checkpoint_path}")
        logger.info(f"  Available files: {[f.name for f in checkpoint_path.iterdir()]}")
        return False

    def _extract_checkpoint_step(self, checkpoint_path: Path) -> int:
        """Extract checkpoint step number from checkpoint path."""
        import re

        path_str = str(checkpoint_path.name)

        # Try different patterns for checkpoint step
        step_patterns = [
            r"checkpoint[_-](\d+)",
            r"step[_-](\d+)",
        ]

        for pattern in step_patterns:
            match = re.search(pattern, path_str)
            if match:
                return int(match.group(1))

        # If no pattern matches, try to extract any number
        numbers = re.findall(r"\d+", path_str)
        if numbers:
            return int(numbers[-1])  # Take the last number found

        # Default to 0 if cannot be determined
        logger.warning(f"Could not extract step from {path_str}, using step=0")
        return 0

    def load_model_checkpoint(self, metadata: dict):
        """Simplified model loading."""
        logger.info(f"Loading checkpoint: {metadata['model_id']}")

        # Load tokenizer
        tokenizer = RHMTokenizer.from_pretrained(metadata["checkpoint_path"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load model based on model_type
        if self.config.model_type == "clm":
            model = AutoModelForCausalLM.from_pretrained(
                metadata["checkpoint_path"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
        elif self.config.model_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(
                metadata["checkpoint_path"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        model = model.to(self.device)
        model.eval()

        # Resize embeddings if needed
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer
