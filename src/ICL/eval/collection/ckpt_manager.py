"""Checkpoint management with explicit L,M values - no auto-discovery."""

import logging
import typing as t
import warnings
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from ICL import settings
from ICL.eval.collection.data_schema import (
    ModelMetadata,
    determine_training_phase,
    generate_model_variant_name,
)
from ICL.train.tokenizer import RHMTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoint discovery with explicit L,M values."""

    def __init__(self, config):
        """Initialize checkpoint manager with evaluation configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self._checkpoint_cache: dict[str, tuple[t.Any, t.Any]] = {}

        # Validate that config has explicit L,M,model_type
        if not hasattr(config, "config_L") or not hasattr(config, "config_m"):
            raise ValueError("Config must have explicit config_L and config_m values")

        # if not hasattr(config, "model_type") or not config.model_type:
        #     raise ValueError("Config must have explicit model_type value")

        if config.config_L is None or config.config_m is None:
            raise ValueError("config_L and config_m cannot be None - must be explicit")

        if config.model_type not in ["clm", "mlm"]:
            raise ValueError(f"model_type must be 'clm' or 'mlm', got '{config.model_type}'")

    def discover_checkpoints(self) -> list[ModelMetadata]:
        """Discover all available checkpoints for the specified model variant using explicit L,M."""
        all_metadata = []

        # Validate model directory exists (using explicit L,M)
        if not self.config.model_base_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.config.model_base_dir}")

        checkpoint_dirs = self._find_checkpoint_directories(self.config.model_base_dir)

        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {self.config.model_base_dir}")

        # Load minimal model config to get the essential fields
        model_config = self._load_model_config_for_variant()

        # Create metadata for each checkpoint
        max_step = max(self._extract_checkpoint_step(path) for path in checkpoint_dirs)

        for checkpoint_path in checkpoint_dirs:
            try:
                checkpoint_step = self._extract_checkpoint_step(checkpoint_path)
                training_phase = determine_training_phase(checkpoint_step, max_step)

                # Generate model ID
                model_id = self._generate_model_id(self.config.model_variant, checkpoint_step, checkpoint_path)

                metadata = ModelMetadata(
                    dataset_type=self.config.dataset_type,
                    num_seeds=self.config.num_seeds,
                    seed=self.config.seed,
                    config_L=self.config.config_L,  # EXPLICIT from config
                    config_m=self.config.config_m,  # EXPLICIT from config
                    task_name=model_config["task_name"],
                    model_variant=self.config.model_variant,
                    shuffle_before_packing=model_config["shuffle_before_packing"],
                    seed_balanced_batching=model_config["seed_balanced_batching"],
                    checkpoint_step=checkpoint_step,
                    checkpoint_path=checkpoint_path,
                    model_id=model_id,
                )

                all_metadata.append(metadata)

            except Exception as e:
                warnings.warn(f"Failed to process checkpoint {checkpoint_path}: {e}")
                continue

        # Sort by checkpoint step
        all_metadata.sort(key=lambda x: x.checkpoint_step)

        logger.info(f"Discovered {len(all_metadata)} checkpoints for {self.config.model_variant}")
        self._print_discovery_summary(all_metadata)

        return all_metadata

    def _load_model_config_for_variant(self) -> dict[str, t.Any]:
        """Load minimal model config for the current variant using explicit L,M,model_type."""
        # Use explicit L,M to construct shared identifier
        shared_id = (
            f"{self.config.dataset_type}_{self.config.num_seeds}_L{self.config.config_L}_M{self.config.config_m}"
        )

        # Load from unified training.yaml using explicit model_type
        training_config_path = settings.PATH.conf_dir / shared_id / "training.yaml"

        if not training_config_path.exists():
            raise FileNotFoundError(f"Training config file not found: {training_config_path}")

        # Load unified training config for the specified model type
        model_config = self._load_unified_training_config(training_config_path, self.config.model_type)

        # Verify that the loaded config generates the expected variant name
        expected_variant = generate_model_variant_name(model_config)
        if expected_variant != self.config.model_variant:
            logger.warning(f"Config mismatch: expected {expected_variant}, got {self.config.model_variant}")

        return model_config

    def _load_unified_training_config(self, config_path: Path, model_type: str) -> dict[str, t.Any]:
        """Load unified training config and extract model-specific configuration with validation."""
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(
                f"Training config file not found: {config_path}\nExpected: training.yaml with unified configuration"
            )

        # Load and validate YAML
        try:
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read {config_path}: {e}")

        if not full_config:
            raise ValueError(f"Empty or invalid training.yaml at {config_path}")

        # Validate task_specific section exists
        if "task_specific" not in full_config:
            raise ValueError(
                f"Missing 'task_specific' section in {config_path}\nFound top-level keys: {list(full_config.keys())}"
            )

        task_specific = full_config["task_specific"]
        if model_type not in task_specific:
            available_types = list(task_specific.keys())
            raise ValueError(
                f"Model type '{model_type}' not found in task_specific section of {config_path}\n"
                f"Available model types: {available_types}"
            )

        # Get shared parameters (everything except task_specific)
        shared_params = {k: v for k, v in full_config.items() if k != "task_specific"}

        # Get task-specific parameters
        task_params = task_specific[model_type]

        # Merge shared + task-specific parameters
        merged_config = {**shared_params, **task_params}

        # Validate required fields
        required_fields = ["task_name", "shuffle_before_packing", "seed_balanced_batching"]
        missing_fields = [field for field in required_fields if field not in merged_config]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {config_path} for model_type '{model_type}': {missing_fields}"
            )

        # Extract only the essential fields needed for model variant generation
        essential_config = {
            "task_name": merged_config["task_name"],
            "shuffle_before_packing": merged_config["shuffle_before_packing"],
            "seed_balanced_batching": merged_config["seed_balanced_batching"],
        }

        return essential_config

    def _find_checkpoint_directories(self, model_base_dir: Path) -> list[Path]:
        """Find all checkpoint directories within the model variant directory."""
        checkpoint_patterns = [
            "checkpoint-*",  # Standard HuggingFace format
            "step_*",  # Alternative naming
            "step-*",  # Another alternative
        ]

        found_checkpoints = []
        for pattern in checkpoint_patterns:
            found_checkpoints.extend(model_base_dir.glob(pattern))

        # Filter to only directories that look like valid checkpoints
        valid_checkpoints = []
        for checkpoint_path in found_checkpoints:
            if checkpoint_path.is_dir() and self._is_valid_checkpoint(checkpoint_path):
                valid_checkpoints.append(checkpoint_path)

        logger.info(f"Found {len(valid_checkpoints)} valid checkpoints in {model_base_dir}")
        return sorted(valid_checkpoints)

    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """Check if path contains a valid model checkpoint."""
        # Check for essential files
        config_file = checkpoint_path / "config.json"
        if not config_file.exists():
            return False

        # Check for model weights (at least one format)
        model_files = ["pytorch_model.bin", "model.safetensors", "model.pt", "pytorch_model.safetensors"]

        has_model = any((checkpoint_path / f).exists() for f in model_files)
        return has_model

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

    def _generate_model_id(self, model_variant: str, checkpoint_step: int, checkpoint_path: Path) -> str:
        """Generate unique model identifier."""
        components = [
            model_variant,
            f"step{checkpoint_step}",
        ]

        # Add path hash for uniqueness in case of conflicts
        path_hash = str(abs(hash(str(checkpoint_path))))[-6:]
        components.append(path_hash)

        return "_".join(components)

    def _print_discovery_summary(self, metadata_list: list[ModelMetadata]) -> None:
        """Print summary of discovered checkpoints."""
        if not metadata_list:
            logger.info("No checkpoints found")
            return

        steps = [meta.checkpoint_step for meta in metadata_list]
        logger.info("\nCheckpoint Discovery Summary:")
        logger.info("-" * 50)
        logger.info(f"Model variant: {self.config.model_variant}")
        logger.info(f"Model type: {self.config.model_type} (explicit)")
        logger.info(f"Using explicit L={self.config.config_L}, M={self.config.config_m}")
        logger.info(f"Task name: {metadata_list[0].task_name}")
        logger.info(f"Shuffle before packing: {metadata_list[0].shuffle_before_packing}")
        logger.info(f"Seed balanced batching: {metadata_list[0].seed_balanced_batching}")
        logger.info(f"Total checkpoints: {len(steps)}")
        logger.info(f"Step range: {min(steps)} - {max(steps)}")
        logger.info(f"Steps: {sorted(steps)}")

    def load_model_checkpoint(self, metadata: ModelMetadata, cache: bool = True) -> tuple[t.Any, t.Any]:
        """Load model and tokenizer from checkpoint."""
        if cache and metadata.model_id in self._checkpoint_cache:
            return self._checkpoint_cache[metadata.model_id]

        logger.info(f"Loading checkpoint: {metadata.model_id}")

        try:
            # Load tokenizer
            tokenizer = RHMTokenizer.from_pretrained(metadata.checkpoint_path, trust_remote_code=True)

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Load model based on task type
            if metadata.task_name == "clm":
                model = AutoModelForCausalLM.from_pretrained(
                    metadata.checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
            elif metadata.task_name == "mlm":
                model = AutoModelForMaskedLM.from_pretrained(
                    metadata.checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
            else:
                raise ValueError(f"Unsupported task type: {metadata.task_name}")

            # Move to device
            model = model.to(self.device)
            model.eval()

            # Resize embeddings if tokenizer was modified
            if len(tokenizer) != model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(tokenizer))

            if cache:
                self._checkpoint_cache[metadata.model_id] = (model, tokenizer)

            return model, tokenizer

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {metadata.checkpoint_path}: {e}")

    def validate_checkpoint_completeness(self, metadata_list: list[ModelMetadata]) -> dict[str, t.Any]:
        """Validate checkpoint discovery and provide statistics."""
        validation_results = {
            "total_checkpoints": len(metadata_list),
            "model_variant": self.config.model_variant,
            "explicit_config": {"L": self.config.config_L, "M": self.config.config_m},
            "step_coverage": {},
            "training_phases": {},
        }

        if not metadata_list:
            return validation_results

        # Analyze step coverage
        steps = [meta.checkpoint_step for meta in metadata_list]
        validation_results["step_coverage"] = {
            "min_step": min(steps),
            "max_step": max(steps),
            "step_count": len(steps),
            "steps": sorted(steps),
        }

        # Analyze training phase distribution
        from collections import defaultdict

        phase_counts = defaultdict(int)
        for meta in metadata_list:
            phase = determine_training_phase(meta.checkpoint_step, max(steps))
            phase_counts[phase] += 1

        validation_results["training_phases"] = dict(phase_counts)

        return validation_results

    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        for model, tokenizer in self._checkpoint_cache.values():
            if hasattr(model, "cpu"):
                model.cpu()
            del model, tokenizer

        self._checkpoint_cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleared checkpoint cache")
