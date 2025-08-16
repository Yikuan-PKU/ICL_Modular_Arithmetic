"""Checkpoint management and model loading utilities."""

import json
import re
import typing as t
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ICL.eval.collection.data_schema import EvaluationConfig, ModelMetadata


class CheckpointManager:
    """Manages model checkpoint discovery, loading, and metadata extraction."""

    def __init__(self, config: EvaluationConfig):
        """Initialize checkpoint manager with evaluation configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self._checkpoint_cache: dict[str, tuple[t.Any, t.Any]] = {}

    def discover_checkpoints(self) -> list[ModelMetadata]:
        """Discover all available checkpoints matching target configurations."""
        all_metadata = []

        for checkpoint_dir in self.config.checkpoint_base_dirs:
            metadata = self._scan_checkpoint_directory(checkpoint_dir)
            all_metadata.extend(metadata)

        # Filter by target configurations and diversity levels
        filtered_metadata = self._filter_by_targets(all_metadata)

        print(
            f"Discovered {len(filtered_metadata)} checkpoints across {len(self.config.checkpoint_base_dirs)} directories"
        )
        self._print_discovery_summary(filtered_metadata)

        return filtered_metadata

    def _scan_checkpoint_directory(self, checkpoint_dir: Path) -> list[ModelMetadata]:
        """Scan a single checkpoint directory for model files with metadata.json."""
        metadata_list = []

        # Look for metadata.json files in subdirectories
        metadata_files = list(checkpoint_dir.glob("*/metadata.json"))

        if not metadata_files:
            # Fallback: look for metadata.json in the root directory
            root_metadata = checkpoint_dir / "metadata.json"
            if root_metadata.exists():
                metadata_files = [root_metadata]

        for metadata_file in metadata_files:
            experiment_dir = metadata_file.parent

            try:
                # Load metadata from JSON file
                with open(metadata_file) as f:
                    metadata_json = json.load(f)

                # Find checkpoint subdirectories in this experiment directory
                checkpoint_subdirs = self._find_checkpoint_subdirectories(experiment_dir)

                if not checkpoint_subdirs:
                    warnings.warn(f"No valid checkpoints found in {experiment_dir}")
                    continue

                # Create metadata for each checkpoint in this experiment
                for checkpoint_path in checkpoint_subdirs:
                    if self._is_valid_checkpoint(checkpoint_path):
                        try:
                            metadata = self._extract_checkpoint_metadata_from_json(
                                checkpoint_path, metadata_json, metadata_file
                            )
                            if metadata:
                                metadata_list.append(metadata)
                        except Exception as e:
                            warnings.warn(f"Failed to process checkpoint {checkpoint_path}: {e}")

            except Exception as e:
                warnings.warn(f"Failed to load metadata from {metadata_file}: {e}")
                continue

        return metadata_list

    def _find_checkpoint_subdirectories(self, experiment_dir: Path) -> list[Path]:
        """Find valid checkpoint subdirectories within an experiment directory."""
        checkpoint_patterns = [
            "checkpoint-*",  # HuggingFace standard: checkpoint-1000
            "step_*",  # Custom step naming: step_1000
            "step-*",  # Alternative step naming: step-1000
            "*_steps",  # Alternative format: 1000_steps
        ]

        found_checkpoints = []
        for pattern in checkpoint_patterns:
            found_checkpoints.extend(experiment_dir.glob(pattern))

        # If no pattern matches, check if the experiment_dir itself is a checkpoint
        if not found_checkpoints and self._is_valid_checkpoint(experiment_dir):
            found_checkpoints = [experiment_dir]

        return sorted(found_checkpoints)

    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """Check if path contains a valid model checkpoint."""
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors", "model.pt"]

        # Check for config file
        if not any((checkpoint_path / f).exists() for f in required_files):
            return False

        # Check for model weights
        return any((checkpoint_path / f).exists() for f in model_files)

    def _extract_checkpoint_metadata(self, checkpoint_path: Path) -> ModelMetadata | None:
        """Extract metadata from checkpoint path and config files."""
        # First, try to find metadata.json in parent directories
        metadata_file = self._find_metadata_file(checkpoint_path)

        if metadata_file:
            try:
                with open(metadata_file) as f:
                    metadata_json = json.load(f)
                return self._extract_checkpoint_metadata_from_json(checkpoint_path, metadata_json, metadata_file)
            except Exception as e:
                warnings.warn(f"Failed to load metadata from {metadata_file}: {e}")

        # Fallback to original path-based parsing if no metadata.json found
        warnings.warn(f"No metadata.json found for {checkpoint_path}, attempting path-based parsing")
        return self._extract_checkpoint_metadata_from_path(checkpoint_path)

    def _find_metadata_file(self, checkpoint_path: Path) -> Path | None:
        """Find metadata.json file for a checkpoint by searching parent directories."""
        # Check immediate parent directory
        parent_metadata = checkpoint_path.parent / "metadata.json"
        if parent_metadata.exists():
            return parent_metadata

        # Check grandparent directory (in case of nested structure)
        grandparent_metadata = checkpoint_path.parent.parent / "metadata.json"
        if grandparent_metadata.exists():
            return grandparent_metadata

        # Check if checkpoint_path itself contains metadata.json
        direct_metadata = checkpoint_path / "metadata.json"
        if direct_metadata.exists():
            return direct_metadata

        return None

    def _extract_checkpoint_metadata_from_path(self, checkpoint_path: Path) -> ModelMetadata | None:
        """Fallback method: extract metadata from path structure (original implementation)."""
        # Try to extract config info from path structure
        config_info = self._parse_checkpoint_path(checkpoint_path)

        # Try to load config.json for additional info
        config_json_path = checkpoint_path / "config.json"
        if config_json_path.exists():
            try:
                with open(config_json_path) as f:
                    model_config = json.load(f)
                config_info.update(self._parse_model_config(model_config))
            except Exception as e:
                warnings.warn(f"Failed to parse config.json at {config_json_path}: {e}")

        # Try to extract training info from training_args.json
        training_args_path = checkpoint_path / "training_args.json"
        if training_args_path.exists():
            try:
                with open(training_args_path) as f:
                    training_args = json.load(f)
                config_info.update(self._parse_training_args(training_args))
            except Exception:
                pass  # Optional file

        # Validate required fields
        required_fields = ["config_L", "config_m", "n_train", "checkpoint_step", "model_type"]
        if not all(field in config_info for field in required_fields):
            warnings.warn(f"Missing required metadata fields for {checkpoint_path}")
            return None

        # Generate model ID
        model_id = self._generate_model_id(config_info, checkpoint_path)

        return ModelMetadata(
            model_id=model_id,
            config_L=config_info["config_L"],
            config_m=config_info["config_m"],
            n_train=config_info["n_train"],
            checkpoint_step=config_info["checkpoint_step"],
            checkpoint_path=checkpoint_path,
            model_type=config_info["model_type"],
            training_seed=config_info.get("training_seed"),
            eval_seed=config_info.get("eval_seed"),
        )

    def _parse_checkpoint_path(self, checkpoint_path: Path) -> dict[str, t.Any]:
        """Extract configuration info from checkpoint path structure."""
        path_str = str(checkpoint_path)
        config_info = {}

        # Extract L and m from path patterns like "L2_m3" or "config_L2_m3"
        config_pattern = r"(?:config_)?L(\d+)_m(\d+)"
        config_match = re.search(config_pattern, path_str)
        if config_match:
            config_info["config_L"] = int(config_match.group(1))
            config_info["config_m"] = int(config_match.group(2))

        # Extract N_train from patterns like "ntrain64" or "n_train_128"
        ntrain_pattern = r"n?_?train[_-]?(\d+)"
        ntrain_match = re.search(ntrain_pattern, path_str, re.IGNORECASE)
        if ntrain_match:
            config_info["n_train"] = int(ntrain_match.group(1))

        # Extract checkpoint step from patterns like "checkpoint-1000" or "step_1000"
        step_patterns = [
            r"checkpoint[_-](\d+)",
            r"step[_-](\d+)",
            r"(\d+)_steps",
        ]
        for pattern in step_patterns:
            step_match = re.search(pattern, path_str)
            if step_match:
                config_info["checkpoint_step"] = int(step_match.group(1))
                break

        # Infer model type from path
        if "causal" in path_str.lower() or "clm" in path_str.lower():
            config_info["model_type"] = "causal_lm"
        elif "masked" in path_str.lower() or "mlm" in path_str.lower():
            config_info["model_type"] = "mlm"

        return config_info

    def _parse_model_config(self, model_config: dict) -> dict[str, t.Any]:
        """Extract relevant info from model config.json."""
        config_info = {}

        # Try to infer model type from architecture
        if "architectures" in model_config:
            arch = model_config["architectures"][0].lower()
            if "causallm" in arch or "gpt" in arch:
                config_info["model_type"] = "causal_lm"
            elif "maskedlm" in arch or "bert" in arch:
                config_info["model_type"] = "mlm"

        return config_info

    def _parse_training_args(self, training_args: dict) -> dict[str, t.Any]:
        """Extract relevant info from training_args.json."""
        config_info = {}

        if "seed" in training_args:
            config_info["training_seed"] = training_args["seed"]

        if "eval_steps" in training_args:
            config_info["eval_steps"] = training_args["eval_steps"]

        return config_info

    def _generate_model_id(self, config_info: dict, checkpoint_path: Path) -> str:
        """Generate unique model identifier."""
        components = [
            config_info["model_type"],
            f"L{config_info['config_L']}",
            f"m{config_info['config_m']}",
            f"ntrain{config_info['n_train']}",
            f"step{config_info['checkpoint_step']}",
        ]

        # Add path hash for uniqueness
        path_hash = str(abs(hash(str(checkpoint_path))))[-6:]
        components.append(path_hash)

        return "_".join(components)

    def _filter_by_targets(self, metadata_list: list[ModelMetadata]) -> list[ModelMetadata]:
        """Filter checkpoints by target configurations and diversity levels."""
        if not self.config.target_configs:
            return metadata_list

        filtered = []
        for metadata in metadata_list:
            config = (metadata.config_L, metadata.config_m)

            # Check if config matches targets
            if config in self.config.target_configs:
                # Check if diversity level matches
                if metadata.n_train in self.config.diversity_levels:
                    # Check if model type matches
                    if metadata.model_type in self.config.model_types:
                        filtered.append(metadata)

        return filtered

    def _print_discovery_summary(self, metadata_list: list[ModelMetadata]) -> None:
        """Print summary of discovered checkpoints."""
        if not metadata_list:
            print("No checkpoints found matching criteria")
            return

        # Group by configuration
        by_config = defaultdict(list)
        for meta in metadata_list:
            key = (meta.config_L, meta.config_m, meta.n_train, meta.model_type)
            by_config[key].append(meta)

        print("\nCheckpoint Discovery Summary:")
        print("-" * 50)
        for (L, m, n_train, model_type), metas in sorted(by_config.items()):
            steps = [meta.checkpoint_step for meta in metas]
            print(
                f"L={L}, m={m}, N_train={n_train}, {model_type}: {len(steps)} checkpoints (steps: {min(steps)}-{max(steps)})"
            )

    def load_model_checkpoint(self, metadata: ModelMetadata, cache: bool = True) -> tuple[t.Any, t.Any]:
        """Load model and tokenizer from checkpoint."""
        if cache and metadata.model_id in self._checkpoint_cache:
            return self._checkpoint_cache[metadata.model_id]

        print(f"Loading checkpoint: {metadata.model_id}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(metadata.checkpoint_path, trust_remote_code=True)

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Load model based on type
            if metadata.model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    metadata.checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
            elif metadata.model_type == "mlm":
                model = AutoModelForMaskedLM.from_pretrained(
                    metadata.checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
            else:
                raise ValueError(f"Unsupported model type: {metadata.model_type}")

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
        """Validate that we have complete checkpoint coverage."""
        validation_results = {
            "total_checkpoints": len(metadata_list),
            "configs_found": set(),
            "missing_configs": [],
            "diversity_coverage": defaultdict(set),
            "model_type_coverage": defaultdict(set),
        }

        # Analyze what we have
        for meta in metadata_list:
            config = (meta.config_L, meta.config_m)
            validation_results["configs_found"].add(config)
            validation_results["diversity_coverage"][config].add(meta.n_train)
            validation_results["model_type_coverage"][config].add(meta.model_type)

        # Check for missing target configs
        for target_config in self.config.target_configs:
            if target_config not in validation_results["configs_found"]:
                validation_results["missing_configs"].append(target_config)

        # Check diversity coverage
        incomplete_diversity = []
        for config in validation_results["configs_found"]:
            found_diversity = validation_results["diversity_coverage"][config]
            expected_diversity = set(self.config.diversity_levels)
            if not expected_diversity.issubset(found_diversity):
                missing = expected_diversity - found_diversity
                incomplete_diversity.append((config, list(missing)))

        validation_results["incomplete_diversity"] = incomplete_diversity

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

    def _extract_checkpoint_metadata_from_json(
        self, checkpoint_path: Path, metadata_json: dict, metadata_file: Path
    ) -> ModelMetadata | None:
        """Extract metadata from JSON file for a specific checkpoint."""
        # Validate required fields in metadata JSON
        required_fields = ["config_L", "config_m", "n_train", "model_type"]
        missing_fields = [field for field in required_fields if field not in metadata_json]

        if missing_fields:
            warnings.warn(f"Missing required fields in {metadata_file}: {missing_fields}")
            return None

        # Extract checkpoint step from path
        checkpoint_step = self._extract_checkpoint_step_from_path(checkpoint_path)
        if checkpoint_step is None:
            warnings.warn(f"Could not extract checkpoint step from path: {checkpoint_path}")
            return None

        # Validate metadata values
        try:
            config_L = int(metadata_json["config_L"])
            config_m = int(metadata_json["config_m"])
            n_train = int(metadata_json["n_train"])
            model_type = str(metadata_json["model_type"])

            if model_type not in ["causal_lm", "mlm"]:
                warnings.warn(f"Invalid model_type '{model_type}' in {metadata_file}. Expected 'causal_lm' or 'mlm'")
                return None

        except (ValueError, TypeError) as e:
            warnings.warn(f"Invalid metadata values in {metadata_file}: {e}")
            return None

        # Generate model ID
        model_id = self._generate_model_id_from_metadata(
            config_L, config_m, n_train, checkpoint_step, model_type, checkpoint_path
        )

        # Extract optional fields
        training_seed = metadata_json.get("training_seed")
        eval_seed = metadata_json.get("eval_seed")

        return ModelMetadata(
            model_id=model_id,
            config_L=config_L,
            config_m=config_m,
            n_train=n_train,
            checkpoint_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            training_seed=training_seed,
            eval_seed=eval_seed,
        )

    def _extract_checkpoint_step_from_path(self, checkpoint_path: Path) -> int | None:
        """Extract checkpoint step number from checkpoint path."""
        path_str = str(checkpoint_path.name)

        # Try different patterns for checkpoint step
        step_patterns = [
            r"checkpoint[_-](\d+)",
            r"step[_-](\d+)",
            r"(\d+)[_-]?steps?",
        ]

        for pattern in step_patterns:
            match = re.search(pattern, path_str)
            if match:
                return int(match.group(1))

        # If no step found in directory name, try to infer from parent structure
        # or use a default value
        return 0  # Default step if cannot be determined

    def _generate_model_id_from_metadata(
        self, config_L: int, config_m: int, n_train: int, checkpoint_step: int, model_type: str, checkpoint_path: Path
    ) -> str:
        """Generate unique model identifier from metadata."""
        components = [
            model_type,
            f"L{config_L}",
            f"m{config_m}",
            f"ntrain{n_train}",
            f"step{checkpoint_step}",
        ]

        # Add path hash for uniqueness in case of conflicts
        path_hash = str(abs(hash(str(checkpoint_path))))[-6:]
        components.append(path_hash)

        return "_".join(components)
