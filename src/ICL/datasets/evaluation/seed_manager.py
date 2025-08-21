import random
import typing as t
from dataclasses import dataclass, field

from ICL.datasets.evaluation.eval_config import ICLEvalConfig


@dataclass
class SeedAllocation:
    """Tracks seed allocation for different evaluation types."""

    # Training seeds (from existing train/validation splits)
    train_seeds: list[int] = field(default_factory=list)

    # OOD seeds (newly generated)
    ood_same_rule_seeds: list[int] = field(default_factory=list)
    ood_transfer_seeds: list[int] = field(default_factory=list)

    # Seed ranges for tracking
    train_seed_range: tuple[int, int] | None = None
    ood_same_rule_range: tuple[int, int] | None = None
    ood_transfer_range: tuple[int, int] | None = None

    def get_all_seeds(self) -> list[int]:
        """Get all allocated seeds."""
        all_seeds = []
        all_seeds.extend(self.train_seeds)
        all_seeds.extend(self.ood_same_rule_seeds)
        all_seeds.extend(self.ood_transfer_seeds)
        return sorted(set(all_seeds))

    def check_seed_overlap(self) -> list[int]:
        """Check for overlapping seeds between different types."""
        all_seeds = self.get_all_seeds()
        if len(all_seeds) != len(set(all_seeds)):
            # Find duplicates
            seen = set()
            duplicates = []
            for seed in all_seeds:
                if seed in seen:
                    duplicates.append(seed)
                seen.add(seed)
            return duplicates
        return []


class EvalSeedManager:
    """Manages seed allocation for ICL evaluation dataset generation."""

    def __init__(self, config: ICLEvalConfig, train_seeds: list[int]):
        """Initialize seed manager.

        Args:
            config: ICL evaluation configuration
            train_seeds: Seeds used in training dataset generation

        """
        self.config = config
        self.train_seeds = sorted(train_seeds)
        self.allocation = SeedAllocation(train_seeds=self.train_seeds.copy())

        # Set random seed for reproducible OOD seed generation
        random.seed(config.base_seed)

        self._generate_ood_seeds()
        self._validate_allocation()

    def _generate_ood_seeds(self) -> None:
        """Generate OOD seeds ensuring no overlap with training seeds."""
        # Generate OOD same rule seeds
        if self.config.ood_same_rule.enable:
            ood_same_seeds = self._generate_seed_range(
                base_seed=self.config.ood_same_rule.ood_seed_offset,
                count=self.config.ood_same_rule.num_ood_seeds,
                avoid_seeds=self.train_seeds,
            )
            self.allocation.ood_same_rule_seeds = ood_same_seeds
            self.allocation.ood_same_rule_range = (min(ood_same_seeds), max(ood_same_seeds)) if ood_same_seeds else None

        # Generate OOD transfer seeds
        if self.config.ood_transfer.enable:
            # Avoid both training seeds and OOD same rule seeds
            avoid_seeds = self.train_seeds + self.allocation.ood_same_rule_seeds

            ood_transfer_seeds = self._generate_seed_range(
                base_seed=self.config.ood_transfer.transfer_seed_offset,
                count=self.config.ood_transfer.num_transfer_seeds,
                avoid_seeds=avoid_seeds,
            )
            self.allocation.ood_transfer_seeds = ood_transfer_seeds
            self.allocation.ood_transfer_range = (
                (min(ood_transfer_seeds), max(ood_transfer_seeds)) if ood_transfer_seeds else None
            )

        # Set training seed range
        if self.train_seeds:
            self.allocation.train_seed_range = (min(self.train_seeds), max(self.train_seeds))

    def _generate_seed_range(self, base_seed: int, count: int, avoid_seeds: list[int]) -> list[int]:
        """Generate a range of seeds avoiding conflicts."""
        if count <= 0:
            return []

        avoid_set = set(avoid_seeds)
        generated_seeds = []

        # Use deterministic generation for reproducibility
        current_seed = base_seed
        attempts = 0
        max_attempts = count * 100  # Safety limit

        while len(generated_seeds) < count and attempts < max_attempts:
            if current_seed not in avoid_set:
                generated_seeds.append(current_seed)
                avoid_set.add(current_seed)  # Avoid duplicates in this generation

            current_seed += 1
            attempts += 1

        if len(generated_seeds) < count:
            raise RuntimeError(
                f"Failed to generate {count} unique seeds starting from {base_seed}. "
                f"Only generated {len(generated_seeds)} after {max_attempts} attempts."
            )

        return sorted(generated_seeds)

    def _validate_allocation(self) -> None:
        """Validate seed allocation for conflicts."""
        overlaps = self.allocation.check_seed_overlap()
        if overlaps:
            raise ValueError(f"Seed overlap detected: {overlaps}")

        # Validate minimum requirements
        if self.config.memorization.enable and not self.train_seeds:
            raise ValueError("Memorization enabled but no training seeds available")

        if self.config.id_generalization.enable and not self.train_seeds:
            raise ValueError("ID generalization enabled but no training seeds available")

        if (
            self.config.ood_same_rule.enable
            and len(self.allocation.ood_same_rule_seeds) < self.config.ood_same_rule.num_ood_seeds
        ):
            raise ValueError(
                f"OOD same rule: requested {self.config.ood_same_rule.num_ood_seeds} seeds, "
                f"generated {len(self.allocation.ood_same_rule_seeds)}"
            )

        if (
            self.config.ood_transfer.enable
            and len(self.allocation.ood_transfer_seeds) < self.config.ood_transfer.num_transfer_seeds
        ):
            raise ValueError(
                f"OOD transfer: requested {self.config.ood_transfer.num_transfer_seeds} seeds, "
                f"generated {len(self.allocation.ood_transfer_seeds)}"
            )

    def get_seeds_for_type(self, eval_type: str) -> list[int]:
        """Get seeds for specific evaluation type."""
        if eval_type == "memorization" or eval_type == "id_generalization":
            return self.allocation.train_seeds
        if eval_type == "ood_same_rule":
            return self.allocation.ood_same_rule_seeds
        if eval_type == "ood_transfer":
            return self.allocation.ood_transfer_seeds
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    def get_allocation_summary(self) -> dict[str, t.Any]:
        """Get summary of seed allocation."""
        return {
            "train_seeds": {
                "seeds": self.allocation.train_seeds,
                "count": len(self.allocation.train_seeds),
                "range": self.allocation.train_seed_range,
            },
            "ood_same_rule_seeds": {
                "seeds": self.allocation.ood_same_rule_seeds,
                "count": len(self.allocation.ood_same_rule_seeds),
                "range": self.allocation.ood_same_rule_range,
            },
            "ood_transfer_seeds": {
                "seeds": self.allocation.ood_transfer_seeds,
                "count": len(self.allocation.ood_transfer_seeds),
                "range": self.allocation.ood_transfer_range,
            },
            "total_unique_seeds": len(self.allocation.get_all_seeds()),
            "overlaps": self.allocation.check_seed_overlap(),
        }

    def save_allocation(self, output_path) -> None:
        """Save seed allocation to JSON file."""
        import json
        from datetime import datetime
        from pathlib import Path

        output_path = Path(output_path)

        allocation_data = {
            "seed_allocation": self.get_allocation_summary(),
            "config_summary": {
                "base_seed": self.config.base_seed,
                "enabled_types": self.config.get_enabled_types(),
                "ood_same_rule_offset": self.config.ood_same_rule.ood_seed_offset,
                "ood_transfer_offset": self.config.ood_transfer.transfer_seed_offset,
            },
            "created_at": datetime.now().isoformat(),
        }

        with output_path.open("w") as f:
            json.dump(allocation_data, f, indent=2)


def load_train_seeds_from_split(split_info_path) -> list[int]:
    """Load training seeds from split_info.json file."""
    import json
    from pathlib import Path

    split_info_path = Path(split_info_path)

    if not split_info_path.exists():
        raise FileNotFoundError(f"Split info not found: {split_info_path}")

    with split_info_path.open("r") as f:
        split_data = json.load(f)

    # Extract seeds from per_seed_splits
    if "per_seed_splits" in split_data:
        return sorted([int(seed) for seed in split_data["per_seed_splits"].keys()])
    raise ValueError(f"No per_seed_splits found in {split_info_path}")


def discover_train_seeds_from_directory(train_dir) -> list[int]:
    """Discover training seeds from train directory structure."""
    import re
    from pathlib import Path

    train_dir = Path(train_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Look for seed_* directories
    seed_pattern = re.compile(r"seed_(\d+)")
    seeds = []

    for item in train_dir.iterdir():
        if item.is_dir():
            match = seed_pattern.match(item.name)
            if match:
                seeds.append(int(match.group(1)))

    if not seeds:
        raise ValueError(f"No seed directories found in {train_dir}")

    return sorted(seeds)
