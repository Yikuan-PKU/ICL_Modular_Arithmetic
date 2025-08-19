import pickle
from pathlib import Path
from typing import Any

from datasets import Dataset

from ICL.datasets.RHM import RandomHierarchyModel


def generate_raw_rhm_dataset(
    config_list: list[tuple[int, int]],
    samples_per_config: int = 1000,
    output_dir: str = "./raw_rhm_data",
    vocab_size: int = 32,
    num_classes: int = 10,
    tuple_size: int = 2,
    seed_sample: int = 42,
    save_intermediate: bool = True,
) -> tuple[Dataset, dict[str, Any]]:
    """Generate raw RHM dataset for multiple hierarchical configurations.

    Args:
        config_list: List of (L, m) tuples where L=hierarchy depth, m=multiplicity
        samples_per_config: Number of samples to generate per configuration
        output_dir: Directory to save the raw dataset
        vocab_size: Vocabulary size for RHM (default: 32)
        num_classes: Number of classes for RHM (default: 10)
        tuple_size: Size of low-level representations (default: 2)
        seed_sample: Seed for sample generation (default: 42)
        save_intermediate: Whether to save intermediate results per config

    Returns:
        dataset: HuggingFace Dataset containing all raw sequences
        metadata: Dictionary with comprehensive metadata

    """
    print("=" * 60)
    print("GENERATING RAW RHM DATASET")
    print("=" * 60)

    # Convert to Path object
    output_path = Path(output_dir)

    # Initialize storage for all sequences and metadata
    all_sequences = []
    all_task_ids = []
    all_config_L = []
    all_config_m = []
    all_sequence_lengths = []
    all_rules = {}

    # Statistics tracking
    total_sequences = 0
    config_stats = {}

    print(f"Configurations to generate: {len(config_list)}")
    print(f"Samples per configuration: {samples_per_config}")
    print(f"Total target sequences: {len(config_list) * samples_per_config}")
    print()

    # Generate data for each configuration
    for task_id, (L, m) in enumerate(config_list):
        print(f"Generating Task {task_id}: L={L} (depth), m={m} (multiplicity)")
        print("-" * 50)

        try:
            # Create RHM instance for this configuration
            rhm = RandomHierarchyModel(
                num_features=vocab_size,  # vocabulary size (0-31, +1 shift makes it 1-32)
                num_classes=num_classes,  # number of classes
                num_synonyms=m,  # multiplicity parameter
                tuple_size=tuple_size,  # size of low-level representations
                num_layers=L,  # hierarchy depth
                seed_rules=task_id,  # unique rules per task
                seed_sample=seed_sample,  # consistent sampling across tasks
                train_size=samples_per_config,
                test_size=0,
                input_format="long",  # integer sequences (1-based indexing)
                replacement=True,  # allow sampling with replacement
            )

            # Extract generated data
            sequences = rhm.features  # Shape: [samples_per_config, variable_length]
            labels = rhm.labels  # Shape: [samples_per_config] (not used in LM)
            rules = rhm.rules  # Production rules dictionary

            # Convert tensors to lists for HuggingFace compatibility
            if hasattr(sequences, "tolist"):
                sequences_list = sequences.tolist()
            else:
                sequences_list = [list(seq) for seq in sequences]

            # Calculate statistics for this configuration
            seq_lengths = [len(seq) for seq in sequences_list]
            config_stats[task_id] = {
                "L": L,
                "m": m,
                "num_sequences": len(sequences_list),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "avg_length": sum(seq_lengths) / len(seq_lengths),
                "total_tokens": sum(seq_lengths),
            }

            # Store rules for this configuration
            all_rules[task_id] = {
                "L": L,
                "m": m,
                "rules_dict": rules,
                "vocab_range": f"1-{vocab_size}",  # RHM uses 1-based indexing
                "num_sequences": len(sequences_list),
            }

            # Add to master dataset
            all_sequences.extend(sequences_list)
            all_task_ids.extend([task_id] * len(sequences_list))
            all_config_L.extend([L] * len(sequences_list))
            all_config_m.extend([m] * len(sequences_list))
            all_sequence_lengths.extend(seq_lengths)

            total_sequences += len(sequences_list)

            # Print statistics for this configuration
            print(f"  ✓ Generated {len(sequences_list)} sequences")
            print(f"  ✓ Length range: {min(seq_lengths)}-{max(seq_lengths)} tokens")
            print(f"  ✓ Average length: {sum(seq_lengths) / len(seq_lengths):.1f} tokens")
            print(f"  ✓ Total tokens: {sum(seq_lengths):,}")

            # Save intermediate results if requested
            if save_intermediate:
                config_dir = output_path / "intermediate" / f"task_{task_id}_L{L}_m{m}"
                config_dir.mkdir(parents=True, exist_ok=True)

                # Save sequences and metadata for this config
                config_data = {
                    "sequences": sequences_list,
                    "task_id": task_id,
                    "L": L,
                    "m": m,
                    "rules": rules,
                    "stats": config_stats[task_id],
                }

                with (config_dir / "config_data.pkl").open("wb") as f:
                    pickle.dump(config_data, f)

                print(f"  ✓ Saved intermediate results to {config_dir}")

            print()

        except Exception as e:
            print(f"  ✗ Error generating task {task_id} (L={L}, m={m}): {e}")
            print("  ✗ Skipping this configuration...")
            print()
            continue

    # Create comprehensive metadata
    metadata = {
        "generation_params": {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "tuple_size": tuple_size,
            "seed_sample": seed_sample,
            "samples_per_config": samples_per_config,
        },
        "configurations": [{"task_id": i, "L": L, "m": m} for i, (L, m) in enumerate(config_list)],
        "config_stats": config_stats,
        "rules": all_rules,
        "dataset_stats": {
            "total_sequences": total_sequences,
            "total_configs": len(config_list),
            "successful_configs": len(config_stats),
            "min_seq_length": min(all_sequence_lengths) if all_sequence_lengths else 0,
            "max_seq_length": max(all_sequence_lengths) if all_sequence_lengths else 0,
            "avg_seq_length": sum(all_sequence_lengths) / len(all_sequence_lengths) if all_sequence_lengths else 0,
            "total_tokens": sum(all_sequence_lengths),
            "vocab_range": f"1-{vocab_size} (0 reserved for special tokens)",
        },
    }

    # Create HuggingFace Dataset
    print("Creating HuggingFace Dataset...")
    dataset_dict = {
        "input_ids": all_sequences,  # Raw integer sequences
        "task_id": all_task_ids,  # Which configuration generated this sequence
        "config_L": all_config_L,  # Hierarchy depth for this sequence
        "config_m": all_config_m,  # Multiplicity for this sequence
        "length": all_sequence_lengths,  # Length of this sequence
    }

    dataset = Dataset.from_dict(dataset_dict)

    # Save complete dataset and metadata
    output_path.mkdir(parents=True, exist_ok=True)

    # Save HuggingFace dataset
    dataset.save_to_disk(str(output_path / "dataset"))
    print(f"✓ Saved HuggingFace dataset to {output_path / 'dataset'}")

    # Save metadata
    with (output_path / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {output_path / 'metadata.pkl'}")

    # Save human-readable summary
    with (output_path / "dataset_summary.txt").open("w") as f:
        f.write("RHM DATASET GENERATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total sequences: {total_sequences:,}\n")
        f.write(f"Total tokens: {sum(all_sequence_lengths):,}\n")
        f.write(f"Vocabulary: 1-{vocab_size} (0 reserved)\n")
        f.write(f"Sequence length range: {min(all_sequence_lengths)}-{max(all_sequence_lengths)}\n")
        f.write(f"Average sequence length: {sum(all_sequence_lengths) / len(all_sequence_lengths):.1f}\n\n")

        f.write("CONFIGURATION DETAILS:\n")
        f.write("-" * 30 + "\n")
        for task_id, stats in config_stats.items():
            f.write(f"Task {task_id}: L={stats['L']}, m={stats['m']}\n")
            f.write(f"  Sequences: {stats['num_sequences']:,}\n")
            f.write(f"  Length: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.1f})\n")
            f.write(f"  Tokens: {stats['total_tokens']:,}\n\n")

    print(f"✓ Saved summary to {output_path / 'dataset_summary.txt'}")

    # Final summary
    print("\n" + "=" * 60)
    print("RAW DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total sequences generated: {total_sequences:,}")
    print(f"Total tokens: {sum(all_sequence_lengths):,}")
    print(f"Successful configurations: {len(config_stats)}/{len(config_list)}")
    print(f"Dataset saved to: {output_path}")
    print("=" * 60)

    return dataset, metadata


# Example usage function
def generate_example_dataset():
    """Generate an example RHM dataset with multiple configurations"""
    # Define hierarchical configurations to test
    config_list = [
        (2, 2),  # Shallow, low multiplicity
        (3, 2),  # Medium depth, low multiplicity
        (2, 4),  # Shallow, high multiplicity
        (4, 2),  # Deep, low multiplicity
        (3, 3),  # Medium depth, medium multiplicity
    ]

    # Generate the dataset
    dataset, metadata = generate_raw_rhm_dataset(
        config_list=config_list,
        samples_per_config=1000,
        output_dir="./raw_rhm_data",
        vocab_size=32,
        num_classes=10,
        save_intermediate=True,
    )

    return dataset, metadata


if __name__ == "__main__":
    # Generate example dataset
    dataset, metadata = generate_example_dataset()

    # Quick inspection
    print("\nDataset inspection:")
    print(f"Number of sequences: {len(dataset)}")
    print(f"First sequence: {dataset[0]['input_ids'][:20]}...")  # Show first 20 tokens
    print(f"First sequence length: {dataset[0]['length']}")
    print(f"First sequence config: L={dataset[0]['config_L']}, m={dataset[0]['config_m']}")
