# Dataset Generation for Hierarchical Rule Learning

This module generates Random Hierarchy Model (RHM) datasets for studying in-context learning (ICL) emergence across different rule configurations and probability distributions.

## 🏗️ Code Design Overview

### Core Architecture

The dataset generation follows a **shared experiment identification** pattern with **YAML-driven configuration**:

- **Shared Arguments**: All scripts use identical experiment identifiers (`--dataset-type`, `--model-type`, etc.)
- **Auto-Generated Paths**: No manual directory management - all paths derived from experiment parameters
- **YAML Configuration**: Scientific parameters separated from infrastructure code
- **Modular Design**: Each pipeline stage (generate → train → evaluate) runs independently

### Key Components

```
shared_experiment.py    # ExperimentConfig class & path generation
shared_args.py         # Minimal shared argument parser  
generate_dataset.py    # Main dataset generation script
config/                # YAML configuration files per experiment
datasets/              # Generated datasets (auto-named)
```

### Experiment Naming Convention

**Format**: `{dataset_type}_{model_type}_{mixture_type}_{total_rules}_{seed}`

**Examples**:
- `uniform_clm_allmix_576_42` - Uniform distribution, causal LM, mixed complexity
- `zipf_mlm_depthmix_L2_288_1337` - Zipf distribution, masked LM, depth-focused mixing

## ⚙️ Configuration System

### YAML Structure
Each experiment requires a configuration file at:
```
config/{experiment_name}/dataset.yaml
```

### Key Parameters

```yaml
# RHM model parameters
rhm_params:
  vocab_size: 32              # Vocabulary size (1-32, 0 reserved)
  num_classes: 10             # Number of output classes
  tuple_size: 2               # Size of low-level representations
  samples_per_config: 1000    # Samples per (L,m) configuration

# Hierarchical configurations to generate
configurations:
  - L: 2    # L = depth, m = multiplicity
    m: 2
  - L: 3
    m: 2

# Probability distribution for rule sampling
distribution:
  type: "uniform"    # "uniform" or "zipf"
  zipf_alpha: 1.0   # Zipf parameter (if type="zipf")
```

### Distribution Types

- **Uniform** (`--dataset-type uniform`): All synonymic rules equally likely
- **Zipf** (`--dataset-type zipf`): Power-law distribution - first rules much more probable

## 🚀 Usage Examples

### Basic Generation

```bash
# Generate uniform distribution dataset
python generate_dataset.py \
    --dataset-type uniform \
    --model-type clm \
    --mixture-type allmix \
    --total-rules 576 \
    --seed 42

# Generate Zipf distribution dataset  
python generate_dataset.py \
    --dataset-type zipf \
    --model-type clm \
    --mixture-type allmix \
    --total-rules 576 \
    --seed 42
```

### Pipeline Controls

```bash
# Validate configuration without generating
python generate_dataset.py [args] --validate-only

# Skip if dataset already exists
python generate_dataset.py [args] --resume

# Overwrite existing dataset
python generate_dataset.py [args] --overwrite

# Verbose output
python generate_dataset.py [args] --verbose
```

## 📜 Sample Bash Scripts

### Single Experiment

```bash
#!/bin/bash
# generate_single_experiment.sh

set -e  # Exit on error

# Define experiment parameters
DATASET_TYPE="uniform"
MODEL_TYPE="clm" 
MIXTURE_TYPE="allmix"
TOTAL_RULES=576
SEED=42

SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --mixture-type $MIXTURE_TYPE --total-rules $TOTAL_RULES --seed $SEED"

echo "Generating dataset: ${DATASET_TYPE}_${MODEL_TYPE}_${MIXTURE_TYPE}_${TOTAL_RULES}_${SEED}"

# Step 1: Validate configuration
python generate_dataset.py $SHARED_ARGS --validate-only
echo "✅ Configuration validated"

# Step 2: Generate dataset
python generate_dataset.py $SHARED_ARGS --verbose
echo "✅ Dataset generation complete"
```

## 📁 Output Structure

Generated datasets are saved to:
```
datasets/{experiment_name}/
├── dataset/              # HuggingFace dataset files
├── metadata.pkl         # Complete generation metadata  
└── dataset_summary.txt  # Human-readable summary
```

## 📦 Dependencies

- `torch` - PyTorch for RHM implementation
- `datasets` - HuggingFace datasets for data storage
- `numpy` - Zipf distribution generation
- `pyyaml` - YAML configuration parsing