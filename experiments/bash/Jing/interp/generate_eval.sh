#!/bin/bash
#SBATCH --job-name=RHM_uniform
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/datasets/uniform.log

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/datasets"

# Primary experiment parameters
DATASET_TYPE="uniform"      # Options: uniform, zipf
RNG_SEED=42                # Seed for generating random seeds (for reproducibility)
NUM_SEEDS=10                # Number of random seeds to generate
L=4
M=2

# Pipeline control flags
PIPELINE_ARGS="--verbose --overwrite"  # Options: --verbose, --overwrite, --resume

# Construct command arguments
SHARED_ARGS="--dataset-type $DATASET_TYPE --seed $RNG_SEED --num-seeds $NUM_SEEDS --L $L --M $M"

# =============================================================================
# GENERATION STEPS
# =============================================================================

# Step 1: Validate configuration (optional but recommended)
echo "Step 1: Validating configuration..."
python $SCRIPT_ROOT/generate_eval.py $SHARED_ARGS --validate-only

echo "Step 2: Running generation..."
python $SCRIPT_ROOT/generate_eval.py $SHARED_ARGS $PIPELINE_ARGS
