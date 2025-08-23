#!/bin/bash
#SBATCH --job-name=gen_500_L3_M3
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/datasets/uniform_500_L3_M3.log

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/datasets"

# Primary experiment parameters
DATASET_TYPE="uniform"      # Options: uniform, zipf
RNG_SEED=42                # Seed for generating random seeds (for reproducibility)
NUM_SEEDS=500                # Number of random seeds to generate
L=3
M=3

# Pipeline control flags
PIPELINE_ARGS="--verbose --overwrite"  # Options: --verbose, --overwrite, --resume

# Construct command arguments
SHARED_ARGS="--dataset-type $DATASET_TYPE --seed $RNG_SEED --num-seeds $NUM_SEEDS --L $L --M $M"

# =============================================================================
# GENERATING RAW DATASET
# =============================================================================
echo "============== Generating RAW DATA =============="
# Step 1: Validate configuration (optional but recommended)
echo "Step 1: Validating configuration..."
python $SCRIPT_ROOT/generate_raw.py $SHARED_ARGS --validate-only

echo "Step 2: Running generation..."
python $SCRIPT_ROOT/generate_raw.py $SHARED_ARGS $PIPELINE_ARGS



# =============================================================================
# SPLIT STEPS
# =============================================================================

echo "============== SPLITTING DATA =============="
# Step 1: Validate configuration (optional but recommended)
echo "Step 1: Validating configuration..."
python $SCRIPT_ROOT/create_split.py $SHARED_ARGS --validate-only

echo "Step 2: Running splitting..."
python $SCRIPT_ROOT/create_split.py $SHARED_ARGS $PIPELINE_ARGS



# =============================================================================
# GENERATING EVAL TASK
# =============================================================================

echo "============== GENERATING EVAL TASKS =============="
# Step 1: Validate configuration (optional but recommended)
echo "Step 1: Validating configuration..."
python $SCRIPT_ROOT/generate_eval.py $SHARED_ARGS --validate-only

echo "Step 2: Running splitting..."
python $SCRIPT_ROOT/generate_eval.py $SHARED_ARGS $PIPELINE_ARGS
