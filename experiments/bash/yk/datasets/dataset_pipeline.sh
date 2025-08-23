#!/bin/bash
#SBATCH -J RHM_uniform
#SBATCH -p gpu_l40
#SBATCH -N 1
#SBATCH -o RHM_%j.out
#SBATCH -e RHM_%j.err
#SBATCH --no-requeue
#SBATCH -A qi_g1
#SBATCH --qos=qil40
#SBATCH --gres=gpu:2
#SBATCH --overcommit
#SBATCH --mincpus=9



source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate


SCRIPT_ROOT="$HOME/lustre1/ykzhang/ICL_RHM/ICL_Modular_Arithmetic/src/scripts/datasets"


# Primary experiment parameters
DATASET_TYPE="uniform"      # Options: uniform, zipf
RNG_SEED=42                # Seed for generating random seeds (for reproducibility)
NUM_SEEDS=2000                # Number of random seeds to generate
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
