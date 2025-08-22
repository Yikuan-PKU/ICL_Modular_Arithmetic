#!/bin/bash
#SBATCH --job-name=icl_collection
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/eval/collection.log

# Script root directory
SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/eval"

# Define experiment parameters using the new simplified structure
DATASET_TYPE="uniform"
MODEL_TYPE="clm"
NUM_SEEDS=10
SEED=42
L=4
M=2

# Construct shared arguments (updated to match new argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --num-seeds $NUM_SEEDS --seed $SEED --L $L --M $M"

# Optional pipeline controls
PIPELINE_ARGS="--verbose"

# Step 1: Validate configuration
echo "Step 1: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode --validate-only

# Step 2: Run script
echo "Step 2: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode $PIPELINE_ARGS --overwrite
