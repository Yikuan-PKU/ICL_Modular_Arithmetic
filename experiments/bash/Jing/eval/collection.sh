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

# Define experiment parameters (same as training script style)
DATASET_TYPE="uniform"
NUM_SEEDS=10
SEED=42

# Construct shared arguments (matches training argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --num-seeds $NUM_SEEDS --seed $SEED"

# Optional pipeline controls
PIPELINE_ARGS="--verbose"

# Step 1: Discover available combinations (shows auto-discovery)
echo "Step 1: Auto-discovering combinations from model configs..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --list-combinations


# Step 2: Validate configuration
echo "Step 2: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode --validate-only

# Step 3: Run script
echo "Step 2: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode $PIPELINE_ARGS --overwrite
