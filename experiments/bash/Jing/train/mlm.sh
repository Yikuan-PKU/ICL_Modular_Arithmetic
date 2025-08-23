#!/bin/bash
#SBATCH --job-name=train_mlm
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/train/uniform_allmix_576_42_clm.log

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/train"

# Define experiment parameters using the new simplified structure
DATASET_TYPE="uniform"
MODEL_TYPE="mlm"
NUM_SEEDS=10
SEED=42
L=4
M=2

# Construct shared arguments (updated to match new argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --num-seeds $NUM_SEEDS --seed $SEED --L $L --M $M"

# Optional pipeline controls
PIPELINE_ARGS="--verbose"

# Step 1: Validate configuration (optional)
echo "Step 1: Validating training configuration..."
python $SCRIPT_ROOT/train.py $SHARED_ARGS --dry-run


# Step 2: Train model
echo "Step 2: Training model..."
python $SCRIPT_ROOT/train.py $SHARED_ARGS $PIPELINE_ARGS
