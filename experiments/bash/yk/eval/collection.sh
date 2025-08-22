#!/bin/bash
#SBATCH -J train_clm
#SBATCH -p gpu_l40
#SBATCH -N 1
#SBATCH -o RHM_%j.out
#SBATCH -e RHM_%j.err
#SBATCH --no-requeue
#SBATCH -A qi_g1
#SBATCH --qos=qil40
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=9

source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate


SCRIPT_ROOT="$HOME/lustre1/ykzhang/ICL_RHM/ICL_Modular_Arithmetic/src/scripts/eval"

# Define experiment parameters (same as training script style)
DATASET_TYPE="uniform"
NUM_SEEDS=10
SEED=42

# Construct shared arguments (matches training argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --num-seeds $NUM_SEEDS --seed $SEED"

# Optional pipeline controls
PIPELINE_ARGS="--verbose --overwrite"

# Step 1: Discover available combinations (shows auto-discovery)
echo "Step 1: Auto-discovering combinations from model configs..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --list-combinations


# Step 2: Validate configuration
echo "Step 2: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode --validate-only

# Step 3: Run script
echo "Step 2: Validating auto-discovered configuration..."
python $SCRIPT_ROOT/collection.py $SHARED_ARGS --batch-mode $PIPELINE_ARGS
