#!/bin/bash
#SBATCH -J eval_clm
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -o RHM_%j.out
#SBATCH -e RHM_%j.err
#SBATCH --no-requeue
#SBATCH -A qi_g1
#SBATCH --qos=qig4c
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=9

source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate


# Define experiment parameters using the new simplified structure
DATASET_TYPE="uniform"
MODEL_TYPE="clm"
NUM_SEEDS=2000
SEED=42
L=3
M=3

# Construct shared arguments (updated to match new argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --num-seeds $NUM_SEEDS --seed $SEED --L $L --M $M"
# Optional pipeline controls
MODE_ARGS="--batch-mode --verbose --overwrite"

ALL_ARGS="$SHARED_ARGS $MODE_ARGS"

# =============================================================================
# Evaluating model
# =============================================================================
echo "============== Evaluating model =============="

SCRIPT_ROOT="$HOME/lustre1/ykzhang/ICL_RHM/ICL_Modular_Arithmetic/src/scripts/eval"
# Step 1: Discover available combinations using explicit L,M,MODEL_TYPE
echo "Step 1: Discovering combinations using explicit L=$L, M=$M, MODEL_TYPE=$MODEL_TYPE..."
echo "Execution parameters will be loaded from collection.yaml"
echo "Device will be auto-detected (GPU preferred)"
python $SCRIPT_ROOT/collection.py $ALL_ARGS --list-combinations

# Step 2: Validate configuration with explicit L,M,MODEL_TYPE
echo ""
echo "Step 2: Validating configuration with explicit L=$L, M=$M, MODEL_TYPE=$MODEL_TYPE..."
python $SCRIPT_ROOT/collection.py $ALL_ARGS --validate-only

# Step 3: Run collection pipeline with explicit L,M,MODEL_TYPE
echo ""
echo "Step 3: Running collection pipeline with explicit L=$L, M=$M, MODEL_TYPE=$MODEL_TYPE..."
echo "Execution parameters: batch_size, max_sequences, capture_attention loaded from YAML"
python $SCRIPT_ROOT/collection.py $ALL_ARGS

echo ""
echo "Collection completed. Results directory pattern:"
echo "  results/${DATASET_TYPE}_${NUM_SEEDS}_L${L}_M${M}/"