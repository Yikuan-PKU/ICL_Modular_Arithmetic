#!/bin/bash
#SBATCH --job-name=clm500_L3_M3
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/model_pipeline/clm_uniform_500_L3_M3_eval.log


# Define experiment parameters using the new simplified structure
DATASET_TYPE="uniform"
MODEL_TYPE="clm"
NUM_SEEDS=500
SEED=42
L=3
M=3

# Construct shared arguments (updated to match new argument structure)
SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --num-seeds $NUM_SEEDS --seed $SEED --L $L --M $M"
# Optional pipeline controls
PIPELINE_ARGS="--verbose"

# =============================================================================
# Training model
# =============================================================================
# echo "============== Training model =============="

# SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/train"
# # Step 1: Validate configuration (optional)
# echo "Step 1: Validating training configuration..."
# python $SCRIPT_ROOT/train.py $SHARED_ARGS --dry-run

# # Step 2: Train model
# echo "Step 2: Training model..."
# python $SCRIPT_ROOT/train.py $SHARED_ARGS $PIPELINE_ARGS


# =============================================================================
# Evaluating model
# =============================================================================
echo "============== Evaluating model =============="

# Collection mode and pipeline controls (execution params come from YAML)
MODE_ARGS="--batch-mode --verbose --overwrite"

# Combine arguments (execution parameters loaded from YAML)
ALL_ARGS="$SHARED_ARGS $MODE_ARGS"

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/eval"
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