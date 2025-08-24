#!/bin/bash
#SBATCH --job-name=icl_collection
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/eval/collection.log

# Script root directory
SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/eval"

# Define experiment parameters (explicit - no auto-discovery)
DATASET_TYPE="uniform"
NUM_SEEDS=10
SEED=42
L=4          # EXPLICIT hierarchy depth
M=2          # EXPLICIT multiplicity
MODEL_TYPE="clm"  # EXPLICIT model type (clm or mlm)

# Construct shared arguments with EXPLICIT L,M,MODEL_TYPE
SHARED_ARGS="--dataset-type $DATASET_TYPE --num-seeds $NUM_SEEDS --seed $SEED --L $L --M $M --model-type $MODEL_TYPE"

# Collection mode and pipeline controls (execution params come from YAML)
MODE_ARGS="--batch-mode --verbose --resume"

# Combine arguments (execution parameters loaded from YAML)
ALL_ARGS="$SHARED_ARGS $MODE_ARGS"

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