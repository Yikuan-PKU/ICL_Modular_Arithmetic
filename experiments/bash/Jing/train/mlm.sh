#!/bin/bash
#SBATCH --job-name=train_mlm
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/trai/uniform_allmix_576_42_clm.log

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/train"

# Define experiment parameters
DATASET_TYPE="uniform"
MODEL_TYPE="mlm"
MIXTURE_TYPE="allmix"
TOTAL_RULES=576
SEED=42

# Construct shared arguments
SHARED_ARGS="--dataset-type $DATASET_TYPE --model-type $MODEL_TYPE --mixture-type $MIXTURE_TYPE --total-rules $TOTAL_RULES --seed $SEED"

# Optional pipeline controls
PIPELINE_ARGS="--verbose"

# Experiment names for reference
DATASET_NAME="${DATASET_TYPE}_${MIXTURE_TYPE}_${TOTAL_RULES}_${SEED}"
MODEL_NAME="${DATASET_NAME}_${MODEL_TYPE}"

echo "=========================================="
echo "TRAINING RHM MODEL"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME"
echo "GPU allocation: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Step 1: Validate configuration (optional)
echo "Step 1: Validating training configuration..."
python $SCRIPT_ROOT/train.py $SHARED_ARGS --dry-run

if [ $? -ne 0 ]; then
    echo "Configuration validation failed!"
    exit 1
fi

# Step 2: Train model
echo "Step 2: Training model..."
python $SCRIPT_ROOT/train.py $SHARED_ARGS $PIPELINE_ARGS

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: exp/${DATASET_NAME}/${MODEL_TYPE}/"
else
    echo "Training failed!"
    exit 1
fi