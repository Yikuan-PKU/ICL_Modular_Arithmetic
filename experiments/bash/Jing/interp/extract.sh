#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/interp/rest_%a.log
#SBATCH --array=0-3 # 2 models * 3 seeds * 4 tasks = 24 jobs

# Path to your Python scripts
SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/interp"

# Define arrays
MODELS=("mlm")
SEEDS=(500 1000)
TASKS=("ood_same_rule" "ood_transfer")

# Array lengths
NUM_MODELS=${#MODELS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}

# Compute indices for this job
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_SEEDS * NUM_TASKS) ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_TASKS) % NUM_SEEDS ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS ))

# Get values
MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TASK="${TASKS[$TASK_IDX]}"

echo "Running job $SLURM_ARRAY_TASK_ID: model=$MODEL, seed=$SEED, task=$TASK"

# Run Python script
python $SCRIPT_ROOT/run_example.py --model "$MODEL" --seed_num "$SEED" --task "$TASK" --max_seq 50 --resume
