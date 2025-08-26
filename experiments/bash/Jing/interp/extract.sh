#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/ICL/logs/interp/extract%a.log
#SBATCH --array=0-5

SCRIPT_ROOT="/scratch2/jliu/ICL/ICL_Modular_Arithmetic/src/scripts/interp"
# Define arrays
MODELS=("clm" "mlm")
SEEDS=(500 1000 1500)

# Total number of combinations
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#SEEDS[@]}))

# Compute indices
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

# Get values
MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Run Python script
python $SCRIPT_ROOT/run_example.py --model "$MODEL" --seed_num "$SEED"
