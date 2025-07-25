#!/bin/bash
#SBATCH -J ICL
#SBATCH -p gpu_l40
#SBATCH -N 1
#SBATCH -o ICL_%j.out
#SBATCH -e ICL_%j.err
#SBATCH --no-requeue
#SBATCH -A qi_g1
#SBATCH --qos=qil40
#SBATCH --gres=gpu:2
#SBATCH --overcommit
#SBATCH --mincpus=9


nvidia-smi
hostname

source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate
cd $SLURM_SUBMIT_DIR

# Get the number of nodes and GPUs per node
NNODES=$SLURM_NNODES
NPROC_PER_NODE=2
MASTER_PORT=$1
export OMP_NUM_THREADS=$((2 * 2))



source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate
cd $SLURM_SUBMIT_DIR

# Get the number of nodes and GPUs per node
NNODES=$SLURM_NNODES
NPROC_PER_NODE=1
MASTER_PORT=$1
export OMP_NUM_THREADS=$((1 * 2))


# HyperPMs
P=$2
DIM=$3
DEPTH=$4
NHEAD=$5
N_VAR=2
BS=1024
EVAL_BS=1
STEPS=200000
WARM_STEPS=10000
LR=$6
WD=$7
DATA_PCT=$8
NTASKS_PL=$9
NTASKS_RD=${10}
SEED=${11}



# Define log file
LOGFILE="icl_history.log"

echo "$SLURM_GPUS_ON_NODE" >> $LOGFILE
echo "$SLURM_NNODES" >> $LOGFILE

# Build the command string
# When torchrun, these arguments are then parsed inside the Python script using a library like argparse.
CMD="torchrun  --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
play_with_repeated_sequences.py --device='cuda' --mixed_precision=True --dtype='bfloat16' --num_workers=8 \
--n_tasks_rd=$NTASKS_RD --n_tasks_pl=$NTASKS_PL --parallelogram=True --n_var=$N_VAR --p=$P --base=$P --data_pct=$DATA_PCT --split_data=True \
--model='rope_decoder' --act_name='relu' --block_size=512 --n_embd=$DIM --n_layer=$DEPTH --n_head=$NHEAD \
--optim='adamw' --lr=$LR --wd=$WD --dont_decay_embd=False --weight_tying=True --lr_decay='cosine' --clip=0.0 \
--steps=$STEPS --warmup_steps=$WARM_STEPS --steps_per_record=1000 --fake_restart_steps=5000 --n_point_per_row=2 \
--bs=$BS --eval_bs=$EVAL_BS --seed=$SEED --data_seed=0 --reshuffle_step=1 \
--tqdm_bar=False --dist_backend='nccl' "

# Log the start of the command
echo -e "\n $(date), SLURM Job ID $SLURM_JOB_ID: Starting command:" >> $LOGFILE
echo "$CMD" >> $LOGFILE

# Execute the command
eval $CMD

# Check the exit status of the last executed command
if [ $? -eq 0 ]; then
  STATUS="SUCCESS"
else
  STATUS="FAILURE"
fi

# Log the end of the command
echo -e "\n $(date), SLURM Job ID $SLURM_JOB_ID: Finished, STATUS: $STATUS" >> $LOGFILE




























