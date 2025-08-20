#!/bin/bash
#SBATCH -J train_clm
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -o RHM_%j.out
#SBATCH -e RHM_%j.err
#SBATCH --no-requeue
#SBATCH -A qi_g1
#SBATCH --qos=qig4c
#SBATCH --gres=gpu:2
#SBATCH --overcommit
#SBATCH --mincpus=9

source ~/lustre1/ykzhang/apps/python-3.11.11/venvICL/bin/activate


SCRIPT_ROOT="$HOME/lustre1/ykzhang/ICL_RHM/ICL_Modular_Arithmetic/src/scripts/train"


# 检查路径是否存在
echo "检查脚本路径: $SCRIPT_ROOT"
if [ ! -d "$SCRIPT_ROOT" ]; then
    echo "错误: 脚本目录不存在!"
    exit 1
fi


echo "=== 调试环境信息 ==="
echo "工作目录: $(pwd)"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "脚本目录: $SCRIPT_ROOT"
echo "=================="


# 测试基本功能

echo "测试torch导入..."
python -c "import torch; print('torch导入成功'); print('CUDA可用:', torch.cuda.is_available())"

echo "测试numpy导入..."
python -c "import numpy; print('numpy导入成功')"



# 检查脚本是否存在
if [ -f "$SCRIPT_ROOT/train.py" ]; then
    echo "找到脚本文件"
    # 只运行验证步骤进行测试
    python $SCRIPT_ROOT/train.py --help
else
    echo "错误: 找不到脚本文件 $SCRIPT_ROOT/train.py"
    exit 1
fi



# Define experiment parameters
DATASET_TYPE="uniform"
MODEL_TYPE="clm"
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


# Step 2: Train model
echo "Step 2: Training model..."
python $SCRIPT_ROOT/train.py $SHARED_ARGS $PIPELINE_ARGS