#!/bin/bash
#SBATCH -J RHM_uniform
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


SCRIPT_ROOT="$HOME/lustre1/ykzhang/ICL_RHM/ICL_Modular_Arithmetic/src/scripts/datasets"




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
if [ -f "$SCRIPT_ROOT/create_split.py" ]; then
    echo "找到脚本文件"
    # 只运行验证步骤进行测试
    python $SCRIPT_ROOT/create_split.py --help
else
    echo "错误: 找不到脚本文件 $SCRIPT_ROOT/generate_raw.py"
    exit 1
fi




# Primary experiment parameters
DATASET_TYPE="uniform"      # Options: uniform, zipf
RNG_SEED=42                # Seed for generating random seeds (for reproducibility)
NUM_SEEDS=10                # Number of random seeds to generate

# Pipeline control flags
PIPELINE_ARGS="--verbose --overwrite"  # Options: --verbose, --overwrite, --resume

# Construct command arguments
SHARED_ARGS="--dataset-type $DATASET_TYPE --seed $RNG_SEED --num-seeds $NUM_SEEDS"

# =============================================================================
# GENERATION STEPS
# =============================================================================

# Step 1: Validate configuration (optional but recommended)
echo "Step 1: Validating configuration..."
python $SCRIPT_ROOT/create_split.py $SHARED_ARGS --validate-only

echo "Step 2: Running generation..."
python $SCRIPT_ROOT/create_split.py $SHARED_ARGS $PIPELINE_ARGS
