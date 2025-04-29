#!/bin/bash
#SBATCH --job-name=finetune-flan-t5
#SBATCH --output=logs/train_flan_t5_%j.out
#SBATCH --error=logs/train_flan_t5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --time=48:00:00

echo "🔵 Starting job on $(hostname) at $(date)"

# 保证加载正确bashrc
source ~/.bashrc

# 检查conda是否可用
if ! command -v conda &> /dev/null
then
    echo "❌ conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 检查环境是否存在
ENV_NAME="flan-t5-finetune"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ Conda environment '$ENV_NAME' found. Activating..."
else
    echo "⚠️ Conda environment '$ENV_NAME' not found. Initializing environment setup..."
    bash init_env.sh
fi

# 激活环境
conda activate $ENV_NAME

# 显示CUDA信息
nvidia-smi

# 加载Accelerate配置（如果已经有）
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "⚠️ Accelerate config not found. Please run 'accelerate config' manually before first training."
else
    echo "✅ Found Accelerate config."
fi

# 正式启动训练
echo "🚀 Launching training..."
accelerate launch --multi_gpu --mixed_precision fp16 src/train_flan_t5.py
echo "✅ Training completed on $(hostname) at $(date)"