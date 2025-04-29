#!/bin/bash
#SBATCH --job-name=finetune-flan-t5
#SBATCH --output=logs/train_flan_t5_%j.out
#SBATCH --error=logs/train_flan_t5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --account="msccsit2024"
#SBATCH --partition=normal
#SBATCH --time=4:00:00

echo "🔵 Starting job on $(hostname) at $(date)"

# 加载 Anaconda
module load Anaconda3/2023.09-0

# 激活环境（已手动创建）
source ~/.bashrc
conda activate flan-t5-finetune

# 正式训练
echo "🚀 Launching training..."
accelerate launch --main_process_port=0 --multi_gpu --mixed_precision fp16 src/train_flan_t5.py
echo "✅ Training completed on $(hostname) at $(date)"