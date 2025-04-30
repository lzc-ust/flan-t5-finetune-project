#!/bin/bash
#SBATCH --job-name=finetune-flan-t5
#SBATCH --output=logs/train_flan_t5_%j.out
#SBATCH --error=logs/train_flan_t5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --account="msccsit2024"
#SBATCH --partition=normal
#SBATCH --time=16:00:00

echo "🔵 Starting job on $(hostname) at $(date)"

# 切换到项目根目录
cd /home/zlijw/flan-t5-finetune-project/

# 设置 PYTHONPATH，确保 src 模块可被识别
export PYTHONPATH=$(pwd)

# 正式训练（单卡）
echo "🚀 Launching training..."
/home/zlijw/.conda/envs/flan-t5-finetune/bin/python src/train_flan_t5.py

echo "✅ Training completed on $(hostname) at $(date)"