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

echo "🟢 Starting evaluation on $(hostname) at $(date)"

# 激活环境（推荐使用绝对路径）
/home/zlijw/.conda/envs/flan-t5-finetune/bin/python tests/evaluate_on_testset.py

echo "✅ Evaluation complete at $(date)"