#!/bin/bash
# init_env.sh

# Step 1: 创建conda环境（如果还没有）
conda create -n flan-t5-finetune python=3.10 -y

# Step 2: 激活环境
conda activate flan-t5-finetune

# Step 3: 安装基本依赖
pip install -r requirements.txt

# Step 4: 如果需要完全还原，可以选用冻结版
# pip install -r requirements-freeze.txt

echo "✅ Environment setup complete!"