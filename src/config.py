from transformers import AutoConfig

# 从原模型 google/flan-t5-large 拉取配置
config = AutoConfig.from_pretrained("google/flan-t5-large")

# 保存到你的微调模型目录
config.save_pretrained("/home/zlijw/flan-t5-finetune-project/checkpoints/flan-t5-full")