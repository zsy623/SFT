# run_training.sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# 设置环境变量，避免在线下载
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置W&B（可选，如果不用可以注释掉）
# export WANDB_API_KEY="d3efdf53552bd80588b7bb71f7b89750d3e2a1b8"

# 安装必要的包

# echo "开始训练..."
python train_qwen3.py

echo "训练完成！"
# echo "可以使用以下命令监控训练："
# echo "python monitor_training.py"