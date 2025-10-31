# run_training.sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# 设置W&B
export WANDB_API_KEY="d3efdf53552bd80588b7bb71f7b89750d3e2a1b8"

# 启动训练
python train_qwen3.py

# 在另一个终端启动监控
# python live_monitor.py /home/silverbullet/develop/project/sft/output/logs/log_history.json