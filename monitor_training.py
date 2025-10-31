# monitor_training.py
import os
import tensorboard as tb
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_training_curves(log_dir: str):
    """绘制训练曲线"""
    # 读取日志文件
    log_file = os.path.join(log_dir, "logs", "log_history.json")
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    # 提取数据
    steps = []
    losses = []
    learning_rates = []
    
    for log in logs:
        if 'loss' in log:
            steps.append(log.get('step', 0))
            losses.append(log['loss'])
            learning_rates.append(log.get('learning_rate', 0))
    
    # 绘制loss曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'))
    plt.show()

def start_tensorboard(log_dir: str):
    """启动TensorBoard"""
    os.system(f"tensorboard --logdir={log_dir} --port=6006")