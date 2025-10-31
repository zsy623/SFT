# monitor_training.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from datetime import datetime

def plot_training_curves(log_dir: str):
    """绘制训练曲线"""
    # 读取日志文件
    log_file = os.path.join(log_dir, "log_history.json")
    
    try:
        with open(log_file, 'r') as f:
            logs = [json.loads(line) for line in f if line.strip()]
    except:
        print(f"无法读取日志文件: {log_file}")
        return
    
    # 提取数据
    steps = []
    losses = []
    learning_rates = []
    
    for log in logs:
        if 'loss' in log and log['loss'] is not None:
            steps.append(log.get('step', 0))
            losses.append(log['loss'])
            learning_rates.append(log.get('learning_rate', 0))
    
    if not steps:
        print("没有找到训练日志数据")
        return
    
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

def monitor_training_live(log_dir: str, update_interval: int = 30):
    """实时监控训练进度"""
    log_file = os.path.join(log_dir, "log_history.json")
    last_size = 0
    
    print(f"开始监控训练日志: {log_file}")
    print("按 Ctrl+C 停止监控")
    
    while True:
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > last_size:
                    new_lines = lines[last_size:]
                    last_size = len(lines)
                    
                    for line in new_lines:
                        try:
                            log_data = json.loads(line.strip())
                            if 'loss' in log_data and log_data['loss'] is not None:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"Step: {log_data.get('step', 'N/A')}, "
                                      f"Loss: {log_data.get('loss', 'N/A'):.4f}, "
                                      f"LR: {log_data.get('learning_rate', 'N/A'):.2e}")
                        except json.JSONDecodeError:
                            continue
            
            time.sleep(update_interval)
            
        except FileNotFoundError:
            print("日志文件不存在，等待中...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n监控已停止")
            break

def start_tensorboard(log_dir: str):
    """启动TensorBoard"""
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    if os.path.exists(tensorboard_dir):
        os.system(f"tensorboard --logdir={tensorboard_dir} --port=6006 --bind_all")
    else:
        print(f"TensorBoard日志目录不存在: {tensorboard_dir}")

if __name__ == "__main__":
    log_dir = "/home/silverbullet/develop/project/sft/output/logs"
    monitor_training_live(log_dir)