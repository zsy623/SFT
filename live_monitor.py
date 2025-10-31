# live_monitor.py
import time
import json
from datetime import datetime

def monitor_training_progress(log_file: str, update_interval: int = 30):
    """实时监控训练进度"""
    last_size = 0
    
    while True:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > last_size:
                new_lines = lines[last_size:]
                last_size = len(lines)
                
                for line in new_lines:
                    log_data = json.loads(line.strip())
                    if 'loss' in log_data:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Step: {log_data.get('step', 'N/A')}, "
                              f"Loss: {log_data.get('loss', 'N/A'):.4f}, "
                              f"LR: {log_data.get('learning_rate', 'N/A'):.2e}")
            
            time.sleep(update_interval)
            
        except FileNotFoundError:
            print("Log file not found, waiting...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("Monitoring stopped.")
            break