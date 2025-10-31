#!/bin/bash
# /home/silverbullet/develop/project/sft/download_model.sh

#!/bin/bash
MODEL_DIR="/mnt/ssd2/models/Qwen2.5-7B"

# 创建模型目录
mkdir -p $MODEL_DIR

# 下载Qwen2.5-7B模型
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_path = '/mnt/ssd2/models/Qwen2.5-7B'
print('正在下载Qwen2.5-7B模型...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
print('模型下载完成!')

"