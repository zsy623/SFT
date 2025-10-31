# train_qwen3.py
import os
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from data_loader import load_alpaca_zh_dataset
import json
from typing import Dict, List
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class Qwen3Trainer:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
    def setup_model(self):
        """初始化模型和分词器"""
        print("Loading tokenizer and model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # 用于梯度检查点
        )
        
        # 启用梯度检查点以节省显存
        self.model.gradient_checkpointing_enable()
        
        print(f"Model loaded on device: {self.model.device}")
        
    def preprocess_function(self, examples):
        """数据预处理函数"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        
        # 对于因果语言模型，labels就是input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset):
        """准备训练数据集"""
        print("Preprocessing dataset...")
        
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def setup_training_args(self):
        """设置训练参数"""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # 训练参数
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.03,
            
            # 优化器
            optim="adamw_torch",
            
            # 学习率调度
            lr_scheduler_type="cosine",
            
            # 日志和保存
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="no",
            
            # FP16训练
            fp16=True,
            
            # 报告设置
            report_to=["tensorboard", "wandb"],
            
            # 其他
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
    
    def train(self, dataset):
        """开始训练"""
        # 初始化W&B
        wandb.init(project="qwen3-1.7b-sft", name="qwen3_finetuning")
        
        # 准备数据
        train_dataset = self.prepare_dataset(dataset)
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 训练参数
        training_args = self.setup_training_args()
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("Starting training...")
        train_result = trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        trainer.save_state()
        
        # 记录最终指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return metrics

def main():
    # 配置路径
    model_path = "/mnt/ssd2/models/Qwen3-1.7B"
    output_dir = "/home/silverbullet/develop/project/sft/output"
    log_dir = "/home/silverbullet/develop/project/sft/logs"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = Qwen3Trainer(model_path, output_dir)
    trainer.setup_model()
    
    # 加载数据集
    print("Loading dataset...")
    dataset = load_alpaca_zh_dataset()
    
    # 可以选择只使用部分数据用于测试
    # dataset = dataset.select(range(1000))
    
    print(f"Dataset size: {len(dataset)}")
    
    # 开始训练
    metrics = trainer.train(dataset)
    print(f"Training completed! Metrics: {metrics}")

if __name__ == "__main__":
    main()