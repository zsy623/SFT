# dataset_loader.py
from datasets import load_dataset

def load_alpaca_zh_dataset():
    """加载Alpaca中文数据集"""
    dataset = load_dataset("yehaiwu/alpaca_zh", split="train")
    return dataset
