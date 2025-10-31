# data_loader.py
from datasets import Dataset
import pandas as pd

def load_alpaca_zh_dataset():
    """从ModelScope加载Alpaca中文数据集"""
    # 尝试从ModelScope加载数据集
    from modelscope import MsDataset
    print("Loading dataset from ModelScope...")
    dataset = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-zh', split='train')
    # 转换为标准的dataset格式
    dataset = dataset.to_hf_dataset()
    print(f"Successfully loaded dataset with {len(dataset)} samples")
    return dataset
    

def load_dataset_from_local(path: str):
    """从本地文件加载数据集"""
    if path.endswith('.json'):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    elif path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)
    else:
        raise ValueError("Unsupported file format")