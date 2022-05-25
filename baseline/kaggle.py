import json

import pandas as pd
import torch
from tqdm import tqdm


class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, tokenizer, max_length=416, delimiter=" "):  # max wangchan 416 subwords
        self.mapper = {"Fake News": 0, "Fact News": 1}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.label_path = label_path
        self.delimiter = delimiter
        self.max_length = max_length
        self.data = pd.read_csv(self.label_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item["title"]
        label = item["label"]
        feature = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True)
        feature = {k: torch.tensor(v).to(self.device) for k, v in feature.items()}
        feature["labels"] = torch.tensor(label).to(self.device)
        return feature

def read_kaggle(kaggle_dir, train_percentage=None, delimiter=" "):

    train = pd.read_csv(f"{kaggle_dir}/kaggle_train.csv")
    val = pd.read_csv(f"{kaggle_dir}/kaggle_val.csv")
    test = pd.read_csv(f"{kaggle_dir}/kaggle_test.csv")
            
    if train_percentage is not None:
        print("Total full training samples:", train.shape[0])
        train = train[:int(len(train)*train_percentage)]
        print("Truncated to:", train.shape[0])

    return {
        "train": train,
        "val": val,
        "test": test
    }