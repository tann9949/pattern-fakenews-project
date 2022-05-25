import json

import pandas as pd
import torch
from tqdm import tqdm


class LimeSodaDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, tokenizer, max_length=416, delimiter=" "):  # max wangchan 416 subwords
        self.mapper = {"Fake News": 0, "Fact News": 1}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.label_path = label_path
        self.delimiter = delimiter
        self.max_length = max_length
        self.load_dataframe()
        
    def load_dataframe(self):
        print("Loading data...")
        data = []
        with open(self.label_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                if line["Document Tag"] not in self.mapper.keys():
                    continue
                line["label"] = self.mapper[line["Document Tag"]]
                line["text"] = self.delimiter.join([t for t in line["Text"] if len(t.strip()) > 0])
                line.pop("Document Tag")
                line.pop("Text")
                data.append(line)
        self.data = pd.DataFrame(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item["text"]
        label = item["label"]
        feature = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True)
        feature = {k: torch.tensor(v).to(self.device) for k, v in feature.items()}
        feature["labels"] = torch.tensor(label).to(self.device)
        return feature


def read_limesoda(limesoda_dir, train_percentage=None, delimiter=" "):
    train, val, test = [], [], []
    mapper = {"Fake News": 0, "Fact News": 1}
    
    # train
    with open(f"{limesoda_dir}//../tempLimesoda/train_v1.jsonl", "r") as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            line["label"] = mapper[line["Document Tag"]]
            line["text"] = delimiter.join([t for t in line["Text"] if len(t.strip()) > 0])
            line.pop("Document Tag")
            line.pop("Text")
            train.append(line)
           
    # val
    with open(f"{limesoda_dir}//../tempLimesoda/val_v1.jsonl", "r") as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            line["label"] = mapper[line["Document Tag"]]
            line["text"] = delimiter.join([t for t in line["Text"] if len(t) > 0])
            line.pop("Document Tag")
            line.pop("Text")
            val.append(line)
            
    with open(f"{limesoda_dir}//../tempLimesoda/test_v1.jsonl", "r") as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            if line["Document Tag"] not in mapper.keys():
                continue
            line["label"] = mapper[line["Document Tag"]]
            line["text"] = delimiter.join([t for t in line["Text"] if len(t) > 0])
            line.pop("Document Tag")
            line.pop("Text")
            test.append(line)
            
    if train_percentage is not None:
        print("Total full training samples:", len(train))
        train = train[:int(len(train)*train_percentage)]
        print("Truncated to:", len(train))

    return {
        "train": pd.DataFrame(train),
        "val": pd.DataFrame(val),
        "test": pd.DataFrame(test)
    }
