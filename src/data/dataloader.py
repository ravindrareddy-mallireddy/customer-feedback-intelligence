from __future__ import annotations
from typing import Literal
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from src.utils import get_logger

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, num_aspects=6):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_aspects = num_aspects

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

def build_dataloaders(train_df, val_df, test_df, tokenizer, config, task="sentiment"):
    logger = get_logger(__name__, config)
    if task == "sentiment":
        model_cfg = config["models"]["sentiment"]
        batch_size = model_cfg["batch_size"]
        max_length = config["data"]["max_text_length"]

        def _make(df, shuffle):
            texts = df["text"].tolist()
            labels = df["sentiment_label"].tolist()
            ds = SentimentDataset(texts, labels, tokenizer, max_length)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    elif task == "aspect":
        model_cfg = config["models"]["aspect"]
        batch_size = model_cfg["batch_size"]
        max_length = config["data"]["max_text_length"]
        aspects = model_cfg["aspects"]

        def _make(df, shuffle):
            texts = df["text"].tolist()
            labels = df[aspects].values.tolist()
            ds = AspectDataset(texts, labels, tokenizer, max_length, len(aspects))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    else:
        raise ValueError(f"Unknown task: {task}")

    train_dl = _make(train_df, shuffle=True)
    val_dl = _make(val_df, shuffle=False)
    test_dl = _make(test_df, shuffle=False)
    logger.info("DataLoaders built [%s] Train=%d Val=%d Test=%d BatchSize=%d",
        task, len(train_df), len(val_df), len(test_df), batch_size)
    return train_dl, val_dl, test_dl
