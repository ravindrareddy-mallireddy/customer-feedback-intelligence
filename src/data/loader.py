from __future__ import annotations
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.utils import get_logger, load_config, set_seed

class ReviewDatasetLoader:
    KEEP_COLS = ["text", "rating", "title", "parent_asin", "timestamp"]
    CATEGORY_COL = "category"

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        set_seed(config["project"]["random_seed"])
        self.data_cfg = config["data"]
        self.sent_cfg = config["sentiment"]
        self.processed_dir = Path(self.data_cfg["processed_dir"])
        self.raw_dir = Path(self.data_cfg["raw_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.df_train = self.df_val = self.df_test = None

    def run(self):
        self.logger.info("=== Stage 1: Data Pipeline ===")
        raw_df = self._load_all_categories()
        labeled_df = self._add_sentiment_label(raw_df)
        self.df_train, self.df_val, self.df_test = self._split(labeled_df)
        self._save()
        self.logger.info("Pipeline complete. Train=%d  Val=%d  Test=%d",
            len(self.df_train), len(self.df_val), len(self.df_test))
        return self.df_train, self.df_val, self.df_test

    def load_from_disk(self):
        splits = {}
        for split in ("train", "val", "test"):
            path = self.processed_dir / f"{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}. Run .run() first.")
            splits[split] = pd.read_parquet(path)
            self.logger.info("Loaded %s (%d rows)", path.name, len(splits[split]))
        self.df_train, self.df_val, self.df_test = splits["train"], splits["val"], splits["test"]
        return self.df_train, self.df_val, self.df_test

    def _load_all_categories(self):
        categories = self.data_cfg["categories"]
        per_cat = self.data_cfg["reviews_per_category"]
        frames = []
        for cat in categories:
            self.logger.info("Loading category: %s (target %d reviews)", cat, per_cat)
            df = self._load_category(cat, per_cat)
            df[self.CATEGORY_COL] = cat
            frames.append(df)
            self.logger.info("  -> %d rows loaded for %s", len(df), cat)
        combined = pd.concat(frames, ignore_index=True)
        self.logger.info("Total raw reviews: %d", len(combined))
        return combined

    def _load_category(self, category, n):
        cache_path = self.raw_dir / f"{category}.parquet"
        if cache_path.exists():
            self.logger.info("  Cache hit: %s", cache_path)
            return pd.read_parquet(cache_path).head(n)
        try:
            ds = load_dataset(
                self.data_cfg["dataset_name"],
                category,
                split="full",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            self.logger.warning("Failed to load %s: %s. Using fallback.", category, e)
            ds = self._fallback_dataset(category)
        rows = []
        for item in ds:
            text = (item.get("text") or "").strip()
            if not text or len(text) < self.data_cfg["min_text_length"]:
                continue
            row = {col: item.get(col) for col in self.KEEP_COLS}
            rows.append(row)
            if len(rows) >= n:
                break
        df = pd.DataFrame(rows)
        df.to_parquet(cache_path, index=False)
        return df

    def _fallback_dataset(self, category):
        self.logger.info("Using fallback dataset for %s", category)
        ds = load_dataset("amazon_polarity", split="train", streaming=True, trust_remote_code=True)
        def normalise(item):
            return {
                "text": item.get("content", ""),
                "rating": 5 if item.get("label") == 1 else 1,
                "title": item.get("title", ""),
                "parent_asin": "",
                "timestamp": None,
            }
        return ds.map(normalise)

    def _add_sentiment_label(self, df):
        pos = set(self.sent_cfg["labels"]["positive"])
        neu = set(self.sent_cfg["labels"]["neutral"])
        label_map = self.sent_cfg["label_map"]
        def _map(rating):
            try:
                r = int(float(rating))
            except (TypeError, ValueError):
                return None
            if r in pos: return "positive"
            if r in neu: return "neutral"
            if r in {1, 2}: return "negative"
            return None
        df = df.copy()
        df["sentiment"] = df["rating"].apply(_map)
        df = df.dropna(subset=["sentiment"])
        df["sentiment_label"] = df["sentiment"].map(label_map)
        return df

    def _split(self, df):
        test_size = self.data_cfg["test_split"]
        val_size = self.data_cfg["val_split"]
        seed = self.config["project"]["random_seed"]
        train_val, test = train_test_split(df, test_size=test_size,
            stratify=df["sentiment_label"], random_state=seed)
        val_fraction = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_fraction,
            stratify=train_val["sentiment_label"], random_state=seed)
        return train, val, test

    def _save(self):
        for name, df in [("train", self.df_train), ("val", self.df_val), ("test", self.df_test)]:
            path = self.processed_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            self.logger.info("Saved %s -> %s (%d rows)", name, path, len(df))
