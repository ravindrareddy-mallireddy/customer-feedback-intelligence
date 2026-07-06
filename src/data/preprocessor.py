from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd
from src.utils import get_logger, load_config

class ReviewPreprocessor:
    _HTML_TAG = re.compile(r"<[^>]+>")
    _HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
    _CTRL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
    _MULTI_SPACE = re.compile(r" {2,}")
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _URL = re.compile(r"https?://\S+|www\.\S+")

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.data_cfg = config["data"]
        self.min_len = self.data_cfg["min_text_length"]
        self.max_tokens = self.data_cfg["max_text_length"]

    def fit_transform(self, df):
        df = df.copy()
        n_original = len(df)
        self.logger.info("Preprocessing %d reviews ...", n_original)
        df["text"] = df["text"].fillna("").astype(str)
        df["text"] = df["text"].apply(self._clean_text)
        char_max = self.max_tokens * 6
        mask_len = df["text"].str.len().between(self.min_len, char_max)
        df = df[mask_len].copy()
        self.logger.info("  After length filter: %d -> %d rows", n_original, len(df))
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        self.logger.info("  After dedup: %d -> %d rows", before_dedup, len(df))
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        return df

    def validate(self, df):
        report = {}
        null_rates = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        report["null_rates_pct"] = null_rates
        for col, rate in null_rates.items():
            if rate > 5:
                self.logger.warning("  HIGH null rate in '%s': %.1f%%", col, rate)
        if "sentiment" in df.columns:
            dist = df["sentiment"].value_counts(normalize=True).round(4).to_dict()
            report["sentiment_distribution"] = dist
            self.logger.info("  Sentiment dist: %s", dist)
        lengths = df["text"].str.len()
        report["text_length"] = {
            "min": int(lengths.min()), "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "p25": int(lengths.quantile(0.25)), "p50": int(lengths.quantile(0.50)),
            "p75": int(lengths.quantile(0.75)),
        }
        self.logger.info("  Text length stats: %s", report["text_length"])
        if "category" in df.columns:
            report["category_distribution"] = df["category"].value_counts().to_dict()
            self.logger.info("  Category dist: %s", report["category_distribution"])
        report["total_rows"] = len(df)
        self.logger.info("  Total valid rows: %d", len(df))
        return report

    def _clean_text(self, text):
        text = self._HTML_TAG.sub(" ", text)
        text = self._HTML_ENTITY.sub(" ", text)
        text = unicodedata.normalize("NFKC", text)
        text = self._CTRL_CHARS.sub("", text)
        text = self._URL.sub(" ", text)
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()
