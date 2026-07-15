from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils import get_logger


class EmbeddingIndexer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.retrieval_cfg = config["retrieval"]
        self.model_name = self.retrieval_cfg["embedding_model"]
        self.batch_size = self.retrieval_cfg["batch_size"]
        self.save_dir = Path(config["models"]["topic"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.embeddings = None
        self.texts = None
        self.metadata = None

    def build(self):
        self.logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.logger.info("Embedding model loaded.")

    def fit(self, df: pd.DataFrame):
        self.texts = df["text"].tolist()
        self.metadata = df.to_dict(orient="records")
        self.logger.info("Generating embeddings for %d reviews ...", len(self.texts))
        self.embeddings = self.model.encode(
            self.texts, batch_size=self.batch_size,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.logger.info("Embeddings shape: %s", str(self.embeddings.shape))
        return self.embeddings

    def save(self):
        emb_path = self.save_dir / "retrieval_embeddings.npy"
        np.save(emb_path, self.embeddings)
        self.logger.info("Embeddings saved -> %s", emb_path)
        meta_path = self.save_dir / "retrieval_metadata.json"
        import json, math
        clean_meta = []
        for m in self.metadata:
            clean = {}
            for k, v in m.items():
                try:
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        clean[k] = ""
                    else:
                        clean[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
                except:
                    clean[k] = str(v)
            clean_meta.append(clean)
        with open(meta_path, "w") as f:
            json.dump(clean_meta, f)
        self.logger.info("Metadata saved -> %s", meta_path)

    def load(self):
        # Try retrieval_embeddings.npy first, fall back to embeddings.npy
        emb_path = self.save_dir / "retrieval_embeddings.npy"
        if not emb_path.exists():
            emb_path = self.save_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"No embeddings file found in {self.save_dir}")
        self.embeddings = np.load(emb_path)
        self.logger.info("Embeddings loaded: %s", str(self.embeddings.shape))

        meta_path = self.save_dir / "retrieval_metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.texts = [m["text"] for m in self.metadata]
            self.logger.info("Metadata loaded: %d records", len(self.metadata))
        else:
            self.logger.warning("No metadata file found - retrieval will work without metadata")
            self.metadata = [{"text": ""} for _ in range(len(self.embeddings))]
            self.texts = [""] * len(self.embeddings)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
