from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils import get_logger


class EmbeddingIndexer:
    """Generates and stores sentence embeddings for all reviews."""

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
        """Generate embeddings for all reviews.

        Args:
            df: DataFrame with 'text' column and metadata columns.
        """
        self.texts = df["text"].tolist()
        self.metadata = df.to_dict(orient="records")

        self.logger.info("Generating embeddings for %d reviews ...", len(self.texts))
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.logger.info("Embeddings shape: %s", str(self.embeddings.shape))
        return self.embeddings

    def save(self):
        """Save embeddings and metadata to disk."""
        emb_path = self.save_dir / "retrieval_embeddings.npy"
        np.save(emb_path, self.embeddings)
        self.logger.info("Embeddings saved -> %s", emb_path)

        meta_path = self.save_dir / "retrieval_metadata.json"
        import json
        with open(meta_path, "w") as f:
            # Convert non-serializable types
            clean_meta = []
            for m in self.metadata:
                clean = {}
                for k, v in m.items():
                    try:
                        import math
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            clean[k] = ""
                        else:
                            clean[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
                    except:
                        clean[k] = str(v)
                clean_meta.append(clean)
            json.dump(clean_meta, f)
        self.logger.info("Metadata saved -> %s", meta_path)

    def load(self):
        """Load embeddings and metadata from disk."""
        emb_path = self.save_dir / "retrieval_embeddings.npy"
        self.embeddings = np.load(emb_path)
        self.logger.info("Embeddings loaded: %s", str(self.embeddings.shape))

        meta_path = self.save_dir / "retrieval_metadata.json"
        import json
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.texts = [m["text"] for m in self.metadata]
        self.logger.info("Metadata loaded: %d records", len(self.metadata))

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode query texts at inference time."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
