from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import faiss
import chromadb
from src.utils import get_logger


class ReviewSearcher:
    """Dual search backend: FAISS for speed, ChromaDB for filtered search."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.retrieval_cfg = config["retrieval"]
        self.top_k = self.retrieval_cfg["top_k"]
        self.faiss_path = Path(self.retrieval_cfg["faiss_index_path"])
        self.chroma_dir = self.retrieval_cfg["chroma_persist_dir"]
        self.chroma_collection = self.retrieval_cfg["chroma_collection"]
        self.faiss_index = None
        self.chroma_client = None
        self.chroma_col = None
        self.metadata = None

    # ── FAISS ────────────────────────────────────────────────────────────────

    def build_faiss(self, embeddings: np.ndarray, metadata: list[dict]):
        """Build and save FAISS index from embeddings."""
        self.metadata = metadata
        dim = embeddings.shape[1]
        self.logger.info("Building FAISS index (dim=%d, n=%d) ...", dim, len(embeddings))
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vectors
        self.faiss_index.add(embeddings.astype(np.float32))
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        self.logger.info("FAISS index saved -> %s (%d vectors)", self.faiss_path, self.faiss_index.ntotal)

    def load_faiss(self, metadata: list[dict]):
        """Load FAISS index from disk."""
        self.faiss_index = faiss.read_index(str(self.faiss_path))
        self.metadata = metadata
        self.logger.info("FAISS index loaded: %d vectors", self.faiss_index.ntotal)

    def search_faiss(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
        """Search FAISS index with a query embedding.

        Args:
            query_embedding: 1D numpy array of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score' and metadata fields.
        """
        k = top_k or self.top_k
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.faiss_index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            result["search_backend"] = "faiss"
            results.append(result)
        return results

    # ── ChromaDB ─────────────────────────────────────────────────────────────

    def build_chroma(self, embeddings: np.ndarray, metadata: list[dict], texts: list[str]):
        """Build ChromaDB collection from embeddings."""
        self.logger.info("Building ChromaDB collection ...")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        # Delete existing collection if present
        try:
            self.chroma_client.delete_collection(self.chroma_collection)
        except:
            pass

        self.chroma_col = self.chroma_client.create_collection(
            name=self.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # Add in batches of 500
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            self.chroma_col.add(
                ids=[str(j) for j in range(i, batch_end)],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=[{k: str(v) for k, v in m.items() if k != "text"} for m in metadata[i:batch_end]],
            )
            self.logger.info("  ChromaDB: added %d/%d", batch_end, len(texts))

        self.logger.info("ChromaDB collection built: %d documents", self.chroma_col.count())

    def load_chroma(self):
        """Load existing ChromaDB collection."""
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.chroma_col = self.chroma_client.get_collection(self.chroma_collection)
        self.logger.info("ChromaDB loaded: %d documents", self.chroma_col.count())

    def search_chroma(self, query_embedding: np.ndarray, top_k: int | None = None,
                      where: dict | None = None) -> list[dict]:
        """Search ChromaDB with optional metadata filtering.

        Args:
            query_embedding: 1D numpy array.
            top_k: Number of results.
            where: Optional ChromaDB where filter e.g. {"sentiment": "negative"}.

        Returns:
            List of result dicts.
        """
        k = top_k or self.top_k
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.chroma_col.query(**kwargs)
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            result = dict(meta)
            result["text"] = doc
            result["score"] = float(1 - dist)  # Convert distance to similarity
            result["search_backend"] = "chroma"
            output.append(result)
        return output
