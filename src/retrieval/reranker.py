from __future__ import annotations
from sentence_transformers import CrossEncoder
from src.utils import get_logger


class Reranker:
    """Cross-encoder reranker for improving retrieval precision."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.model_name = config["retrieval"]["reranker_model"]
        self.rerank_top_k = config["retrieval"]["rerank_top_k"]
        self.model = None

    def build(self):
        self.logger.info("Loading reranker: %s", self.model_name)
        self.model = CrossEncoder(self.model_name, max_length=512)
        self.logger.info("Reranker loaded.")

    def rerank(self, query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerank retrieved results using cross-encoder.

        Args:
            query: Original search query string.
            results: List of result dicts from FAISS or ChromaDB search.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of result dicts with updated 'rerank_score'.
        """
        k = top_k or self.rerank_top_k
        if not results:
            return []

        pairs = [[query, r["text"]] for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        self.logger.info(
            "Reranked %d results → top %d (best score=%.4f)",
            len(results), k, reranked[0]["rerank_score"] if reranked else 0
        )
        return reranked[:k]
