from __future__ import annotations
from src.utils import get_logger


class Reranker:
    """Cross-encoder reranker - falls back gracefully on low memory environments."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.model_name = config["retrieval"]["reranker_model"]
        self.rerank_top_k = config["retrieval"]["rerank_top_k"]
        self.model = None
        self.enabled = False

    def build(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=256)
            self.enabled = True
            self.logger.info("Reranker loaded: %s", self.model_name)
        except Exception as e:
            self.logger.warning("Reranker disabled: %s", e)
            self.enabled = False

    def rerank(self, query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
        k = top_k or self.rerank_top_k
        if not results:
            return []
        if not self.enabled or self.model is None:
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:k]
        pairs = [[query, r["text"]] for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        self.logger.info("Reranked %d -> top %d", len(results), k)
        return reranked[:k]
