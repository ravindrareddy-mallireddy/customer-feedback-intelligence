from __future__ import annotations
import json
import time
import numpy as np
from pathlib import Path
from src.utils import get_logger


class RetrievalEvaluator:
    """Evaluates retrieval pipeline with MRR, Hit Rate, and latency metrics."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.retrieval_cfg = config["retrieval"]

    def evaluate(self, searcher, indexer, reranker, gold_queries: list[dict]) -> dict:
        """Run full retrieval evaluation.

        Args:
            searcher: ReviewSearcher instance.
            indexer: EmbeddingIndexer instance.
            reranker: Reranker instance.
            gold_queries: List of dicts with 'query' and 'relevant_keywords'.

        Returns:
            Dict with MRR, Hit Rate, and latency metrics.
        """
        mrr_scores = []
        hit_rates_at_5 = []
        hit_rates_at_10 = []
        latencies = []

        for item in gold_queries:
            query = item["query"]
            keywords = item["relevant_keywords"]

            t0 = time.time()
            query_emb = indexer.encode([query])[0]
            faiss_results = searcher.search_faiss(query_emb, top_k=50)
            reranked = reranker.rerank(query, faiss_results, top_k=10)
            latency = time.time() - t0
            latencies.append(latency)

            # Check relevance by keyword matching
            def is_relevant(text):
                text_lower = text.lower()
                return any(kw.lower() in text_lower for kw in keywords)

            # MRR
            rr = 0.0
            for rank, result in enumerate(reranked, 1):
                if is_relevant(result["text"]):
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

            # Hit Rate @ 5
            top5 = reranked[:5]
            hit5 = any(is_relevant(r["text"]) for r in top5)
            hit_rates_at_5.append(float(hit5))

            # Hit Rate @ 10
            hit10 = any(is_relevant(r["text"]) for r in reranked[:10])
            hit_rates_at_10.append(float(hit10))

        latencies_arr = np.array(latencies)
        metrics = {
            "mrr": round(float(np.mean(mrr_scores)), 4),
            "hit_rate_at_5": round(float(np.mean(hit_rates_at_5)), 4),
            "hit_rate_at_10": round(float(np.mean(hit_rates_at_10)), 4),
            "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)) * 1000, 2),
            "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)) * 1000, 2),
            "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)) * 1000, 2),
            "n_queries": len(gold_queries),
        }

        self.logger.info("MRR          : %.4f", metrics["mrr"])
        self.logger.info("Hit Rate @5  : %.4f", metrics["hit_rate_at_5"])
        self.logger.info("Hit Rate @10 : %.4f", metrics["hit_rate_at_10"])
        self.logger.info("Latency P50  : %.1f ms", metrics["latency_p50_ms"])
        self.logger.info("Latency P95  : %.1f ms", metrics["latency_p95_ms"])
        return metrics


class ModelEvaluator:
    """Consolidates all model evaluation metrics into one report."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

    def consolidate(self, eval_dir: Path) -> dict:
        """Load and consolidate all evaluation JSON files.

        Args:
            eval_dir: Path to evaluation directory.

        Returns:
            Consolidated metrics dict.
        """
        report = {}

        # Sentiment metrics
        sentiment_path = eval_dir / "sentiment_eval.json"
        if sentiment_path.exists():
            with open(sentiment_path) as f:
                report["sentiment"] = json.load(f)
            self.logger.info("Sentiment: accuracy=%.4f f1=%.4f",
                report["sentiment"]["accuracy"],
                report["sentiment"]["f1_weighted"])

        # Aspect metrics
        aspect_path = eval_dir / "aspect_eval.json"
        if aspect_path.exists():
            with open(aspect_path) as f:
                report["aspect"] = json.load(f)
            self.logger.info("Aspect: f1_micro=%.4f f1_macro=%.4f",
                report["aspect"]["f1_micro"],
                report["aspect"]["f1_macro"])

        # Topic metrics
        topic_path = eval_dir / "topic_metrics.json"
        if topic_path.exists():
            with open(topic_path) as f:
                report["topics"] = json.load(f)
            self.logger.info("Topics: n=%d diversity=%.4f outlier_rate=%.4f",
                report["topics"]["n_topics"],
                report["topics"]["topic_diversity"],
                report["topics"]["outlier_rate"])

        return report
