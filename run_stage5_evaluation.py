import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.evaluation.metrics import RetrievalEvaluator, ModelEvaluator
from src.retrieval.embeddings import EmbeddingIndexer
from src.retrieval.search import ReviewSearcher
from src.retrieval.reranker import Reranker
from src.utils import get_logger, load_config, set_seed

# Gold standard queries for retrieval evaluation
GOLD_QUERIES = [
    {"query": "battery life is terrible", "relevant_keywords": ["battery", "charge", "power"]},
    {"query": "great value for money", "relevant_keywords": ["price", "value", "worth", "cheap"]},
    {"query": "fast delivery but poor packaging", "relevant_keywords": ["shipping", "delivery", "package", "box"]},
    {"query": "customer service was unhelpful", "relevant_keywords": ["customer service", "support", "refund", "return"]},
    {"query": "easy to use and setup", "relevant_keywords": ["easy", "simple", "setup", "install"]},
    {"query": "product broke after one week", "relevant_keywords": ["broke", "broken", "stopped working", "defective"]},
    {"query": "size runs small", "relevant_keywords": ["size", "small", "fit", "tight"]},
    {"query": "excellent sound quality", "relevant_keywords": ["sound", "audio", "music", "quality"]},
    {"query": "difficult to assemble", "relevant_keywords": ["assemble", "assembly", "instructions", "difficult"]},
    {"query": "not as described in listing", "relevant_keywords": ["described", "listing", "expected", "misleading"]},
]

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage5_evaluation", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 5 - Evaluation Framework")
    logger.info("=" * 50)

    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Consolidate model metrics
    logger.info("\n--- Model Metrics ---")
    model_evaluator = ModelEvaluator(cfg)
    model_report = model_evaluator.consolidate(eval_dir)

    # Step 2: Load retrieval pipeline
    logger.info("\n--- Retrieval Evaluation ---")
    indexer = EmbeddingIndexer(cfg)
    indexer.build()
    indexer.load()

    searcher = ReviewSearcher(cfg)
    searcher.load_faiss(indexer.metadata)
    searcher.load_chroma()

    reranker = Reranker(cfg)
    reranker.build()

    # Step 3: Run retrieval evaluation
    retrieval_evaluator = RetrievalEvaluator(cfg)
    retrieval_metrics = retrieval_evaluator.evaluate(
        searcher, indexer, reranker, GOLD_QUERIES
    )

    # Step 4: Consolidate full report
    full_report = {
        "models": model_report,
        "retrieval": retrieval_metrics,
    }

    report_path = eval_dir / "full_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Full evaluation report saved -> %s", report_path)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 5 COMPLETE - EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info("\nMODEL PERFORMANCE:")
    if "sentiment" in model_report:
        logger.info("  Sentiment Accuracy  : %.4f", model_report["sentiment"]["accuracy"])
        logger.info("  Sentiment F1        : %.4f", model_report["sentiment"]["f1_weighted"])
    if "aspect" in model_report:
        logger.info("  Aspect F1 Micro     : %.4f", model_report["aspect"]["f1_micro"])
        logger.info("  Aspect F1 Macro     : %.4f", model_report["aspect"]["f1_macro"])
    if "topics" in model_report:
        logger.info("  Topics Found        : %d", model_report["topics"]["n_topics"])
        logger.info("  Topic Diversity     : %.4f", model_report["topics"]["topic_diversity"])

    logger.info("\nRETRIEVAL PERFORMANCE:")
    logger.info("  MRR                 : %.4f", retrieval_metrics["mrr"])
    logger.info("  Hit Rate @5         : %.4f", retrieval_metrics["hit_rate_at_5"])
    logger.info("  Hit Rate @10        : %.4f", retrieval_metrics["hit_rate_at_10"])
    logger.info("  Latency P50         : %.1f ms", retrieval_metrics["latency_p50_ms"])
    logger.info("  Latency P95         : %.1f ms", retrieval_metrics["latency_p95_ms"])
    logger.info("\nNext: run_stage6_dashboard.py")

if __name__ == "__main__":
    main()
