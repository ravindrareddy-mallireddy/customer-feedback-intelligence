import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.retrieval.embeddings import EmbeddingIndexer
from src.retrieval.search import ReviewSearcher
from src.retrieval.reranker import Reranker
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage3_retrieval", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 3 - RAG Retrieval Pipeline")
    logger.info("=" * 50)

    # Load all reviews
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info("Total reviews to index: %d", len(combined))

    # Step 1: Generate embeddings
    indexer = EmbeddingIndexer(cfg)
    indexer.build()

    emb_path = Path(cfg["models"]["topic"]["save_dir"]) / "retrieval_embeddings.npy"
    if emb_path.exists():
        logger.info("Found existing embeddings - loading from disk.")
        indexer.load()
    else:
        logger.info("Generating embeddings ...")
        indexer.fit(combined)
        indexer.save()

    # Step 2: Build FAISS index
    searcher = ReviewSearcher(cfg)
    faiss_path = Path(cfg["retrieval"]["faiss_index_path"])

    if faiss_path.exists():
        logger.info("Found existing FAISS index - loading.")
        searcher.load_faiss(indexer.metadata)
    else:
        logger.info("Building FAISS index ...")
        searcher.build_faiss(indexer.embeddings, indexer.metadata)

    # Step 3: Build ChromaDB index
    chroma_dir = Path(cfg["retrieval"]["chroma_persist_dir"])
    if chroma_dir.exists():
        logger.info("Found existing ChromaDB - loading.")
        searcher.load_chroma()
    else:
        logger.info("Building ChromaDB collection ...")
        searcher.build_chroma(indexer.embeddings, indexer.metadata, indexer.texts)

    # Step 4: Build reranker
    reranker = Reranker(cfg)
    reranker.build()

    # Step 5: Test pipeline with sample queries
    test_queries = [
        "battery life is terrible",
        "great value for money",
        "fast delivery but poor packaging",
        "customer service was unhelpful",
        "easy to use and setup",
    ]

    logger.info("\nTesting retrieval pipeline with sample queries ...")
    results_log = []

    for query in test_queries:
        logger.info("\nQuery: '%s'", query)

        # Encode query
        query_emb = indexer.encode([query])[0]

        # FAISS search
        faiss_results = searcher.search_faiss(query_emb, top_k=20)

        # Rerank
        reranked = reranker.rerank(query, faiss_results, top_k=5)

        logger.info("Top 3 results after reranking:")
        for i, r in enumerate(reranked[:3], 1):
            logger.info("  %d. [score=%.4f] %s...", i, r["rerank_score"], r["text"][:100])

        results_log.append({
            "query": query,
            "top_results": [
                {"text": r["text"][:200], "rerank_score": r["rerank_score"]}
                for r in reranked[:3]
            ]
        })

    # Save results
    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "retrieval_test_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    logger.info("\nRetrieval test results saved.")

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 3 COMPLETE")
    logger.info("=" * 50)
    logger.info("  Reviews indexed : %d", len(combined))
    logger.info("  FAISS vectors   : %d", searcher.faiss_index.ntotal)
    logger.info("  ChromaDB docs   : %d", searcher.chroma_col.count())
    logger.info("\nNext: run_stage4_insights.py")

if __name__ == "__main__":
    main()
