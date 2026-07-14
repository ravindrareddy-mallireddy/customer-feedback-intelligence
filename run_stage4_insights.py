import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.insights.summariser import ClusterSummariser
from src.insights.trends import TrendDetector
from src.insights.reports import ReportGenerator
from src.models.topics import TopicModeler
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage4_insights", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 4 - LLM Insight Generation")
    logger.info("=" * 50)

    # Load data
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info("Loaded %d reviews", len(combined))

    # Load topic model
    logger.info("Loading topic model ...")
    modeler = TopicModeler(cfg)
    modeler.build()
    modeler.load()

    # Get topic assignments
    logger.info("Getting topic assignments ...")
    topics, _ = modeler.topic_model.transform(combined["text"].tolist())
    combined["topic"] = topics

    # Get topic info
    topic_info = modeler.get_topic_info()

    # Step 1: Summarise each cluster
    logger.info("\nSummarising topic clusters ...")
    summariser = ClusterSummariser(cfg)
    cluster_summaries = []

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue

        # Get reviews for this topic
        topic_reviews = combined[combined["topic"] == topic_id]
        sample_reviews = topic_reviews["text"].tolist()[:10]

        # Get sentiment distribution
        if "sentiment" in topic_reviews.columns:
            sentiment_dist = topic_reviews["sentiment"].value_counts().to_dict()
        else:
            sentiment_dist = {"positive": 0, "negative": 0}

        # Get topic words
        topic_words = [w for w, _ in modeler.get_topic_words(topic_id, 8)]

        logger.info("Summarising Topic %d (%d reviews) ...", topic_id, len(topic_reviews))
        summary = summariser.summarise_cluster(
            topic_id=topic_id,
            topic_words=topic_words,
            sample_reviews=sample_reviews,
            sentiment_dist=sentiment_dist,
        )
        cluster_summaries.append(summary)

        # Rate limit
        import time
        time.sleep(2)

    # Step 2: Detect trends
    logger.info("\nDetecting trends across clusters ...")
    detector = TrendDetector(cfg)
    trends = detector.detect_trends(cluster_summaries)
    logger.info("Trends detected: %s", trends.get("top_trends", []))

    # Step 3: Generate weekly report
    logger.info("\nGenerating weekly report ...")
    reporter = ReportGenerator(cfg)

    total = len(combined)
    if "sentiment" in combined.columns:
        pos_pct = (combined["sentiment"] == "positive").mean() * 100
        neg_pct = (combined["sentiment"] == "negative").mean() * 100
    else:
        pos_pct = neg_pct = 0

    stats = {
        "total_reviews": total,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "n_topics": len(cluster_summaries),
    }

    report = reporter.generate_weekly_report(cluster_summaries, trends, stats)

    # Save everything
    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "cluster_summaries.json", "w") as f:
        json.dump(cluster_summaries, f, indent=2)
    logger.info("Cluster summaries saved.")

    with open(eval_dir / "trends.json", "w") as f:
        json.dump(trends, f, indent=2)
    logger.info("Trends saved.")

    with open(eval_dir / "weekly_report.md", "w") as f:
        f.write(report)
    logger.info("Weekly report saved.")

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 4 COMPLETE")
    logger.info("=" * 50)
    logger.info("  Clusters summarised : %d", len(cluster_summaries))
    logger.info("  Trends detected     : %d", len(trends.get("top_trends", [])))
    logger.info("  Report generated    : data/evaluation/weekly_report.md")
    logger.info("\nNext: run_stage5_evaluation.py")

if __name__ == "__main__":
    main()
