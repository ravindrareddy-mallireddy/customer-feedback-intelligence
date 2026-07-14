import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.models.topics import TopicModeler, evaluate_topics
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage2_topics", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 2 - Topic Modelling (BERTopic)")
    logger.info("=" * 50)

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    texts = combined["text"].tolist()
    logger.info("Total texts for topic modelling: %d", len(texts))

    modeler = TopicModeler(cfg)
    save_dir = Path(cfg["models"]["topic"]["save_dir"])
    model_path = save_dir / "bertopic_model"

    if model_path.exists():
        logger.info("Found existing topic model - loading.")
        modeler.build()
        modeler.load()
        topics = modeler.topic_model.transform(texts)[0]
    else:
        modeler.build()
        topics, probs = modeler.fit(texts)
        modeler.save()

    topic_info = modeler.get_topic_info()
    logger.info("\nTop 10 topics:")
    for _, row in topic_info.head(11).iterrows():
        if row["Topic"] == -1:
            continue
        words = [w for w, _ in modeler.get_topic_words(row["Topic"], 5)]
        logger.info("  Topic %2d (n=%4d): %s", row["Topic"], row["Count"], ", ".join(words))

    metrics = evaluate_topics(modeler, texts, topics)

    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    topic_records = []
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        words = [w for w, _ in modeler.get_topic_words(row["Topic"], 10)]
        topic_records.append({
            "topic_id": int(row["Topic"]),
            "count": int(row["Count"]),
            "top_words": words,
        })
    with open(eval_dir / "topic_info.json", "w") as f:
        json.dump(topic_records, f, indent=2)
    with open(eval_dir / "topic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2 TOPICS COMPLETE")
    logger.info("=" * 50)
    logger.info("  Topics found   : %d", metrics["n_topics"])
    logger.info("  Outlier rate   : %.1f%%", metrics["outlier_rate"] * 100)
    logger.info("  Topic diversity: %.4f", metrics["topic_diversity"])
    logger.info("\nNext: run_stage3_retrieval.py")

if __name__ == "__main__":
    main()
