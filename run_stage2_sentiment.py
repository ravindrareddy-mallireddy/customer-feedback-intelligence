import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data.dataloader import build_dataloaders
from src.models.sentiment import SentimentClassifier, SentimentTrainer, evaluate_sentiment
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage2_sentiment", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 2 - Sentiment Classifier (DistilBERT + LoRA)")
    logger.info("=" * 50)

    # Load data
    processed_dir = Path(cfg["data"]["processed_dir"])
    logger.info("Loading processed splits ...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")
    logger.info("Train=%d  Val=%d  Test=%d", len(train_df), len(val_df), len(test_df))

    # Build classifier and tokenizer
    classifier = SentimentClassifier(cfg)
    classifier.build()

    # Build dataloaders
    logger.info("Building dataloaders ...")
    train_dl, val_dl, test_dl = build_dataloaders(
        train_df, val_df, test_df,
        classifier.tokenizer, cfg, task="sentiment"
    )

    # Check if best model already exists
    best_path = Path(cfg["models"]["sentiment"]["save_dir"]) / "best"
    if best_path.exists():
        logger.info("Found existing best model at %s — loading.", best_path)
        classifier.load(best_path)
    else:
        # Train
        logger.info("Starting training ...")
        trainer = SentimentTrainer(classifier)
        history = trainer.train(train_dl, val_dl)
        logger.info("Training history: %s", {
            k: [round(v, 4) for v in vals] for k, vals in history.items()
        })
        # Load best checkpoint for evaluation
        classifier.load(best_path)

    # Evaluate on test set
    logger.info("\nEvaluating on test set ...")
    metrics = evaluate_sentiment(classifier, test_dl)

    # Save evaluation results
    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "sentiment_eval.json"
    save_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    logger.info("Evaluation results saved -> %s", results_path)

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2 SENTIMENT COMPLETE")
    logger.info("=" * 50)
    logger.info("  Accuracy    : %.4f", metrics["accuracy"])
    logger.info("  F1 Weighted : %.4f", metrics["f1_weighted"])
    logger.info("  F1 Macro    : %.4f", metrics["f1_macro"])
    logger.info("\nNext: run_stage2_aspect.py")

if __name__ == "__main__":
    main()
