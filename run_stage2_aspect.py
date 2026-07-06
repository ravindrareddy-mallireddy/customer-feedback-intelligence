import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data.dataloader import build_dataloaders
from src.models.aspect import (
    AspectClassifier, AspectTrainer, evaluate_aspect, generate_aspect_labels
)
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage2_aspect", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 2 - Aspect Extractor (RoBERTa + LoRA)")
    logger.info("=" * 50)

    # Load data
    processed_dir = Path(cfg["data"]["processed_dir"])
    logger.info("Loading processed splits ...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")

    # Generate pseudo-labels from keywords
    aspects = cfg["models"]["aspect"]["aspects"]
    logger.info("Generating aspect labels via keyword matching ...")
    train_df = generate_aspect_labels(train_df, aspects)
    val_df   = generate_aspect_labels(val_df, aspects)
    test_df  = generate_aspect_labels(test_df, aspects)

    # Log aspect label distribution
    for aspect in aspects:
        pct = train_df[aspect].mean() * 100
        logger.info("  %-20s : %.1f%% positive", aspect, pct)

    # Build classifier
    classifier = AspectClassifier(cfg)
    classifier.build()

    # Build dataloaders
    logger.info("Building dataloaders ...")
    train_dl, val_dl, test_dl = build_dataloaders(
        train_df, val_df, test_df,
        classifier.tokenizer, cfg, task="aspect"
    )

    # Check for existing best model
    best_path = Path(cfg["models"]["aspect"]["save_dir"]) / "best"
    if best_path.exists():
        logger.info("Found existing best model at %s — loading.", best_path)
        classifier.load(best_path)
    else:
        logger.info("Starting training ...")
        trainer = AspectTrainer(classifier)
        history = trainer.train(train_dl, val_dl)
        logger.info("Training history: %s", {
            k: [round(v, 4) for v in vals] for k, vals in history.items()
        })
        classifier.load(best_path)

    # Evaluate
    logger.info("\nEvaluating on test set ...")
    metrics = evaluate_aspect(classifier, test_dl)

    # Save results
    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "aspect_eval.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation results saved -> %s", results_path)

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2 ASPECT COMPLETE")
    logger.info("=" * 50)
    logger.info("  F1 Micro    : %.4f", metrics["f1_micro"])
    logger.info("  F1 Macro    : %.4f", metrics["f1_macro"])
    logger.info("  Hamming Loss: %.4f", metrics["hamming_loss"])
    logger.info("\nPer-aspect F1:")
    for aspect, score in metrics["per_aspect_f1"].items():
        logger.info("  %-20s : %.4f", aspect, score)
    logger.info("\nNext: run_stage2_topics.py")

if __name__ == "__main__":
    main()
