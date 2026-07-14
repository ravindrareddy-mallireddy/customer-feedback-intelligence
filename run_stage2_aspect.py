import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data.dataloader import build_dataloaders
from src.models.aspect import (
    AspectClassifier, AspectTrainer, evaluate_aspect,
    generate_aspect_labels, compute_pos_weights
)
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("stage2_aspect", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("STAGE 2 - Aspect Extractor (DistilBERT + LoRA)")
    logger.info("=" * 50)

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")

    aspects = cfg["models"]["aspect"]["aspects"]
    logger.info("Generating aspect labels ...")
    train_df = generate_aspect_labels(train_df, aspects)
    val_df   = generate_aspect_labels(val_df, aspects)
    test_df  = generate_aspect_labels(test_df, aspects)

    for aspect in aspects:
        pct = train_df[aspect].mean() * 100
        logger.info("  %-20s : %.1f%% positive", aspect, pct)

    # Compute pos_weights to fix class imbalance
    pos_weights = compute_pos_weights(train_df, aspects)
    logger.info("Pos weights: %s", [f"{w:.1f}" for w in pos_weights.tolist()])

    # Delete old checkpoints to avoid loading incompatible weights
    import shutil
    for ckpt in Path(cfg["models"]["aspect"]["save_dir"]).glob("checkpoint_epoch*"):
        shutil.rmtree(ckpt)
    best_path = Path(cfg["models"]["aspect"]["save_dir"]) / "best"
    if best_path.exists():
        shutil.rmtree(best_path)
        logger.info("Cleared old checkpoints.")

    classifier = AspectClassifier(cfg)
    classifier.build()

    logger.info("Building dataloaders ...")
    train_dl, val_dl, test_dl = build_dataloaders(
        train_df, val_df, test_df,
        classifier.tokenizer, cfg, task="aspect"
    )

    logger.info("Starting training ...")
    trainer = AspectTrainer(classifier, pos_weights=pos_weights)
    history = trainer.train(train_dl, val_dl)

    best_path = Path(cfg["models"]["aspect"]["save_dir"]) / "best"
    classifier.load(best_path)

    logger.info("\nEvaluating on test set ...")
    metrics = evaluate_aspect(classifier, test_dl)

    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "aspect_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2 ASPECT COMPLETE")
    logger.info("=" * 50)
    logger.info("  F1 Micro    : %.4f", metrics["f1_micro"])
    logger.info("  F1 Macro    : %.4f", metrics["f1_macro"])
    logger.info("  Hamming Loss: %.4f", metrics["hamming_loss"])
    for aspect, score in metrics["per_aspect_f1"].items():
        logger.info("  %-20s : %.4f", aspect, score)
    logger.info("\nNext: run_stage2_topics.py")

if __name__ == "__main__":
    main()
