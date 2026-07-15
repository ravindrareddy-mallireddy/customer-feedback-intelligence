import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data.llm_labeler_batch import BatchLLMAspectLabeler
from src.data.dataloader import build_dataloaders
from src.models.aspect import AspectClassifier, AspectTrainer, evaluate_aspect, compute_pos_weights
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("relabel_full", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("FULL DATASET LLM ASPECT RELABELING")
    logger.info("=" * 50)

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")

    aspects = cfg["models"]["aspect"]["aspects"]

    # Label train with LLM
    train_labeled_path = processed_dir / "train_llm_batch_labeled.parquet"
    if train_labeled_path.exists():
        existing = pd.read_parquet(train_labeled_path)
        if len(existing) >= len(train_df):
            logger.info("Found complete LLM labels - loading.")
            train_labeled = existing
        else:
            logger.info("Found partial labels (%d/%d) - resuming.", len(existing), len(train_df))
            remaining = train_df.iloc[len(existing):].reset_index(drop=True)
            labeler = BatchLLMAspectLabeler(cfg, batch_size=5)
            new_labels = labeler.label_dataframe(remaining, save_path=None)
            train_labeled = pd.concat([existing, new_labels], ignore_index=True)
            train_labeled.to_parquet(train_labeled_path, index=False)
    else:
        logger.info("Starting batch LLM labeling ...")
        labeler = BatchLLMAspectLabeler(cfg, batch_size=5)
        train_labeled = labeler.label_dataframe(
            train_df, save_path=str(train_labeled_path)
        )

    # Label val and test with LLM too for consistency
    val_labeled_path = processed_dir / "val_llm_batch_labeled.parquet"
    if val_labeled_path.exists():
        val_labeled = pd.read_parquet(val_labeled_path)
    else:
        labeler = BatchLLMAspectLabeler(cfg, batch_size=5)
        val_labeled = labeler.label_dataframe(
            val_df, save_path=str(val_labeled_path)
        )

    test_labeled_path = processed_dir / "test_llm_batch_labeled.parquet"
    if test_labeled_path.exists():
        test_labeled = pd.read_parquet(test_labeled_path)
    else:
        labeler = BatchLLMAspectLabeler(cfg, batch_size=5)
        test_labeled = labeler.label_dataframe(
            test_df, save_path=str(test_labeled_path)
        )

    # Log aspect distribution
    logger.info("Aspect distribution (LLM labels):")
    for aspect in aspects:
        if aspect in train_labeled.columns:
            pct = train_labeled[aspect].mean() * 100
            logger.info("  %-20s : %.1f%% positive", aspect, pct)

    # Compute pos weights
    pos_weights = compute_pos_weights(train_labeled, aspects)
    logger.info("Pos weights: %s", [f"{w:.1f}" for w in pos_weights.tolist()])

    # Clear old model
    import shutil
    best_path = Path(cfg["models"]["aspect"]["save_dir"]) / "best"
    if best_path.exists():
        shutil.rmtree(best_path)

    # Train
    classifier = AspectClassifier(cfg)
    classifier.build()

    train_dl, val_dl, test_dl = build_dataloaders(
        train_labeled, val_labeled, test_labeled,
        classifier.tokenizer, cfg, task="aspect"
    )

    logger.info("Training with LLM labels ...")
    trainer = AspectTrainer(classifier, pos_weights=pos_weights)
    history = trainer.train(train_dl, val_dl)

    classifier.load(best_path)

    logger.info("\nEvaluating on test set ...")
    metrics = evaluate_aspect(classifier, test_dl)

    eval_dir = Path(cfg["data"]["evaluation_dir"])
    with open(eval_dir / "aspect_eval_llm_full.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("RELABELING COMPLETE")
    logger.info("=" * 50)
    logger.info("  F1 Micro    : %.4f", metrics["f1_micro"])
    logger.info("  F1 Macro    : %.4f", metrics["f1_macro"])
    logger.info("  Hamming Loss: %.4f", metrics["hamming_loss"])
    for aspect, score in metrics["per_aspect_f1"].items():
        logger.info("  %-20s : %.4f", aspect, score)

if __name__ == "__main__":
    main()
