import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data.llm_labeler import LLMAspectLabeler
from src.data.dataloader import build_dataloaders
from src.models.aspect import AspectClassifier, AspectTrainer, evaluate_aspect, compute_pos_weights
from src.utils import get_logger, load_config, set_seed

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("relabel_aspects", cfg)
    set_seed(cfg["project"]["random_seed"])

    logger.info("=" * 50)
    logger.info("ASPECT RELABELING WITH LLM")
    logger.info("=" * 50)

    # Load data
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")

    aspects = cfg["models"]["aspect"]["aspects"]

    # Check if labeled data already exists
    labeled_path = processed_dir / "train_llm_labeled.parquet"

    if labeled_path.exists():
        logger.info("Found existing LLM labels - loading.")
        train_labeled = pd.read_parquet(labeled_path)
    else:
        logger.info("Starting LLM labeling (500 reviews, ~18 min) ...")
        labeler = LLMAspectLabeler(cfg)
        train_labeled = labeler.label_batch(
            train_df,
            sample_size=500,
            save_path=str(labeled_path)
        )

    logger.info("LLM labeled dataset: %d rows", len(train_labeled))

    # Log aspect distribution
    for aspect in aspects:
        if aspect in train_labeled.columns:
            pct = train_labeled[aspect].mean() * 100
            logger.info("  %-20s : %.1f%% positive", aspect, pct)

    # Label val and test with keywords (LLM too slow for all)
    from src.models.aspect import generate_aspect_labels
    val_labeled   = generate_aspect_labels(val_df, aspects)
    test_labeled  = generate_aspect_labels(test_df, aspects)

    # Compute pos weights from LLM labels
    pos_weights = compute_pos_weights(train_labeled, aspects)
    logger.info("Pos weights: %s", [f"{w:.1f}" for w in pos_weights.tolist()])

    # Clear old model
    import shutil
    best_path = Path(cfg["models"]["aspect"]["save_dir"]) / "best"
    if best_path.exists():
        shutil.rmtree(best_path)
        logger.info("Cleared old aspect model.")

    # Build and train
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
    with open(eval_dir / "aspect_eval_llm.json", "w") as f:
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
