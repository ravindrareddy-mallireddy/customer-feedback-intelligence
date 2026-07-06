"""
run_stage1.py
-------------
End-to-end runner for Stage 1: Data Pipeline.

What this does:
  1. Downloads Amazon reviews from Hugging Face (streamed — no full dataset download)
  2. Samples up to 10,000 reviews per category × 5 categories = 50,000
  3. Cleans text (HTML, duplicates, length filter)
  4. Adds 3-class sentiment labels from star ratings
  5. Splits 80/10/10 with stratification
  6. Saves train.parquet / val.parquet / test.parquet
  7. Validates data quality
  8. Generates dataset_stats.png

Run:
    python run_stage1.py

Expected time:
    First run: 15-40 minutes (streaming download)
    Subsequent runs: ~30 seconds (reads from cache)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import ReviewDatasetLoader
from src.data.preprocessor import ReviewPreprocessor
from src.data.visualise import plot_dataset_stats
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    cfg = load_config("config.yaml")
    logger = get_logger("stage1", cfg)
    set_seed(cfg["project"]["random_seed"])

    # ── Step 1: Load (or resume from cache) ──────────────────────────────────
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("STAGE 1  —  Customer Feedback Data Pipeline")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    loader = ReviewDatasetLoader(cfg)
    processed_dir = Path(cfg["data"]["processed_dir"])

    if (processed_dir / "train.parquet").exists():
        logger.info("Found existing splits — loading from disk (skip download).")
        train, val, test = loader.load_from_disk()
    else:
        logger.info("No cached splits found — starting fresh download.")
        train, val, test = loader.run()

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    logger.info("\nPreprocessing splits …")
    pre = ReviewPreprocessor(cfg)
    train = pre.fit_transform(train)
    val   = pre.fit_transform(val)
    test  = pre.fit_transform(test)

    # Overwrite parquet files with clean versions
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = processed_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("  Saved clean %s → %s (%d rows)", name, path, len(df))

    # ── Step 3: Validate ─────────────────────────────────────────────────────
    logger.info("\nRunning data validation …")
    combined = pd.concat([train, val, test], ignore_index=True)
    report = pre.validate(combined)

    # Save validation report
    eval_dir = Path(cfg["data"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "stage1_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved → %s", report_path)

    # ── Step 4: Visualise ────────────────────────────────────────────────────
    logger.info("\nGenerating dataset statistics plot …")
    try:
        plot_path = plot_dataset_stats(combined, cfg)
        logger.info("Plot saved → %s", plot_path)
    except Exception as e:
        logger.warning("Could not generate plot: %s", e)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 50)
    logger.info("STAGE 1 COMPLETE")
    logger.info("═" * 50)
    logger.info("  Train : %6d rows", len(train))
    logger.info("  Val   : %6d rows", len(val))
    logger.info("  Test  : %6d rows", len(test))
    logger.info("  Total : %6d rows", len(combined))
    if "sentiment" in combined.columns:
        dist = combined["sentiment"].value_counts()
        for label, count in dist.items():
            logger.info("  %-10s : %6d  (%.1f%%)", label, count, count / len(combined) * 100)
    logger.info("")
    logger.info("Next: run_stage2_sentiment.py")


if __name__ == "__main__":
    main()