from __future__ import annotations
import json
import os
import time
from pathlib import Path
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from src.utils import get_logger, load_config

load_dotenv()


class LLMAspectLabeler:
    """Labels reviews with aspect tags using Groq LLM."""

    ASPECTS = ["price", "quality", "delivery", "customer_service", "packaging", "usability"]

    PROMPT_TEMPLATE = """You are labeling Amazon product reviews for aspect-based sentiment analysis.

Review: "{text}"

For each aspect, respond 1 if the review mentions it, 0 if not.
Respond ONLY with a valid JSON object, no other text:

{{
  "price": 0 or 1,
  "quality": 0 or 1,
  "delivery": 0 or 1,
  "customer_service": 0 or 1,
  "packaging": 0 or 1,
  "usability": 0 or 1
}}

Guidelines:
- price: mentions cost, value, expensive, cheap, worth it, money
- quality: mentions build, material, durability, broke, sturdy, flimsy
- delivery: mentions shipping, arrived, days, late, fast, transit
- customer_service: mentions support, refund, return, representative, contact
- packaging: mentions box, wrapped, packaging, damaged in shipping
- usability: mentions easy, difficult, setup, instructions, intuitive"""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_cfg = config["llm"]

    def label_review(self, text: str) -> dict | None:
        """Label a single review with aspect tags.

        Args:
            text: Review text.

        Returns:
            Dict with aspect binary labels or None if failed.
        """
        prompt = self.PROMPT_TEMPLATE.format(text=text[:300])

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                # Clean up response
                content = content.replace("```json", "").replace("```", "").strip()
                labels = json.loads(content)
                # Validate all aspects present
                if all(k in labels for k in self.ASPECTS):
                    return {k: int(bool(labels[k])) for k in self.ASPECTS}
            except Exception as e:
                self.logger.warning("Attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        return None

    def label_batch(self, df: pd.DataFrame, sample_size: int = 500,
                    save_path: str | None = None) -> pd.DataFrame:
        """Label a sample of reviews with LLM aspect tags.

        Args:
            df: DataFrame with 'text' column.
            sample_size: Number of reviews to label (default 500).
            save_path: Optional path to save labeled data.

        Returns:
            DataFrame with aspect label columns added.
        """
        # Sample stratified by sentiment if available
        if "sentiment" in df.columns:
            sample = df.groupby("sentiment", group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // df["sentiment"].nunique()),
                                   random_state=42)
            ).reset_index(drop=True)
        else:
            sample = df.sample(min(sample_size, len(df)), random_state=42)

        self.logger.info("Labeling %d reviews with LLM ...", len(sample))

        results = []
        failed = 0

        for i, row in sample.iterrows():
            if i % 50 == 0:
                self.logger.info("Progress: %d/%d (failed=%d)", i, len(sample), failed)

            labels = self.label_review(row["text"])

            if labels:
                row_dict = row.to_dict()
                row_dict.update(labels)
                results.append(row_dict)
            else:
                failed += 1
                # Fall back to keyword labels
                from src.models.aspect import generate_aspect_labels
                fallback = generate_aspect_labels(
                    pd.DataFrame([row]), self.ASPECTS
                )
                results.append(fallback.iloc[0].to_dict())

            # Rate limiting - 30 RPM = 2 seconds between calls
            time.sleep(2.1)

        labeled_df = pd.DataFrame(results)
        self.logger.info("Labeled %d reviews (%d failed, used keyword fallback)",
                         len(labeled_df), failed)

        if save_path:
            labeled_df.to_parquet(save_path, index=False)
            self.logger.info("Saved labeled data -> %s", save_path)

        return labeled_df
