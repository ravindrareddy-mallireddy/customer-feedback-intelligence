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


class BatchLLMAspectLabeler:
    """Labels reviews with aspect tags using Groq LLM in batches of 20."""

    ASPECTS = ["price", "quality", "delivery", "customer_service", "packaging", "usability"]

    def __init__(self, config, batch_size=20):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_cfg = config["llm"]
        self.batch_size = batch_size

    def _build_prompt(self, reviews: list[str]) -> str:
        reviews_text = ""
        for i, review in enumerate(reviews, 1):
            reviews_text += f"\nReview {i}: {review[:200]}"

        return f"""You are labeling Amazon product reviews for aspect-based sentiment analysis.

For each review, respond 1 if it mentions the aspect, 0 if not.

Aspects:
- price: cost, value, expensive, cheap, worth it, money, affordable
- quality: build, material, durability, broke, sturdy, flimsy, well made
- delivery: shipping, arrived, days, late, fast, transit, package
- customer_service: support, refund, return, representative, contact, help
- packaging: box, wrapped, packaging, damaged in shipping, bubble wrap
- usability: easy, difficult, setup, instructions, intuitive, user friendly

{reviews_text}

Respond ONLY with a valid JSON array, one object per review, no other text:
[
  {{"price": 0, "quality": 0, "delivery": 0, "customer_service": 0, "packaging": 0, "usability": 0}},
  ...
]"""

    def label_batch(self, reviews: list[str]) -> list[dict] | None:
        prompt = self._build_prompt(reviews)
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                labels = json.loads(content)
                if isinstance(labels, list) and len(labels) == len(reviews):
                    return [{k: int(bool(l.get(k, 0))) for k in self.ASPECTS} for l in labels]
            except Exception as e:
                self.logger.warning("Attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        return None

    def label_dataframe(self, df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
        self.logger.info("Labeling %d reviews in batches of %d ...", len(df), self.batch_size)
        all_labels = []
        failed_batches = 0
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            reviews = batch_df["text"].tolist()
            batch_num = i // self.batch_size + 1

            if batch_num % 10 == 0:
                self.logger.info("Progress: batch %d/%d (failed=%d)", 
                    batch_num, total_batches, failed_batches)

            labels = self.label_batch(reviews)

            if labels:
                all_labels.extend(labels)
            else:
                failed_batches += 1
                # Fallback to keyword labels
                from src.models.aspect import generate_aspect_labels
                fallback = generate_aspect_labels(batch_df, self.ASPECTS)
                for _, row in fallback.iterrows():
                    all_labels.append({k: int(row[k]) for k in self.ASPECTS})

            time.sleep(2.1)

        labels_df = pd.DataFrame(all_labels)
        result = pd.concat([df.reset_index(drop=True), labels_df], axis=1)

        self.logger.info("Done. %d reviews labeled (%d batches failed)", 
            len(result), failed_batches)

        if save_path:
            result.to_parquet(save_path, index=False)
            self.logger.info("Saved -> %s", save_path)

        return result
