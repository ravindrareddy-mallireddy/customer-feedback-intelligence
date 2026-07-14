from __future__ import annotations
import time
import os
from groq import Groq
from src.utils import get_logger
from dotenv import load_dotenv
load_dotenv()


class ClusterSummariser:
    """Summarises topic clusters using Groq LLM."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.llm_cfg = config["llm"]
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def summarise_cluster(self, topic_id: int, topic_words: list[str],
                          sample_reviews: list[str], sentiment_dist: dict) -> dict:
        """Generate a business-readable summary for a topic cluster.

        Args:
            topic_id: Topic ID from BERTopic.
            topic_words: Top words for this topic.
            sample_reviews: Sample reviews from this cluster.
            sentiment_dist: Dict with positive/negative/neutral counts.

        Returns:
            Dict with summary, key_issues, and recommendations.
        """
        reviews_text = "\n".join([f"- {r[:200]}" for r in sample_reviews[:5]])
        total = sum(sentiment_dist.values())
        pos_pct = sentiment_dist.get("positive", 0) / max(total, 1) * 100
        neg_pct = sentiment_dist.get("negative", 0) / max(total, 1) * 100

        prompt = f"""You are a customer insights analyst. Analyze these Amazon product reviews from a specific topic cluster.

Topic keywords: {", ".join(topic_words[:8])}
Sentiment: {pos_pct:.0f}% positive, {neg_pct:.0f}% negative
Sample reviews:
{reviews_text}

Provide a JSON response with exactly these fields:
{{
  "summary": "2-3 sentence business summary of what customers are saying",
  "key_issues": ["issue 1", "issue 2", "issue 3"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "sentiment_label": "positive/negative/mixed"
}}

Respond with valid JSON only, no other text."""

        response = self._call_groq(prompt)
        try:
            import json
            result = json.loads(response)
        except:
            result = {
                "summary": response[:300],
                "key_issues": [],
                "recommendations": [],
                "sentiment_label": "mixed"
            }
        result["topic_id"] = topic_id
        result["topic_words"] = topic_words[:8]
        result["sentiment_distribution"] = sentiment_dist
        return result

    def _call_groq(self, prompt: str, retries: int = 3) -> str:
        """Call Groq API with retry logic."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.llm_cfg["max_tokens"],
                    temperature=self.llm_cfg["temperature"],
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning("Groq API error (attempt %d): %s", attempt + 1, e)
                if attempt < retries - 1:
                    time.sleep(self.llm_cfg["retry_delay"] * (attempt + 1))
        return ""
