from __future__ import annotations
import os
import time
import json
from groq import Groq
from src.utils import get_logger
from dotenv import load_dotenv
load_dotenv()


class TrendDetector:
    """Detects trends and patterns across topic clusters using LLM."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.llm_cfg = config["llm"]
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def detect_trends(self, cluster_summaries: list[dict]) -> dict:
        """Detect cross-cluster trends from cluster summaries.

        Args:
            cluster_summaries: List of dicts from ClusterSummariser.

        Returns:
            Dict with trends, opportunities, and risks.
        """
        summaries_text = ""
        for s in cluster_summaries[:10]:
            summaries_text += f"\nTopic {s['topic_id']} ({s.get('sentiment_label', 'mixed')}): {s.get('summary', '')}"

        prompt = f"""You are a senior product analyst. Based on these customer feedback cluster summaries, identify key trends.

{summaries_text}

Provide a JSON response with exactly these fields:
{{
  "top_trends": ["trend 1", "trend 2", "trend 3"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "risks": ["risk 1", "risk 2"],
  "overall_sentiment": "positive/negative/mixed",
  "executive_summary": "3-4 sentence executive summary for leadership"
}}

Respond with valid JSON only, no other text."""

        response = self._call_groq(prompt)
        try:
            result = json.loads(response)
        except:
            result = {
                "top_trends": [],
                "opportunities": [],
                "risks": [],
                "overall_sentiment": "mixed",
                "executive_summary": response[:500]
            }
        return result

    def _call_groq(self, prompt: str, retries: int = 3) -> str:
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
