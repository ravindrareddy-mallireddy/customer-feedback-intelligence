from __future__ import annotations
import os
import time
import json
from datetime import datetime
from groq import Groq
from src.utils import get_logger
from dotenv import load_dotenv
load_dotenv()


class ReportGenerator:
    """Generates weekly insight reports using LLM."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.llm_cfg = config["llm"]
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_weekly_report(self, cluster_summaries: list[dict],
                                trends: dict, stats: dict) -> str:
        """Generate a formatted weekly insight report.

        Args:
            cluster_summaries: List of cluster summary dicts.
            trends: Trend detection output dict.
            stats: Basic stats dict (total_reviews, sentiment_dist etc).

        Returns:
            Formatted markdown report string.
        """
        top_clusters = cluster_summaries[:5]
        clusters_text = ""
        for c in top_clusters:
            clusters_text += f"\n- Topic {c['topic_id']} ({', '.join(c.get('topic_words', [])[:4])}): {c.get('summary', '')[:150]}"

        prompt = f"""You are a customer insights analyst writing a weekly report for a product team.

DATA SUMMARY:
- Total reviews analyzed: {stats.get('total_reviews', 0)}
- Positive: {stats.get('positive_pct', 0):.0f}%
- Negative: {stats.get('negative_pct', 0):.0f}%
- Topics identified: {stats.get('n_topics', 0)}

TOP TOPIC CLUSTERS:
{clusters_text}

TRENDS DETECTED:
- Top trends: {', '.join(trends.get('top_trends', [])[:3])}
- Opportunities: {', '.join(trends.get('opportunities', [])[:2])}
- Risks: {', '.join(trends.get('risks', [])[:2])}

Write a professional weekly customer feedback report in markdown format with these sections:
1. Executive Summary
2. Key Findings (3-5 bullet points)
3. Top Issues by Category
4. Recommendations (3-5 actionable items)
5. Next Steps

Keep it concise and actionable. Use markdown formatting."""

        report = self._call_groq(prompt)
        date_str = datetime.now().strftime("%Y-%m-%d")
        final_report = f"# Weekly Customer Feedback Report\n**Generated:** {date_str}\n\n{report}"
        return final_report

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
