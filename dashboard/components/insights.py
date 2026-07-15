import streamlit as st
import json
from pathlib import Path


def render(cfg):
    st.title("📝 Insights & Reports")
    st.markdown("AI-generated insights from customer feedback analysis.")

    eval_dir = Path(cfg["data"]["evaluation_dir"])

    # Trends
    trends_path = eval_dir / "trends.json"
    if trends_path.exists():
        with open(trends_path) as f:
            trends = json.load(f)

        st.subheader("🔍 Detected Trends")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Top Trends**")
            for trend in trends.get("top_trends", []):
                st.markdown(f"• {trend}")

        with col2:
            st.markdown("**Opportunities**")
            for opp in trends.get("opportunities", []):
                st.markdown(f"✅ {opp}")

        with col3:
            st.markdown("**Risks**")
            for risk in trends.get("risks", []):
                st.markdown(f"⚠️ {risk}")

        st.markdown("---")
        st.subheader("Executive Summary")
        st.info(trends.get("executive_summary", "No summary available."))

    # Cluster summaries
    st.markdown("---")
    summaries_path = eval_dir / "cluster_summaries.json"
    if summaries_path.exists():
        with open(summaries_path) as f:
            summaries = json.load(f)

        st.subheader("📊 Cluster Insights")
        for summary in summaries[:5]:
            sentiment_label = summary.get("sentiment_label", "mixed")
            icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}.get(sentiment_label, "⚪")
            words = ", ".join(summary.get("topic_words", [])[:4])

            with st.expander(f"{icon} Topic {summary['topic_id']}: {words}"):
                st.markdown(f"**Summary:** {summary.get('summary', '')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Key Issues:**")
                    for issue in summary.get("key_issues", []):
                        st.markdown(f"• {issue}")
                with col2:
                    st.markdown("**Recommendations:**")
                    for rec in summary.get("recommendations", []):
                        st.markdown(f"• {rec}")

    # Weekly report
    st.markdown("---")
    st.subheader("📄 Weekly Report")
    report_path = eval_dir / "weekly_report.md"
    if report_path.exists():
        with open(report_path) as f:
            report = f.read()
        st.download_button(
            label="Download Report",
            data=report,
            file_name="weekly_feedback_report.md",
            mime="text/markdown",
        )
        st.markdown(report)
    else:
        st.warning("Weekly report not found. Run run_stage4_insights.py first.")
