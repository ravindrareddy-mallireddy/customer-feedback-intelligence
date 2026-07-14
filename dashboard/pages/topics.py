import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path


def render(cfg, topic_modeler=None):
    st.title("📈 Topic Explorer")
    st.markdown("Explore the 19 topic clusters discovered across 18,188 reviews.")

    eval_dir = Path(cfg["data"]["evaluation_dir"])
    topic_info_path = eval_dir / "topic_info.json"

    if not topic_info_path.exists():
        st.error("Topic info not found.")
        return

    with open(topic_info_path) as f:
        topic_records = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Topics Found", len(topic_records))
    col2.metric("Topic Diversity", "0.9579")
    col3.metric("Outlier Rate", "22.2%")

    st.markdown("---")
    st.subheader("Topic Sizes")
    topic_ids = [f"Topic {r['topic_id']}" for r in topic_records]
    topic_counts = [r["count"] for r in topic_records]
    topic_labels = [", ".join(r["top_words"][:3]) for r in topic_records]

    fig = px.bar(x=topic_ids, y=topic_counts, hover_data=[topic_labels],
        labels={"x": "Topic", "y": "Number of Reviews", "hover_data_0": "Keywords"},
        color=topic_counts, color_continuous_scale="Blues")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Topic Details")
    selected_topic = st.selectbox("Select a topic",
        options=[f"Topic {r['topic_id']}: {', '.join(r['top_words'][:4])}" for r in topic_records])

    topic_idx = int(selected_topic.split(":")[0].replace("Topic ", ""))
    selected = next(r for r in topic_records if r["topic_id"] == topic_idx)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Reviews in cluster:** {selected['count']}")
        st.markdown("**Top keywords:**")
        for i, word in enumerate(selected["top_words"], 1):
            st.markdown(f"{i}. {word}")
    with col2:
        words = selected["top_words"][:8]
        scores = list(range(len(words), 0, -1))
        fig = go.Figure(go.Bar(x=scores, y=words, orientation="h", marker_color="#5C6BC0"))
        fig.update_layout(title="Keyword Importance", height=300)
        st.plotly_chart(fig, use_container_width=True)

    summaries_path = eval_dir / "cluster_summaries.json"
    if summaries_path.exists():
        with open(summaries_path) as f:
            summaries = json.load(f)
        summary = next((s for s in summaries if s["topic_id"] == topic_idx), None)
        if summary:
            st.markdown("---")
            st.subheader("AI-Generated Summary")
            st.info(summary.get("summary", ""))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Issues:**")
                for issue in summary.get("key_issues", []):
                    st.markdown(f"• {issue}")
            with col2:
                st.markdown("**Recommendations:**")
                for rec in summary.get("recommendations", []):
                    st.markdown(f"• {rec}")
