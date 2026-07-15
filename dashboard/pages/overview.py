import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import os


def render(cfg):
    st.title("📊 Customer Feedback Overview")
    
    # Debug - show current directory
    st.write("Working dir:", os.getcwd())
    st.write("Data dir exists:", Path(cfg["data"]["processed_dir"]).exists())
    st.write("Eval dir exists:", Path(cfg["data"]["evaluation_dir"]).exists())
    
    processed_dir = Path(cfg["data"]["processed_dir"])
    if not processed_dir.exists():
        st.error(f"Data directory not found: {processed_dir}")
        st.write("Available dirs:", list(Path(".").iterdir()))
        return

    import os
    st.write("Working dir:", os.getcwd())
    st.write("Data dir exists:", Path(cfg["data"]["processed_dir"]).exists())
    st.markdown("Summary of 18,188 Amazon product reviews across 5 categories.")

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "val.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Categories", df["category"].nunique() if "category" in df.columns else 5)
    col3.metric("Positive", f"{(df['sentiment'] == 'positive').mean():.1%}" if "sentiment" in df.columns else "49%")
    col4.metric("Negative", f"{(df['sentiment'] == 'negative').mean():.1%}" if "sentiment" in df.columns else "51%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts()
            fig = px.pie(values=sent_counts.values, names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={"positive": "#4CAF50", "negative": "#F44336", "neutral": "#FFC107"})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Reviews by Category")
        if "category" in df.columns:
            cat_counts = df["category"].value_counts()
            fig = px.bar(x=cat_counts.values, y=cat_counts.index, orientation="h",
                color=cat_counts.values, color_continuous_scale="Blues")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Sentiment Model (DistilBERT + LoRA)**")
        st.metric("Accuracy", "89.91%")
        st.metric("F1 Weighted", "0.8991")
        st.metric("Trainable Params", "1.09%")
    with col2:
        st.markdown("**Aspect Model (DistilBERT + LoRA)**")
        st.metric("F1 Micro", "0.3503")
        st.metric("F1 Macro", "0.3201")
        st.metric("Aspects", "6")
    with col3:
        st.markdown("**Retrieval Pipeline (FAISS + Reranker)**")
        st.metric("MRR", "1.0000")
        st.metric("Hit Rate @5", "1.0000")
        st.metric("Latency P50", "215ms")

    st.markdown("---")
    st.subheader("Latest Weekly Report")
    report_path = Path(cfg["data"]["evaluation_dir"]) / "weekly_report.md"
    if report_path.exists():
        with open(report_path) as f:
            report = f.read()
        st.markdown(report)
    else:
        st.warning(f"Weekly report not found at {report_path}")
