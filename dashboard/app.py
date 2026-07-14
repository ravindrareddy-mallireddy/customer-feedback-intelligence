import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load models once and cache
@st.cache_resource
def load_all_models():
    from src.utils import load_config
    from src.models.sentiment import SentimentClassifier
    from src.models.aspect import AspectClassifier
    from src.models.topics import TopicModeler
    from src.retrieval.embeddings import EmbeddingIndexer
    from src.retrieval.search import ReviewSearcher
    from src.retrieval.reranker import Reranker

    cfg = load_config("config.yaml")

    with st.spinner("Loading sentiment model..."):
        sentiment = SentimentClassifier(cfg)
        sentiment.build()
        sentiment.load("models/sentiment/best")

    with st.spinner("Loading aspect model..."):
        aspect = AspectClassifier(cfg)
        aspect.build()
        aspect.load("models/aspect/best")

    with st.spinner("Loading topic model..."):
        topics = TopicModeler(cfg)
        topics.build()
        topics.load()

    with st.spinner("Loading retrieval pipeline..."):
        indexer = EmbeddingIndexer(cfg)
        indexer.build()
        indexer.load()
        searcher = ReviewSearcher(cfg)
        searcher.load_faiss(indexer.metadata)
        searcher.load_chroma()
        reranker = Reranker(cfg)
        reranker.build()

    return cfg, sentiment, aspect, topics, indexer, searcher, reranker

# Sidebar
st.sidebar.title("🧠 Customer Feedback Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Search", "🏷️ Analyze Review", "📈 Topics", "📝 Insights"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("Sentiment Accuracy", "89.91%")
st.sidebar.metric("Retrieval MRR", "1.00")
st.sidebar.metric("Reviews Indexed", "18,188")

# Load models
models = load_all_models()
cfg, sentiment_clf, aspect_clf, topic_modeler, indexer, searcher, reranker = models

# Route pages
if page == "📊 Overview":
    from dashboard.pages import overview
    overview.render(cfg)
elif page == "🔍 Search":
    from dashboard.pages import search
    search.render(cfg, indexer, searcher, reranker)
elif page == "🏷️ Analyze Review":
    from dashboard.pages import analyze
    analyze.render(cfg, sentiment_clf, aspect_clf)
elif page == "📈 Topics":
    from dashboard.pages import topics
    topics.render(cfg, topic_modeler)
elif page == "📝 Insights":
    from dashboard.pages import insights
    insights.render(cfg)
