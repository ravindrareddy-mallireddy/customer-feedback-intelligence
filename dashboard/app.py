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

@st.cache_resource
def load_models_only():
    from src.utils import load_config
    from src.models.sentiment import SentimentClassifier
    from src.models.aspect import AspectClassifier
    from huggingface_hub import snapshot_download

    cfg = load_config("config.yaml")

    with st.spinner("Downloading sentiment model..."):
        snapshot_download(repo_id="rr1371859/customer-feedback-sentiment",
            repo_type="model", local_dir="models/sentiment/best")

    with st.spinner("Downloading aspect model..."):
        snapshot_download(repo_id="rr1371859/customer-feedback-aspect",
            repo_type="model", local_dir="models/aspect/best")

    with st.spinner("Loading sentiment model..."):
        sentiment = SentimentClassifier(cfg)
        sentiment.build()
        sentiment.load("models/sentiment/best")

    with st.spinner("Loading aspect model..."):
        aspect = AspectClassifier(cfg)
        aspect.build()
        aspect.load("models/aspect/best")

    return cfg, sentiment, aspect

@st.cache_resource
def load_search():
    from src.utils import load_config
    from src.retrieval.embeddings import EmbeddingIndexer
    from src.retrieval.search import ReviewSearcher
    from src.retrieval.reranker import Reranker
    from huggingface_hub import snapshot_download

    cfg = load_config("config.yaml")

    with st.spinner("Downloading retrieval index..."):
        snapshot_download(repo_id="rr1371859/customer-feedback-topics",
            repo_type="model", local_dir="models/embeddings")

    indexer = EmbeddingIndexer(cfg)
    indexer.build()
    indexer.load()

    reranker = Reranker(cfg)
    reranker.build()

    searcher = None
    try:
        from src.retrieval.search import ReviewSearcher
        faiss_path = Path(cfg["retrieval"]["faiss_index_path"])
        if faiss_path.exists():
            searcher = ReviewSearcher(cfg)
            searcher.load_faiss(indexer.metadata)
    except Exception as e:
        st.warning(f"Search index not available: {e}")

    return cfg, indexer, searcher, reranker

st.sidebar.title("🧠 Customer Feedback Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["📊 Overview", "🔍 Search", "🏷️ Analyze Review", "📈 Topics", "📝 Insights"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("Sentiment Accuracy", "89.91%")
st.sidebar.metric("Retrieval MRR", "1.00")
st.sidebar.metric("Reviews Indexed", "18,188")

# Load only core models at startup
cfg, sentiment_clf, aspect_clf = load_models_only()

if page == "📊 Overview":
    from dashboard.components import overview
    overview.render(cfg)
elif page == "🔍 Search":
    # Load search only when needed
    cfg, indexer, searcher, reranker = load_search()
    from dashboard.components import search
    search.render(cfg, indexer, searcher, reranker)
elif page == "🏷️ Analyze Review":
    from dashboard.components import analyze
    analyze.render(cfg, sentiment_clf, aspect_clf)
elif page == "📈 Topics":
    from dashboard.components import topics
    topics.render(cfg, None)
elif page == "📝 Insights":
    from dashboard.components import insights
    insights.render(cfg)
