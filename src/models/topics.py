from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import get_logger


class TopicModeler:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.topic_cfg = config["models"]["topic"]
        self.save_dir = Path(self.topic_cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = None
        self.topic_model = None
        self.embeddings = None

    def build(self):
        model_name = self.topic_cfg["embedding_model"]
        self.logger.info("Loading embedding model: %s", model_name)
        self.embedding_model = SentenceTransformer(model_name)
        umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=self.config["project"]["random_seed"],
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.topic_cfg["min_topic_size"],
            metric="euclidean", cluster_selection_method="eom",
            prediction_data=True,
        )
        # Stopword removal for meaningful topic words
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=5,
            ngram_range=(1, 2),
        )
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            nr_topics=self.topic_cfg["num_topics"],
            top_n_words=10,
            verbose=True,
        )
        self.logger.info("BERTopic configured with stopword removal.")

    def fit(self, texts):
        self.logger.info("Generating embeddings for %d texts ...", len(texts))
        self.embeddings = self.embedding_model.encode(
            texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True,
        )
        self.logger.info("Embeddings shape: %s", str(self.embeddings.shape))
        self.logger.info("Fitting BERTopic ...")
        topics, probs = self.topic_model.fit_transform(texts, self.embeddings)
        self.logger.info("Found %d topics.", len(set(topics)) - 1)
        return topics, probs

    def get_topic_info(self):
        return self.topic_model.get_topic_info()

    def get_topic_words(self, topic_id, n=10):
        return self.topic_model.get_topic(topic_id)[:n]

    def save(self):
        model_path = self.save_dir / "bertopic_model"
        self.topic_model.save(str(model_path))
        self.logger.info("BERTopic model saved -> %s", model_path)
        if self.embeddings is not None:
            emb_path = self.save_dir / "embeddings.npy"
            np.save(emb_path, self.embeddings)
            self.logger.info("Embeddings saved -> %s", emb_path)

    def load(self):
        model_path = self.save_dir / "bertopic_model"
        self.topic_model = BERTopic.load(str(model_path))
        self.logger.info("BERTopic model loaded from %s", model_path)
        emb_path = self.save_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            self.logger.info("Embeddings loaded: %s", str(self.embeddings.shape))


def evaluate_topics(topic_modeler, texts, topics):
    logger = get_logger(__name__, topic_modeler.config)
    topic_info = topic_modeler.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    outlier_count = sum(1 for t in topics if t == -1)
    outlier_rate = outlier_count / len(topics)
    all_words = []
    unique_words = set()
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_modeler.get_topic_words(topic_id, 10)]
        all_words.extend(words)
        unique_words.update(words)
    diversity = len(unique_words) / max(len(all_words), 1)
    metrics = {
        "n_topics": n_topics,
        "outlier_rate": round(outlier_rate, 4),
        "topic_diversity": round(diversity, 4),
        "total_documents": len(texts),
        "outlier_count": outlier_count,
    }
    logger.info("Topics found    : %d", n_topics)
    logger.info("Outlier rate    : %.1f%%", outlier_rate * 100)
    logger.info("Topic diversity : %.4f", diversity)
    return metrics
