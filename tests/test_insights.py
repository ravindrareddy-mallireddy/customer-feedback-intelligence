import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.models.aspect import generate_aspect_labels, compute_pos_weights
from src.utils import load_config


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def aspects(cfg):
    return cfg["models"]["aspect"]["aspects"]


def test_generate_labels_returns_all_aspects(aspects):
    df = pd.DataFrame({"text": ["good product"]})
    result = generate_aspect_labels(df, aspects)
    for aspect in aspects:
        assert aspect in result.columns


def test_generate_labels_quality_keywords(aspects):
    df = pd.DataFrame({"text": ["The build quality is very sturdy and durable"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["quality"] == 1


def test_generate_labels_customer_service_keywords(aspects):
    df = pd.DataFrame({"text": ["I contacted customer service for a refund"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["customer_service"] == 1


def test_generate_labels_packaging_keywords(aspects):
    df = pd.DataFrame({"text": ["The packaging was damaged and the box was crushed"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["packaging"] == 1


def test_generate_labels_usability_keywords(aspects):
    df = pd.DataFrame({"text": ["Very easy to use and simple to setup"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["usability"] == 1


def test_generate_labels_case_insensitive(aspects):
    df = pd.DataFrame({"text": ["FAST SHIPPING and QUICK DELIVERY"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["delivery"] == 1


def test_generate_labels_multiple_aspects(aspects):
    df = pd.DataFrame({"text": ["Great price and fast shipping, easy to use"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["price"] == 1
    assert result.iloc[0]["delivery"] == 1
    assert result.iloc[0]["usability"] == 1


def test_pos_weights_all_positive(aspects):
    import torch
    df = pd.DataFrame({"text": ["test"] * 100})
    for aspect in aspects:
        df[aspect] = 1
    weights = compute_pos_weights(df, aspects)
    for w in weights:
        assert float(w) == pytest.approx(0.0, abs=0.1)


def test_pos_weights_highly_imbalanced(aspects):
    import torch
    df = pd.DataFrame({"text": ["test"] * 100})
    for aspect in aspects:
        df[aspect] = 0
    df.loc[:4, "price"] = 1  # 5% positive
    weights = compute_pos_weights(df, aspects)
    price_idx = aspects.index("price")
    assert float(weights[price_idx]) == pytest.approx(19.0, rel=0.1)


def test_topic_config(cfg):
    topic_cfg = cfg["models"]["topic"]
    assert topic_cfg["embedding_model"] == "all-MiniLM-L6-v2"
    assert topic_cfg["num_topics"] == 20
    assert topic_cfg["min_topic_size"] == 10
