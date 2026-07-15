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


def test_generate_aspect_labels_price(aspects):
    df = pd.DataFrame({"text": ["This is very expensive and overpriced"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["price"] == 1


def test_generate_aspect_labels_delivery(aspects):
    df = pd.DataFrame({"text": ["Fast shipping and arrived early"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["delivery"] == 1


def test_generate_aspect_labels_no_match(aspects):
    df = pd.DataFrame({"text": ["I love this product so much"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["price"] == 0
    assert result.iloc[0]["delivery"] == 0


def test_generate_aspect_labels_multiple(aspects):
    df = pd.DataFrame({"text": ["Great price and fast shipping delivery"]})
    result = generate_aspect_labels(df, aspects)
    assert result.iloc[0]["price"] == 1
    assert result.iloc[0]["delivery"] == 1


def test_compute_pos_weights_shape(aspects):
    import torch
    df = pd.DataFrame({"text": ["test"] * 100})
    for aspect in aspects:
        df[aspect] = 0
    df.loc[:9, "price"] = 1
    weights = compute_pos_weights(df, aspects)
    assert isinstance(weights, torch.Tensor)
    assert len(weights) == len(aspects)
    assert weights[aspects.index("price")] == pytest.approx(9.0, rel=0.1)
