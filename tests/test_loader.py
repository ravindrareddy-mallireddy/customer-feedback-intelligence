import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data.loader import ReviewDatasetLoader
from src.utils import load_config


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def loader(cfg):
    return ReviewDatasetLoader(cfg)


def test_sentiment_label_positive(loader):
    df = pd.DataFrame({"text": ["great"], "rating": [5]})
    result = loader._add_sentiment_label(df)
    assert result.iloc[0]["sentiment"] == "positive"


def test_sentiment_label_negative(loader):
    df = pd.DataFrame({"text": ["terrible"], "rating": [1]})
    result = loader._add_sentiment_label(df)
    assert result.iloc[0]["sentiment"] == "negative"


def test_sentiment_label_neutral(loader):
    df = pd.DataFrame({"text": ["okay"], "rating": [3]})
    result = loader._add_sentiment_label(df)
    assert result.iloc[0]["sentiment"] == "neutral"


def test_sentiment_label_drops_invalid(loader):
    df = pd.DataFrame({"text": ["test"], "rating": [None]})
    result = loader._add_sentiment_label(df)
    assert len(result) == 0


def test_sentiment_label_map_values(loader):
    df = pd.DataFrame({
        "text": ["good", "ok", "bad"],
        "rating": [5, 3, 1]
    })
    result = loader._add_sentiment_label(df)
    assert result[result["sentiment"] == "positive"]["sentiment_label"].values[0] == 2
    assert result[result["sentiment"] == "neutral"]["sentiment_label"].values[0] == 1
    assert result[result["sentiment"] == "negative"]["sentiment_label"].values[0] == 0
