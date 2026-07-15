import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data.preprocessor import ReviewPreprocessor
from src.utils import load_config


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def preprocessor(cfg):
    return ReviewPreprocessor(cfg)


def test_clean_text_removes_html(preprocessor):
    result = preprocessor._clean_text("<b>Great product</b>")
    assert "<b>" not in result
    assert "Great product" in result


def test_clean_text_removes_urls(preprocessor):
    result = preprocessor._clean_text("Check https://amazon.com for details")
    assert "https://" not in result


def test_clean_text_removes_control_chars(preprocessor):
    result = preprocessor._clean_text("Good\x00product\x01here")
    assert "\x00" not in result
    assert "\x01" not in result


def test_clean_text_collapses_whitespace(preprocessor):
    result = preprocessor._clean_text("too   many    spaces")
    assert "  " not in result


def test_fit_transform_drops_short_reviews(preprocessor):
    df = pd.DataFrame({
        "text": ["hi", "a" * 100],
        "rating": [5, 4],
        "sentiment": ["positive", "positive"],
        "sentiment_label": [2, 2],
        "category": ["Books", "Books"],
    })
    result = preprocessor.fit_transform(df)
    assert len(result) == 1


def test_fit_transform_deduplicates(preprocessor):
    df = pd.DataFrame({
        "text": ["a" * 100, "a" * 100],
        "rating": [5, 5],
        "sentiment": ["positive", "positive"],
        "sentiment_label": [2, 2],
        "category": ["Books", "Books"],
    })
    result = preprocessor.fit_transform(df)
    assert len(result) == 1


def test_validate_returns_report(preprocessor):
    df = pd.DataFrame({
        "text": ["a" * 100, "b" * 100],
        "rating": [5, 1],
        "sentiment": ["positive", "negative"],
        "sentiment_label": [2, 0],
        "category": ["Books", "Electronics"],
    })
    report = preprocessor.validate(df)
    assert "total_rows" in report
    assert "sentiment_distribution" in report
    assert "text_length" in report
    assert report["total_rows"] == 2
