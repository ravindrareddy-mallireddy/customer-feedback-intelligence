import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.utils import load_config


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def test_sentiment_config_has_correct_classes(cfg):
    assert cfg["sentiment"]["num_classes"] == 3


def test_sentiment_label_map_has_all_classes(cfg):
    label_map = cfg["sentiment"]["label_map"]
    assert "positive" in label_map
    assert "negative" in label_map
    assert "neutral" in label_map


def test_sentiment_label_map_values_unique(cfg):
    label_map = cfg["sentiment"]["label_map"]
    values = list(label_map.values())
    assert len(values) == len(set(values))


def test_aspect_config_has_six_aspects(cfg):
    aspects = cfg["models"]["aspect"]["aspects"]
    assert len(aspects) == 6


def test_aspect_config_contains_expected_aspects(cfg):
    aspects = cfg["models"]["aspect"]["aspects"]
    expected = ["price", "quality", "delivery", "customer_service", "packaging", "usability"]
    for asp in expected:
        assert asp in aspects


def test_sentiment_model_config(cfg):
    model_cfg = cfg["models"]["sentiment"]
    assert model_cfg["base_model"] == "distilbert-base-uncased"
    assert model_cfg["epochs"] == 5
    assert model_cfg["batch_size"] == 16
    assert model_cfg["learning_rate"] == 2.0e-5


def test_aspect_model_config(cfg):
    model_cfg = cfg["models"]["aspect"]
    assert model_cfg["base_model"] == "distilbert-base-uncased"
    assert model_cfg["epochs"] == 5
    assert model_cfg["batch_size"] == 8


def test_lora_config_sentiment(cfg):
    lora = cfg["models"]["sentiment"]["lora"]
    assert lora["r"] == 8
    assert lora["lora_alpha"] == 16
    assert "q_lin" in lora["target_modules"]
    assert "v_lin" in lora["target_modules"]


def test_lora_config_aspect(cfg):
    lora = cfg["models"]["aspect"]["lora"]
    assert lora["r"] == 8
    assert lora["lora_alpha"] == 16
