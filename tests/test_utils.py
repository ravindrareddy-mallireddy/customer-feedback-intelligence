import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.utils import load_config, set_seed, get_device


def test_config_loads():
    cfg = load_config("config.yaml")
    assert cfg is not None
    assert "data" in cfg
    assert "models" in cfg
    assert "sentiment" in cfg
    assert "retrieval" in cfg
    assert "llm" in cfg


def test_config_has_required_keys():
    cfg = load_config("config.yaml")
    assert cfg["data"]["target_reviews"] == 50000
    assert cfg["sentiment"]["num_classes"] == 3
    assert len(cfg["models"]["aspect"]["aspects"]) == 6
    assert cfg["models"]["sentiment"]["base_model"] == "distilbert-base-uncased"


def test_set_seed_reproducible():
    import numpy as np
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert list(a) == list(b)


def test_get_device_returns_device():
    import torch
    device = get_device()
    assert isinstance(device, torch.device)
    assert str(device) in ["cpu", "cuda", "mps"]


def test_config_sentiment_labels():
    cfg = load_config("config.yaml")
    label_map = cfg["sentiment"]["label_map"]
    assert "positive" in label_map
    assert "negative" in label_map
    assert "neutral" in label_map
