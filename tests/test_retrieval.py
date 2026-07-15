import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.retrieval.reranker import Reranker
from src.utils import load_config


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def test_reranker_initializes(cfg):
    reranker = Reranker(cfg)
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.rerank_top_k == 10


def test_reranker_returns_empty_on_no_results(cfg):
    reranker = Reranker(cfg)
    result = reranker.rerank("test query", [])
    assert result == []


def test_reranker_sorts_by_score_when_disabled(cfg):
    reranker = Reranker(cfg)
    reranker.enabled = False
    results = [
        {"text": "bad result", "score": 0.1},
        {"text": "good result", "score": 0.9},
        {"text": "medium result", "score": 0.5},
    ]
    reranked = reranker.rerank("test query", results, top_k=3)
    assert reranked[0]["score"] == 0.9
    assert reranked[1]["score"] == 0.5
    assert reranked[2]["score"] == 0.1


def test_reranker_respects_top_k(cfg):
    reranker = Reranker(cfg)
    reranker.enabled = False
    results = [{"text": f"result {i}", "score": float(i)} for i in range(20)]
    reranked = reranker.rerank("test query", results, top_k=5)
    assert len(reranked) == 5


def test_reranker_top_k_uses_config_default(cfg):
    reranker = Reranker(cfg)
    reranker.enabled = False
    results = [{"text": f"result {i}", "score": float(i)} for i in range(20)]
    reranked = reranker.rerank("test query", results)
    assert len(reranked) == cfg["retrieval"]["rerank_top_k"]
