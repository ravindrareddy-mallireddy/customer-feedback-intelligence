import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml

def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_logger(name, config=None):
    cfg = (config or {}).get("logging", {})
    level_str = cfg.get("level", "INFO")
    fmt = cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log_dir = cfg.get("log_dir", "logs")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level_str))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "pipeline.log")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
