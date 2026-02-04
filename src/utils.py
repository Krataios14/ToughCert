"""Utilities for training/evaluation."""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_state_dict_flexible(path: str, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items() if k != "n_averaged"}
        if "n_averaged" in state:
            state = {k: v for k, v in state.items() if k != "n_averaged"}
    return state
