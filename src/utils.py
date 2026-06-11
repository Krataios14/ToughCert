"""Utilities for training/evaluation.

torch is optional here: set_seed works without it, and the helpers
that genuinely need it import it lazily and fail with a clear message.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime

import numpy as np


def _import_torch(caller: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"{caller} requires torch, which is not installed. "
            "Install it with 'pip install torch' (CPU wheels: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu)."
        ) from exc
    return torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    torch = _import_torch("get_device")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_state_dict_flexible(path: str, device: torch.device) -> dict:
    torch = _import_torch("load_state_dict_flexible")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items() if k != "n_averaged"}
        if "n_averaged" in state:
            state = {k: v for k, v in state.items() if k != "n_averaged"}
    return state
