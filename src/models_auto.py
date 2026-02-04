"""Auto model selection over several tabular regressors."""

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.gbdt import train_gbdt


def _fit_candidate(name: str, x_train: np.ndarray, y_train: np.ndarray, cfg: Dict):
    if name == "gbdt":
        return train_gbdt(x_train, y_train, cfg)
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
        ).fit(x_train, y_train)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
        ).fit(x_train, y_train)
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=cfg.get("seed", 42)).fit(x_train, y_train)
    raise ValueError(f"Unknown candidate: {name}")


def train_auto(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict,
) -> Tuple[object, str, float]:
    candidates = cfg.get("model", {}).get("auto", {}).get("candidates", [])
    if not candidates:
        candidates = ["gbdt", "extra_trees", "random_forest", "ridge"]

    best_name = None
    best_model = None
    best_mae = float("inf")

    for name in candidates:
        model = _fit_candidate(name, x_train, y_train, cfg)
        preds = model.predict(x_val).squeeze()
        mae = float(mean_absolute_error(y_val, preds))
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = model

    return best_model, best_name, best_mae
