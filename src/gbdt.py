"""Gradient-boosted tree baseline for tabular regression."""

from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def train_gbdt(x_train: np.ndarray, y_train: np.ndarray, cfg: Dict) -> HistGradientBoostingRegressor:
    params = cfg.get("gbdt", {})
    model = HistGradientBoostingRegressor(
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", None),
        max_leaf_nodes=params.get("max_leaf_nodes", 63),
        max_bins=params.get("max_bins", 255),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("l2_regularization", 0.1),
        max_iter=params.get("max_iter", 500),
        early_stopping=params.get("early_stopping", True),
        random_state=cfg.get("seed", 42),
    )
    model.fit(x_train, y_train)
    return model


def save_gbdt(model: HistGradientBoostingRegressor, path: str) -> None:
    joblib.dump(model, path)


def load_gbdt(path: str) -> HistGradientBoostingRegressor:
    return joblib.load(path)
