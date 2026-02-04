"""Random search tuning for GBDT."""

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import load_config
from src.data import inverse_transform_target, load_data
from src.gbdt import train_gbdt
from src.schema import apply_schema
from src.utils import ensure_dir, save_json, timestamp


def _sample_params(rng: random.Random) -> Dict:
    return {
        "learning_rate": rng.choice([0.02, 0.03, 0.05, 0.07, 0.1]),
        "max_depth": rng.choice([3, 4, 6, 8, None]),
        "max_leaf_nodes": rng.choice([31, 63, 127]),
        "min_samples_leaf": rng.choice([5, 10, 20, 40]),
        "l2_regularization": rng.choice([0.0, 0.1, 0.5, 1.0]),
        "max_iter": rng.choice([300, 500, 800, 1200]),
        "early_stopping": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    cfg = apply_schema(load_config(args.config))
    rng = random.Random(cfg.get("seed", 42))

    train_ds, val_ds, artifacts, num_features, cat_features = load_data(
        csv_path=cfg["data"]["train_csv"],
        target=cfg["data"]["target"],
        num_features=cfg["data"]["numerical_features"],
        cat_features=cfg["data"]["categorical_features"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
        target_transform=cfg["data"].get("target_transform", "none"),
        target_standardize=cfg["data"].get("target_standardize", False),
        group_columns=cfg["data"].get("group_columns"),
        min_non_null_fraction=cfg["data"].get("min_non_null_fraction"),
    )

    x_train = np.concatenate(
        [train_ds.x_num.numpy(), train_ds.x_num_mask.numpy(), train_ds.x_cat.numpy().astype(np.float32)],
        axis=1,
    )
    x_val = np.concatenate(
        [val_ds.x_num.numpy(), val_ds.x_num_mask.numpy(), val_ds.x_cat.numpy().astype(np.float32)],
        axis=1,
    )
    y_train = train_ds.y.numpy().squeeze()
    y_val = val_ds.y.numpy().squeeze()

    best = {"mae": float("inf"), "params": None, "r2": None, "rmse": None}
    results = []

    for _ in range(args.trials):
        params = _sample_params(rng)
        cfg["model"]["gbdt"] = params
        model = train_gbdt(x_train, y_train, cfg)
        preds = model.predict(x_val).squeeze()

        preds = inverse_transform_target(preds, artifacts, cfg["data"].get("target_standardize", False))
        targets = inverse_transform_target(y_val, artifacts, cfg["data"].get("target_standardize", False))

        mae = float(mean_absolute_error(targets, preds))
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        r2 = float(r2_score(targets, preds))
        results.append({"mae": mae, "rmse": rmse, "r2": r2, "params": params})

        if mae < best["mae"]:
            best = {"mae": mae, "rmse": rmse, "r2": r2, "params": params}

    out_dir = os.path.join("runs", f"tune_{timestamp()}")
    ensure_dir(out_dir)
    save_json({"best": best, "results": results}, os.path.join(out_dir, "tuning_results.json"))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
