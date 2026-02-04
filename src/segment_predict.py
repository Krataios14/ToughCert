"""Predict with segment models, falling back to a global model."""

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import inverse_transform_target, load_artifacts, transform_df
from src.gbdt import load_gbdt
from src.schema import apply_schema


def _load_features(run_dir: str, cfg: Dict):
    num_features = cfg["data"]["numerical_features"]
    cat_features = cfg["data"]["categorical_features"]
    features_path = os.path.join(run_dir, "features.json")
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        num_features = features.get("numerical_features", num_features)
        cat_features = features.get("categorical_features", cat_features)
    return num_features, cat_features


def _predict_with_model(run_dir: str, model_path: str, df: pd.DataFrame, cfg: Dict) -> np.ndarray:
    artifacts = load_artifacts(
        scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
        encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
    )

    num_features, cat_features = _load_features(run_dir, cfg)
    for col in num_features:
        if col not in df.columns:
            df[col] = float("nan")
    for col in cat_features:
        if col not in df.columns:
            df[col] = "UNK"

    x_num, x_num_mask, x_cat = transform_df(df, num_features, cat_features, artifacts)
    x = np.concatenate([x_num, x_num_mask, x_cat.astype(np.float32)], axis=1)

    model = load_gbdt(model_path)
    preds = model.predict(x).squeeze()
    preds = inverse_transform_target(preds, artifacts, cfg["data"].get("target_standardize", False))
    return preds


def _latest_run(root: str, prefix: str) -> str | None:
    if not os.path.exists(root):
        return None
    candidates = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.startswith(prefix)
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(root, candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--segments_root", type=str, default=None)
    parser.add_argument("--global_run", type=str, default=None)
    parser.add_argument("--out", type=str, default="predictions.csv")
    args = parser.parse_args()

    cfg = apply_schema(load_config(args.config))

    df = pd.read_csv(args.data)

    segments_root = args.segments_root or _latest_run("runs", "segments_")
    if segments_root is None:
        raise FileNotFoundError("No segments_ runs found. Provide --segments_root.")

    global_run = args.global_run or _latest_run("runs", "run_")
    if global_run is None:
        raise FileNotFoundError("No global run_ found. Provide --global_run.")

    summary_path = os.path.join(segments_root, "segments_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Build index from segment string to run_dir
    segment_map = {item["segment"]: item["run_dir"] for item in summary}
    seg_cols = cfg.get("data", {}).get("segment_columns", [])

    preds = np.zeros(len(df), dtype=np.float64)
    used_global = np.zeros(len(df), dtype=bool)

    # Precompute global predictions
    global_model_path = os.path.join(global_run, cfg["outputs"]["best_model"])
    global_preds = _predict_with_model(global_run, global_model_path, df.copy(), cfg)

    # Initialize with global
    preds[:] = global_preds
    used_global[:] = True

    # Override with segment models where available
    for col in seg_cols:
        if col not in df.columns:
            continue
        for value, idx in df.groupby(col).groups.items():
            seg_key = f"{col}={value}"
            run_dir = segment_map.get(seg_key)
            if not run_dir:
                continue
            model_path = os.path.join(run_dir, cfg["outputs"]["best_model"])
            seg_df = df.loc[idx].copy()
            seg_preds = _predict_with_model(run_dir, model_path, seg_df, cfg)
            preds[idx] = seg_preds
            used_global[idx] = False

    out_df = df.copy()
    out_df["fracture_toughness_pred"] = preds
    out_df["used_global_model"] = used_global
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
