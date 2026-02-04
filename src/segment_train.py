"""Train per-segment models (e.g., by phase or material condition)."""

import argparse
import json
import os
from typing import Dict, List

import pandas as pd

from src.config import load_config
from src.schema import apply_schema
from src.train import train_from_config
from src.utils import ensure_dir, timestamp


def _safe_name(value: str) -> str:
    return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in value])[:80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--segments", type=str, nargs="*", default=None)
    args = parser.parse_args()

    cfg = apply_schema(load_config(args.config))
    seg_cols: List[str] = args.segments or cfg.get("data", {}).get("segment_columns", [])
    min_rows = int(cfg.get("data", {}).get("min_segment_rows", 30))

    df = pd.read_csv(cfg["data"]["train_csv"])
    out_root = os.path.join("runs", f"segments_{timestamp()}")
    ensure_dir(out_root)

    summary = []

    for col in seg_cols:
        if col not in df.columns:
            continue
        for value, seg_df in df.groupby(col, dropna=False):
            value_name = _safe_name(str(value)) if value == value else "UNK"
            if len(seg_df) < min_rows:
                continue

            seg_dir = os.path.join(out_root, col, value_name)
            ensure_dir(seg_dir)

            seg_csv = os.path.join(seg_dir, "train_segment.csv")
            seg_df.to_csv(seg_csv, index=False)

            seg_cfg = json.loads(json.dumps(cfg))
            seg_cfg["data"]["train_csv"] = seg_csv

            run_dir = train_from_config(seg_cfg)
            seg_cfg_path = os.path.join(run_dir, "segment_config.json")
            with open(seg_cfg_path, "w", encoding="utf-8") as f:
                json.dump(seg_cfg, f, indent=2)

            summary.append({"segment": f"{col}={value_name}", "run_dir": run_dir, "rows": len(seg_df)})

    with open(os.path.join(out_root, "segments_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"segments_root": out_root, "count": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
