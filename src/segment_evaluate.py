"""Evaluate per-segment models created by segment_train."""

import argparse
import json
import os

from src.evaluate import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_root", type=str, required=True)
    args = parser.parse_args()

    summary_path = os.path.join(args.segments_root, "segments_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    results = []
    for item in summary:
        run_dir = item["run_dir"]
        cfg_path = os.path.join(run_dir, "segment_config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        metrics = evaluate_from_config(cfg, run_dir)
        results.append({"segment": item["segment"], "run_dir": run_dir, **metrics})

    out_path = os.path.join(args.segments_root, "segments_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"segments_root": args.segments_root, "count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
