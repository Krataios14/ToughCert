"""Visualization utilities."""

import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.utils import ensure_dir


def _plot_learning_curves(run_dir: str, figures_dir: str) -> None:
    history_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_path):
        return
    history = pd.read_json(history_path, typ="series").to_dict()

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss or not val_loss:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "learning_curves.png"), dpi=200)
    plt.close()


def _plot_parity(run_dir: str, figures_dir: str) -> None:
    preds_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        return
    df = pd.read_csv(preds_path)

    plt.figure(figsize=(5, 5))
    plt.scatter(df["y_true"], df["y_pred"], s=20, alpha=0.7)
    lims = [min(df.min()), max(df.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "parity_plot.png"), dpi=200)
    plt.close()


def _plot_residuals(run_dir: str, figures_dir: str) -> None:
    preds_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        return
    df = pd.read_csv(preds_path)
    df["residual"] = df["y_true"] - df["y_pred"]

    residuals = df["residual"].dropna().to_numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.8)
    if len(residuals) > 1 and np.std(residuals) > 0:
        grid = np.linspace(residuals.min(), residuals.max(), 200)
        kde = stats.gaussian_kde(residuals)
        # Scale the density to the histogram's count axis
        bin_width = (residuals.max() - residuals.min()) / 30 or 1.0
        plt.plot(grid, kde(grid) * len(residuals) * bin_width, lw=1.5)
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residuals.png"), dpi=200)
    plt.close()

def _plot_permutation_importance(run_dir: str, figures_dir: str) -> None:
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return
    metrics = pd.read_json(metrics_path, typ="series").to_dict()
    importances = metrics.get("permutation_importance")
    if not importances:
        return
    df = pd.DataFrame(importances).head(20)

    plt.figure(figsize=(7, 5))
    plt.barh(df["feature"], df["mae_increase"])
    plt.gca().invert_yaxis()  # most important at the top
    plt.xlabel("MAE Increase (Permutation)")
    plt.ylabel("Feature")
    plt.title("Permutation Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "permutation_importance.png"), dpi=200)
    plt.close()


def visualize_from_config(cfg: Dict, run_dir: str) -> None:
    figures_dir = cfg["outputs"]["figures_dir"]
    ensure_dir(figures_dir)
    _plot_learning_curves(run_dir, figures_dir)
    _plot_parity(run_dir, figures_dir)
    _plot_residuals(run_dir, figures_dir)
    _plot_permutation_importance(run_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = cfg["outputs"]["run_dir"]
        run_dir = os.path.join(run_dir, sorted(os.listdir(run_dir))[-1])

    visualize_from_config(cfg, run_dir)


if __name__ == "__main__":
    main()
