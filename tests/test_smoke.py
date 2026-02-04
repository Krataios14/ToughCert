import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.train import train_from_config
from src.evaluate import evaluate_from_config


def test_smoke_train_and_eval(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "C": rng.normal(0.3, 0.05, n),
            "Mn": rng.normal(1.2, 0.2, n),
            "Cr": rng.normal(0.8, 0.1, n),
            "austenitizing_temp": rng.normal(900, 50, n),
            "tempering_temp": rng.normal(550, 30, n),
            "grain_size": rng.normal(10, 2, n),
            "yield_strength": rng.normal(900, 120, n),
            "steel_grade": rng.choice(["A", "B", "C"], n),
            "product_form": rng.choice(["plate", "bar"], n),
        }
    )
    df["fracture_toughness"] = (
        60
        + 0.1 * df["yield_strength"]
        - 0.5 * df["grain_size"]
        + rng.normal(0, 5, n)
    )

    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "seed": 42,
        "data": {
            "train_csv": str(csv_path),
            "target": "fracture_toughness",
            "numerical_features": [
                "C",
                "Mn",
                "Cr",
                "austenitizing_temp",
                "tempering_temp",
                "grain_size",
                "yield_strength",
            ],
            "categorical_features": ["steel_grade", "product_form"],
            "val_split": 0.2,
            "num_workers": 0,
        },
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "mlp_hidden": 128,
            "mlp_layers": 2,
        },
        "train": {
            "batch_size": 64,
            "max_epochs": 3,
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_epochs": 1,
            "patience": 5,
            "grad_clip": 1.0,
            "huber_delta": 1.0,
            "mixed_precision": False,
            "use_swa": False,
            "swa_start": 0.75,
            "mixup_alpha": 0.0,
        },
        "outputs": {
            "run_dir": str(tmp_path / "runs"),
            "best_model": "best_model.pt",
            "scaler": "scaler.joblib",
            "encoder": "encoder.joblib",
            "metrics": "metrics.json",
            "figures_dir": str(tmp_path / "figures"),
        },
    }

    run_dir = train_from_config(cfg)
    assert os.path.exists(os.path.join(run_dir, "best_model.pt"))

    metrics = evaluate_from_config(cfg, run_dir)
    assert "mae" in metrics
