import os
import numpy as np
import pandas as pd

from src.predict import main as predict_main


def test_predict_outputs_file(tmp_path, monkeypatch):
    # Build a tiny fake run directory with artifacts and model
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Minimal config
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
seed: 0

data:
  train_csv: data/train.csv
  target: fracture_toughness
  numerical_features: [C]
  categorical_features: [steel_grade]
  val_split: 0.2
  num_workers: 0

model:
  d_model: 8
  n_heads: 2
  n_layers: 1
  dropout: 0.1
  mlp_hidden: 16
  mlp_layers: 2

train:
  batch_size: 8
  max_epochs: 1
  lr: 0.001
  weight_decay: 0.0
  warmup_epochs: 0
  patience: 1
  grad_clip: 1.0
  huber_delta: 1.0
  mixed_precision: false
  use_swa: false
  swa_start: 0.75
  mixup_alpha: 0.0

outputs:
  run_dir: runs
  best_model: best_model.pt
  scaler: scaler.joblib
  encoder: encoder.joblib
  metrics: metrics.json
  figures_dir: reports/figures
""",
        encoding="ascii",
    )

    # Create artifacts and model via a tiny training run
    from src.train import train_from_config

    df = pd.DataFrame(
        {
            "C": np.random.normal(0.3, 0.05, 30),
            "steel_grade": np.random.choice(["A", "B"], 30),
            "fracture_toughness": np.random.normal(100, 5, 30),
        }
    )
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    # Patch config to point at this CSV and run dir
    cfg_text = cfg_path.read_text(encoding="ascii")
    cfg_text = cfg_text.replace("data/train.csv", str(train_csv))
    cfg_path.write_text(cfg_text, encoding="ascii")

    # Override run_dir output
    cfg_text = cfg_path.read_text(encoding="ascii")
    cfg_text = cfg_text.replace("run_dir: runs", f"run_dir: {run_dir.as_posix()}")
    cfg_path.write_text(cfg_text, encoding="ascii")

    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="ascii"))
    run_path = train_from_config(cfg)

    # Prepare inference data
    infer_df = pd.DataFrame({"C": [0.25, 0.35], "steel_grade": ["A", "B"]})
    infer_csv = tmp_path / "infer.csv"
    infer_df.to_csv(infer_csv, index=False)

    out_csv = tmp_path / "preds.csv"

    # Invoke CLI entrypoint
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--model",
            os.path.join(run_path, "best_model.pt"),
            "--data",
            str(infer_csv),
            "--config",
            str(cfg_path),
            "--out",
            str(out_csv),
        ],
    )

    predict_main()
    assert out_csv.exists()
