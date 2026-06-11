from pathlib import Path

import numpy as np
import pandas as pd

from src.certify import certify_dataframe, load_run
from src.report import render_report


def test_qualify_writes_artifact(pipeline):
    _, run_dir, _ = pipeline
    run_dir = Path(run_dir)
    for name in [
        "conformal_model.joblib",
        "trust.joblib",
        "scaler.joblib",
        "encoder.joblib",
        "features.json",
        "model_card.json",
    ]:
        assert (run_dir / name).exists(), name


def test_model_card_contents(pipeline):
    _, run_dir, _ = pipeline
    run = load_run(run_dir)
    card = run["model_card"]
    assert card["training_data"]["n_specimens"] > 100
    assert card["training_data"]["n_groups"] > 10
    assert card["model_selection"]["selected"] in {"ridge", "gbdt"}
    ev = card["calibration_evidence"]["alpha_0.10"]
    # Finite-sample guarantee is >= 0.8; allow slack for small eval splits
    assert ev["empirical_coverage"] >= 0.7
    assert ev["mean_interval_width"] > 0


def test_certify_outputs(pipeline):
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    out, blocks = certify_dataframe(unseen_df, run)
    assert len(out) == len(unseen_df)
    for col in [
        "predicted_toughness_mpa_m0_5",
        "lower_90",
        "upper_90",
        "lower_95",
        "upper_95",
        "trust_score",
        "trust_tier",
        "nearest_training_anchors",
    ]:
        assert col in out.columns, col
    assert (out["lower_90"] <= out["upper_90"]).all()
    # 95% bounds are at least as wide as 90% bounds
    assert (out["lower_95"] <= out["lower_90"] + 1e-9).all()
    assert (out["upper_95"] >= out["upper_90"] - 1e-9).all()
    assert (out["lower_90"] >= 0).all()
    assert out["trust_tier"].isin(["A", "B", "C"]).all()
    # Provenance anchors carry source citations
    assert out["nearest_training_anchors"].str.len().gt(5).all()
    assert len(blocks) == len(out)


def test_report_renders(pipeline):
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    out, blocks = certify_dataframe(unseen_df, run)
    path = render_report(str(tmp / "report.html"), out, run["model_card"], blocks)
    text = Path(path).read_text(encoding="utf-8")
    assert "ToughCert" in text
    assert "data:image/png;base64," in text
    assert "Intended use" in text
    assert len(text) > 20000  # embedded figures present


def test_certify_handles_raw_columns(pipeline):
    """Certify accepts a frame missing physics columns and rebuilds them."""
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    stripped = unseen_df.drop(columns=[c for c in unseen_df.columns if c.startswith("phys_")])
    out, _ = certify_dataframe(stripped, run)
    assert np.isfinite(out["predicted_toughness_mpa_m0_5"]).all()
