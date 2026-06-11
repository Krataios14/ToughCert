import numpy as np
import pandas as pd
import pytest

from src.certify import load_run
from src.physics import parse_composition
from src.screen import advise_tests, generate_candidates, screen_candidates


def test_generate_candidates_normalized():
    base = [parse_composition("Al20-Co20-Cr20-Fe20-Ni20")]
    cands = generate_candidates(base, n_per_base=10, jitter=0.2, seed=1)
    assert len(cands) == 11  # base + 10 perturbations
    for comp in cands["composition_at_percent"]:
        parsed = parse_composition(comp)
        assert sum(parsed.values()) == pytest.approx(100.0, abs=0.2)
        # Perturbations stay within the jitter envelope after renormalizing
        assert all(10.0 < v < 32.0 for v in parsed.values())


def test_generate_candidates_deterministic():
    base = [parse_composition("Fe50-Ni50")]
    a = generate_candidates(base, n_per_base=5, seed=7)
    b = generate_candidates(base, n_per_base=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_screen_ranks_by_lower_bound(pipeline):
    _, run_dir, _ = pipeline
    run = load_run(run_dir)
    base = [
        parse_composition("Al20-Co20-Cr20-Fe20-Ni20"),
        parse_composition("Co20-Cr20-Fe20-Mn20-Ni20"),
    ]
    cands = generate_candidates(base, n_per_base=8, seed=3)
    result = screen_candidates(cands, run, temperature_k=298.0, min_tier="C")
    assert len(result) > 0
    lows = result["lower_90"].to_numpy()
    assert (lows[:-1] >= lows[1:] - 1e-9).all()  # descending
    assert "density_estimate_g_cm3" not in result.columns  # no constraint given


def test_screen_density_constraint(pipeline):
    _, run_dir, _ = pipeline
    run = load_run(run_dir)
    base = [parse_composition("Al20-Co20-Cr20-Fe20-Ni20")]
    cands = generate_candidates(base, n_per_base=8, seed=4)
    result = screen_candidates(cands, run, max_density=7.0, min_tier="C")
    if len(result):
        assert (result["density_estimate_g_cm3"] <= 7.0).all()


def test_advise_ranks_by_test_value(pipeline):
    _, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    result = advise_tests(unseen_df, run, top=5)
    assert len(result) == 5
    scores = result["test_value_score"].to_numpy()
    assert (scores[:-1] >= scores[1:] - 1e-12).all()
    assert (result["relative_interval_width"] > 0).all()
