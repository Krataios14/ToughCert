import math

import numpy as np
import pandas as pd
import pytest

from src.physics import (
    ELEMENT_PROPS,
    H_MIX,
    add_physics_features,
    composition_descriptors,
    parse_composition,
)


def test_parse_composition():
    comp = parse_composition("Al10.71-Co17.86-Cr17.86-Fe17.86-Ni17.86-Ti17.86")
    assert comp["Al"] == pytest.approx(10.71)
    assert comp["Ti"] == pytest.approx(17.86)
    assert parse_composition(None) == {}
    assert parse_composition("") == {}
    assert parse_composition(float("nan")) == {}


def test_cantor_alloy_descriptors():
    # Equiatomic CoCrFeMnNi (Cantor alloy): known reference values.
    comp = {e: 20.0 for e in ["Co", "Cr", "Fe", "Mn", "Ni"]}
    d = composition_descriptors(comp)
    # Ideal mixing entropy of 5 equiatomic components = ln(5) R
    assert d["phys_smix_r"] == pytest.approx(math.log(5), rel=1e-6)
    # VEC = mean of (9, 6, 8, 7, 10) = 8 -> FCC-stabilizing regime
    assert d["phys_vec"] == pytest.approx(8.0, rel=1e-6)
    # Small atomic size mismatch (delta < 4%) is characteristic of Cantor
    assert 0.0 < d["phys_delta_r"] < 4.0
    # Slightly negative mixing enthalpy, fully covered pair table
    assert d["phys_dhmix"] < 0.0
    assert d["phys_dhmix_coverage"] == pytest.approx(1.0)
    assert d["phys_n_elements"] == 5
    assert d["phys_max_frac"] == pytest.approx(0.2)
    # Solid-solution stability parameter should be comfortably > 1
    assert d["phys_omega"] > 1.0
    # Rule-of-mixtures density near 8 g/cm3 for Cantor
    assert 7.5 < d["phys_density_rom"] < 8.5


def test_empty_composition_gives_nan():
    d = composition_descriptors({})
    assert all(math.isnan(v) for v in d.values())


def test_enthalpy_table_symmetric_and_known():
    assert H_MIX[frozenset(("Al", "Ni"))] == -22
    assert H_MIX[frozenset(("Ni", "Al"))] == -22
    # All table entries reference known elements
    for pair in H_MIX:
        for e in pair:
            assert e in ELEMENT_PROPS


def test_add_physics_features_mechanics():
    df = pd.DataFrame(
        {
            "composition_at_percent": ["Fe50-Ni50", None],
            "grain_size_um": [25.0, np.nan],
            "yield_strength_mpa": [400.0, 500.0],
            "uts_mpa": [600.0, 550.0],
            "youngs_modulus_gpa": [200.0, np.nan],
            "hardness_gpa": [2.0, np.nan],
            "testing_temperature_k": [298.0, 77.0],
        }
    )
    out, added = add_physics_features(df)
    assert set(added).issuperset(
        {"phys_smix_r", "phys_t_homologous", "phys_hall_petch", "phys_strain_hardening"}
    )
    # Row 0: full data
    assert out.loc[0, "phys_hall_petch"] == pytest.approx(1.0 / math.sqrt(25.0))
    assert out.loc[0, "phys_yield_strain"] == pytest.approx(400.0 / 200000.0)
    assert out.loc[0, "phys_strain_hardening"] == pytest.approx(0.5)
    assert out.loc[0, "phys_h_over_e"] == pytest.approx(0.01)
    tm = out.loc[0, "phys_tm_mix"]
    assert out.loc[0, "phys_t_homologous"] == pytest.approx(298.0 / tm)
    # Row 1: no composition -> composition descriptors NaN, mechanics partial
    assert math.isnan(out.loc[1, "phys_smix_r"])
    assert math.isnan(out.loc[1, "phys_t_homologous"])
    assert out.loc[1, "phys_strain_hardening"] == pytest.approx(0.1)
    # No infinities anywhere
    num = out[added].to_numpy(dtype=float)
    assert not np.isinf(num).any()


def test_real_dataset_featurizes():
    from pathlib import Path

    csv = Path(__file__).resolve().parents[1] / "assets" / "combined_fracture_training.csv"
    df = pd.read_csv(csv, skip_blank_lines=True)
    from src.prepare_data import prepare_dataframe

    out, elem_cols, phys_cols = prepare_dataframe(
        df, target_col="fracture_toughness_mpa_m0_5", drop_cols=[]
    )
    assert len(phys_cols) >= 15
    # Most rows have parsable compositions
    assert out["phys_smix_r"].notna().mean() > 0.9
    # Homologous temperature is physically plausible where defined
    th = out["phys_t_homologous"].dropna()
    assert (th > 0).all() and (th < 1.5).all()
