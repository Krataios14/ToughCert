import pandas as pd

from src.prepare_data import prepare_dataframe


def test_prepare_dataframe_parses_composition():
    df = pd.DataFrame(
        {
            "Composition (at. %)": ["Al10.0-Cr20.0-Fe70.0"],
            "Material condition": ["as-cast"],
            "Fracture_toughness_MPa_m0.5": [50.0],
        }
    )

    out, elem_cols, phys_cols = prepare_dataframe(
        df, target_col="fracture_toughness_mpa_m0_5", drop_cols=[]
    )

    assert "elem_al" in out.columns
    assert "elem_cr" in out.columns
    assert "elem_fe" in out.columns
    assert out.loc[0, "elem_al"] == 10.0
    assert out.loc[0, "elem_fe"] == 70.0
    assert "material_condition" in out.columns
    assert "fracture_toughness_mpa_m0_5" in out.columns
    assert len(elem_cols) == 3
    # Raw composition string retained as provenance metadata
    assert "composition_at_percent" in out.columns
    # Physics descriptors appended
    assert "phys_smix_r" in phys_cols
    assert out.loc[0, "phys_vec"] > 0


def test_prepare_dataframe_without_physics():
    df = pd.DataFrame(
        {
            "Composition (at. %)": ["Al10.0-Fe90.0"],
            "Fracture_toughness_MPa_m0.5": [50.0],
        }
    )
    out, _, phys_cols = prepare_dataframe(
        df, target_col="fracture_toughness_mpa_m0_5", drop_cols=[], physics=False
    )
    assert phys_cols == []
    assert not any(c.startswith("phys_") for c in out.columns)
