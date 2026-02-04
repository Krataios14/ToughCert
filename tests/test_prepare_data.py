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

    out, elem_cols = prepare_dataframe(
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
