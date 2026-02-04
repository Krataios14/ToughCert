import numpy as np
import pandas as pd
import pytest

from src.data import load_data


def test_load_data_missing_columns(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "fracture_toughness": [10.0, 12.0]})
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as exc:
        load_data(
            csv_path=str(csv_path),
            target="fracture_toughness",
            num_features=["C"],
            cat_features=[],
            val_split=0.2,
            seed=0,
        )
    assert "Missing feature columns" in str(exc.value)


def test_load_data_shapes(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "C": rng.normal(0.3, 0.05, 50),
            "Mn": rng.normal(1.2, 0.2, 50),
            "steel_grade": rng.choice(["A", "B"], 50),
            "fracture_toughness": rng.normal(100, 5, 50),
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    train_ds, val_ds, _, _, _ = load_data(
        csv_path=str(csv_path),
        target="fracture_toughness",
        num_features=["C", "Mn"],
        cat_features=["steel_grade"],
        val_split=0.2,
        seed=0,
        min_known_numeric=1,
    )

    assert train_ds.x_num.shape[1] == 2
    assert train_ds.x_num_mask.shape[1] == 2
    assert train_ds.x_cat.shape[1] == 1
    assert val_ds.x_num.shape[1] == 2
    assert val_ds.x_num_mask.shape[1] == 2
    assert val_ds.x_cat.shape[1] == 1
