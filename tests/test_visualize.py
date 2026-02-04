import os
import pandas as pd

from src.visualize import visualize_from_config


def test_visualize_outputs(tmp_path):
    run_dir = tmp_path / "run"
    fig_dir = tmp_path / "figs"
    run_dir.mkdir()

    # history.json
    (run_dir / "history.json").write_text(
        '{"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}', encoding="ascii"
    )
    # predictions.csv
    pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 1.9, 3.2]}).to_csv(
        run_dir / "predictions.csv", index=False
    )
    # metrics.json with permutation importance
    (run_dir / "metrics.json").write_text(
        '{"permutation_importance": [{"feature": "C", "type": "numerical", "mae_increase": 0.5}]}',
        encoding="ascii",
    )

    cfg = {"outputs": {"figures_dir": str(fig_dir)}}
    visualize_from_config(cfg, str(run_dir))

    assert os.path.exists(fig_dir / "learning_curves.png")
    assert os.path.exists(fig_dir / "parity_plot.png")
    assert os.path.exists(fig_dir / "residuals.png")
    assert os.path.exists(fig_dir / "permutation_importance.png")
