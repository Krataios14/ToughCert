import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.conformal import (
    GroupCVPlus,
    MondrianCVPlus,
    clopper_pearson,
    cvplus_excess,
    evaluate_group_coverage,
    interval_metrics,
    provable_floor,
    subsample_one_per_group,
)


def _make_data(n=300, n_groups=40, noise=0.5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + noise * rng.randn(n)
    groups = rng.randint(0, n_groups, size=n)
    return X, y, groups


def test_cvplus_coverage_on_fresh_data():
    X, y, groups = _make_data(seed=1)
    Xt, yt, _ = _make_data(n=500, seed=2)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=8, seed=42)
    model.fit(X, y, groups=groups)
    lower, upper = model.predict_interval(Xt, alpha=0.1)
    m = interval_metrics(yt, lower, upper)
    assert m["coverage"] >= 0.85
    assert m["mean_width"] > 0
    assert m["n_unbounded"] == 0


def test_intervals_widen_with_confidence():
    X, y, groups = _make_data(seed=3)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=0)
    model.fit(X, y, groups=groups)
    lo90, hi90 = model.predict_interval(X[:20], alpha=0.10)
    lo95, hi95 = model.predict_interval(X[:20], alpha=0.05)
    assert np.all(lo95 <= lo90 + 1e-12)
    assert np.all(hi95 >= hi90 - 1e-12)
    assert np.all(hi90 > lo90)


def test_multi_alpha_consistent_and_nested():
    X, y, groups = _make_data(seed=9)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=6, seed=1).fit(X, y, groups)
    grid = [0.32, 0.2, 0.1, 0.05]
    multi = model.predict_interval_multi(X[:15], grid)
    for a in grid:
        lo_single, hi_single = model.predict_interval(X[:15], alpha=a)
        np.testing.assert_allclose(multi[a][0], lo_single)
        np.testing.assert_allclose(multi[a][1], hi_single)
    # Nesting across the ladder
    for a_wide, a_narrow in zip(grid[1:], grid[:-1]):
        assert np.all(multi[a_wide][0] <= multi[a_narrow][0] + 1e-12)
        assert np.all(multi[a_wide][1] >= multi[a_narrow][1] - 1e-12)


def test_unbounded_interval_at_tiny_n_alpha():
    """When floor(alpha*(n+1)) < 1 the lower bound must be -inf, not a
    silently clamped order statistic (which would be anti-conservative)."""
    X, y, groups = _make_data(n=20, n_groups=10, seed=4)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=0).fit(X, y, groups)
    lower, upper = model.predict_interval(X[:3], alpha=0.01)
    assert np.all(np.isneginf(lower))
    assert np.all(np.isposinf(upper))
    m = interval_metrics(y[:3], lower, upper)
    assert m["n_unbounded"] == 3
    assert m["coverage"] == 1.0


def test_alpha_validation():
    X, y, groups = _make_data(n=50, seed=6)
    model = GroupCVPlus(lambda: Ridge(), n_folds=4).fit(X, y, groups)
    with pytest.raises(ValueError):
        model.predict_interval(X[:2], alpha=0.0)
    with pytest.raises(ValueError):
        model.predict_interval(X[:2], alpha=1.0)
    # Vacuous-guarantee alphas are allowed and well behaved
    lo, hi = model.predict_interval(X[:2], alpha=0.6)
    assert np.all(lo <= hi)


def test_few_groups_reduces_folds():
    X, y, _ = _make_data(n=60, seed=4)
    groups = np.array([0, 1, 2] * 20)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=10, seed=0)
    model.fit(X, y, groups=groups)
    assert len(model.fold_models_) == 3
    lower, upper = model.predict_interval(X[:5], alpha=0.2)
    assert lower.shape == (5,)
    assert np.all(upper >= lower)


def test_single_group_rejected():
    X, y, _ = _make_data(n=30, seed=5)
    with pytest.raises(ValueError, match="at least 2 distinct groups"):
        GroupCVPlus(lambda: Ridge()).fit(X, y, groups=np.zeros(30))


def test_deterministic_with_seed():
    X, y, groups = _make_data(seed=5)
    a = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=6, seed=7).fit(X, y, groups)
    b = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=6, seed=7).fit(X, y, groups)
    la, ua = a.predict_interval(X[:10], alpha=0.1)
    lb, ub = b.predict_interval(X[:10], alpha=0.1)
    np.testing.assert_allclose(la, lb)
    np.testing.assert_allclose(ua, ub)


def test_requires_fit():
    model = GroupCVPlus(lambda: Ridge())
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 4)))
    with pytest.raises(RuntimeError):
        model.predict_interval(np.zeros((1, 4)))


def test_provable_floor_math():
    # n=147, K=8: excess ~ 0.087, floor at alpha=0.1 ~ 0.71, not 0.80
    excess = cvplus_excess(147, 8)
    assert 0.05 < excess < 0.12
    assert provable_floor(0.1, 147, 8) == pytest.approx(1 - 0.2 - excess)
    # Jackknife+ limit: K=n makes the excess vanish as n grows
    assert cvplus_excess(1000, 1000) < 0.01
    assert provable_floor(0.4, 20, 2) >= 0.0  # clamped at zero


def test_clopper_pearson():
    lo, hi = clopper_pearson(9, 10)
    assert lo < 0.9 < hi
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    lo_small, hi_small = clopper_pearson(2, 3)
    assert hi_small - lo_small > 0.5  # tiny n gives wide bands


def test_subsample_one_per_group():
    X, y, groups = _make_data(n=100, n_groups=12, seed=7)
    Xs, ys, gs = subsample_one_per_group(X, y, groups, seed=0)
    assert len(np.unique(gs)) == len(gs) == len(np.unique(groups))
    # Deterministic
    Xs2, _, _ = subsample_one_per_group(X, y, groups, seed=0)
    np.testing.assert_allclose(Xs, Xs2)


def test_mondrian_routes_and_falls_back():
    X, y, groups = _make_data(n=400, n_groups=60, seed=8)
    bins = np.where(X[:, 0] > 0, "hot", "cold")
    model = MondrianCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=0, min_rows=30, min_groups=8)
    model.fit(X, y, groups, bins)
    assert set(model.bin_models_) == {"hot", "cold"}
    # Unknown bin label routes through the pooled model
    test_bins = np.array(["hot", "cold", "weird"])
    lo, hi = model.predict_interval(X[:3], 0.1, test_bins)
    assert np.all(hi > lo)
    preds = model.predict(X[:3], test_bins)
    pooled_pred = model.pooled_.predict(X[2:3])
    assert preds[2] == pytest.approx(pooled_pred[0])
    summary = model.bin_summary(alpha=0.1)
    assert {"hot", "cold", "__pooled__"} <= set(summary)
    assert all(0 <= v["provable_floor"] < 0.8 for v in summary.values())


def test_mondrian_thin_bin_uses_pooled():
    X, y, groups = _make_data(n=200, n_groups=30, seed=10)
    bins = np.array(["common"] * 195 + ["rare"] * 5)
    model = MondrianCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=0).fit(X, y, groups, bins)
    assert "rare" not in model.bin_models_
    lo, hi = model.predict_interval(X[195:], 0.1, bins[195:])
    plo, phi = model.pooled_.predict_interval(X[195:], 0.1)
    np.testing.assert_allclose(lo, plo)
    np.testing.assert_allclose(hi, phi)


def test_mondrian_single_bin_matches_pooled():
    """With one bin holding everything, Mondrian must equal plain CV+."""
    X, y, groups = _make_data(n=200, n_groups=25, seed=11)
    bins = np.array(["all"] * 200)
    mond = MondrianCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=3).fit(X, y, groups, bins)
    plain = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=3).fit(X, y, groups)
    lo_m, hi_m = mond.predict_interval(X[:10], 0.1, bins[:10])
    lo_p, hi_p = plain.predict_interval(X[:10], 0.1)
    np.testing.assert_allclose(lo_m, lo_p)
    np.testing.assert_allclose(hi_m, hi_p)


def test_evaluate_group_coverage():
    X, y, groups = _make_data(n=400, n_groups=50, seed=8)
    report = evaluate_group_coverage(
        lambda: Ridge(alpha=1.0), X, y, groups, alpha=0.1, n_splits=3, seed=1
    )
    assert report["empirical_coverage"] >= 0.8
    assert 0 < report["provable_floor"] < 0.8
    assert report["mae"] < 1.0
    assert report["n_splits"] == 3


def test_evaluate_group_coverage_multi_alpha():
    X, y, groups = _make_data(n=300, n_groups=40, seed=12)
    report = evaluate_group_coverage(
        lambda: Ridge(alpha=1.0), X, y, groups,
        alpha=0.1, n_splits=2, seed=2, alphas=[0.32, 0.2, 0.1, 0.05],
    )
    per = report["per_alpha"]
    assert set(per) == {"alpha_0.32", "alpha_0.20", "alpha_0.10", "alpha_0.05"}
    # Wider nominal coverage -> wider intervals
    widths = [per[k]["mean_interval_width"] for k in ["alpha_0.32", "alpha_0.10", "alpha_0.05"]]
    assert widths[0] <= widths[1] <= widths[2]
    # Flat fields agree with the primary alpha entry
    assert report["empirical_coverage"] == per["alpha_0.10"]["empirical_coverage"]
