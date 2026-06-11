"""Group-aware conformal prediction (CV+) for finite-sample-valid intervals.

Implements the CV+ / cross-conformal method of Barber, Candes, Ramdas &
Tibshirani, "Predictive inference with the jackknife+", Ann. Statist. 49
(2021), with folds assigned at the group level (here: source publication
plus composition) so specimens from one paper never straddle the
train/calibration boundary. Literature-mined materials data is clustered
(several specimens per paper share lab, method and material), so
row-level exchangeability is violated and naive conformal intervals are
overconfident for genuinely new alloys.

What is and is not guaranteed, stated carefully:

- For K-fold CV+ with equal fold sizes and exchangeable rows, Theorem 4
  of Barber et al. gives coverage at least
  1 - 2*alpha - min{ 2(1-1/K)/(n/K+1), (1-K/n)/(K+1) }.
  The excess term is not negligible at small n (about 0.09 for n=147,
  K=8), so this module reports the honest provable floor rather than
  the often-quoted 1-2*alpha.
- Group-level folding makes fold sizes unequal and the new-group test
  point is not exchangeable with training rows under a two-layer
  hierarchical model; the row-level guarantee extends only
  heuristically. Group-level validity is therefore supported
  empirically, by repeated held-out-group evaluation, and a provably
  valid reference (one subsampled row per group, after Dunn, Wasserman
  & Ramdas, JASA 2022) is available via `subsample_one_per_group`.

MondrianCVPlus partitions the data into pre-committed physical bins
(e.g. brittle vs ductile phase class) and calibrates each bin
separately, with a pooled model as fallback for thin bins. The bin
definition must be chosen on subject-matter grounds before looking at
results; switching schemes post hoc voids the calibration story.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score


ModelFactory = Callable[[], object]


def cvplus_excess(n: int, k: int) -> float:
    """Theorem 4 excess term for K-fold CV+ (Barber et al. 2021)."""
    if n <= 0 or k <= 1:
        return 1.0
    return min(2.0 * (1.0 - 1.0 / k) / (n / k + 1.0), (1.0 - k / n) / (k + 1.0))


def provable_floor(alpha: float, n: int, k: int) -> float:
    """Honest distribution-free coverage floor for K-fold CV+.

    max(0, 1 - 2*alpha - excess(n, K)), under row exchangeability and
    equal fold sizes; group folding is a heuristic extension on top.
    """
    return max(0.0, 1.0 - 2.0 * alpha - cvplus_excess(n, k))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Exact binomial confidence interval for a coverage proportion."""
    if n == 0:
        return 0.0, 1.0
    lo = stats.beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def subsample_one_per_group(
    X: np.ndarray, y: np.ndarray, groups: Sequence, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One random row per group: the provably valid hierarchical reduction.

    Under a two-layer model (groups i.i.d., rows correlated within a
    group), subsampling a single row per group restores exchangeability
    with a test row drawn from a new group (Dunn, Wasserman & Ramdas,
    JASA 2022). The price is throwing away within-group replicates.
    """
    groups = np.asarray(groups)
    rng = np.random.RandomState(seed)
    keep = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        keep.append(rng.choice(idx))
    keep = np.sort(np.array(keep))
    return np.asarray(X)[keep], np.asarray(y)[keep], groups[keep]


def _fold_assignment(groups: np.ndarray, n_folds: int, seed: int) -> Tuple[np.ndarray, int]:
    """Assign each row to a fold so that a group never straddles folds."""
    unique = np.unique(groups)
    n_folds = max(2, min(n_folds, len(unique)))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique)
    group_to_fold = {g: i % n_folds for i, g in enumerate(shuffled)}
    folds = np.array([group_to_fold[g] for g in groups], dtype=int)
    return folds, n_folds


class GroupCVPlus:
    """CV+ conformal regressor with group-level folds.

    Parameters
    ----------
    model_factory : zero-argument callable returning an unfitted
        sklearn-style regressor (fresh instance per call).
    n_folds : number of cross-conformal folds (reduced automatically if
        there are fewer groups).
    seed : fold-assignment seed.
    """

    def __init__(self, model_factory: ModelFactory, n_folds: int = 8, seed: int = 42):
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.seed = seed
        self.fold_models_: list = []
        self.full_model_ = None
        self.residuals_: Optional[np.ndarray] = None
        self.fold_of_sample_: Optional[np.ndarray] = None
        self.n_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[Sequence] = None) -> "GroupCVPlus":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if groups is None:
            groups = np.arange(n)
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups must have the same length as y")
        if len(np.unique(groups)) < 2:
            raise ValueError(
                "GroupCVPlus needs at least 2 distinct groups; "
                f"got {len(np.unique(groups))}"
            )

        folds, n_folds = _fold_assignment(groups, self.n_folds, self.seed)

        self.fold_models_ = []
        residuals = np.empty(n, dtype=np.float64)
        for k in range(n_folds):
            held = folds == k
            model = self.model_factory()
            model.fit(X[~held], y[~held])
            preds = np.asarray(model.predict(X[held]), dtype=np.float64).ravel()
            residuals[held] = np.abs(y[held] - preds)
            self.fold_models_.append(model)

        self.full_model_ = self.model_factory()
        self.full_model_.fit(X, y)
        self.residuals_ = residuals
        self.fold_of_sample_ = folds
        self.n_ = n
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.full_model_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.full_model_.predict(np.asarray(X, dtype=np.float64))).ravel()

    def _fold_predictions(self, X: np.ndarray) -> np.ndarray:
        """Matrix of shape (n_folds, m): each fold model's predictions."""
        X = np.asarray(X, dtype=np.float64)
        return np.stack(
            [np.asarray(m.predict(X), dtype=np.float64).ravel() for m in self.fold_models_]
        )

    def predict_interval_multi(
        self, X: np.ndarray, alphas: Sequence[float]
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """CV+ intervals at several alphas from one set of fold predictions.

        For each test point x the bounds are order statistics of
        { mu_{-k(i)}(x) -/+ R_i } over the n training points. When the
        required order statistic falls outside 1..n (n too small for the
        requested alpha) the bound is -inf/+inf: an honest "no finite
        bound at this confidence", never a silently clamped value.
        Alphas at or above 0.5 are allowed but their 1-2*alpha guarantee
        is vacuous.
        """
        if self.residuals_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        for alpha in alphas:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        fold_preds = self._fold_predictions(X)  # (K, m)
        per_sample = fold_preds[self.fold_of_sample_, :]  # (n, m)
        v_lo = np.sort(per_sample - self.residuals_[:, None], axis=0)
        v_hi = np.sort(per_sample + self.residuals_[:, None], axis=0)

        n = self.n_
        m = per_sample.shape[1]
        out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for alpha in alphas:
            i_lo = int(math.floor(alpha * (n + 1)))
            i_hi = int(math.ceil((1.0 - alpha) * (n + 1)))
            lower = v_lo[i_lo - 1] if i_lo >= 1 else np.full(m, -np.inf)
            upper = v_hi[i_hi - 1] if i_hi <= n else np.full(m, np.inf)
            out[alpha] = (lower, upper)
        return out

    def predict_interval(self, X: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_interval_multi(X, [alpha])[alpha]

    def floor(self, alpha: float) -> float:
        return provable_floor(alpha, self.n_, len(self.fold_models_))


class MondrianCVPlus:
    """Per-bin group-CV+ with a pooled fallback for thin bins.

    Bins must be a pre-committed function of the row's covariates (here:
    brittle vs ductile phase class). Each sufficiently populated bin
    gets its own GroupCVPlus, calibrated only on that bin's groups, so
    over-coverage in an easy bin can no longer subsidize under-coverage
    in a hard one. Rows whose bin has no dedicated model fall back to
    the pooled model fitted on everything.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        n_folds: int = 8,
        seed: int = 42,
        min_rows: int = 30,
        min_groups: int = 8,
    ):
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.seed = seed
        self.min_rows = min_rows
        self.min_groups = min_groups
        self.bin_models_: Dict[str, GroupCVPlus] = {}
        self.pooled_: Optional[GroupCVPlus] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence, bins: Sequence
    ) -> "MondrianCVPlus":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        groups = np.asarray(groups)
        bins = np.asarray(bins).astype(str)
        if not (len(bins) == len(groups) == len(y)):
            raise ValueError("X, y, groups and bins must have equal length")

        self.pooled_ = GroupCVPlus(self.model_factory, self.n_folds, self.seed).fit(
            X, y, groups
        )
        self.bin_models_ = {}
        for label in np.unique(bins):
            mask = bins == label
            n_groups = len(np.unique(groups[mask]))
            if mask.sum() >= self.min_rows and n_groups >= self.min_groups:
                self.bin_models_[label] = GroupCVPlus(
                    self.model_factory, self.n_folds, self.seed
                ).fit(X[mask], y[mask], groups[mask])
        return self

    def _routes(self, bins: Sequence) -> Dict[str, np.ndarray]:
        bins = np.asarray(bins).astype(str)
        routes: Dict[str, np.ndarray] = {}
        handled = np.zeros(len(bins), dtype=bool)
        for label, model in self.bin_models_.items():
            mask = bins == label
            if mask.any():
                routes[label] = mask
                handled |= mask
        if (~handled).any():
            routes["__pooled__"] = ~handled
        return routes

    def _model_for(self, label: str) -> GroupCVPlus:
        return self.pooled_ if label == "__pooled__" else self.bin_models_[label]

    def predict(self, X: np.ndarray, bins: Sequence) -> np.ndarray:
        if self.pooled_ is None:
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        for label, mask in self._routes(bins).items():
            out[mask] = self._model_for(label).predict(X[mask])
        return out

    def predict_interval_multi(
        self, X: np.ndarray, alphas: Sequence[float], bins: Sequence
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        if self.pooled_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        out = {a: (np.empty(m), np.empty(m)) for a in alphas}
        for label, mask in self._routes(bins).items():
            sub = self._model_for(label).predict_interval_multi(X[mask], alphas)
            for a in alphas:
                out[a][0][mask] = sub[a][0]
                out[a][1][mask] = sub[a][1]
        return out

    def predict_interval(
        self, X: np.ndarray, alpha: float, bins: Sequence
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_interval_multi(X, [alpha], bins)[alpha]

    def bin_summary(self, alpha: float = 0.1) -> Dict[str, Dict[str, float]]:
        """Per-bin sample sizes and honest provable floors."""
        out: Dict[str, Dict[str, float]] = {}
        for label, model in self.bin_models_.items():
            out[label] = {
                "n": int(model.n_),
                "n_folds": len(model.fold_models_),
                "provable_floor": model.floor(alpha),
            }
        if self.pooled_ is not None:
            out["__pooled__"] = {
                "n": int(self.pooled_.n_),
                "n_folds": len(self.pooled_.fold_models_),
                "provable_floor": self.pooled_.floor(alpha),
            }
        return out


def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "n_unbounded": int(np.sum(~np.isfinite(width))),
    }


def evaluate_group_coverage(
    model_factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    alpha: float = 0.1,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    n_folds: int = 8,
    seed: int = 42,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alphas: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Held-out-group calibration evidence for a single fixed model.

    Repeatedly splits off whole groups as a test set, fits a fresh
    GroupCVPlus on the remainder, and measures empirical coverage,
    interval width and point accuracy on the unseen groups.

    If `alphas` is given, intervals at every listed alpha are computed
    from the same fitted models (cheap: the fold predictions are reused)
    and returned under 'per_alpha'; the flat fields still describe the
    primary `alpha`. If the model is trained on a transformed target,
    pass `inverse_transform` (monotone increasing): coverage is
    invariant under it, while widths and point metrics are reported in
    original units.

    Note this evaluates ONE pre-committed pipeline. If any data-driven
    choice (model class, conformal variant) is made before deployment,
    selection-inclusive evidence requires nesting that choice inside
    each split; see qualify.nested_evidence.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.RandomState(seed)
    alpha_grid = sorted(set([alpha] + list(alphas or [])), reverse=True)

    per_alpha_cov = {a: [] for a in alpha_grid}
    per_alpha_width = {a: [] for a in alpha_grid}
    maes, all_true, all_pred = [], [], []
    n_total = 0
    for split in range(n_splits):
        test_groups = rng.choice(
            unique, size=max(1, int(round(test_fraction * len(unique)))), replace=False
        )
        test_mask = np.isin(groups, test_groups)
        if test_mask.all() or not test_mask.any():
            continue
        model = GroupCVPlus(model_factory, n_folds=n_folds, seed=seed + split)
        model.fit(X[~test_mask], y[~test_mask], groups=groups[~test_mask])
        intervals = model.predict_interval_multi(X[test_mask], alpha_grid)
        preds = model.predict(X[test_mask])
        y_test = y[test_mask]
        if inverse_transform is not None:
            preds = inverse_transform(preds)
            y_eval = inverse_transform(y_test)
        else:
            y_eval = y_test
        n_total += int(test_mask.sum())
        for a in alpha_grid:
            lower, upper = intervals[a]
            if inverse_transform is not None:
                lower, upper = inverse_transform(lower), inverse_transform(upper)
            m = interval_metrics(y_eval, lower, upper)
            per_alpha_cov[a].append(m["coverage"])
            per_alpha_width[a].append(m["mean_width"])
        maes.append(mean_absolute_error(y_eval, preds))
        all_true.append(y_eval)
        all_pred.append(preds)

    if not maes:
        raise ValueError("No valid evaluation splits could be formed")
    y_cat = np.concatenate(all_true)
    p_cat = np.concatenate(all_pred)
    n_train_typ = int(len(y) * (1 - test_fraction))

    def _entry(a: float) -> Dict[str, float]:
        return {
            "alpha": float(a),
            "nominal_coverage": float(1.0 - a),
            "provable_floor": provable_floor(a, n_train_typ, n_folds),
            "empirical_coverage": float(np.mean(per_alpha_cov[a])),
            "coverage_std": float(np.std(per_alpha_cov[a])),
            "mean_interval_width": float(np.mean(per_alpha_width[a])),
        }

    result: Dict[str, object] = _entry(alpha)
    result.update(
        {
            "mae": float(np.mean(maes)),
            "r2": float(r2_score(y_cat, p_cat)) if len(y_cat) > 1 else float("nan"),
            "n_splits": int(len(maes)),
        }
    )
    if alphas:
        result["per_alpha"] = {f"alpha_{a:.2f}": _entry(a) for a in alpha_grid}
    return result
