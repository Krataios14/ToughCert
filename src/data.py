"""Data loading and preprocessing."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset


@dataclass
class PreprocessArtifacts:
    scaler: RobustScaler
    cat_encoder: "CategoricalEncoder"
    num_medians: np.ndarray
    target_mean: float
    target_std: float
    target_transform: str
    target_min: float
    target_max: float


def _apply_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y


def _invert_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        y = np.clip(y, a_min=-30.0, a_max=30.0)
        return np.expm1(y.astype(np.float64)).astype(np.float64)
    return y


class CategoricalEncoder:
    """Simple per-column categorical encoder with UNK=0."""

    def __init__(self) -> None:
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, cat_features: List[str]) -> None:
        self.maps = {}
        for col in cat_features:
            values = df[col].fillna("UNK").astype(str).unique().tolist()
            mapping = {"UNK": 0}
            for v in values:
                if v not in mapping:
                    mapping[v] = len(mapping)
            self.maps[col] = mapping

    def transform(self, df: pd.DataFrame, cat_features: List[str]) -> np.ndarray:
        arrays = []
        for col in cat_features:
            mapping = self.maps[col]
            col_vals = df[col].fillna("UNK").astype(str)
            encoded = col_vals.map(lambda x: mapping.get(x, 0)).astype(int).to_numpy()
            arrays.append(encoded)
        if not arrays:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.stack(arrays, axis=1).astype(np.int64)

    def cardinalities(self, cat_features: List[str]) -> List[int]:
        return [len(self.maps[col]) for col in cat_features]


class TabularDataset(Dataset):
    def __init__(
        self, x_num: np.ndarray, x_num_mask: np.ndarray, x_cat: np.ndarray, y: np.ndarray
    ) -> None:
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_num_mask = torch.tensor(x_num_mask, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_num_mask[idx], self.x_cat[idx], self.y[idx]


def fit_preprocessor(
    df: pd.DataFrame,
    num_features: List[str],
    cat_features: List[str],
    y: np.ndarray,
    target_transform: str,
    target_standardize: bool,
) -> PreprocessArtifacts:
    scaler = RobustScaler()
    if num_features:
        medians = df[num_features].median(axis=0, numeric_only=True).to_numpy()
    else:
        medians = np.zeros((1,), dtype=np.float32)
    if num_features:
        filled = df[num_features].fillna(pd.Series(medians, index=num_features))
        scaler.fit(filled)
    else:
        scaler.fit(np.zeros((len(df), 1)))

    cat_encoder = CategoricalEncoder()
    cat_encoder.fit(df, cat_features)

    y_t = _apply_target_transform(y.astype(np.float32), target_transform)
    target_min = float(np.min(y_t)) if y_t.size else 0.0
    target_max = float(np.max(y_t)) if y_t.size else 0.0
    if target_standardize:
        target_mean = float(np.mean(y_t))
        target_std = float(np.std(y_t) + 1e-8)
    else:
        target_mean = 0.0
        target_std = 1.0

    return PreprocessArtifacts(
        scaler=scaler,
        cat_encoder=cat_encoder,
        num_medians=medians,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=target_transform,
        target_min=target_min,
        target_max=target_max,
    )


def transform_df(
    df: pd.DataFrame,
    num_features: List[str],
    cat_features: List[str],
    artifacts: PreprocessArtifacts,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_features:
        num_df = df[num_features]
        x_num_mask = (~num_df.isna()).to_numpy(dtype=np.float32)
        filled = num_df.fillna(pd.Series(artifacts.num_medians, index=num_features))
        x_num = artifacts.scaler.transform(filled).astype(np.float32)
    else:
        x_num = np.zeros((len(df), 0), dtype=np.float32)
        x_num_mask = np.zeros((len(df), 0), dtype=np.float32)

    x_cat = artifacts.cat_encoder.transform(df, cat_features)
    return x_num, x_num_mask, x_cat


def transform_target(y: np.ndarray, artifacts: PreprocessArtifacts, standardize: bool) -> np.ndarray:
    y_t = _apply_target_transform(y.astype(np.float32), artifacts.target_transform)
    if standardize:
        y_t = (y_t - artifacts.target_mean) / artifacts.target_std
    return y_t


def inverse_transform_target(y: np.ndarray, artifacts: PreprocessArtifacts, standardize: bool) -> np.ndarray:
    y_t = y.astype(np.float32)
    if standardize:
        y_t = y_t * artifacts.target_std + artifacts.target_mean
    y_t = np.clip(y_t, artifacts.target_min, artifacts.target_max)
    y_t = _invert_target_transform(y_t, artifacts.target_transform)
    return y_t


def load_data(
    csv_path: str,
    target: str,
    num_features: List[str],
    cat_features: List[str],
    val_split: float,
    seed: int,
    target_transform: str = "none",
    target_standardize: bool = False,
    group_columns: List[str] | None = None,
    min_non_null_fraction: float | None = None,
    min_known_numeric: int | None = None,
    min_non_zero_fraction: float | None = None,
) -> Tuple[TabularDataset, TabularDataset, PreprocessArtifacts, List[str], List[str]]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset.")
    missing = [c for c in num_features + cat_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.copy()
    y = df[target].to_numpy(dtype=np.float32)
    df = df.drop(columns=[target])

    # Drop all-NaN numerical columns to avoid failures
    if num_features:
        all_nan = [c for c in num_features if df[c].isna().all()]
        if all_nan:
            num_features = [c for c in num_features if c not in all_nan]

    if group_columns:
        valid_groups = [c for c in group_columns if c in df.columns]
    else:
        valid_groups = []
    if valid_groups:
        group_key = (
            df[valid_groups]
            .astype(str)
            .fillna("UNK")
            .agg(lambda row: "|".join(row.astype(str)), axis=1)
        )
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, y, groups=group_key))
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    else:
        train_df, val_df, y_train, y_val = train_test_split(
            df, y, test_size=val_split, random_state=seed
        )

    if min_non_null_fraction is not None and num_features:
        keep = []
        for c in num_features:
            frac = float(train_df[c].notna().mean())
            if frac >= min_non_null_fraction:
                keep.append(c)
        num_features = keep

    if min_non_zero_fraction is not None and num_features:
        keep = []
        for c in num_features:
            if c.startswith("elem_"):
                s = train_df[c]
                frac = float((s.fillna(0.0) != 0.0).mean())
                if frac >= min_non_zero_fraction:
                    keep.append(c)
            else:
                keep.append(c)
        num_features = keep

    if min_known_numeric is not None and num_features:
        known_counts = train_df[num_features].notna().sum(axis=1)
        train_mask = known_counts >= min_known_numeric
        train_df = train_df.loc[train_mask]
        y_train = y_train[train_mask.to_numpy()]

        known_counts_val = val_df[num_features].notna().sum(axis=1)
        val_mask = known_counts_val >= min_known_numeric
        val_df = val_df.loc[val_mask]
        y_val = y_val[val_mask.to_numpy()]

    artifacts = fit_preprocessor(
        train_df,
        num_features,
        cat_features,
        y_train,
        target_transform,
        target_standardize,
    )

    x_num_train, x_num_mask_train, x_cat_train = transform_df(
        train_df, num_features, cat_features, artifacts
    )
    x_num_val, x_num_mask_val, x_cat_val = transform_df(
        val_df, num_features, cat_features, artifacts
    )

    y_train_t = transform_target(y_train, artifacts, target_standardize)
    y_val_t = transform_target(y_val, artifacts, target_standardize)
    train_ds = TabularDataset(x_num_train, x_num_mask_train, x_cat_train, y_train_t)
    val_ds = TabularDataset(x_num_val, x_num_mask_val, x_cat_val, y_val_t)
    return train_ds, val_ds, artifacts, num_features, cat_features


def datasets_to_arrays(train_ds: TabularDataset, val_ds: TabularDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.concatenate(
        [train_ds.x_num.numpy(), train_ds.x_num_mask.numpy(), train_ds.x_cat.numpy().astype(np.float32)],
        axis=1,
    )
    x_val = np.concatenate(
        [val_ds.x_num.numpy(), val_ds.x_num_mask.numpy(), val_ds.x_cat.numpy().astype(np.float32)],
        axis=1,
    )
    y_train = train_ds.y.numpy().squeeze()
    y_val = val_ds.y.numpy().squeeze()
    return x_train, y_train, x_val, y_val


def save_artifacts(artifacts: PreprocessArtifacts, scaler_path: str, encoder_path: str) -> None:
    joblib.dump(
        {
            "scaler": artifacts.scaler,
            "num_medians": artifacts.num_medians,
            "target_mean": artifacts.target_mean,
            "target_std": artifacts.target_std,
            "target_transform": artifacts.target_transform,
            "target_min": artifacts.target_min,
            "target_max": artifacts.target_max,
        },
        scaler_path,
    )
    joblib.dump(artifacts.cat_encoder, encoder_path)


def load_artifacts(scaler_path: str, encoder_path: str) -> PreprocessArtifacts:
    scaler_blob = joblib.load(scaler_path)
    if isinstance(scaler_blob, dict):
        scaler = scaler_blob.get("scaler")
        num_medians = scaler_blob.get("num_medians")
        target_mean = scaler_blob.get("target_mean", 0.0)
        target_std = scaler_blob.get("target_std", 1.0)
        target_transform = scaler_blob.get("target_transform", "none")
        target_min = scaler_blob.get("target_min", -30.0)
        target_max = scaler_blob.get("target_max", 30.0)
    else:
        scaler = scaler_blob
        num_medians = None
        target_mean = 0.0
        target_std = 1.0
        target_transform = "none"
        target_min = -30.0
        target_max = 30.0
    encoder = joblib.load(encoder_path)
    if num_medians is None:
        num_medians = np.zeros((getattr(scaler, "n_features_in_", 1),), dtype=np.float32)
    return PreprocessArtifacts(
        scaler=scaler,
        cat_encoder=encoder,
        num_medians=num_medians,
        target_mean=float(target_mean),
        target_std=float(target_std),
        target_transform=str(target_transform),
        target_min=float(target_min),
        target_max=float(target_max),
    )
