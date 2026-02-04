"""Prepare raw fracture toughness data into model-ready form."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


COMPOSITION_COL = "composition_at_percent"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    for c in df.columns:
        s = c.strip().lower()
        s = s.replace("%", "percent")
        s = s.replace("(", " ").replace(")", " ")
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        cols.append(s)
    df.columns = cols
    return df


def _parse_composition_entry(entry: str) -> Dict[str, float]:
    if not isinstance(entry, str) or not entry.strip():
        return {}
    parts = entry.split("-")
    result: Dict[str, float] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^([A-Z][a-z]?)([0-9]*\.?[0-9]+)?$", part)
        if not match:
            continue
        element = match.group(1)
        value = match.group(2)
        if value is None:
            continue
        try:
            result[element] = float(value)
        except ValueError:
            continue
    return result


def _expand_composition(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[str]]:
    parsed = df[col].apply(_parse_composition_entry)
    elements = sorted({k for d in parsed for k in d.keys()})
    comp_df = pd.DataFrame([{e: d.get(e, 0.0) for e in elements} for d in parsed])
    comp_df.columns = [f"elem_{e.lower()}" for e in comp_df.columns]
    df = df.drop(columns=[col])
    df = pd.concat([df.reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)
    return df, list(comp_df.columns)


def prepare_dataframe(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = _normalize_columns(df)
    df = df.dropna(how="all")

    if COMPOSITION_COL in df.columns:
        df, elem_cols = _expand_composition(df, COMPOSITION_COL)
    else:
        elem_cols = []

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Coerce known numeric columns to numeric
    for col in df.columns:
        if col == target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col.startswith("elem_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif any(x in col for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, elem_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--unseen", type=str, required=True)
    parser.add_argument("--out_train", type=str, required=True)
    parser.add_argument("--out_unseen", type=str, required=True)
    parser.add_argument("--schema", type=str, default="data/schema.json")
    parser.add_argument("--target", type=str, default="fracture_toughness_mpa_m0_5")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, skip_blank_lines=True)
    unseen_df = pd.read_csv(args.unseen, skip_blank_lines=True)

    drop_cols = ["reference"]

    train_df, elem_cols = prepare_dataframe(train_df, args.target, drop_cols)
    unseen_df, _ = prepare_dataframe(unseen_df, args.target, drop_cols)

    # Align columns
    all_cols = sorted(set(train_df.columns) | set(unseen_df.columns))
    train_df = train_df.reindex(columns=all_cols)
    unseen_df = unseen_df.reindex(columns=all_cols)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    train_df.to_csv(args.out_train, index=False)
    unseen_df.to_csv(args.out_unseen, index=False)

    # Infer feature types
    num_features = [
        c
        for c in train_df.columns
        if c.startswith("elem_") or any(x in c for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"])
    ]
    if args.target in num_features:
        num_features.remove(args.target)
    cat_features = [c for c in train_df.columns if c not in num_features + [args.target]]

    schema = {
        "target": args.target,
        "numerical_features": sorted(num_features),
        "categorical_features": sorted(cat_features),
        "element_features": sorted(elem_cols),
    }

    os.makedirs(os.path.dirname(args.schema), exist_ok=True)
    with open(args.schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


if __name__ == "__main__":
    main()
