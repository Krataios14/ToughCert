"""Schema helpers."""

import json
from typing import Dict


def apply_schema(cfg: Dict) -> Dict:
    schema_path = cfg.get("data", {}).get("schema_json")
    if not schema_path:
        return cfg
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    if "target" in schema:
        cfg["data"]["target"] = schema["target"]
    if "numerical_features" in schema:
        cfg["data"]["numerical_features"] = schema["numerical_features"]
    if "categorical_features" in schema:
        cfg["data"]["categorical_features"] = schema["categorical_features"]
    return cfg
