"""Preprocessing for the modelling table.

Raw tables are always exported untouched - the whole point of simulating
missingness and anomalies is that a downstream user gets to practise on the
mess. This module produces the *additional* cleaned table for people who want
to go straight to modelling.

Steps: drop leakage columns → missing indicators → impute → clip outliers →
encode categoricals → drop constants.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import clip_series, get_logger

__all__ = ["preprocess", "LEAKAGE_COLUMNS"]

logger = get_logger()

# Columns that reveal the outcome and must not be fed to a model that predicts
# it. Kept explicit so the exclusion is auditable rather than implicit.
LEAKAGE_COLUMNS = [
    "delay_minutes", "is_late", "actual_delivery_ts", "actual_duration_min",
    "status", "delivery_cost_usd", "labour_cost_usd", "fuel_cost_usd",
    "toll_cost_usd", "handling_cost_usd", "revenue_usd", "margin_usd",
    "margin_pct", "cost_per_km_usd", "cost_per_kg_usd", "speed_kmh_effective",
    "planned_vs_actual_ratio", "sla_utilisation", "route_on_time_rate",
    "delivery_attempts", "co2_kg", "co2_per_km", "target_risk_score",
]


def preprocess(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Return ``(clean_df, metadata)``.

    ``metadata`` records every decision made (imputation values, encoding maps,
    dropped columns) so the transform can be replayed on new data.
    """
    p = cfg.get("preprocessing", {}) or {}
    if not p.get("enabled", True):
        return df, {"enabled": False}

    out = df.copy()
    meta: dict = {"enabled": True, "n_rows_in": len(df), "n_cols_in": df.shape[1]}

    target_cols = [c for c in out.columns if c.startswith("target_")]
    id_cols = [c for c in out.columns if c.endswith("_id")]

    # ---- 1. drop leakage ----------------------------------------------------
    dropped_leak = [c for c in LEAKAGE_COLUMNS if c in out.columns]
    out = out.drop(columns=dropped_leak)
    meta["dropped_leakage_columns"] = dropped_leak

    # ---- 2. missing indicators ---------------------------------------------
    if p.get("add_missing_indicators", True):
        indicators = {}
        for col in out.columns:
            if col in target_cols or col in id_cols:
                continue
            n_missing = out[col].isna().sum()
            if 0 < n_missing < len(out):
                indicators[f"{col}__was_missing"] = out[col].isna().astype("int8")
        if indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=out.index)], axis=1)
        meta["missing_indicators_added"] = len(indicators)

    # ---- 3. impute ----------------------------------------------------------
    num_strategy = p.get("impute_numeric", "median")
    cat_strategy = p.get("impute_categorical", "mode")
    fills: dict[str, object] = {}

    for col in out.columns:
        if col in target_cols or out[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            if num_strategy == "none":
                continue
            value = {"median": out[col].median(), "mean": out[col].mean(), "zero": 0.0}.get(
                num_strategy, out[col].median())
            if pd.isna(value):
                value = 0.0
            out[col] = out[col].fillna(value)
            fills[col] = float(value)
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        else:
            if cat_strategy == "none":
                continue
            if cat_strategy == "mode":
                modes = out[col].mode(dropna=True)
                value = modes.iloc[0] if len(modes) else "unknown"
            else:
                value = "unknown"
            out[col] = out[col].astype("object").fillna(value)
            fills[col] = str(value)
    meta["imputation_values"] = fills

    # ---- 4. clip outliers ---------------------------------------------------
    q = p.get("clip_outliers_quantile")
    if q:
        clipped = []
        for col in out.select_dtypes(include=[np.number]).columns:
            if col in target_cols or col.endswith("__was_missing"):
                continue
            before = out[col].copy()
            out[col] = clip_series(out[col], float(q))
            if not before.equals(out[col]):
                clipped.append(col)
        meta["clipped_columns"] = clipped
        meta["clip_quantile"] = float(q)

    # ---- 5. encode categoricals --------------------------------------------
    encoding = p.get("encode_categoricals", "ordinal")
    if encoding and encoding != "none":
        cat_cols = [c for c in out.columns
                    if (out[c].dtype == object or isinstance(out[c].dtype, pd.CategoricalDtype))
                    and c not in id_cols and c not in target_cols]
        if encoding == "ordinal":
            maps: dict[str, dict] = {}
            for col in cat_cols:
                codes, uniques = pd.factorize(out[col].astype(str), sort=True)
                out[col] = codes.astype("int32")
                maps[col] = {str(v): int(i) for i, v in enumerate(uniques)}
            meta["ordinal_maps"] = maps
        elif encoding == "onehot":
            low_card = [c for c in cat_cols if out[c].nunique() <= 25]
            out = pd.get_dummies(out, columns=low_card, dummy_na=False, dtype="int8")
            out = out.drop(columns=[c for c in cat_cols if c not in low_card], errors="ignore")
            meta["onehot_columns"] = low_card
    meta["encoding"] = encoding

    # ---- 6. drop constants --------------------------------------------------
    if p.get("drop_constant_columns", True):
        constants = [c for c in out.columns
                     if c not in target_cols and out[c].nunique(dropna=False) <= 1]
        out = out.drop(columns=constants)
        meta["dropped_constant_columns"] = constants

    meta["n_rows_out"] = len(out)
    meta["n_cols_out"] = out.shape[1]
    logger.info("Preprocessed ML table: %d x %d (from %d x %d)",
                len(out), out.shape[1], meta["n_rows_in"], meta["n_cols_in"])
    return out, meta


def save_metadata(meta: dict, path) -> None:
    """Write preprocessing metadata as JSON (numpy types coerced)."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=default), encoding="utf-8")
