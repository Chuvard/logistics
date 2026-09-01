"""Stage 1 - preprocessing and splitting.

Two rules drive every decision here:

1. **Fit on train only.** Imputers, scalers and encoders never see validation or
   test rows. Fitting on the whole frame is the most common silent leak in
   tabular ML and it inflates every metric downstream.
2. **Split by time by default.** Predicting SLA breaches is a forecasting
   problem. A random split lets the model learn from the future, which looks
   great in the report and fails in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler,
)

from ..config import Config
from ..data import Dataset
from ..utils import get_logger

__all__ = ["SplitData", "split_dataset", "build_preprocessor", "preprocess"]

logger = get_logger()


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_train": len(self.X_train), "n_val": len(self.X_val), "n_test": len(self.X_test),
            "n_features": len(self.feature_names), **self.meta,
        }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_dataset(ds: Dataset, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return boolean index arrays for train / val / test plus split metadata."""
    strategy = str(cfg.get("split.strategy", "temporal")).lower()
    test_size = float(cfg.get("split.test_size", 0.2))
    val_size = float(cfg.get("split.val_size", 0.1))
    n = len(ds.X)
    idx = np.arange(n)

    if strategy == "temporal" and ds.time_index is not None:
        order = np.argsort(ds.time_index.to_numpy())
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        meta = {
            "split_strategy": "temporal",
            "train_end": str(ds.time_index.iloc[train_idx].max()),
            "test_start": str(ds.time_index.iloc[test_idx].min()),
        }

    elif strategy == "group" and ds.groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed)
        rest_idx, test_idx = next(gss.split(idx, groups=ds.groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                                  random_state=cfg.seed)
        tr, va = next(inner.split(rest_idx, groups=ds.groups.iloc[rest_idx]))
        train_idx, val_idx = rest_idx[tr], rest_idx[va]
        meta = {"split_strategy": "group", "group_column": cfg.get("split.group_column")}

    else:
        from sklearn.model_selection import train_test_split
        stratify = ds.y if (cfg.get("split.stratify", True)
                            and ds.task_type in {"binary", "multiclass"}) else None
        rest_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=cfg.seed, stratify=stratify)
        strat2 = ds.y.iloc[rest_idx] if stratify is not None else None
        train_idx, val_idx = train_test_split(
            rest_idx, test_size=val_size / (1 - test_size),
            random_state=cfg.seed, stratify=strat2)
        meta = {"split_strategy": "random"}

    if strategy == "temporal" and ds.time_index is None:
        logger.warning("Temporal split requested but no time column - fell back to random")

    return train_idx, val_idx, test_idx, meta


# --------------------------------------------------------------------------- #
# Column typing and transformer
# --------------------------------------------------------------------------- #
def _classify_columns(X: pd.DataFrame, cfg: Config) -> tuple[list[str], list[str], list[str]]:
    """Split columns into numeric, categorical and dropped."""
    max_card = int(cfg.get("preprocessing.drop_high_cardinality_above", 5000))
    numeric, categorical, dropped = [], [], []

    for col in X.columns:
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            # Datetimes are already expanded into cyclical/ordinal features
            # upstream; keeping the raw stamp invites the model to memorise dates.
            dropped.append(col)
        elif pd.api.types.is_bool_dtype(s):
            numeric.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            if s.nunique(dropna=True) > max_card:
                dropped.append(col)
            else:
                categorical.append(col)
    return numeric, categorical, dropped


def _scaler(kind: str):
    return {
        "standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(),
    }.get(str(kind).lower())


def build_preprocessor(X: pd.DataFrame, cfg: Config) -> tuple[ColumnTransformer, dict]:
    """Assemble the fit-on-train ColumnTransformer."""
    numeric, categorical, dropped = _classify_columns(X, cfg)
    p = cfg.get("preprocessing", {}) or {}

    num_steps = [("impute", SimpleImputer(
        strategy=str(p.get("impute_numeric", "median")),
        add_indicator=bool(p.get("add_missing_indicators", True))))]
    scaler = _scaler(p.get("scale_numeric", "standard"))
    if scaler is not None:
        num_steps.append(("scale", scaler))

    encoding = str(p.get("categorical_encoding", "ordinal")).lower()
    if encoding == "onehot":
        max_oh = int(p.get("max_onehot_cardinality", 20))
        low = [c for c in categorical if X[c].nunique(dropna=True) <= max_oh]
        high = [c for c in categorical if c not in low]
        transformers = [
            ("num", Pipeline(num_steps), numeric),
            ("cat_low", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), low),
            ("cat_high", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), high),
        ]
    else:
        transformers = [
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy=str(p.get("impute_categorical", "most_frequent")))),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
        ]

    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    meta = {
        "n_numeric": len(numeric), "n_categorical": len(categorical),
        "dropped_columns": dropped, "encoding": encoding,
        "scaler": str(p.get("scale_numeric", "standard")),
    }
    return ct, meta


def preprocess(ds: Dataset, cfg: Config) -> SplitData:
    """Split, then fit the preprocessor on train and apply it everywhere."""
    train_idx, val_idx, test_idx, split_meta = split_dataset(ds, cfg)

    X_tr_raw = ds.X.iloc[train_idx]
    X_va_raw = ds.X.iloc[val_idx]
    X_te_raw = ds.X.iloc[test_idx]

    ct, meta = build_preprocessor(X_tr_raw, cfg)
    ct.fit(X_tr_raw)  # <- train only

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        arr = ct.transform(frame)
        names = list(ct.get_feature_names_out())
        out = pd.DataFrame(np.asarray(arr, dtype=float), columns=names, index=frame.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, X_val, X_test = _apply(X_tr_raw), _apply(X_va_raw), _apply(X_te_raw)

    # Constant columns survive imputation but tell a model nothing.
    if cfg.get("preprocessing.drop_constant", True):
        keep = X_train.columns[X_train.nunique() > 1]
        dropped_const = [c for c in X_train.columns if c not in set(keep)]
        X_train, X_val, X_test = X_train[keep], X_val[keep], X_test[keep]
        meta["dropped_constant"] = dropped_const

    meta.update(split_meta)
    meta["preprocessor"] = ct

    sd = SplitData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=ds.y.iloc[train_idx].reset_index(drop=True),
        y_val=ds.y.iloc[val_idx].reset_index(drop=True),
        y_test=ds.y.iloc[test_idx].reset_index(drop=True),
        feature_names=list(X_train.columns), meta=meta,
    )
    for frame in (sd.X_train, sd.X_val, sd.X_test):
        frame.reset_index(drop=True, inplace=True)

    logger.info("Split (%s): train %d | val %d | test %d | %d features",
                split_meta.get("split_strategy"), len(X_train), len(X_val),
                len(X_test), len(sd.feature_names))
    return sd
