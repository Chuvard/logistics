"""Evaluation metrics for classification and regression.

Classification reports accuracy, precision, recall, F1, ROC-AUC and average
precision, plus the confusion matrix and calibration curve. Average precision
is included deliberately: on imbalanced problems like fraud (0.1% positive),
ROC-AUC looks impressive while the model is useless, and PR-AUC does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

__all__ = [
    "classification_metrics", "regression_metrics", "confusion_frame",
    "calibration_frame", "primary_metric_value", "METRIC_DIRECTION",
]

# Whether a higher value is better - drives model ranking and promotion.
METRIC_DIRECTION = {
    "accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "f1_macro": 1,
    "roc_auc": 1, "average_precision": 1, "balanced_accuracy": 1, "mcc": 1,
    "brier_score": -1, "log_loss": -1,
    "rmse": -1, "mae": -1, "mape": -1, "median_ae": -1, "max_error": -1,
    "r2": 1, "explained_variance": 1,
}


def _safe(fn, *args, **kwargs) -> float:
    """Metrics fail on degenerate inputs (single class, zero variance). Return
    NaN rather than aborting an entire benchmark run."""
    try:
        value = float(fn(*args, **kwargs))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def classification_metrics(
    y_true, y_pred, y_proba=None, task_type: str = "binary", labels=None
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = "binary" if task_type == "binary" else "macro"

    out = {
        "accuracy": _safe(skm.accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(skm.balanced_accuracy_score, y_true, y_pred),
        "precision": _safe(skm.precision_score, y_true, y_pred, average=average, zero_division=0),
        "recall": _safe(skm.recall_score, y_true, y_pred, average=average, zero_division=0),
        "f1": _safe(skm.f1_score, y_true, y_pred, average=average, zero_division=0),
        "f1_macro": _safe(skm.f1_score, y_true, y_pred, average="macro", zero_division=0),
        "mcc": _safe(skm.matthews_corrcoef, y_true, y_pred),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba)
        if task_type == "binary":
            p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
            out["roc_auc"] = _safe(skm.roc_auc_score, y_true, p1)
            out["average_precision"] = _safe(skm.average_precision_score, y_true, p1)
            out["brier_score"] = _safe(skm.brier_score_loss, y_true, p1)
            out["log_loss"] = _safe(skm.log_loss, y_true, np.clip(p1, 1e-7, 1 - 1e-7))
        else:
            out["roc_auc"] = _safe(skm.roc_auc_score, y_true, proba,
                                   multi_class="ovr", average="macro", labels=labels)
            out["log_loss"] = _safe(skm.log_loss, y_true, proba, labels=labels)

    return {k: round(v, 6) if np.isfinite(v) else None for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAPE explodes when actuals approach zero (common for delay in minutes),
    # so it is computed only over rows where the denominator is meaningful.
    mask = np.abs(y_true) > 1e-6
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.sum() > 0 else float("nan"))

    out = {
        "rmse": _safe(lambda a, b: np.sqrt(skm.mean_squared_error(a, b)), y_true, y_pred),
        "mae": _safe(skm.mean_absolute_error, y_true, y_pred),
        "median_ae": _safe(skm.median_absolute_error, y_true, y_pred),
        "max_error": _safe(skm.max_error, y_true, y_pred),
        "r2": _safe(skm.r2_score, y_true, y_pred),
        "explained_variance": _safe(skm.explained_variance_score, y_true, y_pred),
        "mape": mape,
    }
    return {k: round(v, 6) if np.isfinite(v) else None for k, v in out.items()}


def confusion_frame(y_true, y_pred, labels=None) -> pd.DataFrame:
    labels = labels if labels is not None else sorted(pd.unique(np.asarray(y_true)))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm,
                        index=[f"actual_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def calibration_frame(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed positive rate per probability bin.

    A well-calibrated model sits on the diagonal. Tree ensembles usually do not,
    which is why the platform offers isotonic calibration.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": round(float(p[m].mean()), 4),
            "observed_rate": round(float(y_true[m].mean()), 4),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap"] = (df["mean_predicted"] - df["observed_rate"]).round(4)
    return df


def primary_metric_value(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if value is None or not np.isfinite(value):
        return float("-inf")
    return float(value) * METRIC_DIRECTION.get(name, 1)
