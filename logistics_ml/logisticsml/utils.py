"""Shared plumbing: logging, timing, optional-dependency handling, plotting.

The platform is designed to degrade gracefully. Heavy libraries (XGBoost,
LightGBM, CatBoost, torch, SHAP, UMAP, OR-Tools) are all optional: if one is
absent the corresponding model is skipped with a clear note in the report
rather than crashing a 20-minute pipeline run.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "get_logger", "timed", "optional_import", "have", "dependency_report",
    "save_json", "set_seed", "new_figure", "save_figure", "slugify",
]

_OPTIONAL_CACHE: dict[str, Any] = {}


def get_logger(name: str = "logisticsml") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


@contextmanager
def timed(label: str, log: logging.Logger | None = None):
    log = log or logger
    start = time.perf_counter()
    log.info("→ %s", label)
    try:
        yield
    finally:
        log.info("✓ %s (%.2fs)", label, time.perf_counter() - start)


def optional_import(module: str):
    """Import ``module`` or return ``None``. Result is cached."""
    if module in _OPTIONAL_CACHE:
        return _OPTIONAL_CACHE[module]
    try:
        mod = importlib.import_module(module)
    except Exception:  # ImportError, but some libs raise OSError on bad installs
        mod = None
    _OPTIONAL_CACHE[module] = mod
    return mod


def have(module: str) -> bool:
    return optional_import(module) is not None


OPTIONAL_DEPENDENCIES = [
    ("xgboost", "XGBoost gradient boosting"),
    ("lightgbm", "LightGBM gradient boosting"),
    ("catboost", "CatBoost gradient boosting"),
    ("torch", "PyTorch deep learning (MLP, autoencoder, LSTM, transformer, TabNet)"),
    ("shap", "SHAP explanations"),
    ("lime", "LIME explanations"),
    ("umap", "UMAP dimensionality reduction"),
    ("ortools", "Google OR-Tools routing and MIP solvers"),
    ("pulp", "PuLP linear/mixed-integer programming"),
]


def dependency_report() -> pd.DataFrame:
    """What is installed, what is missing, and what each unlocks."""
    rows = []
    for module, purpose in OPTIONAL_DEPENDENCIES:
        mod = optional_import(module)
        rows.append({
            "package": module,
            "available": mod is not None,
            "version": getattr(mod, "__version__", "") if mod else "",
            "unlocks": purpose,
        })
    return pd.DataFrame(rows)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch = optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)


def save_json(obj: Any, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)).strip("_")


# --------------------------------------------------------------------------- #
# Plotting - matplotlib only, Agg backend, no interactive state
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def new_figure(width: float = 8.0, height: float = 5.0):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save_figure(fig, path: Path, dpi: int = 110) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
