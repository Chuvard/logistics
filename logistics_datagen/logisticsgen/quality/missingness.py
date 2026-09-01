"""Configurable missing-data simulation.

Three mechanisms, following Rubin's taxonomy:

* **MCAR** - Missing Completely At Random. Every row has the same probability.
* **MAR**  - Missing At Random. Probability depends on *another observed* column
  (``driver``). Imputation using the rest of the data is unbiased.
* **MNAR** - Missing Not At Random. Probability depends on the hidden value
  itself (high values or low values go missing). This is the hard case, and the
  reason we record ground truth before masking.

Every masked cell is recorded so downstream imputation benchmarks can score
against the true values, and a per-column report is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..rng import RandomStreams
from ..utils import get_logger

__all__ = ["MissingnessReport", "apply_missingness", "sweep_missingness"]

logger = get_logger()


@dataclass
class MissingnessReport:
    rows: list[dict] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.rows.append(kwargs)

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=[
                "table", "column", "mechanism", "target_rate", "achieved_rate",
                "n_rows", "n_missing", "n_missing_before"])
        return pd.DataFrame(self.rows)


def _rank01(values: pd.Series) -> np.ndarray:
    """Percentile rank in [0, 1]; NaNs sit at the midpoint so they aren't
    preferentially selected by the MNAR tilt."""
    ranked = values.rank(pct=True, na_option="keep").to_numpy(dtype=float)
    return np.nan_to_num(ranked, nan=0.5)


def _mask_mcar(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    return rng.random(n) < rate


def _mask_mar(
    rng: np.random.Generator, df: pd.DataFrame, driver: str, rate: float
) -> np.ndarray:
    """Missingness probability varies across the levels of ``driver``.

    Each level gets a multiplier drawn once, then probabilities are renormalised
    so the *overall* rate still matches the configured target.
    """
    n = len(df)
    if driver not in df.columns:
        logger.warning("MAR driver column %r not found - falling back to MCAR", driver)
        return _mask_mcar(rng, n, rate)

    col = df[driver]
    if pd.api.types.is_numeric_dtype(col) and col.nunique(dropna=True) > 12:
        weight = 0.35 + 1.3 * _rank01(col)
    else:
        levels = col.astype(str).where(col.notna(), "__NA__").to_numpy()
        uniques, inverse = np.unique(levels.astype(str), return_inverse=True)
        multipliers = rng.uniform(0.25, 2.4, len(uniques))
        weight = multipliers[inverse]

    p = np.clip(weight * rate / max(weight.mean(), 1e-9), 0.0, 0.98)
    return rng.random(n) < p


def _mask_mnar(
    rng: np.random.Generator, series: pd.Series, rate: float, direction: str = "high"
) -> np.ndarray:
    """Missingness depends on the value being hidden.

    ``direction='high'`` hides large values (e.g. customers who won't disclose
    high-value goods); ``'low'`` hides small ones (e.g. unhappy customers who
    never submit a rating).
    """
    n = len(series)
    if pd.api.types.is_numeric_dtype(series):
        r = _rank01(series)
    else:
        codes = pd.factorize(series)[0].astype(float)
        r = _rank01(pd.Series(codes))
    tilt = r if direction == "high" else (1.0 - r)
    weight = 0.15 + 1.85 * tilt
    p = np.clip(weight * rate / max(weight.mean(), 1e-9), 0.0, 0.98)
    return rng.random(n) < p


def apply_missingness(
    tables: dict[str, pd.DataFrame],
    cfg: Config,
    streams: RandomStreams,
    rate_override: float | None = None,
) -> tuple[dict[str, pd.DataFrame], MissingnessReport, dict[str, pd.DataFrame]]:
    """Mask cells per the configured rules.

    Returns ``(tables, report, ground_truth)`` where ``ground_truth`` maps
    ``"<table>.<column>"`` to a frame of the original values at masked positions.
    """
    report = MissingnessReport()
    ground_truth: dict[str, pd.DataFrame] = {}

    if not cfg.get("missingness.enabled", True):
        logger.info("Missingness disabled by config")
        return tables, report, ground_truth

    rules = cfg.get("missingness.rules", []) or []
    default_rate = float(cfg.get("missingness.default_rate", 0.05))

    for rule in rules:
        table = rule["table"]
        column = rule["column"]
        mechanism = str(rule.get("mechanism", "MCAR")).upper()
        rate = float(rate_override if rate_override is not None else rule.get("rate", default_rate))

        df = tables.get(table)
        if df is None or df.empty or column not in df.columns:
            logger.debug("Skipping missingness rule %s.%s (absent)", table, column)
            continue

        rng = streams.spawn(f"missing::{table}::{column}::{rate}")
        before = int(df[column].isna().sum())

        if mechanism == "MCAR":
            mask = _mask_mcar(rng, len(df), rate)
        elif mechanism == "MAR":
            mask = _mask_mar(rng, df, rule.get("driver", ""), rate)
        elif mechanism == "MNAR":
            mask = _mask_mnar(rng, df[column], rate, rule.get("direction", "high"))
        else:
            raise ValueError(f"Unknown missingness mechanism: {mechanism!r}")

        mask = mask & df[column].notna().to_numpy()
        if mask.any():
            key_col = df.columns[0]
            ground_truth[f"{table}.{column}"] = pd.DataFrame({
                "row_key": df.loc[mask, key_col].to_numpy(),
                "true_value": df.loc[mask, column].to_numpy(),
                "mechanism": mechanism,
            })
            df.loc[mask, column] = np.nan

        report.add(
            table=table, column=column, mechanism=mechanism, target_rate=rate,
            achieved_rate=round(float(df[column].isna().mean()), 6),
            n_rows=len(df), n_missing=int(df[column].isna().sum()), n_missing_before=before,
        )

    logger.info("Applied %d missingness rules", len(report.rows))
    return tables, report, ground_truth


def sweep_missingness(
    tables: dict[str, pd.DataFrame], cfg: Config, streams: RandomStreams
) -> dict[float, pd.DataFrame]:
    """Produce a report per rate in ``missingness.rate_sweep``.

    Used to benchmark imputation strategies at 1/5/10/20/40% severity without
    regenerating the underlying data. Operates on copies - inputs are untouched.
    """
    out: dict[float, pd.DataFrame] = {}
    for rate in cfg.get("missingness.rate_sweep", []) or []:
        copies = {k: v.copy(deep=True) for k, v in tables.items()}
        _, report, _ = apply_missingness(copies, cfg, streams, rate_override=float(rate))
        out[float(rate)] = report.to_frame()
    return out
