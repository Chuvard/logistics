"""Supervised learning targets.

Six targets spanning regression, binary and multiclass so the downstream ML
platform (prompt 02) has something to work with out of the box:

============================  ============  ===========================
Target                        Task          Business question
============================  ============  ===========================
``target_delay_minutes``      regression    How late will this run?
``target_is_late``            binary        Will we breach the SLA?
``target_eta_minutes``        regression    How long will it take?
``target_delivery_cost_usd``  regression    What will it cost to serve?
``target_risk_bucket``        multiclass    How risky is this delivery?
``target_will_be_returned``   binary        Will it come back to us?
============================  ============  ===========================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger

__all__ = ["add_targets", "TARGET_COLUMNS"]

logger = get_logger()

TARGET_COLUMNS = [
    "target_delay_minutes", "target_is_late", "target_eta_minutes",
    "target_delivery_cost_usd", "target_risk_bucket", "target_will_be_returned",
]


def add_targets(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach configured targets. Rows with unknown outcomes keep NaN targets -
    dropping them is left to the modelling stage, which knows its own needs."""
    t = cfg.get("targets", {}) or {}
    df = df.copy()

    if t.get("delivery_delay_minutes", True):
        df["target_delay_minutes"] = df["delay_minutes"]

    if t.get("is_late", True):
        # Only defined where the order actually completed.
        known = df["actual_delivery_ts"].notna()
        df["target_is_late"] = np.where(known, (df["delay_minutes"] > 0).astype("float"), np.nan)

    if t.get("eta_minutes", True):
        df["target_eta_minutes"] = df["actual_duration_min"]

    if t.get("delivery_cost_usd", True):
        df["target_delivery_cost_usd"] = df["delivery_cost_usd"]

    if t.get("risk_bucket", True):
        # Composite risk: lateness pressure, environment, fragility, difficulty.
        delay = df["delay_minutes"].fillna(0)
        risk = (
            np.clip(delay / 120.0, 0, 1) * 0.42
            + df["env_stress_index"].fillna(0.3) * 0.22
            + df.get("zone_difficulty", pd.Series(0.5, index=df.index)).fillna(0.5) * 0.14
            + df["fragile"].astype(float) * 0.08
            + df["cold_chain_required"].astype(float) * 0.08
            + (df["status"].isin(["failed", "returned"])).astype(float) * 0.30
        )
        df["target_risk_score"] = risk.round(4)
        df["target_risk_bucket"] = pd.cut(
            risk, bins=[-np.inf, 0.25, 0.5, np.inf], labels=["low", "medium", "high"]).astype(str)

    if t.get("will_be_returned", True):
        df["target_will_be_returned"] = df["status"].eq("returned").astype(int)

    present = [c for c in TARGET_COLUMNS if c in df.columns]
    logger.info("Targets added: %s", ", ".join(present))
    return df
