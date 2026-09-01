"""Dataset loading and task construction.

Reads the output of the dataset generator (prompt 01) from Parquet, CSV or
SQLite, and assembles a ``(X, y)`` pair for whichever prediction task is
requested.

Two kinds of task exist:

* **Direct** - the target already sits in ``ml_features`` (late delivery, ETA,
  cost, risk, returns).
* **Auxiliary** - the target has to be *constructed* from another table
  (vehicle failure from maintenance history, inventory shortage from stock
  snapshots, fraud from anomaly flags, warehouse congestion from daily volume
  against capacity). These builders live here so the modelling code never has
  to know where a label came from.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

__all__ = ["Dataset", "load_tables", "load_table", "build_task"]

logger = get_logger()


@dataclass
class Dataset:
    """A task-ready modelling frame."""

    X: pd.DataFrame
    y: pd.Series
    task: dict
    time_index: pd.Series | None = None
    groups: pd.Series | None = None
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.task["name"]

    @property
    def task_type(self) -> str:
        return self.task["type"]

    def describe(self) -> dict:
        out = {
            "task": self.name,
            "type": self.task_type,
            "n_rows": int(len(self.X)),
            "n_features": int(self.X.shape[1]),
            "target": self.task["target"],
        }
        if self.task_type in {"binary", "multiclass"}:
            counts = self.y.value_counts(normalize=True).round(4)
            out["class_balance"] = {str(k): float(v) for k, v in counts.items()}
        else:
            out["target_mean"] = float(self.y.mean())
            out["target_std"] = float(self.y.std())
        return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _data_dir(cfg: Config) -> Path:
    return cfg._resolve(cfg.get("data.path", "./data"))


def load_table(cfg: Config, name: str) -> pd.DataFrame:
    """Load a single table by name from the configured source."""
    source = str(cfg.get("data.source", "parquet")).lower()
    if source == "sqlite":
        db = cfg._resolve(cfg.get("data.sqlite_path"))
        with sqlite3.connect(db) as conn:
            return pd.read_sql(f'SELECT * FROM "{name}"', conn)

    directory = _data_dir(cfg)
    if source == "csv":
        for candidate in (directory / f"{name}.csv", directory / f"{name}.csv.gz"):
            if candidate.exists():
                return pd.read_csv(candidate)
        raise FileNotFoundError(f"No CSV for table {name!r} under {directory}")

    path = directory / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No Parquet for table {name!r} at {path}")
    return pd.read_parquet(path)


def load_tables(cfg: Config, names: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for name in names:
        try:
            out[name] = load_table(cfg, name)
        except FileNotFoundError:
            logger.warning("Table %r not found - skipping", name)
    return out


# --------------------------------------------------------------------------- #
# Auxiliary target builders
# --------------------------------------------------------------------------- #
def _build_vehicle_failure(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Will this vehicle need *unplanned* maintenance?

    Features come from the vehicle master; the label is derived from the
    maintenance log. Planned servicing is excluded - predicting a scheduled
    service is trivial and worthless.
    """
    vehicles = load_table(cfg, "vehicles")
    maint = load_table(cfg, "vehicle_maintenance")

    unplanned = (maint.loc[maint["is_unplanned"] == True]  # noqa: E712
                 .groupby("vehicle_id").size().rename("unplanned_events"))
    df = vehicles.merge(unplanned, left_on="vehicle_id", right_index=True, how="left")
    df["unplanned_events"] = df["unplanned_events"].fillna(0)
    y = (df["unplanned_events"] > 0).astype(int).rename("will_fail")

    drop = ["unplanned_events", "vehicle_id", "anomaly_flags"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def _build_inventory_shortage(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Will this warehouse-SKU stock out at the *next* snapshot?

    The label is the next period's stockout flag, shifted backwards within each
    (warehouse, SKU) series. Using the current flag would be a tautology.
    """
    inv = load_table(cfg, "inventory").sort_values(["warehouse_id", "sku_id", "snapshot_date"])
    inv["will_stockout"] = (inv.groupby(["warehouse_id", "sku_id"])["stockout_flag"]
                            .shift(-1).astype("float"))
    inv = inv.dropna(subset=["will_stockout"])
    y = inv["will_stockout"].astype(int).rename("will_stockout")

    drop = ["will_stockout", "stockout_flag", "inventory_id", "sku_id",
            "warehouse_id", "anomaly_flags"]
    X = inv.drop(columns=[c for c in drop if c in inv.columns])
    return X, y


def _build_fraud(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Is this order fraudulent? Label comes from the generator's ground-truth
    anomaly flag, which is exactly what a labelled fraud dataset looks like."""
    orders = load_table(cfg, "orders")
    flags = orders["anomaly_flags"].fillna("")
    y = flags.str.contains("fraudulent_order").astype(int).rename("is_fraud")

    # The injector sets these three fields as part of the fraud signature, so
    # leaving them in would hand the model the answer.
    # `status` and `anomaly_flags` are outcomes and must go. Declared value and
    # payment method stay: both are known the moment the order is placed, and an
    # implausibly high declared value paid cash-on-delivery is exactly the
    # pattern a fraud model should learn. Stripping them too would leave the
    # task with no signal at all rather than making it honest.
    drop = ["anomaly_flags", "status", "order_id", "is_duplicate",
            "actual_delivery_ts", "delay_minutes", "is_late"]
    X = orders.drop(columns=[c for c in drop if c in orders.columns])
    return X, y


def _build_warehouse_congestion(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Will a site exceed its rated throughput on a given day?

    Aggregates orders to warehouse-day, compares volume against the site's
    capacity, and predicts the breach from *lagged* volume and site attributes.
    """
    orders = load_table(cfg, "orders")
    warehouses = load_table(cfg, "warehouses")

    orders["date"] = pd.to_datetime(orders["order_timestamp"]).dt.normalize()
    daily = (orders.groupby(["warehouse_id", "date"])
             .agg(orders_count=("order_id", "count"),
                  total_weight_kg=("package_weight_kg", "sum"),
                  avg_distance_km=("distance_km", "mean"),
                  late_rate=("is_late", "mean"),
                  cold_chain_share=("cold_chain_required", "mean"))
             .reset_index())

    wh = warehouses.set_index("warehouse_id")
    cap = wh["throughput_capacity_orders_day"].reindex(daily["warehouse_id"]).to_numpy()
    daily["capacity"] = cap
    daily["utilisation"] = daily["orders_count"] / np.maximum(cap, 1)

    # Congestion is defined *relative to each site's own normal day*, not
    # against rated throughput. Rated capacity is sized for peak volumes, so an
    # absolute threshold labels almost nothing as congested and leaves the task
    # with a single class. "Unusually busy for this site" is both learnable and
    # what an operations team actually reacts to.
    quantile = float(cfg.get("tasks.warehouse_congestion.busy_quantile", 0.85))
    site_threshold = daily.groupby("warehouse_id")["orders_count"].transform(
        lambda s: s.quantile(quantile))
    daily["is_congested"] = (daily["orders_count"] > site_threshold).astype(int)

    daily = daily.sort_values(["warehouse_id", "date"])
    grp = daily.groupby("warehouse_id")
    for lag in (1, 2, 7):
        daily[f"orders_lag{lag}"] = grp["orders_count"].shift(lag)
    daily["orders_roll7"] = grp["orders_count"].shift(1).rolling(7, min_periods=2).mean() \
                                               .reset_index(level=0, drop=True)
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    for col in ["warehouse_type", "automation_level", "dock_doors", "staff_headcount",
                "operating_hours_per_day", "region", "capacity_m3"]:
        if col in wh.columns:
            daily[col] = wh[col].reindex(daily["warehouse_id"]).to_numpy()

    daily = daily.dropna(subset=["orders_lag1"])
    y = daily["is_congested"].rename("is_congested")

    # Same-day volume and utilisation define the label - both must go.
    drop = ["is_congested", "utilisation", "orders_count", "warehouse_id",
            "total_weight_kg", "late_rate", "avg_distance_km", "cold_chain_share"]
    X = daily.drop(columns=[c for c in drop if c in daily.columns])
    return X, y


_BUILDERS = {
    "vehicle_failure": _build_vehicle_failure,
    "inventory_shortage": _build_inventory_shortage,
    "fraud": _build_fraud,
    "warehouse_congestion": _build_warehouse_congestion,
}


# --------------------------------------------------------------------------- #
# Task assembly
# --------------------------------------------------------------------------- #
def build_task(cfg: Config, task_name: str | None = None) -> Dataset:
    """Assemble the ``(X, y)`` pair for a task, with leakage columns removed."""
    task = cfg.task(task_name)
    builder_key = task.get("build")

    if builder_key:
        if builder_key not in _BUILDERS:
            raise KeyError(f"No builder registered for {builder_key!r}")
        logger.info("Building auxiliary task %r via %s", task["name"], builder_key)
        X, y = _BUILDERS[builder_key](cfg)
        time_index = None
        groups = None
    else:
        table = cfg.get("data.primary_table", "ml_features")
        df = load_table(cfg, table)

        max_rows = cfg.get("data.max_rows")
        if max_rows and len(df) > int(max_rows):
            df = df.sample(int(max_rows), random_state=cfg.seed).sort_index()
            logger.info("Subsampled %s to %d rows", table, len(df))

        target = task["target"]
        if target not in df.columns:
            raise KeyError(f"Target {target!r} not present in {table!r}")

        df = df.loc[df[target].notna()].copy()
        y = df[target]

        time_col = cfg.get("split.time_column", "order_timestamp")
        time_index = pd.to_datetime(df[time_col]) if time_col in df.columns else None
        group_col = cfg.get("split.group_column")
        groups = df[group_col] if group_col in df.columns else None

        # Every other target is a leak, as is anything the task declares.
        other_targets = [c for c in df.columns if c.startswith("target_") and c != target]
        extra = list(task.get("extra_leakage", []) or [])
        X = df.drop(columns=[target, *other_targets, *extra], errors="ignore")

    # Identifiers carry no signal and would let tree models memorise rows.
    id_cols = list(cfg.get("data.id_columns", []) or [])
    X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")

    # Free-text anomaly flags encode ground truth for other tasks - never a feature.
    X = X.drop(columns=[c for c in X.columns if c.endswith("anomaly_flags")], errors="ignore")

    # ---- leakage guard -----------------------------------------------------
    # See the long note on `data.leakage_columns` in the config. This runs after
    # the target has been extracted, so a task may still predict one of these.
    leak = set(cfg.get("data.leakage_columns", []) or [])
    leak |= {f"{c}__was_missing" for c in leak}          # generator's indicators
    dropped_leak = sorted(c for c in X.columns if c in leak)
    if dropped_leak:
        X = X.drop(columns=dropped_leak)
        logger.info("Leakage guard dropped %d columns (e.g. %s)",
                    len(dropped_leak), ", ".join(dropped_leak[:5]))

    if task["type"] in {"binary", "multiclass"}:
        y = y.astype(str) if task["type"] == "multiclass" else y.astype(int)

    ds = Dataset(X=X.reset_index(drop=True), y=y.reset_index(drop=True), task=task,
                 time_index=time_index.reset_index(drop=True) if time_index is not None else None,
                 groups=groups.reset_index(drop=True) if groups is not None else None)
    logger.info("Task %r: %d rows x %d features | target %r",
                ds.name, len(ds.X), ds.X.shape[1], task["target"])
    return ds
