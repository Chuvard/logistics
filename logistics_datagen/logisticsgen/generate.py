"""End-to-end orchestration.

Stage order matters and is deliberate:

1. **Dimensions** - nothing else can reference what doesn't exist yet.
2. **Environment panels** - orders join against them, so they precede orders.
3. **Order/route core** - built clean, on true values.
4. **Operational facts** - derived from the clean core.
5. **Anomalies** - injected *before* missingness so a masked cell can still hide
   an anomalous value (which is exactly what makes MNAR hard in practice).
6. **Features / targets / preprocessing** - computed on the messy data, as a
   real pipeline would have to.
7. **Export and reports.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import generators as gen
from .config import Config
from .io import export_all
from .pipeline import add_targets, build_feature_table, preprocess
from .pipeline.preprocessing import save_metadata
from .quality import apply_missingness, inject_anomalies, sweep_missingness
from .reports import build_reports
from .rng import RandomStreams
from .utils import get_logger, timed

__all__ = ["GenerationResult", "generate_dataset"]

logger = get_logger()

TABLE_ORDER = [
    "warehouses", "pickup_locations", "delivery_zones", "customers", "vehicles",
    "drivers", "traffic", "weather", "fuel_prices", "regional_holidays",
    "orders", "routes", "gps_tracking", "delivery_history", "customer_feedback",
    "vehicle_maintenance", "inventory", "shift_planning", "operating_costs",
    "courier_performance",
]


@dataclass
class GenerationResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    features: pd.DataFrame | None = None
    ml_table: pd.DataFrame | None = None
    missingness_report: pd.DataFrame | None = None
    anomaly_report: pd.DataFrame | None = None
    sweep: dict[float, pd.DataFrame] = field(default_factory=dict)
    exported: dict[str, list[Path]] = field(default_factory=dict)
    reports: list[Path] = field(default_factory=list)
    preprocessing_meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"table": k, "rows": len(v), "columns": v.shape[1]} for k, v in self.tables.items()]
        ).sort_values("rows", ascending=False).reset_index(drop=True)


def generate_dataset(cfg: Config, export: bool = True, reports: bool = True) -> GenerationResult:
    streams = RandomStreams(cfg.seed)
    out_dir = cfg.output_dir
    tables: dict[str, pd.DataFrame] = {}

    # -- 1. dimensions --------------------------------------------------------
    with timed("Dimensions (warehouses, zones, customers, fleet, drivers)"):
        tables["warehouses"] = gen.build_warehouses(cfg, streams)
        tables["pickup_locations"] = gen.build_pickup_locations(cfg, streams, tables["warehouses"])
        tables["delivery_zones"] = gen.build_delivery_zones(cfg, streams, tables["warehouses"])
        tables["customers"] = gen.build_customers(cfg, streams, tables["delivery_zones"])
        tables["vehicles"] = gen.build_vehicles(cfg, streams, tables["warehouses"])
        tables["drivers"] = gen.build_drivers(cfg, streams, tables["warehouses"])

    # -- 2. environment panels ------------------------------------------------
    with timed("Environment panels (traffic, weather, fuel, holidays)"):
        tables["traffic"] = gen.build_traffic(cfg, streams)
        tables["weather"] = gen.build_weather(cfg, streams)
        tables["fuel_prices"] = gen.build_fuel_prices(cfg, streams)
        tables["regional_holidays"] = gen.build_regional_holidays(cfg, streams)

    # -- 3. order / route core ------------------------------------------------
    with timed(f"Orders ({cfg.volume('deliveries'):,} deliveries)"):
        tables["orders"] = gen.build_orders(
            cfg, streams, tables["customers"], tables["warehouses"], tables["delivery_zones"],
            tables["pickup_locations"], tables["drivers"], tables["vehicles"],
            tables["traffic"], tables["weather"])

    with timed("Routes"):
        routes, orders = gen.build_routes(cfg, streams, tables["orders"], tables["vehicles"])
        tables["routes"] = routes
        tables["orders"] = orders

    # -- 4. operational facts -------------------------------------------------
    with timed("Operational facts (GPS, history, feedback, maintenance, inventory)"):
        tables["gps_tracking"] = gen.build_gps_tracking(cfg, streams, tables["orders"])
        tables["delivery_history"] = gen.build_delivery_history(cfg, streams, tables["orders"])
        tables["customer_feedback"] = gen.build_customer_feedback(cfg, streams, tables["orders"])
        tables["vehicle_maintenance"] = gen.build_vehicle_maintenance(cfg, streams, tables["vehicles"])
        tables["inventory"] = gen.build_inventory(cfg, streams, tables["warehouses"])
        tables["shift_planning"] = gen.build_shift_planning(cfg, streams, tables["drivers"])
        tables["operating_costs"] = gen.build_operating_costs(
            cfg, streams, tables["orders"], tables["warehouses"])
        tables["courier_performance"] = gen.build_courier_performance(
            cfg, streams, tables["orders"], tables["drivers"], tables["customer_feedback"])

    result = GenerationResult(tables=tables)

    # -- 5. anomalies then missingness ---------------------------------------
    with timed("Anomaly injection"):
        tables, anomaly_report = inject_anomalies(tables, cfg, streams)
        result.anomaly_report = anomaly_report.to_frame()

    if cfg.get("missingness.rate_sweep") and cfg.get("missingness.enabled", True):
        with timed("Missingness rate sweep (report only)"):
            result.sweep = sweep_missingness(tables, cfg, streams)

    with timed("Missingness injection"):
        tables, miss_report, ground_truth = apply_missingness(tables, cfg, streams)
        result.missingness_report = miss_report.to_frame()

    # Ground truth for masked cells - needed to score imputation methods.
    if ground_truth:
        gt_dir = out_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for key, frame in ground_truth.items():
            frame.to_parquet(gt_dir / f"{key.replace('.', '__')}.parquet", index=False)
        logger.info("Ground truth for %d masked columns → %s", len(ground_truth), gt_dir)

    # -- 6. features, targets, preprocessing ----------------------------------
    if cfg.get("features.enabled", True):
        with timed("Feature engineering"):
            features = build_feature_table(tables, cfg)
            features = add_targets(features, cfg)
            result.features = features
            tables["ml_features"] = features

        if cfg.get("preprocessing.enabled", True):
            with timed("Preprocessing"):
                ml_table, meta = preprocess(features, cfg)
                result.ml_table = ml_table
                result.preprocessing_meta = meta
                tables["ml_clean"] = ml_table
                save_metadata(meta, out_dir / "reports" / "preprocessing_metadata.json")

    result.tables = tables

    # -- 7. export and report -------------------------------------------------
    if export:
        with timed("Export"):
            result.exported = export_all(tables, cfg, out_dir)
        cfg.dump(out_dir / "resolved_config.yaml")

    if reports and cfg.get("reports.eda", True):
        with timed("Reports"):
            result.reports = build_reports(
                tables, cfg, out_dir,
                missingness_report=result.missingness_report,
                anomaly_report=result.anomaly_report,
                sweep=result.sweep)

    logger.info("Done. %s rows across %d tables",
                f"{sum(len(t) for t in tables.values()):,}", len(tables))
    return result
