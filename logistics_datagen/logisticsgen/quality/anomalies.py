"""Anomaly and operational-disruption injection.

Two families are injected:

1. **Operational disruptions** that a business would recognise - demand spikes,
   extreme weather events, vehicle breakdowns, stockouts, fuel price shocks.
   These change downstream values in physically consistent ways.
2. **Data-quality defects** that a data engineer would recognise - GPS jumps and
   flatlines, duplicate order rows, negative durations, absurd costs.

Every affected row is tagged in the table's ``anomaly_flags`` column (a
semicolon-separated label list), giving supervised anomaly detection a clean
ground-truth label instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..rng import RandomStreams
from ..utils import get_logger, offset_coords

__all__ = ["AnomalyReport", "inject_anomalies"]

logger = get_logger()


@dataclass
class AnomalyReport:
    rows: list[dict] = field(default_factory=list)

    def add(self, name: str, table: str, count: int, rate: float, description: str) -> None:
        self.rows.append({
            "anomaly": name, "table": table, "rows_affected": int(count),
            "rate_of_table": round(float(rate), 6), "description": description,
        })

    def to_frame(self) -> pd.DataFrame:
        cols = ["anomaly", "table", "rows_affected", "rate_of_table", "description"]
        return pd.DataFrame(self.rows, columns=cols)


def _flag(df: pd.DataFrame, mask: np.ndarray, label: str) -> None:
    """Append ``label`` to ``anomaly_flags`` for the masked rows."""
    if "anomaly_flags" not in df.columns:
        df["anomaly_flags"] = ""
    current = df.loc[mask, "anomaly_flags"].fillna("")
    df.loc[mask, "anomaly_flags"] = np.where(current == "", label, current + ";" + label)


def _sample_mask(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    return rng.random(n) < float(rate)


def inject_anomalies(
    tables: dict[str, pd.DataFrame], cfg: Config, streams: RandomStreams
) -> tuple[dict[str, pd.DataFrame], AnomalyReport]:
    report = AnomalyReport()
    if not cfg.get("anomalies.enabled", True):
        logger.info("Anomaly injection disabled by config")
        return tables, report

    a = cfg.get("anomalies", {})
    orders = tables.get("orders")
    routes = tables.get("routes")
    gps = tables.get("gps_tracking")
    vehicles = tables.get("vehicles")
    inventory = tables.get("inventory")
    fuel = tables.get("fuel_prices")
    weather = tables.get("weather")
    traffic = tables.get("traffic")
    shifts = tables.get("shift_planning")

    # ---------------------------------------------------------------- GPS jump
    if gps is not None and not gps.empty and "gps_jump" in a:
        rng = streams.spawn("anom::gps_jump")
        spec = a["gps_jump"]
        mask = _sample_mask(rng, len(gps), spec.get("rate", 0.004))
        k = int(mask.sum())
        if k:
            dist = rng.uniform(spec.get("min_km", 8), spec.get("max_km", 90), k)
            brg = rng.uniform(0, 360, k)
            nlat, nlon = offset_coords(
                gps.loc[mask, "latitude"].to_numpy(), gps.loc[mask, "longitude"].to_numpy(), dist, brg)
            gps.loc[mask, "latitude"] = np.round(nlat, 6)
            gps.loc[mask, "longitude"] = np.round(nlon, 6)
            gps.loc[mask, "speed_kmh"] = np.round(rng.uniform(180, 640, k), 2)
            gps.loc[mask, "accuracy_m"] = np.round(rng.uniform(60, 900, k), 2)
            _flag(gps, mask, "gps_jump")
        report.add("gps_jump", "gps_tracking", k, k / max(len(gps), 1),
                   "Position teleports several km with impossible implied speed")

    # ------------------------------------------------------------ GPS flatline
    if gps is not None and not gps.empty and "gps_flatline" in a:
        rng = streams.spawn("anom::gps_flatline")
        spec = a["gps_flatline"]
        starts = np.flatnonzero(_sample_mask(rng, len(gps), spec.get("rate", 0.002)))
        run = int(spec.get("min_pings", 4))
        idx = np.unique(np.concatenate(
            [np.arange(s, min(s + run, len(gps))) for s in starts])) if starts.size else np.array([], int)
        if idx.size:
            mask = np.zeros(len(gps), bool)
            mask[idx] = True
            gps.loc[mask, "speed_kmh"] = 0.0
            gps.loc[mask, "satellites"] = 0
            _flag(gps, mask, "gps_flatline")
            report.add("gps_flatline", "gps_tracking", int(mask.sum()), mask.mean(),
                       "Consecutive pings frozen - lost fix or stationary sensor fault")

    # -------------------------------------------------------- Duplicate orders
    if orders is not None and not orders.empty and "duplicate_orders" in a:
        rng = streams.spawn("anom::duplicate_orders")
        rate = a["duplicate_orders"].get("rate", 0.003)
        k = int(len(orders) * rate)
        if k:
            pick = rng.choice(len(orders), size=k, replace=False)
            dup = orders.iloc[pick].copy()
            dup["order_id"] = np.char.add(dup["order_id"].to_numpy().astype(str), "-DUP")
            dup["is_duplicate"] = True
            dup["order_timestamp"] = dup["order_timestamp"] + pd.to_timedelta(
                rng.integers(1, 240, k), unit="s")
            dup["anomaly_flags"] = np.where(
                dup["anomaly_flags"].fillna("") == "", "duplicate_order",
                dup["anomaly_flags"] + ";duplicate_order")
            orders = pd.concat([orders, dup], ignore_index=True)
            tables["orders"] = orders
        report.add("duplicate_orders", "orders", k, k / max(len(orders), 1),
                   "Near-identical order re-submitted seconds later (double-click / retry)")

    # ------------------------------------------------------- Vehicle breakdown
    if orders is not None and not orders.empty and "vehicle_breakdown" in a:
        rng = streams.spawn("anom::vehicle_breakdown")
        mask = _sample_mask(rng, len(orders), a["vehicle_breakdown"].get("rate", 0.006)) \
               & orders["vehicle_id"].notna().to_numpy()
        k = int(mask.sum())
        if k:
            extra = rng.uniform(45, 480, k)
            orders.loc[mask, "actual_duration_min"] = orders.loc[mask, "actual_duration_min"].fillna(60) + extra
            orders.loc[mask, "delay_minutes"] = orders.loc[mask, "delay_minutes"].fillna(0) + extra
            orders.loc[mask, "is_late"] = True
            orders.loc[mask, "status"] = np.where(
                rng.random(k) < 0.35, "failed", "delivered_late")
            orders.loc[mask, "labour_cost_usd"] = orders.loc[mask, "labour_cost_usd"] * rng.uniform(1.4, 3.2, k)
            orders.loc[mask, "delivery_cost_usd"] = (
                orders.loc[mask, "fuel_cost_usd"] + orders.loc[mask, "labour_cost_usd"]
                + orders.loc[mask, "toll_cost_usd"] + orders.loc[mask, "handling_cost_usd"])
            _flag(orders, mask, "vehicle_breakdown")
            if vehicles is not None:
                broken = orders.loc[mask, "vehicle_id"].dropna().unique()
                vmask = vehicles["vehicle_id"].isin(broken).to_numpy()
                _flag(vehicles, vmask, "breakdown_history")
        report.add("vehicle_breakdown", "orders", k, k / max(len(orders), 1),
                   "Mid-route mechanical failure: large delay, cost spike, possible failure")

    # ------------------------------------------------------------ Demand spike
    if orders is not None and not orders.empty and "demand_spike" in a:
        rng = streams.spawn("anom::demand_spike")
        spec = a["demand_spike"]
        n_events = int(spec.get("events", 24))
        lo_m, hi_m = spec.get("multiplier", [2.0, 6.0])
        lo_h, hi_h = spec.get("duration_hours", [6, 72])
        ts = orders["order_timestamp"]
        cities = orders["city"].dropna().unique()
        total = 0
        for _ in range(n_events):
            city = rng.choice(cities)
            start = ts.min() + pd.Timedelta(hours=int(rng.integers(
                0, max(int((ts.max() - ts.min()).total_seconds() // 3600), 1))))
            dur = int(rng.integers(lo_h, hi_h + 1))
            mult = float(rng.uniform(lo_m, hi_m))
            window = (orders["city"] == city) & ts.between(start, start + pd.Timedelta(hours=dur))
            mask = window.to_numpy()
            k = int(mask.sum())
            if not k:
                continue
            # Surge congestion: slower, pricier, later.
            orders.loc[mask, "actual_duration_min"] = orders.loc[mask, "actual_duration_min"] * (1 + (mult - 1) * 0.12)
            orders.loc[mask, "delay_minutes"] = orders.loc[mask, "delay_minutes"] + (mult - 1) * 9
            orders.loc[mask, "delivery_cost_usd"] = orders.loc[mask, "delivery_cost_usd"] * (1 + (mult - 1) * 0.08)
            orders.loc[mask, "is_late"] = orders.loc[mask, "delay_minutes"] > 0
            _flag(orders, mask, "demand_spike")
            total += k
        report.add("demand_spike", "orders", total, total / max(len(orders), 1),
                   f"{n_events} localised surge windows (promo, holiday, viral demand)")

    # --------------------------------------------------------- Extreme weather
    if weather is not None and not weather.empty and "extreme_weather" in a:
        rng = streams.spawn("anom::extreme_weather")
        spec = a["extreme_weather"]
        cities = weather["city"].unique()
        wts = weather["timestamp"]
        total = 0
        for _ in range(int(spec.get("events", 18))):
            city = rng.choice(cities)
            start = wts.min() + pd.Timedelta(hours=int(rng.integers(
                0, max(int((wts.max() - wts.min()).total_seconds() // 3600), 1))))
            lo_h, hi_h = spec.get("duration_hours", [12, 96])
            dur = int(rng.integers(lo_h, hi_h + 1))
            mask = ((weather["city"] == city)
                    & wts.between(start, start + pd.Timedelta(hours=dur))).to_numpy()
            k = int(mask.sum())
            if not k:
                continue
            weather.loc[mask, "condition"] = "storm"
            weather.loc[mask, "precipitation_mm"] = np.round(rng.uniform(25, 140, k), 2)
            weather.loc[mask, "wind_speed_kmh"] = np.round(rng.uniform(70, 165, k), 1)
            weather.loc[mask, "visibility_km"] = np.round(rng.uniform(0.05, 1.5, k), 2)
            weather.loc[mask, "severity_index"] = np.round(rng.uniform(0.75, 1.0, k), 4)
            weather.loc[mask, "delay_factor"] = np.round(rng.uniform(1.5, 2.3, k), 3)
            weather.loc[mask, "road_condition"] = "flooded"
            _flag(weather, mask, "extreme_weather_event")
            total += k
        report.add("extreme_weather", "weather", total, total / max(len(weather), 1),
                   "Named-storm style events: visibility collapse and heavy precipitation")

    # -------------------------------------------------------- Traffic gridlock
    if traffic is not None and not traffic.empty and "traffic_gridlock" in a:
        rng = streams.spawn("anom::traffic_gridlock")
        mask = _sample_mask(rng, len(traffic), a["traffic_gridlock"].get("rate", 0.01))
        k = int(mask.sum())
        if k:
            traffic.loc[mask, "congestion_index"] = np.round(rng.uniform(0.88, 1.0, k), 4)
            traffic.loc[mask, "traffic_level"] = "gridlock"
            traffic.loc[mask, "avg_speed_kmh"] = np.round(rng.uniform(1.5, 9.0, k), 2)
            traffic.loc[mask, "delay_factor"] = np.round(rng.uniform(1.9, 3.4, k), 4)
            traffic.loc[mask, "incident_type"] = "closure"
            traffic.loc[mask, "road_closure_count"] = rng.integers(2, 12, k)
            _flag(traffic, mask, "gridlock_event")
        report.add("traffic_gridlock", "traffic", k, k / max(len(traffic), 1),
                   "Total standstill from closure or major incident")

    # -------------------------------------------------------- Driver no-show
    if shifts is not None and not shifts.empty and "driver_no_show" in a:
        rng = streams.spawn("anom::driver_no_show")
        mask = _sample_mask(rng, len(shifts), a["driver_no_show"].get("rate", 0.008))
        k = int(mask.sum())
        if k:
            shifts.loc[mask, "attendance_status"] = "no_show"
            shifts.loc[mask, "actual_hours"] = 0.0
            shifts.loc[mask, "overtime_hours"] = 0.0
            shifts.loc[mask, "understaffed_flag"] = True
            _flag(shifts, mask, "driver_no_show")
        report.add("driver_no_show", "shift_planning", k, k / max(len(shifts), 1),
                   "Rostered driver did not report - shift left uncovered")

    # ----------------------------------------------------- Warehouse stockout
    if inventory is not None and not inventory.empty and "warehouse_stockout" in a:
        rng = streams.spawn("anom::stockout")
        mask = _sample_mask(rng, len(inventory), a["warehouse_stockout"].get("rate", 0.015))
        k = int(mask.sum())
        if k:
            inventory.loc[mask, "units_on_hand"] = 0
            inventory.loc[mask, "units_reserved"] = 0
            inventory.loc[mask, "stockout_flag"] = True
            inventory.loc[mask, "days_of_cover"] = 0.0
            _flag(inventory, mask, "stockout")
        report.add("warehouse_stockout", "inventory", k, k / max(len(inventory), 1),
                   "SKU fully depleted at a site - blocks fulfilment")

    # ------------------------------------------------------ Fuel price shock
    if fuel is not None and not fuel.empty and "fuel_price_shock" in a:
        rng = streams.spawn("anom::fuel_shock")
        spec = a["fuel_price_shock"]
        lo, hi = spec.get("magnitude", [0.15, 0.55])
        total = 0
        dates = fuel["date"]
        countries = fuel["country"].unique()
        for _ in range(int(spec.get("events", 6))):
            start = dates.min() + pd.Timedelta(days=int(rng.integers(0, max((dates.max() - dates.min()).days, 1))))
            dur = int(rng.integers(7, 60))
            # Shocks are regional, not planetary - scope each event to a market.
            hit = rng.choice(countries, size=max(1, int(len(countries) * 0.25)), replace=False)
            mask = (dates.between(start, start + pd.Timedelta(days=dur))
                    & fuel["country"].isin(hit)).to_numpy()
            k = int(mask.sum())
            if not k:
                continue
            fuel.loc[mask, "price_per_unit_usd"] = (
                fuel.loc[mask, "price_per_unit_usd"] * (1 + float(rng.uniform(lo, hi)))).round(4)
            _flag(fuel, mask, "price_shock")
            total += k
        report.add("fuel_price_shock", "fuel_prices", total, total / max(len(fuel), 1),
                   "Geopolitical / supply shock lifting prices for weeks")

    # --------------------------------------------------------- Address error
    if orders is not None and not orders.empty and "address_error" in a:
        rng = streams.spawn("anom::address_error")
        mask = _sample_mask(rng, len(orders), a["address_error"].get("rate", 0.007))
        k = int(mask.sum())
        if k:
            # Coordinates land somewhere plausible but wrong - classic geocode slip.
            nlat, nlon = offset_coords(
                orders.loc[mask, "dest_lat"].to_numpy(), orders.loc[mask, "dest_lon"].to_numpy(),
                rng.uniform(2, 45, k), rng.uniform(0, 360, k))
            orders.loc[mask, "dest_lat"] = np.round(nlat, 6)
            orders.loc[mask, "dest_lon"] = np.round(nlon, 6)
            orders.loc[mask, "delivery_attempts"] = rng.integers(2, 5, k)
            orders.loc[mask, "status"] = np.where(rng.random(k) < 0.55, "failed", orders.loc[mask, "status"])
            _flag(orders, mask, "address_error")
        report.add("address_error", "orders", k, k / max(len(orders), 1),
                   "Mis-geocoded destination causing repeat attempts and failures")

    # ------------------------------------------------------ Fraudulent order
    if orders is not None and not orders.empty and "fraudulent_order" in a:
        rng = streams.spawn("anom::fraud")
        mask = _sample_mask(rng, len(orders), a["fraudulent_order"].get("rate", 0.001))
        k = int(mask.sum())
        if k:
            orders.loc[mask, "declared_value_usd"] = np.round(
                orders.loc[mask, "declared_value_usd"].fillna(100) * rng.uniform(6, 40, k), 2)
            orders.loc[mask, "payment_method"] = "cash_on_delivery"
            orders.loc[mask, "status"] = "returned"
            _flag(orders, mask, "fraudulent_order")
        report.add("fraudulent_order", "orders", k, k / max(len(orders), 1),
                   "Implausible declared value with COD and return - fraud signature")

    # ---------------------------------------------------------- Sensor drift
    if gps is not None and not gps.empty and "sensor_drift" in a:
        rng = streams.spawn("anom::sensor_drift")
        mask = _sample_mask(rng, len(gps), a["sensor_drift"].get("rate", 0.003))
        k = int(mask.sum())
        if k:
            gps.loc[mask, "speed_kmh"] = (gps.loc[mask, "speed_kmh"] * rng.uniform(1.3, 2.6, k)).round(2)
            gps.loc[mask, "altitude_m"] = (gps.loc[mask, "altitude_m"] + rng.uniform(200, 2500, k)).round(1)
            _flag(gps, mask, "sensor_drift")
        report.add("sensor_drift", "gps_tracking", k, k / max(len(gps), 1),
                   "Miscalibrated sensor reporting systematically inflated readings")

    # ------------------------------------------------------ Negative duration
    if orders is not None and not orders.empty and "negative_duration" in a:
        rng = streams.spawn("anom::negative_duration")
        mask = _sample_mask(rng, len(orders), a["negative_duration"].get("rate", 0.0008))
        k = int(mask.sum())
        if k:
            orders.loc[mask, "actual_duration_min"] = -np.abs(rng.uniform(1, 240, k)).round(2)
            _flag(orders, mask, "negative_duration")
        report.add("negative_duration", "orders", k, k / max(len(orders), 1),
                   "Corrupt timestamp ordering yielding impossible negative duration")

    # --------------------------------------------------------- Outlier cost
    if routes is not None and not routes.empty and "outlier_cost" in a:
        rng = streams.spawn("anom::outlier_cost")
        mask = _sample_mask(rng, len(routes), a["outlier_cost"].get("rate", 0.002))
        k = int(mask.sum())
        if k:
            routes.loc[mask, "route_cost_usd"] = (
                routes.loc[mask, "route_cost_usd"] * rng.uniform(12, 90, k)).round(2)
            routes.loc[mask, "cost_per_stop_usd"] = (
                routes.loc[mask, "route_cost_usd"] / routes.loc[mask, "stops"]).round(3)
            _flag(routes, mask, "outlier_cost")
        report.add("outlier_cost", "routes", k, k / max(len(routes), 1),
                   "Billing / unit error producing an order-of-magnitude cost outlier")

    logger.info("Injected %d anomaly classes", len(report.rows))
    return tables, report
