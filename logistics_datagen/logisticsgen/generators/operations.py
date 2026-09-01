"""Operational fact tables that hang off orders, routes, vehicles and warehouses:
GPS tracking, delivery history, customer feedback, vehicle maintenance,
inventory, shift planning, operating costs and courier performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import reference as ref
from ..config import Config
from ..rng import RandomStreams, lognormal_between, truncated_normal, weighted_choice
from ..utils import bearing_deg, haversine_km, jitter_coords, make_ids

__all__ = [
    "build_gps_tracking", "build_delivery_history", "build_customer_feedback",
    "build_vehicle_maintenance", "build_inventory", "build_shift_planning",
    "build_operating_costs", "build_courier_performance",
]


# --------------------------------------------------------------------------- #
# GPS tracking
# --------------------------------------------------------------------------- #
def build_gps_tracking(cfg: Config, streams: RandomStreams, orders: pd.DataFrame) -> pd.DataFrame:
    """Interpolated breadcrumb traces for a configurable subset of orders.

    Points are laid along the origin→destination great circle with lateral
    wander, so a trace looks like a road path rather than a straight ruler line.
    """
    rng = streams.get("gps")
    frac = float(cfg.get("child_tables.gps.order_fraction", 0.15))
    pmin = int(cfg.get("child_tables.gps.pings_per_order.min", 6))
    pmax = int(cfg.get("child_tables.gps.pings_per_order.max", 40))

    pool = orders.loc[orders["vehicle_id"].notna() & orders["actual_delivery_ts"].notna()]
    if pool.empty:
        return pd.DataFrame()
    k = max(1, int(len(pool) * frac))
    sel = pool.iloc[rng.choice(len(pool), size=k, replace=False)].reset_index(drop=True)

    n_pings = rng.integers(pmin, pmax + 1, k)
    total = int(n_pings.sum())
    order_pos = np.repeat(np.arange(k), n_pings)
    # 0..1 progress along each trace
    step = np.concatenate([np.arange(c) for c in n_pings])
    denom = np.repeat(np.maximum(n_pings - 1, 1), n_pings)
    progress = step / denom

    o_lat = sel["origin_lat"].to_numpy()[order_pos]
    o_lon = sel["origin_lon"].to_numpy()[order_pos]
    d_lat = sel["dest_lat"].to_numpy()[order_pos]
    d_lon = sel["dest_lon"].to_numpy()[order_pos]

    lat = o_lat + (d_lat - o_lat) * progress
    lon = o_lon + (d_lon - o_lon) * progress
    # Lateral wander peaks mid-route (you deviate most between endpoints).
    wander = np.sin(progress * np.pi) * rng.normal(0, 0.010, total)
    lat = lat + wander
    lon = lon + wander * rng.uniform(-1.4, 1.4, total)

    start = sel["pickup_timestamp"].to_numpy()[order_pos]
    dur_min = np.nan_to_num(sel["actual_duration_min"].to_numpy()[order_pos], nan=45.0)
    ts = pd.DatetimeIndex(start) + pd.to_timedelta((dur_min * progress * 60).astype(int), unit="s")

    dist_km = np.nan_to_num(sel["distance_km"].to_numpy()[order_pos], nan=5.0)
    nominal_speed = dist_km / np.maximum(dur_min / 60.0, 0.02)
    # Speed dips at both ends (loading / parking) and varies in between.
    speed = np.clip(nominal_speed * (0.35 + 1.3 * np.sin(np.clip(progress, 0.02, 0.98) * np.pi))
                    * rng.uniform(0.7, 1.35, total), 0.0, 130.0)

    heading = bearing_deg(o_lat, o_lon, d_lat, d_lon) + rng.normal(0, 18, total)

    return pd.DataFrame({
        "ping_id": make_ids("GPS", total, 10),
        "order_id": sel["order_id"].to_numpy()[order_pos],
        "route_id": sel["route_id"].to_numpy()[order_pos] if "route_id" in sel else None,
        "vehicle_id": sel["vehicle_id"].to_numpy()[order_pos],
        "driver_id": sel["driver_id"].to_numpy()[order_pos],
        "timestamp": ts,
        "sequence": step,
        "latitude": lat.round(6),
        "longitude": lon.round(6),
        "speed_kmh": speed.round(2),
        "heading_deg": (heading % 360).round(1),
        "altitude_m": truncated_normal(rng, 90, 120, -20, 2400, total).round(1),
        "accuracy_m": np.abs(rng.gamma(2.0, 3.5, total)).round(2),
        "satellites": rng.integers(4, 15, total),
        "ignition_on": progress < 0.97,
        "odometer_delta_km": (dist_km / np.maximum(np.repeat(n_pings, n_pings), 1)).round(4),
        "progress_pct": (progress * 100).round(2),
        "anomaly_flags": "",
    })


# --------------------------------------------------------------------------- #
# Delivery history (event log)
# --------------------------------------------------------------------------- #
def build_delivery_history(cfg: Config, streams: RandomStreams, orders: pd.DataFrame) -> pd.DataFrame:
    """Per-order status event log - the classic parcel-tracking timeline."""
    rng = streams.get("delivery_history")
    emin = int(cfg.get("child_tables.delivery_history.events_per_order.min", 3))
    emax = int(cfg.get("child_tables.delivery_history.events_per_order.max", 8))

    n_orders = len(orders)
    n_events = rng.integers(emin, emax + 1, n_orders)
    total = int(n_events.sum())
    pos = np.repeat(np.arange(n_orders), n_events)
    step = np.concatenate([np.arange(c) for c in n_events])
    denom = np.repeat(np.maximum(n_events - 1, 1), n_events)
    progress = step / denom

    status = orders["status"].to_numpy()[pos]
    is_last = step == (np.repeat(n_events, n_events) - 1)

    # Middle events come from the standard flow; the terminal event mirrors status.
    mid_events = np.asarray(ref.DELIVERY_EVENTS[:7], dtype=object)
    event = mid_events[np.clip((progress * (len(mid_events) - 1)).astype(int), 0, len(mid_events) - 1)]
    terminal = np.select(
        [status == "delivered", status == "delivered_late", status == "failed",
         status == "returned", status == "cancelled"],
        ["delivered", "delivered", "exception", "returned", "exception"], default="in_transit")
    event = np.where(is_last, terminal, event)

    order_ts = orders["order_timestamp"].to_numpy()[pos]
    span_min = np.nan_to_num(orders["actual_duration_min"].to_numpy()[pos], nan=120.0) + \
        (orders["pickup_timestamp"].to_numpy()[pos] - order_ts).astype("timedelta64[m]").astype(float)
    ts = pd.DatetimeIndex(order_ts) + pd.to_timedelta(
        (np.maximum(span_min, 5.0) * progress * 60).astype(int), unit="s")

    return pd.DataFrame({
        "event_id": make_ids("EVT", total, 10),
        "order_id": orders["order_id"].to_numpy()[pos],
        "route_id": orders["route_id"].to_numpy()[pos] if "route_id" in orders else None,
        "driver_id": orders["driver_id"].to_numpy()[pos],
        "warehouse_id": orders["warehouse_id"].to_numpy()[pos],
        "event_type": event,
        "event_timestamp": ts,
        "event_sequence": step + 1,
        "location_city": orders["city"].to_numpy()[pos],
        "scan_source": weighted_choice(
            rng, ["handheld", "dock_scanner", "driver_app", "api", "manual"],
            [0.34, 0.22, 0.28, 0.11, 0.05], total),
        "exception_code": np.where(
            event == "exception",
            weighted_choice(rng, ["ADDR_NOT_FOUND", "CUSTOMER_ABSENT", "DAMAGED",
                                  "REFUSED", "ACCESS_DENIED", "VEHICLE_FAULT"],
                            [0.24, 0.31, 0.13, 0.14, 0.11, 0.07], total),
            None),
        "dwell_minutes": np.abs(rng.gamma(1.8, 5.0, total)).round(2),
        "is_terminal": is_last,
    })


# --------------------------------------------------------------------------- #
# Customer feedback
# --------------------------------------------------------------------------- #
def build_customer_feedback(cfg: Config, streams: RandomStreams, orders: pd.DataFrame) -> pd.DataFrame:
    """Post-delivery survey responses. Ratings are *caused* by lateness and
    failures, which makes them a legitimate modelling target."""
    rng = streams.get("feedback")
    rate = float(cfg.get("child_tables.feedback.response_rate", 0.28))
    k = max(1, int(len(orders) * rate))
    sel = orders.iloc[rng.choice(len(orders), size=k, replace=False)].reset_index(drop=True)

    delay = np.nan_to_num(sel["delay_minutes"].to_numpy(), nan=0.0)
    failed = np.isin(sel["status"].to_numpy(), ["failed", "returned"])

    base = 4.55 - np.clip(delay, 0, 600) / 150.0 - failed * 1.7
    rating = np.clip(np.round(base + rng.normal(0, 0.62, k)), 1, 5)

    sentiment = np.clip((rating - 3.0) / 2.0 + rng.normal(0, 0.16, k), -1, 1)
    themes = np.select(
        [rating >= 5, rating >= 4, rating >= 3, rating >= 2],
        ["praise_speed", "praise_courier", "neutral_ok", "complaint_late"],
        default="complaint_damage")

    return pd.DataFrame({
        "feedback_id": make_ids("FBK", k, 9),
        "order_id": sel["order_id"].to_numpy(),
        "customer_id": sel["customer_id"].to_numpy(),
        "driver_id": sel["driver_id"].to_numpy(),
        "submitted_ts": pd.DatetimeIndex(sel["order_timestamp"])
                        + pd.to_timedelta(rng.integers(60, 10080, k), unit="m"),
        "channel": weighted_choice(rng, ref.FEEDBACK_CHANNELS, [0.42, 0.21, 0.14, 0.09, 0.14], k),
        "rating": rating.astype(int),
        "delivery_speed_rating": np.clip(np.round(rating + rng.normal(0, 0.5, k)), 1, 5).astype(int),
        "courier_rating": np.clip(np.round(rating + rng.normal(0, 0.7, k)), 1, 5).astype(int),
        "packaging_rating": np.clip(np.round(rating + rng.normal(0, 0.9, k)), 1, 5).astype(int),
        "comment_sentiment": sentiment.round(3),
        "theme": themes,
        "would_recommend": rating >= 4,
        "complaint_raised": rating <= 2,
        "refund_requested": (rating <= 2) & (rng.random(k) < 0.42),
        "resolution_hours": np.where(rating <= 2, np.abs(rng.gamma(2.0, 14.0, k)).round(1), np.nan),
    })


# --------------------------------------------------------------------------- #
# Vehicle maintenance
# --------------------------------------------------------------------------- #
def build_vehicle_maintenance(cfg: Config, streams: RandomStreams, vehicles: pd.DataFrame) -> pd.DataFrame:
    rng = streams.get("maintenance")
    emin = int(cfg.get("child_tables.maintenance.events_per_vehicle.min", 0))
    emax = int(cfg.get("child_tables.maintenance.events_per_vehicle.max", 9))

    n_v = len(vehicles)
    age = 2026 - vehicles["model_year"].to_numpy()
    lam = np.clip(0.6 + age * 0.35, 0.2, emax)
    counts = np.clip(rng.poisson(lam), emin, emax)
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame()
    pos = np.repeat(np.arange(n_v), counts)

    mtype = weighted_choice(rng, ref.MAINTENANCE_TYPES,
                            [0.31, 0.15, 0.12, 0.09, 0.07, 0.06, 0.08, 0.12], total)
    unplanned = np.isin(mtype, ["engine_repair", "brake_repair", "refrigeration_unit"])
    cost_index = vehicles["region"].map(ref.REGION_COST_INDEX).to_numpy()[pos]
    cost = lognormal_between(rng, 60, 9000, total) * cost_index * (1 + unplanned * 0.8)
    downtime = np.abs(rng.gamma(1.6, 9.0, total)) + unplanned * 14

    return pd.DataFrame({
        "maintenance_id": make_ids("MNT", total, 9),
        "vehicle_id": vehicles["vehicle_id"].to_numpy()[pos],
        "warehouse_id": vehicles["home_warehouse_id"].to_numpy()[pos],
        "region": vehicles["region"].to_numpy()[pos],
        "maintenance_type": mtype,
        "service_date": pd.to_datetime(cfg.get("time.start_date"))
                        + pd.to_timedelta(rng.integers(0, 730, total), "D"),
        "is_unplanned": unplanned,
        "odometer_at_service_km": (vehicles["odometer_km"].to_numpy()[pos]
                                   * rng.uniform(0.25, 1.0, total)).round(0),
        "cost_usd": cost.round(2),
        "parts_cost_usd": (cost * rng.uniform(0.25, 0.7, total)).round(2),
        "labour_hours": np.abs(rng.gamma(1.8, 2.2, total)).round(2),
        "downtime_hours": downtime.round(2),
        "workshop": weighted_choice(rng, ["in_house", "dealer", "third_party", "roadside"],
                                    [0.42, 0.22, 0.28, 0.08], total),
        "severity": np.select([cost > 4000, cost > 1200, cost > 300],
                              ["critical", "major", "moderate"], default="minor"),
        "next_service_due_km": rng.integers(8000, 40000, total),
        "warranty_covered": rng.random(total) < 0.21,
    })


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def build_inventory(cfg: Config, streams: RandomStreams, warehouses: pd.DataFrame) -> pd.DataFrame:
    """Warehouse x SKU x snapshot-date stock positions."""
    rng = streams.get("inventory")
    days = int(cfg.get("child_tables.inventory.snapshot_days", 90))
    skus = int(cfg.get("child_tables.inventory.skus_per_warehouse", 40))

    end = pd.Timestamp(cfg.get("time.end_date"))
    snapshots = pd.date_range(end - pd.Timedelta(days=days), end, freq="7D", inclusive="left")

    n_wh = len(warehouses)
    total = n_wh * skus * len(snapshots)
    wh_pos = np.repeat(np.arange(n_wh), skus * len(snapshots))
    sku_pos = np.tile(np.repeat(np.arange(skus), len(snapshots)), n_wh)
    snap = np.tile(snapshots.to_numpy(), n_wh * skus)

    category = np.asarray(ref.SKU_CATEGORIES, dtype=object)[sku_pos % len(ref.SKU_CATEGORIES)]
    reorder_point = rng.integers(20, 900, total)
    on_hand = np.maximum(0, (reorder_point * rng.uniform(0.05, 4.5, total)).round(0))

    return pd.DataFrame({
        "inventory_id": make_ids("INV", total, 10),
        "warehouse_id": warehouses["warehouse_id"].to_numpy()[wh_pos],
        "region": warehouses["region"].to_numpy()[wh_pos],
        "sku_id": np.char.add("SKU-", np.char.zfill(sku_pos.astype(str), 5)),
        "sku_category": category,
        "snapshot_date": pd.DatetimeIndex(snap),
        "units_on_hand": on_hand,
        "units_reserved": (on_hand * rng.uniform(0, 0.35, total)).round(0),
        "units_inbound": rng.poisson(45, total),
        "reorder_point": reorder_point,
        "safety_stock": (reorder_point * rng.uniform(0.15, 0.6, total)).round(0),
        "unit_cost_usd": lognormal_between(rng, 0.5, 900, total).round(2),
        "shelf_life_days": np.where(np.isin(category, ["grocery_chilled", "pharma"]),
                                    rng.integers(3, 90, total), rng.integers(180, 2000, total)),
        "storage_temp_c": np.where(category == "grocery_chilled", rng.uniform(0, 6, total).round(1),
                          np.where(category == "pharma", rng.uniform(2, 8, total).round(1), np.nan)),
        "stockout_flag": on_hand <= 0,
        "days_of_cover": (on_hand / np.maximum(rng.uniform(1, 90, total), 1)).round(2),
        "turnover_rate": np.abs(rng.gamma(2.0, 1.6, total)).round(3),
        "abc_class": weighted_choice(rng, ["A", "B", "C"], [0.2, 0.3, 0.5], total),
    })


# --------------------------------------------------------------------------- #
# Shift planning
# --------------------------------------------------------------------------- #
def build_shift_planning(cfg: Config, streams: RandomStreams, drivers: pd.DataFrame) -> pd.DataFrame:
    """Driver rosters over the final 12 weeks of the window."""
    rng = streams.get("shifts")
    end = pd.Timestamp(cfg.get("time.end_date"))
    days = pd.date_range(end - pd.Timedelta(days=84), end, freq="D", inclusive="left")

    n_d = len(drivers)
    # Roughly 5 shifts a week per driver, scaled by contracted hours.
    p_working = np.clip(drivers["max_weekly_hours"].to_numpy() / 56.0, 0.2, 0.95)
    grid_d = np.repeat(np.arange(n_d), len(days))
    grid_t = np.tile(days.to_numpy(), n_d)
    keep = rng.random(grid_d.size) < np.repeat(p_working, len(days))
    grid_d, grid_t = grid_d[keep], grid_t[keep]
    total = grid_d.size
    if total == 0:
        return pd.DataFrame()

    shift_type = np.asarray(drivers["preferred_shift"].to_numpy())[grid_d]
    shift_type = np.where(rng.random(total) < 0.25,
                          weighted_choice(rng, ref.SHIFT_TYPES, [0.3, 0.28, 0.16, 0.14, 0.12], total),
                          shift_type)
    start_hour = np.select(
        [shift_type == "morning", shift_type == "afternoon", shift_type == "night", shift_type == "split"],
        [6, 13, 21, 9], default=10)
    planned_hours = np.clip(rng.normal(8.0, 1.4, total), 3.0, 12.0)
    overtime = np.clip(rng.gamma(1.2, 0.7, total) - 0.4, 0, 5)

    return pd.DataFrame({
        "shift_id": make_ids("SHF", total, 10),
        "driver_id": drivers["driver_id"].to_numpy()[grid_d],
        "warehouse_id": drivers["home_warehouse_id"].to_numpy()[grid_d],
        "region": drivers["region"].to_numpy()[grid_d],
        "shift_date": pd.DatetimeIndex(grid_t),
        "shift_type": shift_type,
        "planned_start_hour": start_hour,
        "planned_hours": planned_hours.round(2),
        "actual_hours": (planned_hours + overtime - rng.uniform(0, 0.6, total)).clip(0).round(2),
        "overtime_hours": overtime.round(2),
        "break_minutes": rng.choice([0, 20, 30, 45, 60], total, p=[0.06, 0.2, 0.34, 0.26, 0.14]),
        "attendance_status": weighted_choice(
            rng, ["worked", "sick_leave", "annual_leave", "no_show", "swapped"],
            [0.885, 0.031, 0.049, 0.008, 0.027], total),
        "assigned_vehicle_type": weighted_choice(rng, ref.VEHICLE_TYPES, ref.VEHICLE_TYPE_WEIGHTS, total),
        "planned_stops": rng.integers(6, 65, total),
        "labour_cost_usd": (planned_hours * drivers["hourly_cost_usd"].to_numpy()[grid_d]
                            + overtime * drivers["hourly_cost_usd"].to_numpy()[grid_d] * 1.5).round(2),
        "understaffed_flag": rng.random(total) < 0.09,
    })


# --------------------------------------------------------------------------- #
# Operating costs
# --------------------------------------------------------------------------- #
def build_operating_costs(
    cfg: Config, streams: RandomStreams, orders: pd.DataFrame, warehouses: pd.DataFrame
) -> pd.DataFrame:
    """Monthly cost ledger per warehouse x category, anchored to real volumes."""
    rng = streams.get("operating_costs")
    if orders.empty:
        return pd.DataFrame()

    monthly = (orders.assign(month=orders["order_timestamp"].dt.to_period("M").dt.to_timestamp())
               .groupby(["warehouse_id", "month"], sort=False)
               .agg(orders_count=("order_id", "count"),
                    fuel=("fuel_cost_usd", "sum"),
                    labour=("labour_cost_usd", "sum"),
                    tolls=("toll_cost_usd", "sum"),
                    distance=("distance_km", "sum"))
               .reset_index())

    wh = warehouses.set_index("warehouse_id")
    rows = []
    for category in ref.COST_CATEGORIES:
        block = monthly.copy()
        cost_index = wh["cost_index"].reindex(block["warehouse_id"]).to_numpy()
        m = len(block)
        if category == "fuel":
            amount = block["fuel"].to_numpy()
        elif category == "labour":
            amount = block["labour"].to_numpy()
        elif category == "tolls":
            amount = block["tolls"].to_numpy()
        elif category == "maintenance":
            amount = block["distance"].to_numpy() * rng.uniform(0.04, 0.14, m) * cost_index
        elif category == "insurance":
            amount = block["orders_count"].to_numpy() * rng.uniform(0.15, 0.6, m) * cost_index
        elif category == "warehousing":
            amount = wh["monthly_fixed_cost_usd"].reindex(block["warehouse_id"]).to_numpy() \
                     * rng.uniform(0.8, 1.25, m)
        elif category == "penalties":
            amount = block["orders_count"].to_numpy() * rng.uniform(0.02, 0.35, m) * cost_index
        else:  # overhead
            amount = block["orders_count"].to_numpy() * rng.uniform(0.4, 1.8, m) * cost_index
        block["cost_category"] = category
        block["amount_usd"] = np.round(np.maximum(amount, 0), 2)
        rows.append(block)

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"month": "period_start"})
    out["region"] = wh["region"].reindex(out["warehouse_id"]).to_numpy()
    out["currency"] = "USD"
    out["cost_per_order_usd"] = (out["amount_usd"] / out["orders_count"].clip(lower=1)).round(4)
    out["budget_usd"] = (out["amount_usd"] * rng.uniform(0.85, 1.2, len(out))).round(2)
    out["variance_usd"] = (out["amount_usd"] - out["budget_usd"]).round(2)
    out["overhead_usd"] = (out["amount_usd"] * rng.uniform(0.03, 0.12, len(out))).round(2)
    out = out.drop(columns=["fuel", "labour", "tolls", "distance"])
    out.insert(0, "cost_id", make_ids("CST", len(out), 9))
    return out


# --------------------------------------------------------------------------- #
# Courier performance
# --------------------------------------------------------------------------- #
def build_courier_performance(
    cfg: Config, streams: RandomStreams, orders: pd.DataFrame,
    drivers: pd.DataFrame, feedback: pd.DataFrame
) -> pd.DataFrame:
    """Monthly driver scorecard aggregated from the order book."""
    rng = streams.get("courier_performance")
    work = orders.loc[orders["driver_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["month"] = work["order_timestamp"].dt.to_period("M").dt.to_timestamp()

    agg = work.groupby(["driver_id", "month"], sort=False).agg(
        deliveries=("order_id", "count"),
        late_deliveries=("is_late", "sum"),
        failed_deliveries=("status", lambda s: int((s == "failed").sum())),
        total_distance_km=("distance_km", "sum"),
        total_duration_min=("actual_duration_min", "sum"),
        total_cost_usd=("delivery_cost_usd", "sum"),
        total_revenue_usd=("revenue_usd", "sum"),
        avg_delay_minutes=("delay_minutes", "mean"),
        co2_kg=("co2_kg", "sum"),
    ).reset_index()

    if not feedback.empty:
        fb = (feedback.dropna(subset=["driver_id"])
              .assign(month=lambda d: d["submitted_ts"].dt.to_period("M").dt.to_timestamp())
              .groupby(["driver_id", "month"], sort=False)
              .agg(avg_customer_rating=("rating", "mean"),
                   complaints=("complaint_raised", "sum")).reset_index())
        agg = agg.merge(fb, on=["driver_id", "month"], how="left")
    else:
        agg["avg_customer_rating"] = np.nan
        agg["complaints"] = 0

    m = len(agg)
    agg["on_time_rate"] = (1 - agg["late_deliveries"] / agg["deliveries"]).round(4)
    agg["success_rate"] = (1 - agg["failed_deliveries"] / agg["deliveries"]).round(4)
    agg["deliveries_per_hour"] = (agg["deliveries"] /
                                  (agg["total_duration_min"] / 60).clip(lower=0.5)).round(3)
    agg["cost_per_delivery_usd"] = (agg["total_cost_usd"] / agg["deliveries"]).round(3)
    agg["margin_usd"] = (agg["total_revenue_usd"] - agg["total_cost_usd"]).round(2)
    agg["km_per_delivery"] = (agg["total_distance_km"] / agg["deliveries"]).round(3)
    agg["avg_delay_minutes"] = agg["avg_delay_minutes"].round(2)
    agg["safety_incidents"] = rng.poisson(0.05, m)
    agg["training_completed"] = rng.random(m) < 0.18
    agg["utilisation_pct"] = ((agg["total_duration_min"] / (22 * 8 * 60)) * 100).clip(0, 160).round(2)

    dr = drivers.set_index("driver_id")
    agg["region"] = dr["region"].reindex(agg["driver_id"]).to_numpy()
    agg["employment_type"] = dr["employment_type"].reindex(agg["driver_id"]).to_numpy()
    agg["home_warehouse_id"] = dr["home_warehouse_id"].reindex(agg["driver_id"]).to_numpy()

    # Composite 0-100 scorecard used by the dashboard and the ranking models.
    agg["performance_score"] = (
        agg["on_time_rate"].fillna(0) * 45
        + agg["success_rate"].fillna(0) * 25
        + agg["avg_customer_rating"].fillna(4.0) / 5 * 20
        + np.clip(agg["deliveries_per_hour"] / 4, 0, 1) * 10
    ).round(2)
    agg["rank_in_region"] = agg.groupby(["region", "month"])["performance_score"] \
                               .rank(ascending=False, method="dense").astype(int)
    agg.insert(0, "performance_id", make_ids("PRF", m, 9))
    return agg
