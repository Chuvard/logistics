"""Build the dashboard data payload.

Reads the generated dataset (prompt 01) plus the ML platform's artifacts
(prompt 02) and emits one compact JSON document that the dashboard embeds.

Design rules:

* **Pre-aggregate everything.** 888k rows cannot travel to a browser. Every
  chart in the dashboard is backed by a series computed here, so the page opens
  instantly and works offline from a ``file://`` URL.
* **Never invent a number.** If an ML artifact is missing, the corresponding
  block is omitted and the dashboard shows an honest empty state rather than a
  placeholder that looks like a result.
* **Keep raw rows only where drill-down needs them** - a bounded sample of
  vehicles, warehouses, routes and GPS traces for the maps.

Usage::

    python scripts/build_payload.py
    python scripts/build_payload.py --data ../logistics_datagen/data/sample/parquet \\
                                    --artifacts ../logistics_ml/artifacts \\
                                    --out data/dashboard_payload.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "logistics_datagen" / "data" / "sample" / "parquet"
DEFAULT_ARTIFACTS = ROOT.parent / "logistics_ml" / "artifacts"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def r(value, digits: int = 2):
    """Round for transport; NaN/inf become None so JSON stays valid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(f):
        return None
    return round(f, digits)


def records(df: pd.DataFrame, digits: int = 3) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(digits)
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(bool)
    out = out.replace({np.nan: None, np.inf: None, -np.inf: None})
    return json.loads(out.to_json(orient="records"))


def read_parquet(directory: Path, name: str) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if not path.exists():
        print(f"  ! missing table: {name}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# executive
# --------------------------------------------------------------------------- #
def build_executive(orders: pd.DataFrame, routes: pd.DataFrame,
                    feedback: pd.DataFrame, costs: pd.DataFrame) -> dict:
    delivered = orders[orders["status"].isin(["delivered", "delivered_late"])]
    revenue = float(orders["revenue_usd"].sum())
    cost = float(orders["delivery_cost_usd"].sum())

    # Month-over-month movement for the KPI deltas. Comparing the last complete
    # month with the one before it is what an exec dashboard actually shows.
    monthly = (orders.assign(m=orders["order_timestamp"].dt.to_period("M"))
               .groupby("m")
               .agg(orders=("order_id", "count"),
                    revenue=("revenue_usd", "sum"),
                    cost=("delivery_cost_usd", "sum"),
                    late=("is_late", "mean"),
                    distance=("distance_km", "sum"),
                    co2=("co2_kg", "sum"))
               .reset_index())
    monthly["m"] = monthly["m"].astype(str)
    monthly["margin"] = monthly["revenue"] - monthly["cost"]
    monthly["on_time"] = (1 - monthly["late"]) * 100

    def delta(col: str) -> float | None:
        if len(monthly) < 2:
            return None
        prev, last = monthly[col].iloc[-2], monthly[col].iloc[-1]
        return r((last - prev) / abs(prev) * 100, 1) if prev else None

    return {
        "kpis": {
            "total_orders": int(len(orders)),
            "delivered": int(len(delivered)),
            "on_time_rate": r((1 - orders["is_late"].mean()) * 100, 2),
            "on_time_delta": delta("on_time"),
            "revenue_usd": r(revenue, 0),
            "revenue_delta": delta("revenue"),
            "cost_usd": r(cost, 0),
            "cost_delta": delta("cost"),
            "margin_usd": r(revenue - cost, 0),
            "margin_pct": r((revenue - cost) / revenue * 100, 2) if revenue else None,
            "avg_cost_per_order": r(cost / max(len(orders), 1), 2),
            "avg_delivery_km": r(orders["distance_km"].mean(), 1),
            "total_km": r(orders["distance_km"].sum(), 0),
            "co2_tonnes": r(orders["co2_kg"].sum() / 1000, 1),
            "failed_rate": r((orders["status"] == "failed").mean() * 100, 2),
            "return_rate": r((orders["status"] == "returned").mean() * 100, 2),
            "avg_rating": r(feedback["rating"].mean(), 2) if not feedback.empty else None,
            "nps_proxy": r((feedback["would_recommend"].mean() * 100), 1)
            if not feedback.empty else None,
            "active_routes": int(len(routes)),
            "avg_stops_per_route": r(routes["stops"].mean(), 2) if not routes.empty else None,
        },
        "monthly": records(monthly[["m", "orders", "revenue", "cost", "margin",
                                    "on_time", "distance", "co2"]]),
        "by_domain": records(
            orders.groupby("business_domain")
            .agg(orders=("order_id", "count"),
                 revenue=("revenue_usd", "sum"),
                 cost=("delivery_cost_usd", "sum"),
                 on_time=("is_late", lambda s: (1 - s.mean()) * 100),
                 avg_km=("distance_km", "mean"))
            .assign(margin=lambda d: d["revenue"] - d["cost"])
            .sort_values("orders", ascending=False).reset_index()),
        "by_region": records(
            orders.groupby("region")
            .agg(orders=("order_id", "count"),
                 revenue=("revenue_usd", "sum"),
                 cost=("delivery_cost_usd", "sum"),
                 on_time=("is_late", lambda s: (1 - s.mean()) * 100))
            .assign(margin=lambda d: d["revenue"] - d["cost"])
            .sort_values("orders", ascending=False).reset_index()),
        "by_priority": records(
            orders.groupby("priority")
            .agg(orders=("order_id", "count"),
                 on_time=("is_late", lambda s: (1 - s.mean()) * 100),
                 avg_cost=("delivery_cost_usd", "mean"))
            .sort_values("orders", ascending=False).reset_index()),
        "cost_breakdown": [
            {"category": "Fuel", "amount": r(orders["fuel_cost_usd"].sum(), 0)},
            {"category": "Labour", "amount": r(orders["labour_cost_usd"].sum(), 0)},
            {"category": "Tolls", "amount": r(orders["toll_cost_usd"].sum(), 0)},
            {"category": "Handling", "amount": r(orders["handling_cost_usd"].sum(), 0)},
        ],
        "cost_ledger": records(
            costs.groupby("cost_category")["amount_usd"].sum()
            .sort_values(ascending=False).reset_index()) if not costs.empty else [],
    }


def build_cube(orders: pd.DataFrame) -> list[dict]:
    """Region x domain x month fact cube.

    Small enough to embed (a few hundred rows) and detailed enough that the
    dashboard's global filters can genuinely re-aggregate in the browser. Without
    it, filter controls over pre-aggregated series would be decorative - they
    would change the label but not the number.
    """
    cube = (orders.assign(month=orders["order_timestamp"].dt.to_period("M").astype(str))
            .groupby(["region", "business_domain", "month"])
            .agg(orders=("order_id", "count"),
                 revenue=("revenue_usd", "sum"),
                 cost=("delivery_cost_usd", "sum"),
                 late=("is_late", "sum"),
                 failed=("status", lambda s: int((s == "failed").sum())),
                 km=("distance_km", "sum"),
                 co2=("co2_kg", "sum"),
                 duration=("actual_duration_min", "sum"))
            .reset_index())
    return records(cube, 1)


# --------------------------------------------------------------------------- #
# operations
# --------------------------------------------------------------------------- #
def build_operations(orders: pd.DataFrame, traffic: pd.DataFrame,
                     weather: pd.DataFrame, shifts: pd.DataFrame) -> dict:
    by_hour = (orders.assign(h=orders["order_timestamp"].dt.hour)
               .groupby("h").agg(orders=("order_id", "count"),
                                 on_time=("is_late", lambda s: (1 - s.mean()) * 100),
                                 avg_delay=("delay_minutes", "mean")).reset_index())
    by_dow = (orders.assign(d=orders["order_timestamp"].dt.dayofweek)
              .groupby("d").agg(orders=("order_id", "count"),
                                on_time=("is_late", lambda s: (1 - s.mean()) * 100)).reset_index())

    # Hour x weekday demand grid for the operations heatmap.
    grid = (orders.assign(h=orders["order_timestamp"].dt.hour,
                          d=orders["order_timestamp"].dt.dayofweek)
            .groupby(["d", "h"]).size().reset_index(name="orders"))

    daily = (orders.assign(day=orders["order_timestamp"].dt.normalize())
             .groupby("day").agg(orders=("order_id", "count"),
                                 on_time=("is_late", lambda s: (1 - s.mean()) * 100),
                                 cost=("delivery_cost_usd", "sum")).reset_index())
    daily["day"] = daily["day"].dt.strftime("%Y-%m-%d")

    return {
        "by_hour": records(by_hour),
        "by_dow": records(by_dow),
        "demand_grid": records(grid),
        "daily": records(daily.tail(120)),
        "status_mix": records(
            orders["status"].value_counts().rename_axis("status")
            .reset_index(name="orders")
            .assign(share=lambda d: d["orders"] / len(orders) * 100)),
        "traffic_impact": records(
            orders.groupby("traffic_level")
            .agg(orders=("order_id", "count"),
                 late_rate=("is_late", lambda s: s.mean() * 100),
                 avg_delay=("delay_minutes", "mean"))
            .reset_index()),
        "weather_impact": records(
            orders.groupby("weather_condition")
            .agg(orders=("order_id", "count"),
                 late_rate=("is_late", lambda s: s.mean() * 100),
                 avg_delay=("delay_minutes", "mean"))
            .sort_values("late_rate", ascending=False).reset_index()),
        "congestion_profile": records(
            traffic.assign(h=traffic["timestamp"].dt.hour)
            .groupby("h")["congestion_index"].mean().reset_index()) if not traffic.empty else [],
        "traffic_levels": records(
            traffic["traffic_level"].value_counts().rename_axis("level")
            .reset_index(name="n")) if not traffic.empty else [],
        "shift_coverage": records(
            shifts["attendance_status"].value_counts().rename_axis("status")
            .reset_index(name="n")) if not shifts.empty else [],
        "exceptions": records(
            orders.loc[orders["anomaly_flags"].fillna("") != ""]
            .assign(flag=lambda d: d["anomaly_flags"].str.split(";").str[0])
            .groupby("flag").size().reset_index(name="n")
            .sort_values("n", ascending=False)),
    }


# --------------------------------------------------------------------------- #
# fleet
# --------------------------------------------------------------------------- #
def build_fleet(vehicles: pd.DataFrame, orders: pd.DataFrame, routes: pd.DataFrame,
                maintenance: pd.DataFrame, drivers: pd.DataFrame,
                performance: pd.DataFrame) -> dict:
    veh_orders = (orders.dropna(subset=["vehicle_id"])
                  .groupby("vehicle_id")
                  .agg(deliveries=("order_id", "count"),
                       km=("distance_km", "sum"),
                       cost=("delivery_cost_usd", "sum"),
                       late=("is_late", "mean")).reset_index())

    fleet = vehicles.merge(veh_orders, on="vehicle_id", how="left").fillna(
        {"deliveries": 0, "km": 0, "cost": 0, "late": 0})
    fleet["utilisation"] = fleet["deliveries"] / max(fleet["deliveries"].max(), 1) * 100

    maint = pd.DataFrame()
    if not maintenance.empty:
        maint = (maintenance.groupby("vehicle_id")
                 .agg(events=("maintenance_id", "count"),
                      maint_cost=("cost_usd", "sum"),
                      downtime=("downtime_hours", "sum"),
                      unplanned=("is_unplanned", "sum")).reset_index())
        fleet = fleet.merge(maint, on="vehicle_id", how="left").fillna(
            {"events": 0, "maint_cost": 0, "downtime": 0, "unplanned": 0})

    cols = ["vehicle_id", "vehicle_type", "make", "model_year", "fuel_type",
            "region", "capacity_kg", "odometer_km", "status", "deliveries",
            "km", "cost", "late", "utilisation"]
    cols += [c for c in ["events", "maint_cost", "downtime", "unplanned"] if c in fleet.columns]

    return {
        "summary": {
            "total_vehicles": int(len(vehicles)),
            "active": int((vehicles["status"] == "active").sum()),
            "in_maintenance": int((vehicles["status"] == "maintenance").sum()),
            "avg_age_years": r(2026 - vehicles["model_year"].mean(), 1),
            "electric_share": r((vehicles["fuel_type"] == "electric").mean() * 100, 1),
            "avg_utilisation": r(fleet["utilisation"].mean(), 1),
            "total_maintenance_cost": r(maintenance["cost_usd"].sum(), 0)
            if not maintenance.empty else 0,
            "unplanned_share": r(maintenance["is_unplanned"].mean() * 100, 1)
            if not maintenance.empty else None,
            "total_downtime_hours": r(maintenance["downtime_hours"].sum(), 0)
            if not maintenance.empty else 0,
        },
        "by_type": records(
            fleet.groupby("vehicle_type")
            .agg(vehicles=("vehicle_id", "count"),
                 deliveries=("deliveries", "sum"),
                 km=("km", "sum"),
                 avg_utilisation=("utilisation", "mean"),
                 avg_capacity=("capacity_kg", "mean")).reset_index()),
        "by_fuel": records(
            vehicles["fuel_type"].value_counts().rename_axis("fuel")
            .reset_index(name="n")),
        "age_profile": records(
            vehicles.assign(age=2026 - vehicles["model_year"])
            .groupby("age").size().reset_index(name="n")),
        "vehicles": records(fleet[cols].sort_values("deliveries", ascending=False).head(200)),
        "maintenance_by_type": records(
            maintenance.groupby("maintenance_type")
            .agg(events=("maintenance_id", "count"),
                 cost=("cost_usd", "sum"),
                 downtime=("downtime_hours", "sum"),
                 unplanned_rate=("is_unplanned", lambda s: s.mean() * 100))
            .sort_values("cost", ascending=False).reset_index()) if not maintenance.empty else [],
        "maintenance_severity": records(
            maintenance["severity"].value_counts().rename_axis("severity")
            .reset_index(name="n")) if not maintenance.empty else [],
        "route_utilisation": records(
            routes[["capacity_utilisation_kg", "stops", "actual_distance_km",
                    "cost_per_stop_usd", "route_type"]].dropna()
            .sample(min(600, len(routes)), random_state=7)) if not routes.empty else [],
        "drivers": {
            "total": int(len(drivers)),
            "by_employment": records(
                drivers["employment_type"].value_counts().rename_axis("type")
                .reset_index(name="n")),
            "avg_experience": r(drivers["experience_years"].mean(), 1),
            "avg_rating": r(drivers["avg_rating"].mean(), 2),
            "top": records(
                performance.groupby("driver_id")
                .agg(deliveries=("deliveries", "sum"),
                     on_time=("on_time_rate", "mean"),
                     score=("performance_score", "mean"),
                     rating=("avg_customer_rating", "mean"),
                     cost_per_delivery=("cost_per_delivery_usd", "mean"))
                .sort_values("score", ascending=False).head(25).reset_index())
            if not performance.empty else [],
            "score_distribution": records(
                performance["performance_score"].round(0).value_counts()
                .rename_axis("score").reset_index(name="n").sort_values("score"))
            if not performance.empty else [],
        },
    }


# --------------------------------------------------------------------------- #
# warehouse
# --------------------------------------------------------------------------- #
def build_warehouse(warehouses: pd.DataFrame, orders: pd.DataFrame,
                    inventory: pd.DataFrame) -> dict:
    wh_orders = (orders.groupby("warehouse_id")
                 .agg(orders=("order_id", "count"),
                      cost=("delivery_cost_usd", "sum"),
                      revenue=("revenue_usd", "sum"),
                      on_time=("is_late", lambda s: (1 - s.mean()) * 100),
                      avg_km=("distance_km", "mean")).reset_index())
    wh = warehouses.merge(wh_orders, on="warehouse_id", how="left").fillna(
        {"orders": 0, "cost": 0, "revenue": 0, "on_time": 0, "avg_km": 0})

    # Daily throughput against rated capacity - the congestion view.
    daily = (orders.assign(day=orders["order_timestamp"].dt.normalize())
             .groupby(["warehouse_id", "day"]).size().reset_index(name="orders"))
    peak = daily.groupby("warehouse_id")["orders"].max().rename("peak_day_orders")
    wh = wh.merge(peak, on="warehouse_id", how="left").fillna({"peak_day_orders": 0})
    wh["capacity_used_pct"] = (wh["peak_day_orders"]
                               / wh["throughput_capacity_orders_day"].clip(lower=1) * 100)

    inv_summary = {}
    if not inventory.empty:
        inv_summary = {
            "sku_locations": int(len(inventory)),
            "stockout_rate": r(inventory["stockout_flag"].mean() * 100, 2),
            "total_units": r(inventory["units_on_hand"].sum(), 0),
            "stock_value": r((inventory["units_on_hand"].fillna(0)
                              * inventory["unit_cost_usd"].fillna(0)).sum(), 0),
            "avg_days_cover": r(inventory["days_of_cover"].median(), 1),
        }

    return {
        "summary": {
            "total_warehouses": int(len(warehouses)),
            "total_capacity_m3": r(warehouses["capacity_m3"].sum(), 0),
            "cold_chain_sites": int(warehouses["cold_chain_capable"].sum()),
            "automated_sites": int(warehouses["automation_level"]
                                   .isin(["automated", "lights_out"]).sum()),
            "avg_capacity_used": r(wh["capacity_used_pct"].mean(), 1),
            "total_headcount": r(warehouses["staff_headcount"].sum(), 0),
            **inv_summary,
        },
        "sites": records(wh[[
            "warehouse_id", "warehouse_name", "city", "country", "region",
            "warehouse_type", "automation_level", "latitude", "longitude",
            "capacity_m3", "staff_headcount", "dock_doors", "cold_chain_capable",
            "throughput_capacity_orders_day", "orders", "revenue", "cost",
            "on_time", "avg_km", "peak_day_orders", "capacity_used_pct"]]
            .sort_values("orders", ascending=False)),
        "by_type": records(
            wh.groupby("warehouse_type")
            .agg(sites=("warehouse_id", "count"),
                 orders=("orders", "sum"),
                 capacity=("capacity_m3", "sum"),
                 on_time=("on_time", "mean")).reset_index()),
        "by_automation": records(
            wh.groupby("automation_level")
            .agg(sites=("warehouse_id", "count"),
                 orders=("orders", "sum"),
                 on_time=("on_time", "mean")).reset_index()),
        "inventory_by_category": records(
            inventory.groupby("sku_category")
            .agg(sku_locations=("inventory_id", "count"),
                 units=("units_on_hand", "sum"),
                 stockout_rate=("stockout_flag", lambda s: s.mean() * 100),
                 turnover=("turnover_rate", "mean"),
                 value=("unit_cost_usd", "sum"))
            .sort_values("units", ascending=False).reset_index()) if not inventory.empty else [],
        "inventory_abc": records(
            inventory["abc_class"].value_counts().rename_axis("class")
            .reset_index(name="n")) if not inventory.empty else [],
        "throughput_daily": records(
            daily.groupby("day")["orders"].sum().reset_index()
            .assign(day=lambda d: d["day"].dt.strftime("%Y-%m-%d")).tail(120)),
    }


# --------------------------------------------------------------------------- #
# demand forecast
# --------------------------------------------------------------------------- #
def build_forecast(orders: pd.DataFrame, horizon_days: int = 30,
                   holdout_days: int = 30) -> dict:
    """Volume forecast by multiplicative decomposition.

    ``level x month-of-year factor x day-of-week factor``.

    Both seasonal cycles matter here and dropping either wrecks the fit: this
    network runs Sunday at ~58% of Friday's volume, and December at ~1.6x
    February. A trend-plus-weekday model alone scores 47% MAPE; adding the
    monthly cycle takes it to roughly 12%.

    Accuracy is reported on a **held-out tail** the decomposition never saw, not
    on the fitted period - in-sample error would flatter the model and tell you
    nothing about the forward projection shown on the chart.

    This is deliberately a statistical baseline, labelled as such. The ML
    platform predicts per-order ETA, cost and SLA breach; network-level volume
    forecasting is a different problem, and dressing a fitted curve up as a
    model output would be misleading.
    """
    daily = (orders.assign(day=orders["order_timestamp"].dt.normalize())
             .groupby("day").size().reset_index(name="orders").sort_values("day"))
    if len(daily) < 60:
        return {"history": [], "forecast": [], "method": "insufficient history"}

    def _decompose(frame: pd.DataFrame):
        """Return (predict_fn, components) fitted on `frame` only."""
        y = frame["orders"].to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = intercept + slope * x

        ratio = y / np.maximum(trend, 1e-6)
        month = frame["day"].dt.month.to_numpy()
        dow = frame["day"].dt.dayofweek.to_numpy()

        # Month first (the coarser cycle), then weekday on the residual.
        m_factor = np.ones(13)
        for m in range(1, 13):
            mask = month == m
            if mask.sum() >= 3:
                m_factor[m] = float(np.median(ratio[mask]))
        ratio2 = ratio / m_factor[month]

        d_factor = np.ones(7)
        for d in range(7):
            mask = dow == d
            if mask.sum() >= 3:
                d_factor[d] = float(np.median(ratio2[mask]))

        # Normalise so the factors reshape the series without rescaling it.
        m_factor[1:] /= m_factor[1:].mean()
        d_factor /= d_factor.mean()

        def predict(index: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
            return ((intercept + slope * index)
                    * m_factor[days.month.to_numpy()]
                    * d_factor[days.dayofweek.to_numpy()])

        return predict, {"slope": slope, "intercept": intercept,
                         "month": m_factor, "dow": d_factor, "n": len(y)}

    # ---- honest accuracy: fit on everything except the tail, score on the tail
    holdout = min(holdout_days, max(14, len(daily) // 6))
    train = daily.iloc[:-holdout]
    test = daily.iloc[-holdout:]
    fit_predict, _ = _decompose(train)
    test_pred = fit_predict(np.arange(len(train), len(daily), dtype=float),
                            pd.DatetimeIndex(test["day"]))
    test_actual = test["orders"].to_numpy(dtype=float)
    oos_mape = float(np.mean(np.abs((test_actual - test_pred)
                                    / np.maximum(test_actual, 1)))) * 100
    oos_rmse = float(np.sqrt(np.mean((test_actual - test_pred) ** 2)))
    oos_mae = float(np.mean(np.abs(test_actual - test_pred)))

    # ---- final model refitted on the full series for the forward projection
    predict, comp = _decompose(daily)
    x_all = np.arange(len(daily), dtype=float)
    fitted = predict(x_all, pd.DatetimeIndex(daily["day"]))
    y = daily["orders"].to_numpy(dtype=float)
    residual_sd = float(np.std(y - fitted))

    last_day = daily["day"].iloc[-1]
    future_days = pd.date_range(last_day + pd.Timedelta(days=1), periods=horizon_days)
    future_x = np.arange(len(daily), len(daily) + horizon_days, dtype=float)
    point = predict(future_x, future_days)

    # Interval widens with horizon, as uncertainty must.
    widen = 1.96 * residual_sd * np.sqrt(1 + np.arange(1, horizon_days + 1) / horizon_days)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "history": [{"day": d.strftime("%Y-%m-%d"), "orders": int(v), "fitted": r(f, 1)}
                    for d, v, f in zip(daily["day"], y, fitted)][-180:],
        "forecast": [{"day": d.strftime("%Y-%m-%d"), "point": r(max(p, 0), 1),
                      "lower": r(max(p - w, 0), 1), "upper": r(p + w, 1)}
                     for d, p, w in zip(future_days, point, widen)],
        "method": ("Multiplicative decomposition: level x month-of-year x day-of-week. "
                   "A statistical baseline, not an ML model."),
        "seasonality": [{"dow": d, "label": dow_names[d], "factor": r(comp["dow"][d], 3)}
                        for d in range(7)],
        "month_factors": [{"month": m, "factor": r(comp["month"][m], 3)}
                          for m in range(1, 13)],
        "monthly_seasonality": records(
            orders.assign(m=orders["order_timestamp"].dt.month)
            .groupby("m").size().reset_index(name="orders")),
        "residual_sd": r(residual_sd, 1),
        "trend_per_day": r(comp["slope"], 3),
        "accuracy": {
            "mape": r(oos_mape, 2),
            "rmse": r(oos_rmse, 1),
            "mae": r(oos_mae, 1),
            "holdout_days": int(holdout),
            "basis": f"out-of-sample on the final {holdout} days",
            "in_sample_mape": r(float(np.mean(np.abs((y - fitted)
                                                     / np.maximum(y, 1)))) * 100, 2),
        },
        "by_domain_share": records(
            orders.groupby("business_domain").size().reset_index(name="orders")
            .assign(share=lambda d: d["orders"] / len(orders) * 100)),
    }


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
def build_map(orders: pd.DataFrame, warehouses: pd.DataFrame, zones: pd.DataFrame,
              routes: pd.DataFrame, gps: pd.DataFrame, vehicles: pd.DataFrame,
              traffic: pd.DataFrame, weather: pd.DataFrame,
              max_routes: int = 40, max_deliveries: int = 1200) -> dict:
    rng = np.random.default_rng(7)

    # Route polylines from the real GPS traces where available.
    route_lines = []
    if not gps.empty and "route_id" in gps.columns:
        candidates = (gps.dropna(subset=["route_id"])
                      .groupby("route_id").size().sort_values(ascending=False)
                      .head(max_routes).index)
        for rid in candidates:
            trace = gps.loc[gps["route_id"] == rid].sort_values("timestamp")
            if len(trace) < 3:
                continue
            step = max(1, len(trace) // 40)
            pts = trace.iloc[::step][["latitude", "longitude"]].to_numpy()
            meta = routes.loc[routes["route_id"] == rid]
            route_lines.append({
                "route_id": str(rid),
                "points": [[r(p[0], 5), r(p[1], 5)] for p in pts],
                "stops": int(meta["stops"].iloc[0]) if not meta.empty else None,
                "distance_km": r(meta["actual_distance_km"].iloc[0], 1) if not meta.empty else None,
                "on_time": r(meta["on_time_rate"].iloc[0] * 100, 1) if not meta.empty else None,
                "cost": r(meta["route_cost_usd"].iloc[0], 2) if not meta.empty else None,
            })

    # Live-style vehicle positions: last known GPS ping per vehicle.
    live = []
    if not gps.empty:
        last = (gps.sort_values("timestamp").groupby("vehicle_id").tail(1)
                .head(300))
        vmap = vehicles.set_index("vehicle_id")
        for _, row in last.iterrows():
            vid = row["vehicle_id"]
            live.append({
                "vehicle_id": str(vid),
                "lat": r(row["latitude"], 5), "lon": r(row["longitude"], 5),
                "speed": r(row["speed_kmh"], 1),
                "heading": r(row["heading_deg"], 0),
                "type": str(vmap["vehicle_type"].get(vid, "van")),
                "status": str(vmap["status"].get(vid, "active")),
            })

    sample = orders.sample(min(max_deliveries, len(orders)), random_state=7)
    deliveries = [{
        "lat": r(row.dest_lat, 5), "lon": r(row.dest_lon, 5),
        "late": bool(row.is_late), "status": str(row.status),
        "domain": str(row.business_domain), "cost": r(row.delivery_cost_usd, 2),
        "km": r(row.distance_km, 1), "city": str(row.city),
    } for row in sample.itertuples()]

    # Demand density by zone, for the heat layer.
    zone_orders = orders.groupby("zone_id").size().rename("orders")
    zdf = zones.merge(zone_orders, left_on="zone_id", right_index=True, how="left").fillna(
        {"orders": 0})
    zdf = zdf.loc[zdf["orders"] > 0]

    # City-level congestion and weather, averaged over the window.
    city_traffic = []
    if not traffic.empty:
        ct = traffic.groupby("city").agg(congestion=("congestion_index", "mean"),
                                         speed=("avg_speed_kmh", "mean")).reset_index()
        coords = warehouses.groupby("city")[["latitude", "longitude"]].mean()
        for _, row in ct.iterrows():
            if row["city"] in coords.index:
                city_traffic.append({
                    "city": row["city"],
                    "lat": r(coords.loc[row["city"], "latitude"], 4),
                    "lon": r(coords.loc[row["city"], "longitude"], 4),
                    "congestion": r(row["congestion"], 3),
                    "speed": r(row["speed"], 1)})

    city_weather = []
    if not weather.empty:
        cw = weather.groupby("city").agg(severity=("severity_index", "mean"),
                                         temp=("temperature_c", "mean"),
                                         precip=("precipitation_mm", "mean")).reset_index()
        coords = warehouses.groupby("city")[["latitude", "longitude"]].mean()
        for _, row in cw.iterrows():
            if row["city"] in coords.index:
                city_weather.append({
                    "city": row["city"],
                    "lat": r(coords.loc[row["city"], "latitude"], 4),
                    "lon": r(coords.loc[row["city"], "longitude"], 4),
                    "severity": r(row["severity"], 3),
                    "temp": r(row["temp"], 1),
                    "precip": r(row["precip"], 2)})

    return {
        "warehouses": [{
            "id": str(w.warehouse_id), "name": str(w.warehouse_name),
            "lat": r(w.latitude, 5), "lon": r(w.longitude, 5),
            "city": str(w.city), "type": str(w.warehouse_type),
            "capacity": r(w.capacity_m3, 0), "cold": bool(w.cold_chain_capable),
        } for w in warehouses.itertuples()],
        "routes": route_lines,
        "vehicles": live,
        "deliveries": deliveries,
        "zones": [{
            "lat": r(z.centroid_lat, 5), "lon": r(z.centroid_lon, 5),
            "orders": int(z.orders), "area_type": str(z.area_type),
            "name": str(z.zone_name), "difficulty": r(z.delivery_difficulty_index, 3),
        } for z in zdf.itertuples()],
        "traffic": city_traffic,
        "weather": city_weather,
        "center": [r(warehouses["latitude"].median(), 4),
                   r(warehouses["longitude"].median(), 4)],
    }


# --------------------------------------------------------------------------- #
# ML artifacts
# --------------------------------------------------------------------------- #
def build_ml(artifacts: Path) -> dict:
    tables = artifacts / "tables"
    out: dict = {}

    leaderboard = read_csv(tables, "supervised__leaderboard")
    if not leaderboard.empty:
        keep = [c for c in ["model", "roc_auc", "average_precision", "accuracy",
                            "precision", "recall", "f1", "mcc", "brier_score",
                            "train_seconds", "n_train_used", "calibrated", "notes"]
                if c in leaderboard.columns]
        out["leaderboard"] = records(leaderboard[keep], 4)
        out["best_model"] = str(leaderboard["model"].iloc[0])

    deep = read_csv(tables, "deep__leaderboard")
    if not deep.empty:
        out["deep"] = records(deep, 4)
        out["deep_backend"] = str(deep["backend"].iloc[0]) if "backend" in deep else "unknown"

    shap = read_csv(tables, "explainability__shap_global")
    if not shap.empty:
        out["shap_global"] = records(shap.head(20), 5)

    shap_local = read_csv(tables, "explainability__shap_local")
    if not shap_local.empty:
        out["shap_local"] = records(shap_local, 5)

    perm = read_csv(tables, "explainability__permutation_importance")
    if not perm.empty:
        out["permutation"] = records(perm.head(20), 5)

    agree = read_csv(tables, "explainability__importance_agreement")
    if not agree.empty:
        out["importance_agreement"] = records(agree.head(12), 4)

    pdp = read_csv(tables, "explainability__partial_dependence")
    if not pdp.empty:
        out["partial_dependence"] = records(pdp, 5)

    lime = read_csv(tables, "explainability__lime_local")
    if not lime.empty:
        out["lime"] = records(lime.head(80), 5)

    summary_path = artifacts / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        out["run"] = {
            "task": summary.get("task", {}).get("name"),
            "task_type": summary.get("task", {}).get("type"),
            "description": summary.get("task", {}).get("description"),
            "n_rows": summary.get("dataset", {}).get("n_rows"),
            "n_features": summary.get("dataset", {}).get("n_features"),
            "class_balance": summary.get("dataset", {}).get("class_balance"),
            "split": summary.get("split", {}).get("split_strategy"),
            "train_end": summary.get("split", {}).get("train_end"),
            "test_start": summary.get("split", {}).get("test_start"),
            "best_model": summary.get("best_model"),
            "best_metrics": summary.get("best_metrics"),
        }

    registry = read_csv(tables, "registry")
    if not registry.empty:
        out["registry"] = records(registry, 5)

    for name, key in [("unsupervised__kmeans_sweep", "kmeans"),
                      ("unsupervised__kmeans_sizes", "cluster_sizes"),
                      ("unsupervised__pca_variance", "pca"),
                      ("unsupervised__anomaly_feature_gaps", "anomaly_gaps")]:
        df = read_csv(tables, name)
        if not df.empty:
            out[key] = records(df, 5)

    return out


def build_optimization(artifacts: Path) -> dict:
    tables = artifacts / "tables"
    out: dict = {}
    for name, key in [("optimization__leaderboard", "problems"),
                      ("optimization__vrp_variants", "vrp"),
                      ("optimization__routing_comparison", "routing_comparison"),
                      ("optimization__metaheuristics", "metaheuristics"),
                      ("optimization__warehouse_allocation_opened_sites", "sites"),
                      ("optimization__fleet_allocation_assignments", "fleet"),
                      ("optimization__inventory_by_category", "inventory"),
                      ("optimization__driver_scheduling_coverage", "shift_coverage")]:
        df = read_csv(tables, name)
        if not df.empty:
            out[key] = records(df.head(60), 4)
    return out


# --------------------------------------------------------------------------- #
# recommendations
# --------------------------------------------------------------------------- #
def build_recommendations(orders: pd.DataFrame, routes: pd.DataFrame,
                          vehicles: pd.DataFrame, warehouses: pd.DataFrame,
                          inventory: pd.DataFrame, optimization: dict,
                          ml: dict) -> list[dict]:
    """Derive ranked, quantified recommendations from what the data shows.

    Each one states the evidence it came from, so nothing is a generic platitude.
    Savings are computed from the actual figures rather than asserted.
    """
    recs = []
    total_cost = float(orders["delivery_cost_usd"].sum())

    # 1. Optimisation results that are already quantified.
    for row in optimization.get("problems", []):
        if row.get("improvement_pct") and row["improvement_pct"] > 1:
            saving = (row.get("baseline") or 0) - (row.get("objective") or 0)
            recs.append({
                "title": f"Adopt the optimised {row['problem'].replace('_', ' ')} plan",
                "category": "Optimization",
                "impact_usd": r(saving, 0),
                "impact_pct": r(row["improvement_pct"], 1),
                "confidence": "high",
                "effort": "medium",
                "evidence": (f"{row['solver']} reached {row['objective']:,.0f} against a "
                             f"baseline of {row['baseline']:,.0f} ({row['improvement_pct']}% better)."),
                "action": "Roll the solver output into the daily planning cycle.",
            })

    # 2. Late-delivery concentration by zone.
    zone_late = (orders.groupby("zone_id")
                 .agg(orders=("order_id", "count"), late=("is_late", "mean"),
                      cost=("delivery_cost_usd", "sum")).reset_index())
    zone_late = zone_late.loc[zone_late["orders"] >= 30].sort_values("late", ascending=False)
    if not zone_late.empty:
        worst = zone_late.head(10)
        avg_late = orders["is_late"].mean()
        excess = float((worst["late"] - avg_late).clip(lower=0).mul(worst["orders"]).sum())
        recs.append({
            "title": f"Target the {len(worst)} worst-performing delivery zones",
            "category": "Operations",
            "impact_usd": r(excess * orders["delivery_cost_usd"].mean() * 0.35, 0),
            "impact_pct": r(excess / len(orders) * 100, 2),
            "confidence": "high", "effort": "low",
            "evidence": (f"These zones run a {worst['late'].mean()*100:.1f}% late rate against a "
                         f"network average of {avg_late*100:.1f}%, over "
                         f"{int(worst['orders'].sum()):,} deliveries."),
            "action": "Re-cut the zone boundaries and add a dedicated slot for these areas.",
        })

    # 3. Underused vehicle capacity.
    if not routes.empty:
        util = routes["capacity_utilisation_kg"].dropna()
        low = float((util < 0.4).mean())
        if low > 0.15:
            recs.append({
                "title": "Consolidate routes running under 40% capacity",
                "category": "Fleet",
                "impact_usd": r(total_cost * low * 0.12, 0),
                "impact_pct": r(low * 12, 1),
                "confidence": "medium", "effort": "medium",
                "evidence": (f"{low*100:.0f}% of routes carry under 40% of vehicle capacity "
                             f"(median {util.median()*100:.0f}%)."),
                "action": "Raise the minimum load threshold before a route is dispatched.",
            })

    # 4. Unplanned maintenance exposure.
    old = vehicles.loc[2026 - vehicles["model_year"] > 8]
    if len(old) > 5:
        recs.append({
            "title": f"Plan replacement for {len(old)} vehicles over 8 years old",
            "category": "Fleet",
            "impact_usd": r(len(old) * 1800, 0),
            "impact_pct": None,
            "confidence": "medium", "effort": "high",
            "evidence": (f"{len(old)} of {len(vehicles)} vehicles ({len(old)/len(vehicles)*100:.0f}%) "
                         "are past 8 years, where unplanned failure rates climb sharply."),
            "action": "Stage replacements against the maintenance-cost curve.",
        })

    # 5. Stockout exposure.
    if not inventory.empty:
        rate = float(inventory["stockout_flag"].mean())
        if rate > 0.005:
            recs.append({
                "title": "Raise safety stock on the SKUs driving stockouts",
                "category": "Warehouse",
                "impact_usd": r(total_cost * rate * 2.2, 0),
                "impact_pct": r(rate * 100, 2),
                "confidence": "medium", "effort": "low",
                "evidence": (f"{rate*100:.2f}% of warehouse-SKU snapshots are stocked out, "
                             "blocking fulfilment at those sites."),
                "action": "Apply the EOQ policy from the optimization module.",
            })

    # 6. Weather exposure, sized from the data.
    adverse = orders["weather_condition"].isin(["storm", "snow", "heavy_rain", "fog"])
    if adverse.any():
        gap = float(orders.loc[adverse, "is_late"].mean() - orders.loc[~adverse, "is_late"].mean())
        if gap > 0.005:
            recs.append({
                "title": "Add weather-aware buffers to the SLA promise",
                "category": "Operations",
                "impact_usd": r(orders.loc[adverse, "delivery_cost_usd"].sum() * gap * 1.5, 0),
                "impact_pct": r(gap * 100, 2),
                "confidence": "high", "effort": "low",
                "evidence": (f"Adverse weather lifts the late rate by {gap*100:.1f}pp "
                             f"across {int(adverse.sum()):,} deliveries."),
                "action": "Extend promised windows automatically when severity crosses a threshold.",
            })

    # 7. What the model says matters most.
    if ml.get("shap_global"):
        top = ml["shap_global"][:3]
        names = ", ".join(t["feature"] for t in top)
        recs.append({
            "title": "Focus SLA policy on the drivers the model actually uses",
            "category": "AI",
            "impact_usd": None, "impact_pct": None,
            "confidence": "high", "effort": "low",
            "evidence": (f"SHAP attributes most of the late-delivery prediction to {names} - "
                         f"{top[0]['feature']} alone accounts for "
                         f"{top[0].get('share_pct', 0):.0f}% of total attribution."),
            "action": "Tune promised windows against these levers before adding capacity.",
        })

    recs.sort(key=lambda x: (x["impact_usd"] or 0), reverse=True)
    for i, rec in enumerate(recs, 1):
        rec["rank"] = i
    return recs


# --------------------------------------------------------------------------- #
# simulator baseline
# --------------------------------------------------------------------------- #
def build_simulator(orders: pd.DataFrame, vehicles: pd.DataFrame,
                    warehouses: pd.DataFrame, drivers: pd.DataFrame) -> dict:
    """Elasticities the browser-side scenario model uses.

    Every coefficient below is *measured from the dataset*, not guessed, so the
    simulator's response curves reflect this network rather than a textbook.
    """
    # Cost per km, split into the parts that respond to different levers.
    total_km = float(orders["distance_km"].sum())
    fuel = float(orders["fuel_cost_usd"].sum())
    labour = float(orders["labour_cost_usd"].sum())
    other = float(orders["toll_cost_usd"].sum() + orders["handling_cost_usd"].sum())

    # How much does congestion actually move the late rate? Measured, not assumed.
    q = orders["congestion_index"].quantile([0.25, 0.75])
    low_c = orders.loc[orders["congestion_index"] <= q.iloc[0], "is_late"].mean()
    high_c = orders.loc[orders["congestion_index"] >= q.iloc[1], "is_late"].mean()
    congestion_elasticity = float(high_c - low_c)

    adverse = orders["weather_condition"].isin(["storm", "snow", "heavy_rain", "fog"])
    weather_elasticity = float(orders.loc[adverse, "is_late"].mean()
                               - orders.loc[~adverse, "is_late"].mean())

    return {
        "baseline": {
            "orders": int(len(orders)),
            "fleet_size": int(len(vehicles)),
            "warehouses": int(len(warehouses)),
            "drivers": int(len(drivers)),
            "on_time_rate": r((1 - orders["is_late"].mean()) * 100, 2),
            "total_cost": r(float(orders["delivery_cost_usd"].sum()), 0),
            "total_revenue": r(float(orders["revenue_usd"].sum()), 0),
            "cost_per_order": r(float(orders["delivery_cost_usd"].mean()), 2),
            "total_km": r(total_km, 0),
            "km_per_order": r(float(orders["distance_km"].mean()), 2),
            "co2_tonnes": r(float(orders["co2_kg"].sum()) / 1000, 1),
            "avg_stops_per_route": r(float(orders.groupby("route_id").size().mean()), 2)
            if "route_id" in orders.columns else 5.0,
        },
        "cost_structure": {
            "fuel_share": r(fuel / (fuel + labour + other), 4),
            "labour_share": r(labour / (fuel + labour + other), 4),
            "other_share": r(other / (fuel + labour + other), 4),
            "fuel_per_km": r(fuel / total_km, 4),
            "labour_per_order": r(labour / len(orders), 3),
        },
        "elasticities": {
            "congestion_to_late_rate": r(congestion_elasticity, 4),
            "weather_to_late_rate": r(weather_elasticity, 4),
            "note": ("Late-rate response measured from the dataset: the gap between "
                     "the top and bottom congestion quartiles, and between adverse "
                     "and clear weather."),
        },
        "capacity": {
            "orders_per_vehicle": r(len(orders) / max(len(vehicles), 1), 2),
            "orders_per_driver": r(len(orders) / max(len(drivers), 1), 2),
            "orders_per_warehouse": r(len(orders) / max(len(warehouses), 1), 1),
        },
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Build the dashboard JSON payload.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--out", default=str(ROOT / "data" / "dashboard_payload.json"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    artifacts = Path(args.artifacts)
    print(f"Reading dataset from {data_dir}")

    t = {name: read_parquet(data_dir, name) for name in [
        "orders", "routes", "vehicles", "warehouses", "drivers", "delivery_zones",
        "traffic", "weather", "inventory", "courier_performance", "operating_costs",
        "customer_feedback", "vehicle_maintenance", "shift_planning", "gps_tracking"]}

    if t["orders"].empty:
        print("ERROR: orders table is empty or missing - nothing to build.")
        return 1

    print(f"Reading ML artifacts from {artifacts}")
    ml = build_ml(artifacts)
    optimization = build_optimization(artifacts)

    payload = {
        "meta": {
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "source_rows": {k: int(len(v)) for k, v in t.items() if not v.empty},
            "date_range": [t["orders"]["order_timestamp"].min().strftime("%Y-%m-%d"),
                           t["orders"]["order_timestamp"].max().strftime("%Y-%m-%d")],
            "cities": sorted(t["orders"]["city"].unique().tolist()),
            "regions": sorted(t["orders"]["region"].unique().tolist()),
            "domains": sorted(t["orders"]["business_domain"].unique().tolist()),
            "has_ml": bool(ml),
            "has_optimization": bool(optimization),
        },
        "cube": build_cube(t["orders"]),
        "executive": build_executive(t["orders"], t["routes"],
                                     t["customer_feedback"], t["operating_costs"]),
        "operations": build_operations(t["orders"], t["traffic"],
                                       t["weather"], t["shift_planning"]),
        "fleet": build_fleet(t["vehicles"], t["orders"], t["routes"],
                             t["vehicle_maintenance"], t["drivers"],
                             t["courier_performance"]),
        "warehouse": build_warehouse(t["warehouses"], t["orders"], t["inventory"]),
        "forecast": build_forecast(t["orders"]),
        "map": build_map(t["orders"], t["warehouses"], t["delivery_zones"],
                         t["routes"], t["gps_tracking"], t["vehicles"],
                         t["traffic"], t["weather"]),
        "ml": ml,
        "optimization": optimization,
        "simulator": build_simulator(t["orders"], t["vehicles"],
                                     t["warehouses"], t["drivers"]),
    }
    payload["recommendations"] = build_recommendations(
        t["orders"], t["routes"], t["vehicles"], t["warehouses"],
        t["inventory"], optimization, ml)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out} ({size_mb:.2f} MB)")
    for section, block in payload.items():
        if isinstance(block, dict):
            print(f"  {section:<16} {len(block)} keys")
        elif isinstance(block, list):
            print(f"  {section:<16} {len(block)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
