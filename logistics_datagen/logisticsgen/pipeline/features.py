"""Feature engineering.

Builds an analysis-ready wide table by denormalising orders against every
dimension, then deriving the feature families a logistics model actually wants:
temporal, geospatial, package, resource, environmental, historical and
efficiency ratios.

Everything here is leakage-aware: rolling aggregates are computed with a
``shift(1)`` so a row never sees its own outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, haversine_km

__all__ = ["build_feature_table"]

logger = get_logger()


def _cyclical(df: pd.DataFrame, col: str, period: int, name: str) -> None:
    """Encode a cyclical integer column as sin/cos so 23:00 sits next to 00:00."""
    radians = 2 * np.pi * df[col] / period
    df[f"{name}_sin"] = np.sin(radians).round(6)
    df[f"{name}_cos"] = np.cos(radians).round(6)


def build_feature_table(tables: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    orders = tables["orders"].copy()
    if orders.empty:
        return orders

    customers = tables.get("customers", pd.DataFrame())
    warehouses = tables.get("warehouses", pd.DataFrame())
    zones = tables.get("delivery_zones", pd.DataFrame())
    drivers = tables.get("drivers", pd.DataFrame())
    vehicles = tables.get("vehicles", pd.DataFrame())
    routes = tables.get("routes", pd.DataFrame())
    holidays = tables.get("regional_holidays", pd.DataFrame())

    df = orders

    # ---------------------------------------------------------------- temporal
    ts = df["order_timestamp"]
    df["order_year"] = ts.dt.year
    df["order_month"] = ts.dt.month
    df["order_day"] = ts.dt.day
    df["order_hour"] = ts.dt.hour
    df["order_dayofweek"] = ts.dt.dayofweek
    df["order_dayofyear"] = ts.dt.dayofyear
    df["order_week"] = ts.dt.isocalendar().week.astype(int)
    df["order_quarter"] = ts.dt.quarter
    df["is_weekend"] = df["order_dayofweek"] >= 5
    df["is_business_hours"] = df["order_hour"].between(8, 18)
    df["is_night_shift"] = (df["order_hour"] >= 21) | (df["order_hour"] < 5)
    df["is_peak_hour"] = df["order_hour"].isin([8, 9, 17, 18])
    df["days_since_start"] = (ts - ts.min()).dt.days
    df["lead_time_hours"] = (
        (df["pickup_timestamp"] - ts).dt.total_seconds() / 3600).round(3)

    if cfg.get("features.cyclical_time", True):
        _cyclical(df, "order_hour", 24, "hour")
        _cyclical(df, "order_dayofweek", 7, "dow")
        _cyclical(df, "order_month", 12, "month")

    # ---------------------------------------------------------------- holidays
    if not holidays.empty:
        hol = holidays[["country", "date", "demand_multiplier", "holiday_type"]].copy()
        hol["date"] = hol["date"].dt.normalize()
        df["_order_date"] = ts.dt.normalize()
        df = df.merge(hol, left_on=["country", "_order_date"], right_on=["country", "date"], how="left")
        df["is_holiday"] = df["holiday_type"].notna()
        df["holiday_demand_multiplier"] = df["demand_multiplier"].fillna(1.0)
        df["is_shopping_event"] = df["holiday_type"].eq("shopping_event").fillna(False)
        df = df.drop(columns=["date", "demand_multiplier", "holiday_type", "_order_date"])
    else:
        df["is_holiday"] = False
        df["holiday_demand_multiplier"] = 1.0
        df["is_shopping_event"] = False

    # ------------------------------------------------------------- geospatial
    buckets = cfg.get("features.distance_buckets_km", [5, 15, 40, 100, 300])
    df["distance_bucket"] = pd.cut(
        df["distance_km"], bins=[-np.inf, *buckets, np.inf],
        labels=[f"<{buckets[0]}km", *[f"{buckets[i]}-{buckets[i+1]}km" for i in range(len(buckets) - 1)],
                f">{buckets[-1]}km"]).astype(str)
    df["detour_ratio"] = (df["distance_km"] / df["straight_line_km"].clip(lower=0.05)).round(4)
    df["is_long_haul"] = df["distance_km"] > 150
    df["is_last_mile"] = df["distance_km"] <= 15

    if not zones.empty:
        z = zones.set_index("zone_id")
        for col, out in [
            ("population_density_per_km2", "zone_population_density"),
            ("delivery_difficulty_index", "zone_difficulty"),
            ("parking_availability", "zone_parking"),
            ("avg_income_index", "zone_income_index"),
            ("avg_floor_count", "zone_avg_floors"),
            ("toll_zone", "zone_toll"),
            ("low_emission_zone", "zone_lez"),
            ("area_km2", "zone_area_km2"),
        ]:
            if col in z.columns:
                df[out] = z[col].reindex(df["zone_id"]).to_numpy()

    if not warehouses.empty:
        w = warehouses.set_index("warehouse_id")
        df["warehouse_type"] = w["warehouse_type"].reindex(df["warehouse_id"]).to_numpy()
        df["warehouse_automation"] = w["automation_level"].reindex(df["warehouse_id"]).to_numpy()
        df["warehouse_capacity_m3"] = w["capacity_m3"].reindex(df["warehouse_id"]).to_numpy()
        df["warehouse_throughput_cap"] = w["throughput_capacity_orders_day"].reindex(df["warehouse_id"]).to_numpy()
        df["warehouse_cost_index"] = w["cost_index"].reindex(df["warehouse_id"]).to_numpy()

    # ---------------------------------------------------------------- package
    df["weight_per_item_kg"] = (df["package_weight_kg"] / df["items_count"].clip(lower=1)).round(4)
    df["density_kg_per_m3"] = (
        df["package_weight_kg"] / df["package_volume_m3"].clip(lower=1e-4)).round(3)
    df["value_per_kg_usd"] = (
        df["declared_value_usd"] / df["package_weight_kg"].clip(lower=0.01)).round(3)
    df["is_heavy"] = df["package_weight_kg"] > 50
    df["is_high_value"] = df["declared_value_usd"] > df["declared_value_usd"].quantile(0.9)
    df["special_handling"] = df["cold_chain_required"] | df["fragile"] | df["signature_required"]
    df["log_weight"] = np.log1p(df["package_weight_kg"].clip(lower=0)).round(4)
    df["log_value"] = np.log1p(df["declared_value_usd"].clip(lower=0)).round(4)

    # --------------------------------------------------------------- resources
    if not drivers.empty:
        d = drivers.set_index("driver_id")
        df["driver_experience_years"] = d["experience_years"].reindex(df["driver_id"]).to_numpy()
        df["driver_skill_score"] = d["skill_score"].reindex(df["driver_id"]).to_numpy()
        df["driver_safety_score"] = d["safety_score"].reindex(df["driver_id"]).to_numpy()
        df["driver_avg_rating"] = d["avg_rating"].reindex(df["driver_id"]).to_numpy()
        df["driver_employment_type"] = d["employment_type"].reindex(df["driver_id"]).to_numpy()
        df["driver_hourly_cost_usd"] = d["hourly_cost_usd"].reindex(df["driver_id"]).to_numpy()

    if not vehicles.empty:
        v = vehicles.set_index("vehicle_id")
        df["vehicle_type"] = v["vehicle_type"].reindex(df["vehicle_id"]).to_numpy()
        df["vehicle_age_years"] = (2026 - v["model_year"].reindex(df["vehicle_id"]).to_numpy())
        df["vehicle_capacity_kg"] = v["capacity_kg"].reindex(df["vehicle_id"]).to_numpy()
        df["vehicle_odometer_km"] = v["odometer_km"].reindex(df["vehicle_id"]).to_numpy()
        df["vehicle_consumption"] = v["avg_consumption_l_per_100km"].reindex(df["vehicle_id"]).to_numpy()
        df["vehicle_is_electric"] = v["fuel_type"].reindex(df["vehicle_id"]).eq("electric").to_numpy()
        df["load_factor"] = (df["package_weight_kg"] / df["vehicle_capacity_kg"].clip(lower=1)).round(5)

    if not customers.empty:
        c = customers.set_index("customer_id")
        df["customer_segment"] = c["segment"].reindex(df["customer_id"]).to_numpy()
        df["customer_loyalty_tier"] = c["loyalty_tier"].reindex(df["customer_id"]).to_numpy()
        df["customer_lifetime_orders"] = c["lifetime_orders"].reindex(df["customer_id"]).to_numpy()
        df["customer_avg_rating_given"] = c["avg_rating_given"].reindex(df["customer_id"]).to_numpy()
        df["customer_churn_risk"] = c["churn_risk_score"].reindex(df["customer_id"]).to_numpy()
        df["customer_is_business"] = c["is_business"].reindex(df["customer_id"]).to_numpy()
        df["customer_tenure_days"] = (
            ts - c["signup_date"].reindex(df["customer_id"]).to_numpy()).dt.days

    # ------------------------------------------------------------------ routes
    if not routes.empty and "route_id" in df.columns:
        r = routes.set_index("route_id")
        df["route_stops"] = r["stops"].reindex(df["route_id"]).to_numpy()
        df["route_utilisation_kg"] = r["capacity_utilisation_kg"].reindex(df["route_id"]).to_numpy()
        df["route_type"] = r["route_type"].reindex(df["route_id"]).to_numpy()
        df["route_optimisation_engine"] = r["optimisation_engine"].reindex(df["route_id"]).to_numpy()
        df["route_on_time_rate"] = r["on_time_rate"].reindex(df["route_id"]).to_numpy()
        df["stops_ahead"] = (df["route_stops"] - df["stop_sequence"]).clip(lower=0)

    # ----------------------------------------------------------- environmental
    df["weather_is_adverse"] = df["weather_condition"].isin(
        ["rain", "heavy_rain", "snow", "storm", "fog"])
    df["traffic_is_heavy"] = df["traffic_level"].isin(["heavy", "gridlock"])
    df["env_stress_index"] = (
        df["weather_severity_index"].fillna(0) * 0.55 + df["congestion_index"].fillna(0.4) * 0.45
    ).round(4)
    df["temp_extreme"] = (df["temperature_c"] < 0) | (df["temperature_c"] > 33)

    # --------------------------------------------------- historical (leak-free)
    windows = cfg.get("features.rolling_windows_days", [7, 30])
    df = df.sort_values("order_timestamp").reset_index(drop=True)
    for key, label in [("zone_id", "zone"), ("driver_id", "driver"), ("warehouse_id", "warehouse")]:
        if key not in df.columns:
            continue
        grp = df.groupby(key, sort=False)
        # shift(1) => strictly past information only
        df[f"{label}_prior_orders"] = grp.cumcount()
        df[f"{label}_prior_late_rate"] = (
            grp["is_late"].apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True).round(4))
        for w in windows:
            df[f"{label}_roll{w}d_delay_mean"] = (
                grp["delay_minutes"].apply(lambda s: s.shift(1).rolling(w * 5, min_periods=3).mean())
                .reset_index(level=0, drop=True).round(3))

    # ------------------------------------------------------------- efficiency
    df["cost_per_km_usd"] = (df["delivery_cost_usd"] / df["distance_km"].clip(lower=0.1)).round(4)
    df["cost_per_kg_usd"] = (df["delivery_cost_usd"] / df["package_weight_kg"].clip(lower=0.01)).round(4)
    df["margin_usd"] = (df["revenue_usd"] - df["delivery_cost_usd"]).round(3)
    df["margin_pct"] = (df["margin_usd"] / df["revenue_usd"].clip(lower=0.01)).round(4)
    df["speed_kmh_effective"] = (
        df["distance_km"] / (df["actual_duration_min"].abs().clip(lower=1) / 60)).round(3)
    df["planned_vs_actual_ratio"] = (
        df["actual_duration_min"] / df["planned_duration_min"].clip(lower=1)).round(4)
    df["sla_utilisation"] = (
        df["actual_duration_min"] / (df["sla_hours"] * 60).clip(lower=1)).round(4)
    df["co2_per_km"] = (df["co2_kg"] / df["distance_km"].clip(lower=0.1)).round(5)

    logger.info("Feature table: %d rows x %d columns", len(df), df.shape[1])
    return df
