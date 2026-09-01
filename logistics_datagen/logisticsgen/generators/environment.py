"""Environment tables: traffic, weather, fuel prices and regional holidays.

These are *exogenous* panels on a (place x time) grid. Orders later join against
them so that congestion, storms and fuel shocks show up as genuine causal
drivers of delay and cost rather than as independent noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import reference as ref
from ..config import Config
from ..rng import RandomStreams, truncated_normal, weighted_choice
from ..utils import make_ids

__all__ = ["build_traffic", "build_weather", "build_fuel_prices", "build_regional_holidays"]


def _grid(cfg: Config, resolution_hours: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg.get("time.start_date"))
    end = pd.Timestamp(cfg.get("time.end_date"))
    return pd.date_range(start, end, freq=f"{resolution_hours}h", inclusive="left")


# --------------------------------------------------------------------------- #
# Traffic
# --------------------------------------------------------------------------- #
def build_traffic(cfg: Config, streams: RandomStreams) -> pd.DataFrame:
    """City x timestamp congestion panel with rush-hour and weekend structure."""
    rng = streams.get("traffic")
    res = int(cfg.get("child_tables.traffic.resolution_hours", 1))
    ts = _grid(cfg, res)
    n_cities = len(ref.CITIES)

    city_rep = np.repeat(np.arange(n_cities), len(ts))
    ts_rep = np.tile(ts.to_numpy(), n_cities)
    ts_idx = pd.DatetimeIndex(ts_rep)
    n = city_rep.size

    hour = ts_idx.hour.to_numpy()
    dow = ts_idx.dayofweek.to_numpy()

    # Twin-peaked commuter profile, flattened at weekends.
    morning = np.exp(-0.5 * ((hour - 8.2) / 1.7) ** 2)
    evening = np.exp(-0.5 * ((hour - 17.6) / 2.1) ** 2)
    midday = 0.35 * np.exp(-0.5 * ((hour - 13.0) / 3.2) ** 2)
    weekday_amp = np.where(dow < 5, 1.0, 0.45)
    urbanity = ref.CITY_URBAN_SHARE[city_rep]

    # Calibrated so a quiet rural night reads free-flow and a weekday peak reads
    # heavy. True gridlock is left to the anomaly injector, where it belongs -
    # a baseline that produces gridlock half the time carries no information.
    congestion = np.clip(
        0.07 + urbanity * 0.22 + weekday_amp * (0.30 * morning + 0.34 * evening + 0.5 * midday)
        + rng.normal(0, 0.06, n), 0.0, 1.0)

    level = np.select(
        [congestion < 0.22, congestion < 0.42, congestion < 0.62, congestion < 0.82],
        ["free_flow", "light", "moderate", "heavy"], default="gridlock")

    incident = np.where(
        rng.random(n) < 0.035,
        weighted_choice(rng, ref.INCIDENT_TYPES[1:], [0.34, 0.30, 0.16, 0.12, 0.08], n),
        "none")

    free_flow_speed = np.where(urbanity > 0.85, 46.0, np.where(urbanity > 0.7, 54.0, 66.0))
    avg_speed = free_flow_speed * (1.0 - 0.62 * congestion) * np.where(incident == "none", 1.0, 0.78)

    return pd.DataFrame({
        "traffic_id": make_ids("TRF", n, 9),
        "city": np.asarray(ref.CITY_NAMES, dtype=object)[city_rep],
        "country": np.asarray(ref.COUNTRIES, dtype=object)[city_rep],
        "region": np.asarray(ref.REGIONS, dtype=object)[city_rep],
        "timestamp": ts_idx,
        "hour_of_day": hour,
        "day_of_week": dow,
        "congestion_index": congestion.round(4),
        "traffic_level": level,
        "avg_speed_kmh": avg_speed.round(2),
        "incident_type": incident,
        "road_closure_count": rng.poisson(0.25 + congestion * 0.8, n),
        "delay_factor": np.array([ref.TRAFFIC_DELAY_FACTOR[x] for x in level])
                        * (1 + rng.normal(0, 0.04, n)).clip(0.8, 1.3),
    }).round({"delay_factor": 4})


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def build_weather(cfg: Config, streams: RandomStreams) -> pd.DataFrame:
    """City x timestamp weather panel with latitude-aware seasonal temperature."""
    rng = streams.get("weather")
    res = int(cfg.get("child_tables.weather.resolution_hours", 3))
    ts = _grid(cfg, res)
    n_cities = len(ref.CITIES)

    city_rep = np.repeat(np.arange(n_cities), len(ts))
    ts_idx = pd.DatetimeIndex(np.tile(ts.to_numpy(), n_cities))
    n = city_rep.size

    lat = ref.CITY_LAT[city_rep]
    doy = ts_idx.dayofyear.to_numpy()
    hour = ts_idx.hour.to_numpy()

    # Seasonal sine flipped below the equator, plus a diurnal swing.
    seasonal = np.cos(2 * np.pi * (doy - 200) / 365.25) * np.sign(lat)
    baseline = 27.0 - 0.35 * np.abs(lat)
    amplitude = 2.0 + 0.22 * np.abs(lat)
    diurnal = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = baseline - amplitude * seasonal + diurnal + rng.normal(0, 2.6, n)

    cold = temperature < 1.5
    condition = weighted_choice(rng, ref.WEATHER_CONDITIONS, ref.WEATHER_WEIGHTS, n)
    # Snow only where it can actually snow; otherwise downgrade to rain.
    condition = np.where((condition == "snow") & ~cold, "rain", condition)
    condition = np.where((condition == "heat_wave") & (temperature < 30), "clear", condition)

    wet = np.isin(condition, ["rain", "heavy_rain", "snow", "storm"])
    precipitation = np.where(wet, np.abs(rng.gamma(1.7, 2.4, n)), 0.0)
    precipitation = np.where(condition == "heavy_rain", precipitation * 3.2, precipitation)

    visibility = np.clip(
        np.where(condition == "fog", rng.uniform(0.05, 1.2, n),
                 np.where(wet, rng.uniform(1.5, 9.0, n), rng.uniform(8.0, 30.0, n))), 0.02, 40.0)

    return pd.DataFrame({
        "weather_id": make_ids("WTH", n, 9),
        "city": np.asarray(ref.CITY_NAMES, dtype=object)[city_rep],
        "country": np.asarray(ref.COUNTRIES, dtype=object)[city_rep],
        "region": np.asarray(ref.REGIONS, dtype=object)[city_rep],
        "timestamp": ts_idx,
        "condition": condition,
        "temperature_c": temperature.round(2),
        "feels_like_c": (temperature - np.clip(rng.normal(1.4, 1.6, n), -4, 6)).round(2),
        "precipitation_mm": precipitation.round(2),
        "snow_depth_cm": np.where(condition == "snow", np.abs(rng.gamma(1.4, 3.0, n)).round(1), 0.0),
        "wind_speed_kmh": np.abs(rng.gamma(2.2, 6.0, n) + (condition == "storm") * 40).round(1),
        "visibility_km": visibility.round(2),
        "humidity_pct": np.clip(truncated_normal(rng, 68, 16, 10, 100, n) + wet * 12, 5, 100).round(1),
        "road_condition": np.select(
            [condition == "snow", np.isin(condition, ["heavy_rain", "storm"]), condition == "rain"],
            ["icy", "flooded", "wet"], default="dry"),
        "severity_index": np.clip(
            (precipitation / 25) + (30 - np.minimum(visibility, 30)) / 60
            + np.maximum(0, -temperature) / 20, 0, 1).round(4),
        "delay_factor": np.array([ref.WEATHER_DELAY_FACTOR[c] for c in condition]).round(3),
    })


# --------------------------------------------------------------------------- #
# Fuel prices
# --------------------------------------------------------------------------- #
def build_fuel_prices(cfg: Config, streams: RandomStreams) -> pd.DataFrame:
    """Daily country x fuel-type price series as a mean-reverting random walk."""
    rng = streams.get("fuel_prices")
    start = pd.Timestamp(cfg.get("time.start_date"))
    end = pd.Timestamp(cfg.get("time.end_date"))
    days = pd.date_range(start, end, freq="D", inclusive="left")
    countries = sorted(set(ref.COUNTRIES))
    fuels = ["diesel", "petrol", "cng", "electricity_kwh"]

    frames = []
    anchors = {"diesel": 1.62, "petrol": 1.71, "cng": 0.98, "electricity_kwh": 0.29}
    for country in countries:
        c_idx = ref.COUNTRIES.index(country)
        cost_idx = ref.REGION_COST_INDEX.get(ref.REGIONS[c_idx], 1.0)
        for fuel in fuels:
            anchor = anchors[fuel] * cost_idx * rng.uniform(0.85, 1.2)
            # Ornstein-Uhlenbeck style walk: drifts but never runs away.
            shocks = rng.normal(0, anchor * 0.012, len(days))
            series = np.empty(len(days))
            level = anchor
            for i in range(len(days)):
                level += 0.04 * (anchor - level) + shocks[i]
                series[i] = level
            frames.append(pd.DataFrame({
                "date": days, "country": country, "fuel_type": fuel,
                "price_per_unit_usd": np.clip(series, anchor * 0.4, anchor * 2.5).round(4),
                "currency": "USD",
                "tax_rate": round(float(rng.uniform(0.12, 0.55)), 3),
            }))

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "fuel_price_id", make_ids("FUE", len(out), 8))
    out["price_index"] = (out["price_per_unit_usd"]
                          / out.groupby("fuel_type")["price_per_unit_usd"].transform("mean")).round(4)
    return out


# --------------------------------------------------------------------------- #
# Regional holidays
# --------------------------------------------------------------------------- #
def build_regional_holidays(cfg: Config, streams: RandomStreams) -> pd.DataFrame:
    """Holiday calendar per country, with a demand multiplier and closure flag."""
    rng = streams.get("holidays")
    start = pd.Timestamp(cfg.get("time.start_date"))
    end = pd.Timestamp(cfg.get("time.end_date"))
    years = range(start.year, end.year + 1)

    rows = []
    for country in sorted(set(ref.COUNTRIES)):
        c_idx = ref.COUNTRIES.index(country)
        region = ref.REGIONS[c_idx]
        for year in years:
            for month, day, name, scope in ref.HOLIDAY_TEMPLATES:
                applies = (
                    scope == "global"
                    or scope == country
                    or scope == region
                    or (scope == "christian" and region in {"EMEA", "NA", "LATAM"})
                )
                if not applies:
                    continue
                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                if not (start <= date < end):
                    continue
                is_shopping = name in {"Black Friday", "Singles Day"}
                rows.append({
                    "country": country,
                    "region": region,
                    "date": date,
                    "holiday_name": name,
                    "holiday_type": "shopping_event" if is_shopping else "public_holiday",
                    "is_working_day": bool(is_shopping),
                    "warehouse_closure": (not is_shopping) and bool(rng.random() < 0.62),
                    "demand_multiplier": round(
                        float(rng.uniform(2.1, 4.4) if is_shopping else rng.uniform(0.25, 0.85)), 3),
                    "staffing_multiplier": round(
                        float(rng.uniform(1.3, 2.2) if is_shopping else rng.uniform(0.3, 0.8)), 3),
                })

    out = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    out.insert(0, "holiday_id", make_ids("HOL", len(out), 6))
    return out
