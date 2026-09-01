"""Dimension tables: warehouses, pickup locations, delivery zones, customers,
vehicles and drivers.

These are generated first because every fact table keys off them. All builders
are pure functions of ``(cfg, streams)`` plus already-built parents, so they can
be unit-tested and regenerated independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import reference as ref
from ..config import Config
from ..rng import RandomStreams, lognormal_between, truncated_normal, weighted_choice
from ..utils import jitter_coords, make_ids

__all__ = [
    "build_warehouses", "build_pickup_locations", "build_delivery_zones",
    "build_customers", "build_vehicles", "build_drivers",
]


def _city_assignment(rng: np.random.Generator, n: int) -> np.ndarray:
    """Pick home cities with a Zipf-ish tilt so a few hubs dominate volume."""
    weights = 1.0 / np.power(np.arange(1, len(ref.CITIES) + 1), 0.35)
    weights = weights / weights.sum()
    return rng.choice(len(ref.CITIES), size=n, p=weights)


def _people_names(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(ref.FIRST_NAMES, dtype=object)[rng.integers(0, len(ref.FIRST_NAMES), n)]
    last = np.asarray(ref.LAST_NAMES, dtype=object)[rng.integers(0, len(ref.LAST_NAMES), n)]
    return first, last


def _street_addresses(rng: np.random.Generator, n: int) -> np.ndarray:
    numbers = rng.integers(1, 480, n).astype(str)
    stems = np.asarray(ref.STREET_STEMS, dtype=object)[rng.integers(0, len(ref.STREET_STEMS), n)]
    types = np.asarray(ref.STREET_TYPES, dtype=object)[rng.integers(0, len(ref.STREET_TYPES), n)]
    return pd.Series(numbers) + " " + pd.Series(stems) + " " + pd.Series(types)


# --------------------------------------------------------------------------- #
# Warehouses
# --------------------------------------------------------------------------- #
def build_warehouses(cfg: Config, streams: RandomStreams) -> pd.DataFrame:
    rng = streams.get("warehouses")
    n = cfg.volume("warehouses")
    city_idx = _city_assignment(rng, n)
    lat, lon = jitter_coords(rng, ref.CITY_LAT[city_idx], ref.CITY_LON[city_idx], radius_km=18)

    wtype = weighted_choice(rng, ref.WAREHOUSE_TYPES, ref.WAREHOUSE_TYPE_WEIGHTS, n)
    automation = weighted_choice(rng, ref.AUTOMATION_LEVELS, [0.34, 0.36, 0.24, 0.06], n)
    capacity = np.where(
        wtype == "micro_hub",
        lognormal_between(rng, 400, 3_000, n),
        lognormal_between(rng, 5_000, 120_000, n),
    ).round(0)

    df = pd.DataFrame({
        "warehouse_id": make_ids("WH", n, 5),
        "warehouse_name": [f"{ref.CITY_NAMES[i]} DC {k+1}" for k, i in enumerate(city_idx)],
        "warehouse_type": wtype,
        "city": np.asarray(ref.CITY_NAMES, dtype=object)[city_idx],
        "country": np.asarray(ref.COUNTRIES, dtype=object)[city_idx],
        "region": np.asarray(ref.REGIONS, dtype=object)[city_idx],
        "latitude": lat.round(6),
        "longitude": lon.round(6),
        "capacity_m3": capacity,
        "storage_slots": (capacity * rng.uniform(1.5, 4.0, n)).round(0),
        "automation_level": automation,
        "dock_doors": rng.integers(2, 60, n),
        "staff_headcount": (capacity / rng.uniform(120, 900, n)).round(0).clip(3, None),
        "operating_hours_per_day": rng.choice([8, 12, 16, 24], n, p=[0.18, 0.31, 0.27, 0.24]),
        "cold_chain_capable": (wtype == "cold_storage") | (rng.random(n) < 0.22),
        "hazmat_certified": rng.random(n) < 0.14,
        "opened_date": pd.to_datetime("2005-01-01") + pd.to_timedelta(rng.integers(0, 7000, n), "D"),
        "monthly_fixed_cost_usd": (capacity * rng.uniform(0.9, 3.4, n)).round(2),
        "throughput_capacity_orders_day": (capacity * rng.uniform(0.05, 0.4, n)).round(0).clip(20, None),
    })
    df["cost_index"] = df["region"].map(ref.REGION_COST_INDEX).astype(float)
    return df


# --------------------------------------------------------------------------- #
# Pickup locations
# --------------------------------------------------------------------------- #
def build_pickup_locations(cfg: Config, streams: RandomStreams, warehouses: pd.DataFrame) -> pd.DataFrame:
    rng = streams.get("pickup_locations")
    n = cfg.volume("pickup_locations")
    parent = rng.integers(0, len(warehouses), n)
    plat, plon = jitter_coords(
        rng, warehouses["latitude"].to_numpy()[parent],
        warehouses["longitude"].to_numpy()[parent], radius_km=18,
    )
    return pd.DataFrame({
        "pickup_location_id": make_ids("PU", n, 6),
        "warehouse_id": warehouses["warehouse_id"].to_numpy()[parent],
        "location_type": weighted_choice(
            rng, ["locker", "partner_store", "depot", "curbside", "merchant_site"],
            [0.24, 0.30, 0.20, 0.10, 0.16], n),
        "address": _street_addresses(rng, n).to_numpy(),
        "city": warehouses["city"].to_numpy()[parent],
        "country": warehouses["country"].to_numpy()[parent],
        "latitude": plat.round(6),
        "longitude": plon.round(6),
        "capacity_parcels": rng.integers(20, 900, n),
        "avg_dwell_minutes": truncated_normal(rng, 4.5, 2.2, 0.5, 25, n).round(2),
        "is_24_7": rng.random(n) < 0.31,
        "accessibility_score": truncated_normal(rng, 0.72, 0.16, 0.05, 1.0, n).round(3),
    })


# --------------------------------------------------------------------------- #
# Delivery zones
# --------------------------------------------------------------------------- #
def build_delivery_zones(cfg: Config, streams: RandomStreams, warehouses: pd.DataFrame) -> pd.DataFrame:
    """Delivery zones, restricted to cities the network actually serves.

    Zones (and therefore customers) are placed only in cities that have at
    least one warehouse, weighted by how many. Scattering demand into cities
    with no local site would force every order to be fulfilled from hundreds of
    kilometres away and corrupt every distance-derived field downstream.
    """
    rng = streams.get("delivery_zones")
    n = cfg.volume("delivery_zones")

    served = warehouses["city"].value_counts()
    city_pool = np.array([ref.CITY_NAMES.index(c) for c in served.index])
    weights = served.to_numpy(dtype=float)
    city_idx = rng.choice(city_pool, size=n, p=weights / weights.sum())

    urban_share = ref.CITY_URBAN_SHARE[city_idx]
    roll = rng.random(n)
    area_type = np.where(roll < urban_share, "urban",
                         np.where(roll < urban_share + (1 - urban_share) * 0.6, "suburban", "rural"))

    # Scatter radius follows the settlement pattern: inner-city zones sit close
    # to the centre, rural zones sprawl. This is what makes urban deliveries
    # short-distance and rural ones long, rather than an arbitrary constant.
    radius = np.select([area_type == "urban", area_type == "suburban"], [11.0, 28.0], default=70.0)
    zlat, zlon = jitter_coords(rng, ref.CITY_LAT[city_idx], ref.CITY_LON[city_idx], radius_km=radius)

    pop_density = np.where(
        area_type == "urban", lognormal_between(rng, 1800, 21000, n),
        np.where(area_type == "suburban", lognormal_between(rng, 350, 2200, n),
                 lognormal_between(rng, 8, 340, n)))

    return pd.DataFrame({
        "zone_id": make_ids("ZN", n, 6),
        "zone_name": [f"{ref.CITY_NAMES[i]}-Z{k%999:03d}" for k, i in enumerate(city_idx)],
        "city": np.asarray(ref.CITY_NAMES, dtype=object)[city_idx],
        "country": np.asarray(ref.COUNTRIES, dtype=object)[city_idx],
        "region": np.asarray(ref.REGIONS, dtype=object)[city_idx],
        "area_type": area_type,
        "centroid_lat": zlat.round(6),
        "centroid_lon": zlon.round(6),
        "area_km2": np.where(area_type == "urban", rng.uniform(0.4, 9), rng.uniform(4, 260)).round(3),
        "population_density_per_km2": pop_density.round(1),
        "avg_income_index": truncated_normal(rng, 1.0, 0.28, 0.3, 2.6, n).round(3),
        "delivery_difficulty_index": np.clip(
            truncated_normal(rng, 0.5, 0.18, 0, 1, n)
            + (area_type == "urban") * 0.12 + (area_type == "rural") * 0.08, 0, 1).round(3),
        "parking_availability": truncated_normal(rng, 0.6, 0.22, 0.02, 1.0, n).round(3),
        "avg_floor_count": np.where(area_type == "urban", rng.integers(1, 22, n), rng.integers(1, 4, n)),
        "toll_zone": rng.random(n) < 0.13,
        "low_emission_zone": (area_type == "urban") & (rng.random(n) < 0.35),
        "service_start_hour": rng.choice([6, 7, 8, 9], n, p=[0.2, 0.35, 0.3, 0.15]),
        "service_end_hour": rng.choice([18, 20, 21, 22, 23], n, p=[0.18, 0.3, 0.25, 0.17, 0.1]),
    })


# --------------------------------------------------------------------------- #
# Customers
# --------------------------------------------------------------------------- #
def build_customers(cfg: Config, streams: RandomStreams, zones: pd.DataFrame) -> pd.DataFrame:
    rng = streams.get("customers")
    n = cfg.volume("customers")
    zone_idx = rng.integers(0, len(zones), n)
    first, last = _people_names(rng, n)
    segment = weighted_choice(rng, ref.CUSTOMER_SEGMENTS, ref.CUSTOMER_SEGMENT_WEIGHTS, n)

    clat, clon = jitter_coords(
        rng, zones["centroid_lat"].to_numpy()[zone_idx],
        zones["centroid_lon"].to_numpy()[zone_idx], radius_km=3.0,
    )
    signup = pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 2900, n), "D")
    lifetime_orders = np.where(
        segment == "enterprise", lognormal_between(rng, 20, 4000, n),
        np.where(segment == "smb", lognormal_between(rng, 5, 600, n),
                 lognormal_between(rng, 1, 180, n))).round(0)

    return pd.DataFrame({
        "customer_id": make_ids("CUS", n, 7),
        "customer_name": pd.Series(first) + " " + pd.Series(last),
        "segment": segment,
        "loyalty_tier": weighted_choice(rng, ref.LOYALTY_TIERS, [0.40, 0.24, 0.18, 0.13, 0.05], n),
        "zone_id": zones["zone_id"].to_numpy()[zone_idx],
        "city": zones["city"].to_numpy()[zone_idx],
        "country": zones["country"].to_numpy()[zone_idx],
        "region": zones["region"].to_numpy()[zone_idx],
        "address": _street_addresses(rng, n).to_numpy(),
        "latitude": clat.round(6),
        "longitude": clon.round(6),
        "signup_date": signup,
        "lifetime_orders": lifetime_orders,
        "lifetime_value_usd": (lifetime_orders * lognormal_between(rng, 12, 900, n)).round(2),
        "avg_rating_given": truncated_normal(rng, 4.15, 0.72, 1.0, 5.0, n).round(2),
        "preferred_delivery_window": weighted_choice(
            rng, ["morning", "afternoon", "evening", "any"], [0.22, 0.24, 0.29, 0.25], n),
        "is_business": np.isin(segment, ["smb", "enterprise", "public_sector"]),
        "contract_sla_hours": np.where(
            np.isin(segment, ["enterprise", "public_sector"]),
            rng.choice([12, 24, 48], n), np.nan),
        "churn_risk_score": truncated_normal(rng, 0.28, 0.17, 0, 1, n).round(3),
        "email": pd.Series(first).str.lower() + "." + pd.Series(last).str.lower()
                 + rng.integers(1, 999, n).astype(str) + "@example.com",
    })


# --------------------------------------------------------------------------- #
# Vehicles
# --------------------------------------------------------------------------- #
def build_vehicles(cfg: Config, streams: RandomStreams, warehouses: pd.DataFrame) -> pd.DataFrame:
    rng = streams.get("vehicles")
    n = cfg.volume("vehicles")
    vtype = weighted_choice(rng, ref.VEHICLE_TYPES, ref.VEHICLE_TYPE_WEIGHTS, n)
    home = rng.integers(0, len(warehouses), n)
    year = rng.integers(2012, 2026, n)
    age_years = 2026 - year
    odometer = (age_years * rng.uniform(9_000, 62_000, n)).round(0)

    fuel = np.where(vtype == "bike", "none",
           np.where(vtype == "ev_van", "electric",
           np.where(vtype == "car", rng.choice(["petrol", "hybrid", "electric"], n, p=[0.5, 0.3, 0.2]),
                    rng.choice(["diesel", "cng", "electric"], n, p=[0.82, 0.1, 0.08]))))

    eff_map = cfg.get("operations.fuel_efficiency_l_per_100km", {})
    base_eff = np.array([eff_map.get(t, 10.0) for t in vtype], dtype=float)
    # Older vehicles burn more; EVs and bikes stay at zero litres.
    consumption = np.where(base_eff > 0, base_eff * (1 + age_years * 0.012) * rng.uniform(0.9, 1.15, n), 0.0)

    return pd.DataFrame({
        "vehicle_id": make_ids("VEH", n, 6),
        "vehicle_type": vtype,
        "make": np.asarray(ref.VEHICLE_MAKES, dtype=object)[rng.integers(0, len(ref.VEHICLE_MAKES), n)],
        "model_year": year,
        "fuel_type": fuel,
        "home_warehouse_id": warehouses["warehouse_id"].to_numpy()[home],
        "region": warehouses["region"].to_numpy()[home],
        "capacity_kg": np.array([ref.VEHICLE_CAPACITY_KG[t] for t in vtype]) * rng.uniform(0.85, 1.15, n),
        "capacity_m3": np.array([ref.VEHICLE_CAPACITY_M3[t] for t in vtype]) * rng.uniform(0.85, 1.15, n),
        "odometer_km": odometer,
        "avg_consumption_l_per_100km": consumption.round(2),
        "battery_capacity_kwh": np.where(fuel == "electric", rng.uniform(40, 180, n).round(1), np.nan),
        "refrigeration_unit": vtype == "refrigerated_truck",
        "telematics_installed": rng.random(n) < 0.86,
        "insurance_annual_usd": (rng.uniform(600, 7200, n) * (1 + age_years * 0.02)).round(2),
        "purchase_cost_usd": (np.array([ref.VEHICLE_CAPACITY_KG[t] for t in vtype]) * rng.uniform(6, 22, n) + 2500).round(2),
        "acquired_date": pd.to_datetime(year.astype(str) + "-01-01")
                         + pd.to_timedelta(rng.integers(0, 364, n), "D"),
        "status": weighted_choice(rng, ["active", "maintenance", "idle", "retired"],
                                  [0.80, 0.09, 0.08, 0.03], n),
        "last_inspection_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 580, n), "D"),
        "co2_g_per_km": (consumption * 26.8).round(1),
    })


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def build_drivers(cfg: Config, streams: RandomStreams, warehouses: pd.DataFrame) -> pd.DataFrame:
    rng = streams.get("drivers")
    n = cfg.volume("drivers")
    first, last = _people_names(rng, n)
    home = rng.integers(0, len(warehouses), n)
    employment = weighted_choice(rng, ref.EMPLOYMENT_TYPES, ref.EMPLOYMENT_WEIGHTS, n)
    experience = np.where(
        employment == "gig", truncated_normal(rng, 1.6, 1.3, 0.0, 12, n),
        truncated_normal(rng, 6.4, 4.1, 0.0, 35, n)).round(1)

    lo, hi = cfg.get("operations.driver_hourly_cost_usd.min", 14), cfg.get("operations.driver_hourly_cost_usd.max", 42)
    region = warehouses["region"].to_numpy()[home]
    cost_index = np.array([ref.REGION_COST_INDEX.get(r, 1.0) for r in region])
    hourly = np.clip(rng.uniform(lo, hi, n) * cost_index * (1 + experience * 0.011), lo * 0.5, hi * 1.6)

    # Performance correlates with experience but keeps a healthy spread.
    skill = np.clip(0.55 + experience * 0.018 + rng.normal(0, 0.11, n), 0.15, 0.99)

    return pd.DataFrame({
        "driver_id": make_ids("DRV", n, 6),
        "driver_name": pd.Series(first) + " " + pd.Series(last),
        "home_warehouse_id": warehouses["warehouse_id"].to_numpy()[home],
        "region": region,
        "employment_type": employment,
        "licence_class": weighted_choice(rng, ref.LICENCE_CLASSES, [0.34, 0.26, 0.22, 0.06, 0.12], n),
        "experience_years": experience,
        "hire_date": pd.to_datetime("2026-01-01") - pd.to_timedelta((experience * 365).astype(int), "D"),
        "age": (20 + experience + truncated_normal(rng, 6, 5, 0, 30, n)).round(0).clip(18, 68),
        "hourly_cost_usd": hourly.round(2),
        "skill_score": skill.round(3),
        "safety_score": np.clip(skill + rng.normal(0, 0.07, n), 0.1, 1.0).round(3),
        "avg_rating": np.clip(3.0 + skill * 2.0 + rng.normal(0, 0.18, n), 1.0, 5.0).round(2),
        "accidents_last_2y": rng.poisson(np.clip(0.9 - skill, 0.02, None) * 1.6, n),
        "training_hours_last_year": truncated_normal(rng, 22, 12, 0, 120, n).round(0),
        "max_weekly_hours": rng.choice([20, 30, 40, 48], n, p=[0.14, 0.18, 0.50, 0.18]),
        "preferred_shift": weighted_choice(rng, ref.SHIFT_TYPES, [0.34, 0.28, 0.15, 0.13, 0.10], n),
        "hazmat_certified": rng.random(n) < 0.11,
        "cold_chain_certified": rng.random(n) < 0.19,
        "languages_spoken": rng.integers(1, 4, n),
        "active": rng.random(n) < 0.93,
    })
