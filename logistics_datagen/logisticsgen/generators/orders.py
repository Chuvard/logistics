"""Orders and Routes - the fact core of the dataset.

The delivery-duration model is the most important piece of realism here. Rather
than sampling durations directly, we *build* them:

    travel_time = distance / effective_speed
    effective_speed = base_speed(area) / (traffic_factor * weather_factor)
    total_time = travel_time + service_time + handover + queueing
    delay = total_time - planned_time  (+ disruption shocks)

Because traffic and weather come from the exogenous panels, downstream models
have a genuine signal to learn instead of memorising noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import reference as ref
from ..config import Config
from ..rng import RandomStreams, lognormal_between, truncated_normal, weighted_choice
from ..utils import haversine_km, make_ids, random_timestamps

__all__ = ["build_orders", "build_routes"]


def _domain_draw(cfg: Config, rng: np.random.Generator, n: int) -> np.ndarray:
    domains = cfg.require("domains")
    names = list(domains.keys())
    weights = [float(domains[k].get("weight", 1.0)) for k in names]
    return weighted_choice(rng, names, weights, n)


def _per_domain_attribute(
    cfg: Config, rng: np.random.Generator, domain: np.ndarray, key: str
) -> np.ndarray:
    """Draw a lognormal value per row using that row's domain-specific range."""
    domains = cfg.require("domains")
    out = np.empty(domain.size, dtype=float)
    for name, spec in domains.items():
        mask = domain == name
        k = int(mask.sum())
        if k == 0:
            continue
        low, high = spec[key]
        out[mask] = lognormal_between(rng, float(low), float(high), k)
    return out


def _domain_probability(cfg: Config, domain: np.ndarray, key: str) -> np.ndarray:
    domains = cfg.require("domains")
    lookup = {name: float(spec.get(key, 0.0)) for name, spec in domains.items()}
    return np.array([lookup[d] for d in domain], dtype=float)


def _assign_routes(
    cfg: Config,
    rng: np.random.Generator,
    origin_idx: np.ndarray,
    order_ts: pd.DatetimeIndex,
    vehicles: pd.DataFrame,
    drivers: pd.DataFrame,
    warehouses: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster orders into realistic multi-stop routes *before* resourcing them.

    Assigning a vehicle uniformly at random per order would give every vehicle
    roughly one stop a day, which is nothing like real operations. Instead we
    group orders by (origin warehouse, calendar day), cut each group into runs
    of ``stops_per_route`` consecutive orders, and give each run a single
    vehicle and driver drawn from that warehouse's own fleet.

    Returns ``(route_code, stop_sequence, vehicle_idx, driver_idx)`` aligned to
    the original order row order.
    """
    lo = int(cfg.get("operations.stops_per_route.min", 4))
    hi = int(cfg.get("operations.stops_per_route.max", 28))
    n = origin_idx.size

    frame = pd.DataFrame({
        "wh": origin_idx,
        "date": order_ts.normalize(),
        "ts": order_ts,
        "pos": np.arange(n),
    }).sort_values(["wh", "date", "ts"], kind="mergesort")

    grouped = frame.groupby(["wh", "date"], sort=False)
    within = grouped.cumcount().to_numpy()
    sizes = grouped.size().to_numpy()
    group_code = np.repeat(np.arange(sizes.size), sizes)

    # One target run-length per warehouse-day, so a route has a consistent shape.
    target = np.repeat(rng.integers(lo, hi + 1, sizes.size), sizes)
    chunk = within // target
    stop_sequence = (within % target) + 1

    route_code = pd.factorize(group_code * (int(chunk.max()) + 1) + chunk)[0]

    # ---- resource each route from the fleet homed at its warehouse ----------
    first_of_route = np.unique(route_code, return_index=True)[1]
    route_wh = frame["wh"].to_numpy()[first_of_route]
    n_routes = route_wh.size

    wh_ids = warehouses["warehouse_id"].to_numpy()
    veh_pool: dict[int, np.ndarray] = {}
    drv_pool: dict[int, np.ndarray] = {}
    veh_home = vehicles["home_warehouse_id"].to_numpy()
    drv_home = drivers["home_warehouse_id"].to_numpy()
    for w in np.unique(route_wh):
        wid = wh_ids[w]
        v = np.flatnonzero(veh_home == wid)
        d = np.flatnonzero(drv_home == wid)
        # Fall back to the whole fleet if a site happens to have none assigned.
        veh_pool[w] = v if v.size else np.arange(len(vehicles))
        drv_pool[w] = d if d.size else np.arange(len(drivers))

    route_veh = np.empty(n_routes, dtype=int)
    route_drv = np.empty(n_routes, dtype=int)
    for w in np.unique(route_wh):
        mask = route_wh == w
        route_veh[mask] = rng.choice(veh_pool[w], size=int(mask.sum()))
        route_drv[mask] = rng.choice(drv_pool[w], size=int(mask.sum()))

    # ---- map back to original row order ------------------------------------
    pos = frame["pos"].to_numpy()
    out_route = np.empty(n, dtype=int)
    out_stop = np.empty(n, dtype=int)
    out_veh = np.empty(n, dtype=int)
    out_drv = np.empty(n, dtype=int)
    out_route[pos] = route_code
    out_stop[pos] = stop_sequence
    out_veh[pos] = route_veh[route_code]
    out_drv[pos] = route_drv[route_code]
    return out_route, out_stop, out_veh, out_drv


def build_orders(
    cfg: Config,
    streams: RandomStreams,
    customers: pd.DataFrame,
    warehouses: pd.DataFrame,
    zones: pd.DataFrame,
    pickup_locations: pd.DataFrame,
    drivers: pd.DataFrame,
    vehicles: pd.DataFrame,
    traffic: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    rng = streams.get("orders")
    n = cfg.volume("deliveries")

    # ---- who / where -------------------------------------------------------
    cust_idx = rng.integers(0, len(customers), n)
    customer_id = customers["customer_id"].to_numpy()[cust_idx]
    dest_lat = customers["latitude"].to_numpy()[cust_idx]
    dest_lon = customers["longitude"].to_numpy()[cust_idx]
    city = customers["city"].to_numpy()[cust_idx]
    country = customers["country"].to_numpy()[cust_idx]
    region = customers["region"].to_numpy()[cust_idx]
    zone_id = customers["zone_id"].to_numpy()[cust_idx]

    zone_lookup = zones.set_index("zone_id")
    z = zone_lookup.reindex(zone_id)
    area_type = z["area_type"].to_numpy()
    difficulty = z["delivery_difficulty_index"].to_numpy()
    parking = z["parking_availability"].to_numpy()
    floors = z["avg_floor_count"].to_numpy()
    toll_zone = z["toll_zone"].to_numpy()

    # Business domain is drawn up front because it governs both the package
    # profile and how far upstream the shipment originates.
    domain = _domain_draw(cfg, rng, n)

    # Origin warehouse: serve each city from its *nearest* sites. Sampling a
    # warehouse uniformly would happily ship Tokyo→London and destroy every
    # distance-derived field, so candidates are ranked by great-circle distance
    # and weighted towards the closest. A small long-haul share draws from the
    # next tier out, which is where the transportation/manufacturing domains live.
    wh_lat = warehouses["latitude"].to_numpy()
    wh_lon = warehouses["longitude"].to_numpy()
    n_wh = len(warehouses)
    near_k = min(6, n_wh)
    far_k = min(20, n_wh)

    # Long-haul is a freight phenomenon. A grocery order is never trunked in
    # from 800 km away, but a manufacturing shipment routinely is.
    long_haul_by_domain = {
        "transportation": 0.38, "manufacturing_supply": 0.32,
        "warehouse_distribution": 0.16, "retail_fulfillment": 0.06,
        "ecommerce": 0.03, "pharmaceutical": 0.03, "courier_delivery": 0.01,
        "grocery_delivery": 0.002, "ride_sharing_logistics": 0.0,
    }
    origin_idx = np.empty(n, dtype=int)
    is_long_haul = rng.random(n) < np.array(
        [long_haul_by_domain.get(d, 0.02) for d in domain], dtype=float)
    for c in np.unique(city):
        mask = city == c
        ci = ref.CITY_NAMES.index(c)
        d = haversine_km(ref.CITY_LAT[ci], ref.CITY_LON[ci], wh_lat, wh_lon)
        ranked = np.argsort(d)
        # Only sites genuinely local to the city serve it. If a city has fewer
        # than `near_k` warehouses inside the radius, it is served by the ones
        # it has - not by whatever happens to rank 6th on another continent.
        near_pool = ranked[:near_k][d[ranked[:near_k]] < 120.0]
        if near_pool.size == 0:
            near_pool = ranked[:1]
        # Long-haul still has to be drivable: cap the far tier at 1,200 km.
        far_candidates = ranked[near_k:far_k]
        far_pool = far_candidates[d[far_candidates] < 1200.0]
        if far_pool.size == 0:
            far_pool = near_pool

        w = 1.0 / (1.0 + d[near_pool])
        w = w / w.sum()

        local = mask & ~is_long_haul
        if local.any():
            origin_idx[local] = rng.choice(near_pool, size=int(local.sum()), p=w)
        far = mask & is_long_haul
        if far.any():
            origin_idx[far] = rng.choice(far_pool, size=int(far.sum()))

    warehouse_id = warehouses["warehouse_id"].to_numpy()[origin_idx]
    orig_lat = warehouses["latitude"].to_numpy()[origin_idx]
    orig_lon = warehouses["longitude"].to_numpy()[origin_idx]

    pu_idx = rng.integers(0, len(pickup_locations), n)
    uses_pickup = rng.random(n) < 0.18
    pickup_location_id = np.where(uses_pickup, pickup_locations["pickup_location_id"].to_numpy()[pu_idx], None)

    # ---- what --------------------------------------------------------------
    weight_kg = _per_domain_attribute(cfg, rng, domain, "weight_kg")
    value_usd = _per_domain_attribute(cfg, rng, domain, "value_usd")
    sla_hours_base = _per_domain_attribute(cfg, rng, domain, "sla_hours")
    cold_chain = rng.random(n) < _domain_probability(cfg, domain, "cold_chain")
    fragile = rng.random(n) < _domain_probability(cfg, domain, "fragile")

    priority = weighted_choice(rng, ref.PRIORITIES, ref.PRIORITY_WEIGHTS, n)
    sla_hours = sla_hours_base * np.array([ref.PRIORITY_SLA_FACTOR[p] for p in priority])
    sla_hours = np.clip(sla_hours, 1.0, 336.0)

    density = rng.uniform(90, 420, n)                      # kg per m3
    volume_m3 = np.clip(weight_kg / density, 0.0005, 60.0)
    package_type = np.select(
        [cold_chain, weight_kg < 0.6, weight_kg < 5, weight_kg < 25, weight_kg < 120, weight_kg < 800],
        ["cold_box", "envelope", "small_box", "medium_box", "large_box", "pallet"], default="crate")
    items_count = np.clip(rng.poisson(2.4, n) + 1, 1, 60)

    # ---- when --------------------------------------------------------------
    order_ts = random_timestamps(
        rng,
        pd.Timestamp(cfg.get("time.start_date")),
        pd.Timestamp(cfg.get("time.end_date")),
        n,
        cfg.get("time.weekday_weights"),
        cfg.get("time.month_weights"),
        cfg.get("time.hour_weights"),
    )

    # ---- distance ----------------------------------------------------------
    straight_km = haversine_km(orig_lat, orig_lon, dest_lat, dest_lon)
    # Road networks are never straight; detour factor is worst in dense cities.
    detour = np.where(area_type == "urban", rng.uniform(1.25, 1.65, n),
             np.where(area_type == "suburban", rng.uniform(1.15, 1.42, n),
                      rng.uniform(1.08, 1.28, n)))
    distance_km = np.clip(straight_km * detour, 0.3, None)

    # ---- environment join --------------------------------------------------
    traffic_res = int(cfg.get("child_tables.traffic.resolution_hours", 1))
    weather_res = int(cfg.get("child_tables.weather.resolution_hours", 3))
    t_key = order_ts.floor(f"{traffic_res}h")
    w_key = order_ts.floor(f"{weather_res}h")

    t_small = traffic.set_index(["city", "timestamp"])[
        ["congestion_index", "traffic_level", "delay_factor", "avg_speed_kmh"]]
    joined_t = t_small.reindex(pd.MultiIndex.from_arrays([city, t_key]))
    congestion = np.nan_to_num(joined_t["congestion_index"].to_numpy(), nan=0.4)
    traffic_level = pd.Series(joined_t["traffic_level"].to_numpy()).fillna("moderate").to_numpy()
    traffic_delay_factor = np.nan_to_num(joined_t["delay_factor"].to_numpy(), nan=1.1)

    w_small = weather.set_index(["city", "timestamp"])[
        ["condition", "temperature_c", "precipitation_mm", "severity_index", "delay_factor"]]
    joined_w = w_small.reindex(pd.MultiIndex.from_arrays([city, w_key]))
    weather_condition = pd.Series(joined_w["condition"].to_numpy()).fillna("clear").to_numpy()
    temperature_c = np.nan_to_num(joined_w["temperature_c"].to_numpy(), nan=15.0)
    precipitation_mm = np.nan_to_num(joined_w["precipitation_mm"].to_numpy(), nan=0.0)
    weather_severity = np.nan_to_num(joined_w["severity_index"].to_numpy(), nan=0.05)
    weather_delay_factor = np.nan_to_num(joined_w["delay_factor"].to_numpy(), nan=1.0)

    # ---- resources (route-clustered, not per-order random) -----------------
    route_code, stop_sequence, veh_idx, drv_idx = _assign_routes(
        cfg, rng, origin_idx, order_ts, vehicles, drivers, warehouses)
    driver_id = drivers["driver_id"].to_numpy()[drv_idx]
    vehicle_id = vehicles["vehicle_id"].to_numpy()[veh_idx]
    driver_skill = drivers["skill_score"].to_numpy()[drv_idx]
    vehicle_type = vehicles["vehicle_type"].to_numpy()[veh_idx]

    # ---- duration model ----------------------------------------------------
    speeds = cfg.get("operations.avg_speed_kmh", {})
    base_speed = np.select(
        [area_type == "urban", area_type == "suburban"],
        [speeds.get("urban", 26.0), speeds.get("suburban", 42.0)],
        default=speeds.get("rural", 63.0)).astype(float)
    effective_speed = np.clip(
        base_speed / (traffic_delay_factor * weather_delay_factor) * (0.9 + driver_skill * 0.22),
        4.0, 110.0)
    travel_minutes = distance_km / effective_speed * 60.0

    st = cfg.get("operations.service_time_min", {})
    service_minutes = np.clip(
        rng.normal(st.get("mean", 7.0), st.get("sd", 3.5), n)
        + floors * 0.55 + (1 - parking) * 3.4 + difficulty * 4.6
        + fragile * 1.8 + cold_chain * 2.4 + np.log1p(weight_kg) * 0.9,
        0.8, 180.0)
    handling_minutes = np.clip(rng.gamma(2.0, 4.0, n) + items_count * 0.35, 0.5, 120.0)

    actual_duration_min = travel_minutes + service_minutes + handling_minutes
    # Planned duration is what the TMS promised - optimistic and less informed.
    planned_duration_min = (
        distance_km / (base_speed * 1.05) * 60.0
        + st.get("mean", 7.0) + handling_minutes * 0.8
    ) * rng.uniform(0.92, 1.08, n)

    # ---- SLA calibration ---------------------------------------------------
    # Lateness must be an *emergent* property of the physics above, not a coin
    # flip bolted on afterwards - otherwise `is_late` and `status` can disagree
    # and no model can learn the relationship. So we compute true end-to-end
    # completion time first, then scale the promised SLA by a single constant
    # chosen so the fleet hits the configured on-time rate. Relative SLA
    # structure across priorities and domains is preserved exactly.
    ops = cfg.get("operations", {})
    pickup_lag_min = np.clip(rng.gamma(2.0, 55.0, n), 5, 4320)
    completion_min = pickup_lag_min + actual_duration_min
    target_on_time = float(ops.get("on_time_base_rate", 0.885))
    ratio = completion_min / np.maximum(sla_hours * 60.0, 1.0)
    calibration = float(np.quantile(ratio, np.clip(target_on_time, 0.01, 0.99)))
    sla_hours = sla_hours * max(calibration, 1e-6)

    pickup_ts = order_ts + pd.to_timedelta(pickup_lag_min.astype(int), unit="m")
    promised_ts = order_ts + pd.to_timedelta((sla_hours * 60).astype(int), unit="m")
    actual_ts_all = pickup_ts + pd.to_timedelta(actual_duration_min.astype(int), unit="m")
    delay_all = (actual_ts_all - promised_ts).total_seconds() / 60.0

    # ---- status ------------------------------------------------------------
    roll = rng.random(n)
    # Failure risk rises with zone difficulty; returns with fragility.
    p_failed = ops.get("failed_delivery_rate", 0.021) * (1 + difficulty)
    p_returned = ops.get("return_rate", 0.014) * (1 + fragile * 0.6)
    p_cancelled = 0.011

    late = delay_all > 0
    status = np.select(
        [roll < p_failed,
         roll < p_failed + p_returned,
         roll < p_failed + p_returned + p_cancelled,
         late],
        ["failed", "returned", "cancelled", "delivered_late"], default="delivered")

    delivered_mask = np.isin(status, ["delivered", "delivered_late"])
    actual_ts = pd.Series(actual_ts_all).where(pd.Series(delivered_mask), pd.NaT)
    delay_minutes = pd.Series(delay_all).where(pd.Series(delivered_mask), np.nan)

    # ---- cost --------------------------------------------------------------
    cost_index = np.array([ref.REGION_COST_INDEX.get(r, 1.0) for r in region])
    fuel_l_per_100 = vehicles["avg_consumption_l_per_100km"].to_numpy()[veh_idx]
    fuel_cost = distance_km * fuel_l_per_100 / 100.0 * 1.65 * cost_index
    labour_cost = (actual_duration_min / 60.0) * drivers["hourly_cost_usd"].to_numpy()[drv_idx]
    toll_cost = np.where(toll_zone, rng.uniform(1.5, 14.0, n) * cost_index, 0.0)
    handling_cost = (weight_kg * 0.012 + items_count * 0.22 + cold_chain * 3.1) * cost_index
    delivery_cost = fuel_cost + labour_cost + toll_cost + handling_cost

    revenue = np.clip(
        delivery_cost * rng.uniform(1.05, 2.4, n)
        + np.array([ref.PRIORITY_SLA_FACTOR[p] for p in priority]) ** -1 * 2.5,
        0.5, None)

    orders = pd.DataFrame({
        "order_id": make_ids("ORD", n, 9),
        "route_id": np.char.add("RTE-", np.char.zfill(route_code.astype(str), 9)),
        "stop_sequence": stop_sequence,
        "customer_id": customer_id,
        "warehouse_id": warehouse_id,
        "pickup_location_id": pickup_location_id,
        "zone_id": zone_id,
        "driver_id": np.where(delivered_mask | (status == "failed"), driver_id, None),
        "vehicle_id": np.where(delivered_mask | (status == "failed"), vehicle_id, None),
        "business_domain": domain,
        "priority": priority,
        "status": status,
        "order_timestamp": order_ts,
        "pickup_timestamp": pickup_ts,
        "promised_delivery_ts": promised_ts,
        "delivery_window_end": promised_ts,
        "actual_delivery_ts": actual_ts.to_numpy(),
        "sla_hours": sla_hours.round(2),
        "package_type": package_type,
        "package_weight_kg": weight_kg.round(3),
        "package_volume_m3": volume_m3.round(4),
        "items_count": items_count,
        "declared_value_usd": value_usd.round(2),
        "cold_chain_required": cold_chain,
        "fragile": fragile,
        "signature_required": rng.random(n) < 0.24,
        "payment_method": weighted_choice(rng, ref.PAYMENT_METHODS, [0.46, 0.14, 0.18, 0.13, 0.09], n),
        "origin_lat": orig_lat.round(6),
        "origin_lon": orig_lon.round(6),
        "dest_lat": dest_lat.round(6),
        "dest_lon": dest_lon.round(6),
        "city": city,
        "country": country,
        "region": region,
        "area_type": area_type,
        "straight_line_km": straight_km.round(3),
        "distance_km": distance_km.round(3),
        "planned_duration_min": planned_duration_min.round(2),
        "actual_duration_min": np.where(delivered_mask, actual_duration_min.round(2), np.nan),
        "travel_minutes": travel_minutes.round(2),
        "service_minutes": service_minutes.round(2),
        "handling_minutes": handling_minutes.round(2),
        "traffic_level": traffic_level,
        "congestion_index": congestion.round(4),
        "weather_condition": weather_condition,
        "temperature_c": temperature_c.round(2),
        "precipitation_mm": precipitation_mm.round(2),
        "weather_severity_index": weather_severity.round(4),
        "delivery_attempts": np.where(status == "failed", rng.integers(1, 4, n), 1),
        "fuel_cost_usd": fuel_cost.round(3),
        "labour_cost_usd": labour_cost.round(3),
        "toll_cost_usd": toll_cost.round(3),
        "handling_cost_usd": handling_cost.round(3),
        "delivery_cost_usd": delivery_cost.round(3),
        "revenue_usd": revenue.round(2),
        "delay_minutes": delay_minutes.round(2).to_numpy(),
        "is_late": np.where(delivered_mask, status == "delivered_late", False),
        "co2_kg": (distance_km * fuel_l_per_100 / 100.0 * cfg.get("operations.co2_kg_per_litre", 2.68)).round(3),
        "is_duplicate": False,
        "anomaly_flags": "",
    })
    return orders


def build_routes(
    cfg: Config, streams: RandomStreams, orders: pd.DataFrame, vehicles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group delivered orders into multi-stop routes.

    Returns ``(routes, orders)`` because orders gain a ``route_id`` and a
    ``stop_sequence`` in the process.
    """
    rng = streams.get("routes")

    # ``route_id`` and ``stop_sequence`` were assigned during order generation
    # (see ``_assign_routes``), so a route already is one vehicle, one driver,
    # one warehouse, one day. Here we only roll the stops up into route facts.
    work = orders.loc[
        orders["vehicle_id"].notna(),
        ["order_id", "route_id", "stop_sequence", "vehicle_id", "driver_id",
         "warehouse_id", "order_timestamp", "distance_km", "actual_duration_min",
         "planned_duration_min", "package_weight_kg", "package_volume_m3",
         "delivery_cost_usd", "is_late", "city", "region", "area_type"],
    ].copy()
    work["route_date"] = work["order_timestamp"].dt.floor("D")
    work = work.sort_values(["route_id", "stop_sequence"], kind="mergesort")

    agg = work.groupby("route_id", sort=False).agg(
        vehicle_id=("vehicle_id", "first"),
        driver_id=("driver_id", "first"),
        warehouse_id=("warehouse_id", "first"),
        route_date=("route_date", "first"),
        city=("city", "first"),
        region=("region", "first"),
        area_type=("area_type", "first"),
        stops=("order_id", "count"),
        planned_distance_km=("distance_km", "sum"),
        planned_duration_min=("planned_duration_min", "sum"),
        actual_duration_min=("actual_duration_min", "sum"),
        total_weight_kg=("package_weight_kg", "sum"),
        total_volume_m3=("package_volume_m3", "sum"),
        route_cost_usd=("delivery_cost_usd", "sum"),
        late_stops=("is_late", "sum"),
        start_ts=("order_timestamp", "min"),
        end_ts=("order_timestamp", "max"),
    ).reset_index()

    m = len(agg)
    # Multi-stop routes share legs, so actual km < naive sum of point-to-points.
    consolidation = np.clip(1.0 - 0.028 * (agg["stops"].to_numpy() - 1), 0.42, 1.0)
    agg["actual_distance_km"] = (agg["planned_distance_km"] * consolidation
                                 * rng.uniform(0.96, 1.22, m)).round(3)
    agg["planned_distance_km"] = agg["planned_distance_km"].round(3)

    cap_kg = vehicles.set_index("vehicle_id")["capacity_kg"].reindex(agg["vehicle_id"]).to_numpy()
    cap_m3 = vehicles.set_index("vehicle_id")["capacity_m3"].reindex(agg["vehicle_id"]).to_numpy()
    agg["capacity_utilisation_kg"] = (agg["total_weight_kg"] / np.maximum(cap_kg, 1e-6)).clip(0, 3).round(4)
    agg["capacity_utilisation_m3"] = (agg["total_volume_m3"] / np.maximum(cap_m3, 1e-6)).clip(0, 3).round(4)
    agg["overloaded"] = agg["capacity_utilisation_kg"] > 1.0
    agg["on_time_rate"] = (1 - agg["late_stops"] / agg["stops"]).round(4)
    agg["cost_per_stop_usd"] = (agg["route_cost_usd"] / agg["stops"]).round(3)
    agg["cost_per_km_usd"] = (agg["route_cost_usd"] / agg["actual_distance_km"].clip(lower=0.1)).round(3)
    agg["route_type"] = np.select(
        [agg["stops"] <= 3, agg["stops"] <= 10, agg["stops"] <= 20],
        ["direct", "short_multistop", "standard_multistop"], default="dense_multistop")
    agg["optimisation_engine"] = rng.choice(
        ["manual", "heuristic_v1", "or_tools", "ml_assisted"], m, p=[0.18, 0.34, 0.33, 0.15])
    agg["planned_vs_actual_km_gap"] = (agg["actual_distance_km"] - agg["planned_distance_km"]).round(3)
    agg["disrupted"] = (agg["on_time_rate"] < 0.7) | agg["overloaded"]
    agg["anomaly_flags"] = ""

    # Cancelled / in-transit orders never got a vehicle, so clear their route link.
    unresourced = orders["vehicle_id"].isna()
    orders.loc[unresourced, ["route_id", "stop_sequence"]] = np.nan
    return agg, orders
