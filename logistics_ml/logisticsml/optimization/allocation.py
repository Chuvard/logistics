"""Linear and mixed-integer programming models.

Four classical operations-research problems, all solved with OR-Tools' CP-SAT or
linear solver where available and with a documented greedy heuristic otherwise:

* **Driver scheduling** - set-cover style shift assignment under labour rules.
* **Fleet allocation** - assign vehicles to routes minimising cost, respecting
  capacity and cold-chain requirements (an assignment problem).
* **Warehouse allocation** - the capacitated facility location problem: which
  sites to open and which zones each should serve.
* **Inventory optimisation** - EOQ with safety stock and a reorder point, plus
  a newsvendor comparison for perishables.

Every solver reports its objective *and* the baseline it beat, so the value is
quantified rather than claimed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils import get_logger, optional_import

__all__ = [
    "OptimisationResult", "solve_driver_scheduling", "solve_fleet_allocation",
    "solve_warehouse_allocation", "optimise_inventory",
]

logger = get_logger()


@dataclass
class OptimisationResult:
    problem: str
    solver: str
    objective: float
    baseline: float | None = None
    status: str = "optimal"
    solve_seconds: float = 0.0
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def improvement_pct(self) -> float | None:
        if self.baseline in (None, 0):
            return None
        return round((self.baseline - self.objective) / abs(self.baseline) * 100, 2)

    def summary(self) -> dict:
        return {
            "problem": self.problem, "solver": self.solver,
            "objective": round(self.objective, 2),
            "baseline": round(self.baseline, 2) if self.baseline is not None else None,
            "improvement_pct": self.improvement_pct,
            "status": self.status,
            "solve_seconds": round(self.solve_seconds, 3),
        }


def _cp_sat():
    ortools = optional_import("ortools")
    if ortools is None:
        return None
    try:
        from ortools.sat.python import cp_model
        return cp_model
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Driver scheduling
# --------------------------------------------------------------------------- #
def solve_driver_scheduling(drivers: pd.DataFrame, cfg_opt: dict,
                            seed: int = 0) -> OptimisationResult:
    """Assign drivers to shifts at minimum cost.

    Constraints: every shift needs a minimum crew; no driver exceeds their
    contracted shift count; no driver works two consecutive shifts (a proxy for
    mandatory rest); cost is the driver's hourly rate times shift length.
    """
    rng = np.random.default_rng(seed)
    n_drivers = min(int(cfg_opt.get("n_drivers", 25)), len(drivers))
    n_shifts = int(cfg_opt.get("n_shifts", 21))
    max_per_driver = int(cfg_opt.get("max_shifts_per_driver", 5))
    min_crew = int(cfg_opt.get("min_drivers_per_shift", 2))

    pool = drivers.head(n_drivers).reset_index(drop=True)
    hourly = pool["hourly_cost_usd"].fillna(pool["hourly_cost_usd"].median()).to_numpy(float)
    shift_hours = 8.0
    cost = np.round(hourly * shift_hours, 2)

    # Availability - not everyone can work every shift.
    available = rng.random((n_drivers, n_shifts)) > 0.18

    t0 = time.perf_counter()
    cp_model = _cp_sat()

    if cp_model is not None:
        model = cp_model.CpModel()
        x = {(d, s): model.NewBoolVar(f"x_{d}_{s}")
             for d in range(n_drivers) for s in range(n_shifts)}

        for d in range(n_drivers):
            for s in range(n_shifts):
                if not available[d, s]:
                    model.Add(x[d, s] == 0)
            model.Add(sum(x[d, s] for s in range(n_shifts)) <= max_per_driver)
            # Rest rule: no back-to-back shifts.
            for s in range(n_shifts - 1):
                model.Add(x[d, s] + x[d, s + 1] <= 1)

        for s in range(n_shifts):
            model.Add(sum(x[d, s] for d in range(n_drivers)) >= min_crew)

        model.Minimize(sum(int(cost[d] * 100) * x[d, s]
                           for d in range(n_drivers) for s in range(n_shifts)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(cfg_opt.get("time_limit_seconds", 10))
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assign = [(d, s) for d in range(n_drivers) for s in range(n_shifts)
                      if solver.Value(x[d, s])]
            objective = sum(cost[d] for d, _ in assign)
            table = pd.DataFrame({
                "driver_id": [pool["driver_id"].iloc[d] for d, _ in assign],
                "shift_index": [s for _, s in assign],
                "cost_usd": [cost[d] for d, _ in assign]})
            per_shift = (table.groupby("shift_index").size()
                         .rename("drivers_assigned").reset_index())
            # Baseline: fill each shift with the first available drivers.
            baseline = _greedy_schedule_cost(available, cost, n_shifts, min_crew, max_per_driver)
            return OptimisationResult(
                "driver_scheduling", "ortools_cp_sat", float(objective), baseline,
                "optimal" if status == cp_model.OPTIMAL else "feasible",
                time.perf_counter() - t0,
                {"assignments": table, "coverage": per_shift},
                {"n_drivers": n_drivers, "n_shifts": n_shifts,
                 "total_assignments": len(assign)})

    # Greedy fallback: cheapest available driver first, respecting the rules.
    objective, table = _greedy_schedule(available, cost, pool, n_shifts, min_crew, max_per_driver)
    baseline = _greedy_schedule_cost(available, cost, n_shifts, min_crew, max_per_driver,
                                     cheapest_first=False)
    return OptimisationResult(
        "driver_scheduling", "greedy_heuristic", objective, baseline, "heuristic",
        time.perf_counter() - t0, {"assignments": table},
        {"n_drivers": n_drivers, "n_shifts": n_shifts})


def _greedy_schedule(available, cost, pool, n_shifts, min_crew, max_per_driver,
                     cheapest_first: bool = True):
    used = np.zeros(len(cost), dtype=int)
    last_shift = np.full(len(cost), -10)
    rows = []
    order_base = np.argsort(cost) if cheapest_first else np.arange(len(cost))
    for s in range(n_shifts):
        picked = 0
        for d in order_base:
            if picked >= min_crew:
                break
            if available[d, s] and used[d] < max_per_driver and last_shift[d] != s - 1:
                rows.append({"driver_id": pool["driver_id"].iloc[d],
                             "shift_index": s, "cost_usd": float(cost[d])})
                used[d] += 1
                last_shift[d] = s
                picked += 1
    table = pd.DataFrame(rows)
    return (float(table["cost_usd"].sum()) if not table.empty else 0.0), table


def _greedy_schedule_cost(available, cost, n_shifts, min_crew, max_per_driver,
                          cheapest_first: bool = False) -> float:
    used = np.zeros(len(cost), dtype=int)
    last_shift = np.full(len(cost), -10)
    total = 0.0
    order_base = np.argsort(cost) if cheapest_first else np.arange(len(cost))
    for s in range(n_shifts):
        picked = 0
        for d in order_base:
            if picked >= min_crew:
                break
            if available[d, s] and used[d] < max_per_driver and last_shift[d] != s - 1:
                total += float(cost[d])
                used[d] += 1
                last_shift[d] = s
                picked += 1
    return total


# --------------------------------------------------------------------------- #
# Fleet allocation
# --------------------------------------------------------------------------- #
def solve_fleet_allocation(routes: pd.DataFrame, vehicles: pd.DataFrame,
                           cfg_opt: dict, seed: int = 0,
                           cold_route_ids: set | None = None) -> OptimisationResult:
    """Assign vehicles to routes at minimum operating cost.

    A vehicle can only take a route if it has the capacity, and a cold-chain
    route requires a refrigerated vehicle. Cost combines fuel consumption over
    the route distance with a fixed dispatch charge.

    Routes with no feasible vehicle are reported as *unassigned* rather than
    being forced onto an incompatible vehicle at a fake penalty cost - a
    dispatcher would subcontract them, and burying a 1e7 penalty in the
    objective would make the reported saving meaningless.
    """
    n_routes = min(int(cfg_opt.get("n_routes", 40)), len(routes))
    n_vehicles = min(int(cfg_opt.get("n_vehicles", 60)), len(vehicles))

    r = routes.nlargest(n_routes, "stops").reset_index(drop=True)
    # Sample the fleet rather than taking the first N: `head()` inherits the
    # generator's ordering and can hand back a fleet with almost no heavy or
    # refrigerated vehicles, creating a capacity shortage that is an artefact
    # of slicing rather than a real property of the business.
    v = (vehicles.sample(n_vehicles, random_state=seed) if len(vehicles) > n_vehicles
         else vehicles).reset_index(drop=True)

    demand = r["total_weight_kg"].fillna(0).to_numpy(float)
    distance = r["actual_distance_km"].fillna(r["planned_distance_km"]).fillna(50).to_numpy(float)
    capacity = v["capacity_kg"].fillna(1000).to_numpy(float)
    consumption = v["avg_consumption_l_per_100km"].fillna(10).to_numpy(float)
    refrigerated = v["refrigeration_unit"].fillna(False).to_numpy(bool)

    # Cold-chain requirement comes from the actual orders on each route when the
    # caller supplies it; otherwise nothing is flagged rather than guessed.
    if cold_route_ids:
        cold_needed = r["route_id"].isin(cold_route_ids).to_numpy()
    else:
        cold_needed = np.zeros(n_routes, dtype=bool)

    # Energy cost per 100 km. The generator records litres/100km, which is zero
    # for electric and pedal vehicles - charging an EV nothing per kilometre
    # would make it infinitely preferable and collapse the allocation onto the
    # handful of EVs in the fleet. Electric vehicles are costed on electricity
    # instead, and bikes on the rider alone.
    fuel_price = float(cfg_opt.get("fuel_price_per_litre", 1.65))
    kwh_price = float(cfg_opt.get("electricity_price_per_kwh", 0.29))
    kwh_per_100km = float(cfg_opt.get("ev_kwh_per_100km", 22.0))

    is_electric = (v["fuel_type"].fillna("") == "electric").to_numpy()
    energy_per_100km = consumption * fuel_price
    energy_per_100km = np.where(is_electric, kwh_per_100km * kwh_price, energy_per_100km)

    dispatch_fee = float(cfg_opt.get("dispatch_fee_usd", 25.0))
    cost = (distance[:, None] * energy_per_100km[None, :] / 100.0) + dispatch_fee
    feasible = (capacity[None, :] >= demand[:, None])
    feasible &= ~(cold_needed[:, None] & ~refrigerated[None, :])

    servable = feasible.any(axis=1)
    n_unservable = int((~servable).sum())
    if n_unservable:
        logger.warning("Fleet allocation: %d route(s) have no compatible vehicle "
                       "and are reported unassigned", n_unservable)

    BIG = 1e7
    cost_matrix = np.where(feasible, cost, BIG)

    # Subcontracting is priced *per route*, at a premium over what that specific
    # job would cost in-house - which is how spot rates actually work. A flat
    # price would make every long route cheaper to subcontract and every short
    # one cheaper to keep, which says nothing about fleet allocation. Pricing it
    # this way means the fleet is used wherever possible and subcontracting only
    # appears when capacity genuinely runs out.
    premium = float(cfg_opt.get("subcontract_premium", 1.5))
    flat = cfg_opt.get("subcontract_cost_usd")
    if flat:
        subcontract = np.full(n_routes, float(flat))
    else:
        cheapest = np.where(feasible, cost, np.inf).min(axis=1)
        fallback = float(np.median(cost)) * premium
        subcontract = np.where(np.isfinite(cheapest), cheapest * premium, fallback)

    t0 = time.perf_counter()
    cp_model = _cp_sat()

    if cp_model is not None:
        model = cp_model.CpModel()
        x = {(i, j): model.NewBoolVar(f"x_{i}_{j}")
             for i in range(n_routes) for j in range(n_vehicles)}
        # Leaving a route unassigned is allowed but expensive - it represents
        # subcontracting. Forcing every route onto an owned vehicle can be
        # genuinely infeasible (not enough capacity in the fleet), and a hard
        # constraint would make the whole model return nothing at all.
        unassigned = [model.NewBoolVar(f"u_{i}") for i in range(n_routes)]

        for i in range(n_routes):
            model.Add(sum(x[i, j] for j in range(n_vehicles)) + unassigned[i] == 1)
            for j in range(n_vehicles):
                if not feasible[i, j]:
                    model.Add(x[i, j] == 0)
        for j in range(n_vehicles):
            model.Add(sum(x[i, j] for i in range(n_routes)) <= 1)   # one route per vehicle per day

        model.Minimize(
            sum(int(cost[i, j] * 100) * x[i, j]
                for i in range(n_routes) for j in range(n_vehicles) if feasible[i, j])
            + sum(int(subcontract[i] * 100) * unassigned[i] for i in range(n_routes)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(cfg_opt.get("time_limit_seconds", 10))
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            pairs = [(i, j) for i in range(n_routes) for j in range(n_vehicles)
                     if solver.Value(x[i, j])]
            sub_idx = [i for i, u in enumerate(unassigned) if solver.Value(u)]
            n_sub = len(sub_idx)
            objective = float(sum(cost[i, j] for i, j in pairs)
                              + sum(subcontract[i] for i in sub_idx))
            table = pd.DataFrame({
                "route_id": [r["route_id"].iloc[i] for i, _ in pairs],
                "vehicle_id": [v["vehicle_id"].iloc[j] for _, j in pairs],
                "route_distance_km": [round(distance[i], 2) for i, _ in pairs],
                "load_kg": [round(demand[i], 1) for i, _ in pairs],
                "vehicle_capacity_kg": [round(capacity[j], 1) for _, j in pairs],
                "cost_usd": [round(cost_matrix[i, j], 2) for i, j in pairs]})
            baseline = _greedy_assignment_cost(cost, feasible, subcontract)
            return OptimisationResult(
                "fleet_allocation", "ortools_cp_sat", objective, baseline,
                "optimal" if status == cp_model.OPTIMAL else "feasible",
                time.perf_counter() - t0, {"assignments": table},
                {"n_routes": n_routes, "n_vehicles": n_vehicles,
                 "n_assigned": len(pairs), "n_subcontracted": n_sub,
                 "n_infeasible_routes": n_unservable,
                 "utilisation_pct": round(float(np.mean(
                     [demand[i] / capacity[j] * 100 for i, j in pairs])), 2) if pairs else 0.0})

    # Hungarian algorithm via scipy is exact for the pure assignment case.
    from scipy.optimize import linear_sum_assignment
    rows_i, cols_j = linear_sum_assignment(cost_matrix)
    # Discard pairs the solver only picked because it had to fill the matrix.
    keep = [(i, j) for i, j in zip(rows_i, cols_j) if feasible[i, j]]
    served = {i for i, _ in keep}
    n_sub = n_routes - len(keep)
    objective = float(sum(cost[i, j] for i, j in keep)
                      + sum(subcontract[i] for i in range(n_routes) if i not in served))
    table = pd.DataFrame({
        "route_id": [r["route_id"].iloc[i] for i, _ in keep],
        "vehicle_id": [v["vehicle_id"].iloc[j] for _, j in keep],
        "cost_usd": [round(cost[i, j], 2) for i, j in keep]})
    baseline = _greedy_assignment_cost(cost, feasible, subcontract)
    return OptimisationResult(
        "fleet_allocation", "scipy_hungarian", objective, baseline, "optimal",
        time.perf_counter() - t0, {"assignments": table},
        {"n_routes": n_routes, "n_vehicles": n_vehicles,
         "n_assigned": len(keep), "n_subcontracted": n_sub})


def _greedy_assignment_cost(cost: np.ndarray, feasible: np.ndarray,
                            subcontract: np.ndarray) -> float:
    """First-come-first-served assignment - what a dispatcher does by hand.

    Unserved routes are charged the same subcontracting rate the optimiser pays,
    so the two objectives are directly comparable.
    """
    taken: set[int] = set()
    total = 0.0
    for i in range(cost.shape[0]):
        options = [j for j in range(cost.shape[1]) if j not in taken and feasible[i, j]]
        if not options:
            total += float(subcontract[i])
            continue
        j = options[0]
        taken.add(j)
        total += float(cost[i, j])
    return total


# --------------------------------------------------------------------------- #
# Warehouse allocation (capacitated facility location)
# --------------------------------------------------------------------------- #
def solve_warehouse_allocation(warehouses: pd.DataFrame, zones: pd.DataFrame,
                               cfg_opt: dict, seed: int = 0) -> OptimisationResult:
    """Which sites to open, and which zones each should serve.

    Objective: fixed opening cost of chosen sites plus distance-proportional
    service cost for every zone assignment, subject to a cap on how many sites
    may be open and on each site's throughput.
    """
    n_sites = min(int(cfg_opt.get("n_candidate_sites", 12)), len(warehouses))
    n_zones = min(int(cfg_opt.get("n_zones", 60)), len(zones))
    max_open = int(cfg_opt.get("max_open_sites", 5))

    w = warehouses.head(n_sites).reset_index(drop=True)
    z = zones.head(n_zones).reset_index(drop=True)

    from .routing import EARTH_RADIUS_KM
    lat_w, lon_w = w["latitude"].to_numpy(float), w["longitude"].to_numpy(float)
    lat_z, lon_z = z["centroid_lat"].to_numpy(float), z["centroid_lon"].to_numpy(float)
    p, q = np.radians(lat_z)[:, None], np.radians(lat_w)[None, :]
    dlam = np.radians(lon_w)[None, :] - np.radians(lon_z)[:, None]
    a = np.sin((q - p) / 2) ** 2 + np.cos(p) * np.cos(q) * np.sin(dlam / 2) ** 2
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    demand = (z["population_density_per_km2"].fillna(500).to_numpy(float)
              * z["area_km2"].fillna(5).to_numpy(float) / 1000.0)
    service_cost = dist * demand[:, None] * 0.02
    fixed_cost = w["monthly_fixed_cost_usd"].fillna(50000).to_numpy(float) / 1000.0
    site_capacity = w["throughput_capacity_orders_day"].fillna(500).to_numpy(float)

    t0 = time.perf_counter()
    cp_model = _cp_sat()

    if cp_model is not None:
        model = cp_model.CpModel()
        open_var = [model.NewBoolVar(f"open_{j}") for j in range(n_sites)]
        assign = {(i, j): model.NewBoolVar(f"a_{i}_{j}")
                  for i in range(n_zones) for j in range(n_sites)}

        for i in range(n_zones):
            model.Add(sum(assign[i, j] for j in range(n_sites)) == 1)
            for j in range(n_sites):
                model.Add(assign[i, j] <= open_var[j])     # can't serve from a closed site
        model.Add(sum(open_var) <= max_open)
        for j in range(n_sites):
            model.Add(sum(int(demand[i]) * assign[i, j] for i in range(n_zones))
                      <= int(site_capacity[j]))

        model.Minimize(
            sum(int(fixed_cost[j] * 100) * open_var[j] for j in range(n_sites))
            + sum(int(service_cost[i, j] * 100) * assign[i, j]
                  for i in range(n_zones) for j in range(n_sites)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(cfg_opt.get("time_limit_seconds", 15))
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            opened = [j for j in range(n_sites) if solver.Value(open_var[j])]
            pairs = [(i, j) for i in range(n_zones) for j in range(n_sites)
                     if solver.Value(assign[i, j])]
            objective = (sum(fixed_cost[j] for j in opened)
                         + sum(service_cost[i, j] for i, j in pairs))
            site_tbl = pd.DataFrame({
                "warehouse_id": [w["warehouse_id"].iloc[j] for j in opened],
                "city": [w["city"].iloc[j] for j in opened],
                "fixed_cost": [round(fixed_cost[j], 2) for j in opened],
                "zones_served": [sum(1 for _, jj in pairs if jj == j) for j in opened],
                "demand_served": [round(sum(demand[i] for i, jj in pairs if jj == j), 1)
                                  for j in opened]})
            baseline = _greedy_site_cost(service_cost, fixed_cost, max_open)
            return OptimisationResult(
                "warehouse_allocation", "ortools_cp_sat", float(objective), baseline,
                "optimal" if status == cp_model.OPTIMAL else "feasible",
                time.perf_counter() - t0, {"opened_sites": site_tbl},
                {"n_sites_opened": len(opened), "n_candidates": n_sites, "n_zones": n_zones,
                 "avg_distance_km": round(float(np.mean([dist[i, j] for i, j in pairs])), 2)})

    # Greedy fallback: open the `max_open` cheapest-to-serve sites.
    totals = service_cost.sum(axis=0) + fixed_cost
    opened = list(np.argsort(totals)[:max_open])
    nearest = np.array([min(opened, key=lambda j: service_cost[i, j]) for i in range(n_zones)])
    objective = float(sum(fixed_cost[j] for j in opened)
                      + sum(service_cost[i, nearest[i]] for i in range(n_zones)))
    site_tbl = pd.DataFrame({
        "warehouse_id": [w["warehouse_id"].iloc[j] for j in opened],
        "city": [w["city"].iloc[j] for j in opened],
        "zones_served": [int((nearest == j).sum()) for j in opened]})
    return OptimisationResult(
        "warehouse_allocation", "greedy_heuristic", objective,
        _greedy_site_cost(service_cost, fixed_cost, max_open), "heuristic",
        time.perf_counter() - t0, {"opened_sites": site_tbl},
        {"n_sites_opened": len(opened)})


def _greedy_site_cost(service_cost: np.ndarray, fixed_cost: np.ndarray,
                      max_open: int) -> float:
    """Baseline: classic greedy facility location.

    Repeatedly open whichever site reduces total cost the most, until `max_open`
    sites are open. This is the standard heuristic (and a strong one - greedy is
    within a constant factor of optimal for uncapacitated facility location), so
    the improvement the MIP reports is measured against a real opponent rather
    than a straw man. It also opens the same number of sites as the optimiser,
    making the two objectives directly comparable.
    """
    n_zones, n_sites = service_cost.shape
    opened: list[int] = []
    best_per_zone = np.full(n_zones, np.inf)
    total = 0.0

    for _ in range(min(max_open, n_sites)):
        best_site, best_total = None, total
        for j in range(n_sites):
            if j in opened:
                continue
            candidate = fixed_cost[list(opened) + [j]].sum() + \
                np.minimum(best_per_zone, service_cost[:, j]).sum()
            if best_site is None or candidate < best_total:
                best_site, best_total = j, candidate
        if best_site is None:
            break
        opened.append(best_site)
        best_per_zone = np.minimum(best_per_zone, service_cost[:, best_site])
        total = best_total

    return float(total)


# --------------------------------------------------------------------------- #
# Inventory optimisation
# --------------------------------------------------------------------------- #
def optimise_inventory(inventory: pd.DataFrame, cfg_opt: dict) -> OptimisationResult:
    """EOQ + safety stock + reorder point, per warehouse-SKU.

    EOQ           = sqrt(2 * D * S / H)
    safety stock  = z * sigma_demand * sqrt(lead_time)
    reorder point = mean daily demand * lead time + safety stock

    The baseline is the generator's existing reorder points, so the reported
    saving is against what the simulated business is already doing.
    """
    t0 = time.perf_counter()
    holding = float(cfg_opt.get("holding_cost_per_unit_year", 2.4))
    order_cost = float(cfg_opt.get("order_cost", 65.0))
    service_level = float(cfg_opt.get("service_level", 0.95))
    lead_time = float(cfg_opt.get("lead_time_days", 5))
    z = float(norm.ppf(service_level))

    inv = inventory.copy()
    inv = inv.dropna(subset=["units_on_hand"])

    # Estimate demand from the turnover rate the generator records.
    daily_demand = (inv["turnover_rate"].fillna(1.0) * inv["units_on_hand"].fillna(0).clip(lower=1)
                    / 30.0).clip(lower=0.1)
    demand_sd = daily_demand * 0.35          # assumed CV; tune per SKU in production

    grouped = pd.DataFrame({
        "warehouse_id": inv["warehouse_id"],
        "sku_id": inv["sku_id"],
        "sku_category": inv.get("sku_category", "unknown"),
        "daily_demand": daily_demand,
        "demand_sd": demand_sd,
        "unit_cost_usd": inv["unit_cost_usd"].fillna(10.0),
        "current_reorder_point": inv["reorder_point"].fillna(0),
        "current_safety_stock": inv.get("safety_stock", pd.Series(0, index=inv.index)).fillna(0),
    }).groupby(["warehouse_id", "sku_id"], as_index=False).agg(
        sku_category=("sku_category", "first"),
        daily_demand=("daily_demand", "mean"),
        demand_sd=("demand_sd", "mean"),
        unit_cost_usd=("unit_cost_usd", "mean"),
        current_reorder_point=("current_reorder_point", "mean"),
        current_safety_stock=("current_safety_stock", "mean"))

    annual_demand = grouped["daily_demand"] * 365
    holding_per_unit = holding + grouped["unit_cost_usd"] * 0.18   # carrying = 18% of value
    grouped["eoq"] = np.sqrt(2 * annual_demand * order_cost / holding_per_unit).round(1)
    grouped["safety_stock"] = (z * grouped["demand_sd"] * math.sqrt(lead_time)).round(1)
    grouped["reorder_point"] = (grouped["daily_demand"] * lead_time
                                + grouped["safety_stock"]).round(1)
    grouped["orders_per_year"] = (annual_demand / grouped["eoq"].clip(lower=1)).round(2)
    grouped["annual_cost_usd"] = (
        grouped["orders_per_year"] * order_cost
        + (grouped["eoq"] / 2 + grouped["safety_stock"]) * holding_per_unit).round(2)

    # Baseline: current reorder points with an order quantity of one reorder point.
    current_q = grouped["current_reorder_point"].clip(lower=1)
    baseline_cost = ((annual_demand / current_q) * order_cost
                     + (current_q / 2 + grouped["current_safety_stock"]) * holding_per_unit)

    objective = float(grouped["annual_cost_usd"].sum())
    baseline = float(baseline_cost.sum())

    by_category = (grouped.groupby("sku_category")
                   .agg(skus=("sku_id", "count"),
                        avg_eoq=("eoq", "mean"),
                        avg_reorder_point=("reorder_point", "mean"),
                        annual_cost_usd=("annual_cost_usd", "sum"))
                   .round(2).reset_index())

    return OptimisationResult(
        "inventory_optimisation", "eoq_analytic", objective, baseline, "closed_form",
        time.perf_counter() - t0,
        {"policy": grouped.head(200).round(2), "by_category": by_category},
        {"service_level": service_level, "lead_time_days": lead_time,
         "n_sku_locations": int(len(grouped)), "z_score": round(z, 3)})
