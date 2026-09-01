"""Vehicle routing: TSP, CVRP and VRP with time windows.

Solved with Google OR-Tools when available. A greedy nearest-neighbour +
2-opt local search is provided as a fallback so the stage always produces a
route and a comparable cost - and because it doubles as the baseline that makes
the OR-Tools improvement measurable rather than asserted.

Instances are built from real generated orders (coordinates, weights, promised
windows) rather than random points, so the distances and loads are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import get_logger, optional_import

__all__ = ["RoutingInstance", "RoutingSolution", "build_instance", "solve_tsp",
           "solve_cvrp", "solve_vrptw", "nearest_neighbour_route", "two_opt"]

logger = get_logger()

EARTH_RADIUS_KM = 6371.0088


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    p = np.radians(lat)[:, None]
    q = np.radians(lat)[None, :]
    dphi = q - p
    dlam = np.radians(lon)[None, :] - np.radians(lon)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p) * np.cos(q) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class RoutingInstance:
    """A concrete routing problem: a depot, stops, demands and time windows."""

    coords: np.ndarray                       # (n, 2) lat/lon, index 0 = depot
    distance_km: np.ndarray                  # (n, n)
    demands_kg: np.ndarray                   # (n,), depot = 0
    time_windows: np.ndarray | None = None   # (n, 2) minutes from horizon start
    service_minutes: np.ndarray | None = None
    n_vehicles: int = 5
    capacity_kg: float = 1200.0
    depot: int = 0
    labels: list[str] = field(default_factory=list)
    avg_speed_kmh: float = 35.0

    @property
    def n_stops(self) -> int:
        return len(self.coords)

    def travel_minutes(self) -> np.ndarray:
        return self.distance_km / max(self.avg_speed_kmh, 1e-6) * 60.0


@dataclass
class RoutingSolution:
    method: str
    routes: list[list[int]]
    total_distance_km: float
    feasible: bool = True
    solve_seconds: float = 0.0
    n_vehicles_used: int = 0
    load_per_route: list[float] = field(default_factory=list)
    max_lateness_min: float = 0.0
    notes: str = ""

    def summary(self) -> dict:
        return {
            "method": self.method,
            "total_distance_km": round(self.total_distance_km, 3),
            "n_vehicles_used": self.n_vehicles_used,
            "feasible": self.feasible,
            "solve_seconds": round(self.solve_seconds, 3),
            "max_load_kg": round(max(self.load_per_route), 1) if self.load_per_route else 0.0,
            "max_lateness_min": round(self.max_lateness_min, 1),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Instance construction
# --------------------------------------------------------------------------- #
def build_instance(orders: pd.DataFrame, cfg_opt: dict, seed: int = 0) -> RoutingInstance:
    """Carve a routing instance out of the real order book.

    Picks the busiest warehouse-day so the stops genuinely cluster - a random
    sample of orders across a continent makes for a meaningless VRP.
    """
    n_stops = int(cfg_opt.get("n_stops", 40))
    rng = np.random.default_rng(seed)

    df = orders.dropna(subset=["dest_lat", "dest_lon", "origin_lat", "origin_lon"]).copy()
    df["date"] = pd.to_datetime(df["order_timestamp"]).dt.normalize()
    busiest = (df.groupby(["warehouse_id", "date"]).size()
               .sort_values(ascending=False).index[0])
    pool = df.loc[(df["warehouse_id"] == busiest[0]) & (df["date"] == busiest[1])]

    # Top up from the same warehouse on other days if one day is too small.
    if len(pool) < n_stops:
        extra = df.loc[(df["warehouse_id"] == busiest[0]) & (df["date"] != busiest[1])]
        pool = pd.concat([pool, extra.head(n_stops - len(pool))])
    if len(pool) > n_stops:
        pool = pool.iloc[rng.choice(len(pool), n_stops, replace=False)]

    depot = np.array([[pool["origin_lat"].iloc[0], pool["origin_lon"].iloc[0]]])
    stops = pool[["dest_lat", "dest_lon"]].to_numpy(dtype=float)
    coords = np.vstack([depot, stops])

    demands = np.concatenate([[0.0], pool["package_weight_kg"].fillna(1.0).to_numpy(dtype=float)])
    service = np.concatenate([[0.0], np.full(len(pool), 8.0)])

    # Time windows from the promised delivery timestamp, relative to the
    # earliest order that day, with a two-hour window around the promise.
    tw = None
    if "promised_delivery_ts" in pool.columns:
        promised = pd.to_datetime(pool["promised_delivery_ts"])
        base = pd.to_datetime(pool["order_timestamp"]).min()
        centre = ((promised - base).dt.total_seconds() / 60).to_numpy()
        centre = np.clip(centre, 60, 60 * 24)
        tw = np.vstack([[0, 60 * 24],
                        np.column_stack([np.maximum(centre - 120, 0), centre + 120])])

    return RoutingInstance(
        coords=coords,
        distance_km=_haversine_matrix(coords[:, 0], coords[:, 1]),
        demands_kg=demands,
        time_windows=tw,
        service_minutes=service,
        n_vehicles=int(cfg_opt.get("n_vehicles", 5)),
        capacity_kg=float(cfg_opt.get("vehicle_capacity_kg", 1200)),
        labels=["depot"] + [str(x) for x in pool["order_id"].tolist()],
    )


def route_distance(route: list[int], D: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))


# --------------------------------------------------------------------------- #
# Heuristic baseline
# --------------------------------------------------------------------------- #
def nearest_neighbour_route(D: np.ndarray, start: int = 0,
                            nodes: list[int] | None = None) -> list[int]:
    nodes = list(nodes) if nodes is not None else list(range(len(D)))
    unvisited = set(nodes) - {start}
    route = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(start)
    return route


def two_opt(route: list[int], D: np.ndarray, max_passes: int = 40) -> list[int]:
    """Classic 2-opt: repeatedly reverse a segment when it shortens the tour."""
    best = route[:]
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[j], best[j + 1]
                if D[a, b] + D[c, d] > D[a, c] + D[b, d] + 1e-12:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best


def _greedy_cvrp(inst: RoutingInstance) -> RoutingSolution:
    """Capacity-aware sweep: fill a vehicle by nearest neighbour until the next
    stop would breach capacity, then start a new route."""
    D = inst.distance_km
    remaining = set(range(1, inst.n_stops))
    routes, loads = [], []

    while remaining and len(routes) < inst.n_vehicles:
        route, load, current = [inst.depot], 0.0, inst.depot
        while True:
            feasible = [j for j in remaining if load + inst.demands_kg[j] <= inst.capacity_kg]
            if not feasible:
                break
            nxt = min(feasible, key=lambda j: D[current, j])
            route.append(nxt)
            load += inst.demands_kg[nxt]
            remaining.discard(nxt)
            current = nxt
        route.append(inst.depot)
        if len(route) > 2:
            routes.append(two_opt(route, D))
            loads.append(load)
        else:
            break

    total = sum(route_distance(r, D) for r in routes)
    return RoutingSolution(
        method="greedy_nn_2opt", routes=routes, total_distance_km=total,
        feasible=not remaining, n_vehicles_used=len(routes), load_per_route=loads,
        notes="fallback heuristic" + ("" if not remaining else
                                      f"; {len(remaining)} stops unserved"))


# --------------------------------------------------------------------------- #
# OR-Tools solvers
# --------------------------------------------------------------------------- #
def _ortools_solve(inst: RoutingInstance, variant: str, time_limit: int) -> RoutingSolution | None:
    ortools = optional_import("ortools")
    if ortools is None:
        return None
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None

    import time as _time
    t0 = _time.perf_counter()

    n_vehicles = 1 if variant == "tsp" else inst.n_vehicles
    manager = pywrapcp.RoutingIndexManager(inst.n_stops, n_vehicles, inst.depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distances are scaled to integers - OR-Tools works in integer arithmetic.
    scaled = np.round(inst.distance_km * 1000).astype(np.int64)

    def distance_cb(from_index, to_index):
        return int(scaled[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    if variant in {"cvrp", "vrptw"}:
        demands = np.round(inst.demands_kg).astype(np.int64)

        def demand_cb(from_index):
            return int(demands[manager.IndexToNode(from_index)])

        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitCallback(demand_cb), 0,
            [int(inst.capacity_kg)] * n_vehicles, True, "Capacity")

    if variant == "vrptw" and inst.time_windows is not None:
        travel = inst.travel_minutes()
        service = inst.service_minutes if inst.service_minutes is not None \
            else np.zeros(inst.n_stops)

        def time_cb(from_index, to_index):
            i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            return int(round(travel[i, j] + service[i]))

        routing.AddDimension(routing.RegisterTransitCallback(time_cb),
                             int(60 * 8), int(60 * 26), False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for node in range(inst.n_stops):
            lo, hi = inst.time_windows[node]
            time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(lo), int(hi))

        # Dropping a stop is allowed at a stiff penalty - without this an
        # infeasible window makes the whole model return no solution at all.
        for node in range(1, inst.n_stops):
            routing.AddDisjunction([manager.NodeToIndex(node)], 5_000_000)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    routes, loads, dropped = [], [], 0
    for v in range(n_vehicles):
        index = routing.Start(v)
        route, load = [], 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            load += inst.demands_kg[node]
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        if len(route) > 2:
            routes.append(route)
            loads.append(load)

    served = {n for r in routes for n in r}
    dropped = inst.n_stops - len(served)

    total = sum(route_distance(r, inst.distance_km) for r in routes)
    return RoutingSolution(
        method=f"ortools_{variant}", routes=routes, total_distance_km=total,
        feasible=dropped == 0, solve_seconds=_time.perf_counter() - t0,
        n_vehicles_used=len(routes), load_per_route=loads,
        notes="" if dropped == 0 else f"{dropped} stop(s) dropped as infeasible")


def solve_tsp(inst: RoutingInstance, time_limit: int = 10) -> RoutingSolution:
    sol = _ortools_solve(inst, "tsp", time_limit)
    if sol is not None:
        return sol
    route = two_opt(nearest_neighbour_route(inst.distance_km), inst.distance_km)
    return RoutingSolution("greedy_nn_2opt_tsp", [route],
                           route_distance(route, inst.distance_km),
                           n_vehicles_used=1, notes="OR-Tools unavailable")


def solve_cvrp(inst: RoutingInstance, time_limit: int = 12) -> RoutingSolution:
    sol = _ortools_solve(inst, "cvrp", time_limit)
    return sol if sol is not None else _greedy_cvrp(inst)


def solve_vrptw(inst: RoutingInstance, time_limit: int = 15) -> RoutingSolution:
    sol = _ortools_solve(inst, "vrptw", time_limit)
    if sol is not None:
        return sol
    fallback = _greedy_cvrp(inst)
    fallback.method = "greedy_nn_2opt_vrptw"
    fallback.notes = "OR-Tools unavailable; time windows not enforced"
    return fallback
