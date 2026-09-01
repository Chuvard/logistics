"""Stage 6 orchestration - runs every optimisation problem and collects results.

The routing section deliberately runs OR-Tools *and* the three metaheuristics on
the identical instance, so the leaderboard shows which approach actually wins on
this problem rather than assuming the fanciest one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, new_figure, save_figure
from . import metaheuristics as mh
from .allocation import (
    optimise_inventory, solve_driver_scheduling, solve_fleet_allocation,
    solve_warehouse_allocation,
)
from .routing import build_instance, solve_cvrp, solve_tsp, solve_vrptw

__all__ = ["OptimisationResults", "run_optimization"]

logger = get_logger()


@dataclass
class OptimisationResults:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summaries: list[dict] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(self.summaries) if self.summaries else pd.DataFrame()


def run_optimization(tables: dict[str, pd.DataFrame], cfg: Config,
                     out_dir: Path) -> OptimisationResults:
    res = OptimisationResults()
    if not cfg.get("optimization.enabled", True):
        logger.info("Optimization stage disabled by config")
        return res

    opt = cfg.get("optimization", {}) or {}
    seed = cfg.seed
    plot_dir = out_dir / "plots" / "optimization"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("eda.dpi", 110))
    make_plots = bool(cfg.get("reports.include_plots", True))

    # ---------------------------------------------------------------- routing
    vrp_cfg = opt.get("vrp", {}) or {}
    instance = None
    if vrp_cfg.get("enabled", True) and "orders" in tables:
        try:
            instance = build_instance(tables["orders"], vrp_cfg, seed)
            logger.info("VRP instance: %d stops, %d vehicles, capacity %.0f kg",
                        instance.n_stops - 1, instance.n_vehicles, instance.capacity_kg)
            tl = int(vrp_cfg.get("time_limit_seconds", 12))
            rows = []
            solvers = {"tsp": solve_tsp, "cvrp": solve_cvrp, "vrptw": solve_vrptw}
            for variant in vrp_cfg.get("variants", ["tsp", "cvrp", "vrptw"]):
                fn = solvers.get(variant)
                if fn is None:
                    continue
                sol = fn(instance, tl)
                rows.append({"variant": variant, **sol.summary()})
                res.details[f"routes_{variant}"] = sol.routes
                logger.info("  %-6s %-22s %.2f km, %d vehicles",
                            variant, sol.method, sol.total_distance_km, sol.n_vehicles_used)
            res.tables["vrp_variants"] = pd.DataFrame(rows)
        except Exception as exc:
            res.skipped["vrp"] = f"{type(exc).__name__}: {exc}"
            logger.error("VRP failed: %s", exc)

    # -------------------------------------------------------- metaheuristics
    meta_cfg = opt.get("metaheuristics", {}) or {}
    if meta_cfg.get("enabled", True) and instance is not None:
        rows, histories = [], {}
        try:
            # Shared reference point for all three methods.
            from .routing import nearest_neighbour_route, route_distance
            greedy_km = route_distance(
                nearest_neighbour_route(instance.distance_km), instance.distance_km)

            ga_cfg = meta_cfg.get("genetic_algorithm", {}) or {}
            r = mh.genetic_algorithm(
                instance, int(ga_cfg.get("population", 120)),
                int(ga_cfg.get("generations", 220)),
                float(ga_cfg.get("mutation_rate", 0.18)),
                int(ga_cfg.get("elite", 6)), seed)
            r.baseline_distance_km = greedy_km
            rows.append(r.summary()); histories["genetic_algorithm"] = r.history

            sa_cfg = meta_cfg.get("simulated_annealing", {}) or {}
            r = mh.simulated_annealing(
                instance, int(sa_cfg.get("iterations", 12000)),
                float(sa_cfg.get("t_start", 120.0)), float(sa_cfg.get("t_end", 0.5)), seed)
            r.baseline_distance_km = greedy_km
            rows.append(r.summary()); histories["simulated_annealing"] = r.history

            pso_cfg = meta_cfg.get("particle_swarm", {}) or {}
            r = mh.particle_swarm(
                instance, int(pso_cfg.get("particles", 60)),
                int(pso_cfg.get("iterations", 300)),
                float(pso_cfg.get("inertia", 0.72)), float(pso_cfg.get("c1", 1.5)),
                float(pso_cfg.get("c2", 1.5)), seed)
            r.baseline_distance_km = greedy_km
            rows.append(r.summary()); histories["particle_swarm"] = r.history

            res.tables["metaheuristics"] = pd.DataFrame(rows)
            for name, row in zip(histories, rows):
                logger.info("  %-20s %.2f km (%.1fs)", name,
                            row["best_distance_km"], row["solve_seconds"])

            # Head-to-head against the exact TSP solve on the same instance.
            if "vrp_variants" in res.tables:
                tsp_row = res.tables["vrp_variants"].query("variant == 'tsp'")
                if not tsp_row.empty:
                    ref = float(tsp_row["total_distance_km"].iloc[0])
                    comp = pd.DataFrame(rows)[["method", "best_distance_km", "solve_seconds"]]
                    comp["vs_ortools_pct"] = ((comp["best_distance_km"] - ref) / ref * 100).round(2)
                    comp = pd.concat([
                        pd.DataFrame([{"method": "ortools_tsp", "best_distance_km": round(ref, 3),
                                       "solve_seconds": float(tsp_row["solve_seconds"].iloc[0]),
                                       "vs_ortools_pct": 0.0}]), comp], ignore_index=True)
                    res.tables["routing_comparison"] = comp.sort_values("best_distance_km")

            if make_plots and histories:
                fig, ax = new_figure(8, 4.5)
                for name, hist in histories.items():
                    x = np.linspace(0, 100, len(hist))
                    ax.plot(x, hist, label=name)
                ax.set_xlabel("search progress (%)")
                ax.set_ylabel("best tour length (km)")
                ax.set_title("Metaheuristic convergence")
                ax.legend(fontsize=8)
                res.plots.append(save_figure(fig, plot_dir / "metaheuristic_convergence.png", dpi))
        except Exception as exc:
            res.skipped["metaheuristics"] = f"{type(exc).__name__}: {exc}"
            logger.error("Metaheuristics failed: %s", exc)

    # ------------------------------------------------------ driver scheduling
    ds_cfg = opt.get("driver_scheduling", {}) or {}
    if ds_cfg.get("enabled", True) and "drivers" in tables:
        try:
            r = solve_driver_scheduling(tables["drivers"], ds_cfg, seed)
            res.summaries.append(r.summary())
            res.tables.update({f"driver_scheduling_{k}": v for k, v in r.tables.items()})
            logger.info("  driver_scheduling: $%.0f (%s, %s%% better than baseline)",
                        r.objective, r.solver, r.improvement_pct)
        except Exception as exc:
            res.skipped["driver_scheduling"] = f"{type(exc).__name__}: {exc}"
            logger.error("Driver scheduling failed: %s", exc)

    # -------------------------------------------------------- fleet allocation
    fa_cfg = opt.get("fleet_allocation", {}) or {}
    if fa_cfg.get("enabled", True) and {"routes", "vehicles"} <= tables.keys():
        try:
            # A route needs a refrigerated vehicle only when cold-chain freight
            # dominates it. A single chilled parcel among ten travels in a cold
            # box - requiring a reefer for every route with one chilled item
            # would make the fleet look artificially short of capacity.
            cold_ids = None
            orders = tables.get("orders")
            if orders is not None and {"route_id", "cold_chain_required"} <= set(orders.columns):
                share = orders.groupby("route_id")["cold_chain_required"].mean()
                threshold = float(fa_cfg.get("cold_chain_route_threshold", 0.5))
                cold_ids = set(share.loc[share >= threshold].index)
            r = solve_fleet_allocation(tables["routes"], tables["vehicles"], fa_cfg, seed,
                                       cold_route_ids=cold_ids)
            res.summaries.append(r.summary())
            res.tables.update({f"fleet_allocation_{k}": v for k, v in r.tables.items()})
            logger.info("  fleet_allocation: $%.0f (%s, %s%% better)",
                        r.objective, r.solver, r.improvement_pct)
        except Exception as exc:
            res.skipped["fleet_allocation"] = f"{type(exc).__name__}: {exc}"
            logger.error("Fleet allocation failed: %s", exc)

    # ---------------------------------------------------- warehouse allocation
    wa_cfg = opt.get("warehouse_allocation", {}) or {}
    if wa_cfg.get("enabled", True) and {"warehouses", "delivery_zones"} <= tables.keys():
        try:
            r = solve_warehouse_allocation(tables["warehouses"], tables["delivery_zones"],
                                           wa_cfg, seed)
            res.summaries.append(r.summary())
            res.tables.update({f"warehouse_allocation_{k}": v for k, v in r.tables.items()})
            logger.info("  warehouse_allocation: opened %s sites, objective %.0f",
                        r.details.get("n_sites_opened"), r.objective)
        except Exception as exc:
            res.skipped["warehouse_allocation"] = f"{type(exc).__name__}: {exc}"
            logger.error("Warehouse allocation failed: %s", exc)

    # ------------------------------------------------------------- inventory
    inv_cfg = opt.get("inventory", {}) or {}
    if inv_cfg.get("enabled", True) and "inventory" in tables:
        try:
            r = optimise_inventory(tables["inventory"], inv_cfg)
            res.summaries.append(r.summary())
            res.tables.update({f"inventory_{k}": v for k, v in r.tables.items()})
            logger.info("  inventory: $%.0f/yr vs $%.0f baseline (%s%% saving)",
                        r.objective, r.baseline, r.improvement_pct)
        except Exception as exc:
            res.skipped["inventory"] = f"{type(exc).__name__}: {exc}"
            logger.error("Inventory optimisation failed: %s", exc)

    # ------------------------------------------------------------- route plot
    if make_plots and instance is not None and "routes_cvrp" in res.details:
        try:
            fig, ax = new_figure(7, 6.5)
            coords = instance.coords
            colours = ["#4c7ef3", "#e0574c", "#3fa87a", "#e0a24c", "#7a5cf0", "#4cc7e0"]
            for k, route in enumerate(res.details["routes_cvrp"]):
                path = coords[route]
                ax.plot(path[:, 1], path[:, 0], "-o", ms=3.5, lw=1.2,
                        color=colours[k % len(colours)], label=f"vehicle {k+1}")
            ax.plot(coords[0, 1], coords[0, 0], "s", ms=11, color="#111", label="depot")
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_title("CVRP solution")
            ax.legend(fontsize=7)
            res.plots.append(save_figure(fig, plot_dir / "cvrp_routes.png", dpi))
        except Exception as exc:
            logger.debug("Route plot failed: %s", exc)

    return res
