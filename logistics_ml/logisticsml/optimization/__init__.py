"""Stage 6 - mathematical optimization: routing, scheduling, allocation, inventory."""

from .allocation import (
    OptimisationResult, optimise_inventory, solve_driver_scheduling,
    solve_fleet_allocation, solve_warehouse_allocation,
)
from .metaheuristics import genetic_algorithm, particle_swarm, simulated_annealing
from .routing import build_instance, solve_cvrp, solve_tsp, solve_vrptw
from .runner import OptimisationResults, run_optimization

__all__ = [
    "run_optimization", "OptimisationResults", "OptimisationResult",
    "build_instance", "solve_tsp", "solve_cvrp", "solve_vrptw",
    "genetic_algorithm", "simulated_annealing", "particle_swarm",
    "solve_driver_scheduling", "solve_fleet_allocation",
    "solve_warehouse_allocation", "optimise_inventory",
]
