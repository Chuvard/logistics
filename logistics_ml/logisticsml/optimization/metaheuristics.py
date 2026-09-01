"""Genetic algorithm, simulated annealing and particle swarm optimisation.

All three are implemented from scratch (no black-box library) so the mechanics
are visible and auditable, and all three are benchmarked on the *same* routing
instance as OR-Tools. That makes the comparison honest: metaheuristics are
usually beaten by a dedicated solver on a well-posed VRP, and the report should
show that rather than hide it.

PSO operates on a continuous space, so it uses random-key encoding: each stop
gets a real-valued key and the permutation is the argsort of the keys. This is
the standard way to apply a continuous optimiser to a combinatorial problem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .routing import RoutingInstance, RoutingSolution, nearest_neighbour_route, route_distance, two_opt

__all__ = ["genetic_algorithm", "simulated_annealing", "particle_swarm", "MetaResult"]


@dataclass
class MetaResult:
    method: str
    best_distance_km: float
    solve_seconds: float
    iterations: int
    history: list[float]
    route: list[int]

    # Cost of the shared greedy nearest-neighbour tour, set by the caller. All
    # three methods are scored against this same reference; using each method's
    # own starting point would make PSO (which starts from a random permutation)
    # look dramatically better than GA and SA, which start from the greedy tour.
    baseline_distance_km: float = 0.0

    def summary(self) -> dict:
        base = self.baseline_distance_km or (self.history[0] if self.history else 0.0)
        return {
            "method": self.method,
            "best_distance_km": round(self.best_distance_km, 3),
            "baseline_greedy_km": round(base, 3),
            "solve_seconds": round(self.solve_seconds, 2),
            "iterations": self.iterations,
            "vs_greedy_pct": round((base - self.best_distance_km) / max(base, 1e-9) * 100, 2),
        }


def _tour_length(order: np.ndarray, D: np.ndarray) -> float:
    """Closed tour: depot -> stops in order -> depot."""
    route = np.concatenate([[0], order, [0]])
    return float(D[route[:-1], route[1:]].sum())


# --------------------------------------------------------------------------- #
# Genetic algorithm
# --------------------------------------------------------------------------- #
def genetic_algorithm(inst: RoutingInstance, population: int = 120, generations: int = 220,
                      mutation_rate: float = 0.18, elite: int = 6,
                      seed: int = 0) -> MetaResult:
    """Order-crossover GA with elitism and swap mutation."""
    rng = np.random.default_rng(seed)
    D = inst.distance_km
    stops = np.arange(1, inst.n_stops)
    t0 = time.perf_counter()

    # Seed one individual with the greedy tour - a good gene pool converges faster.
    pop = [rng.permutation(stops) for _ in range(population - 1)]
    greedy = np.array(nearest_neighbour_route(D)[1:-1])
    pop.append(greedy)
    pop = np.array(pop)

    fitness = np.array([_tour_length(ind, D) for ind in pop])
    history = [float(fitness.min())]

    def _ox(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order crossover - preserves relative ordering, always valid."""
        n = len(p1)
        a, b = sorted(rng.choice(n, 2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[a:b] = p1[a:b]
        fill = [g for g in p2 if g not in set(p1[a:b])]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child

    for _ in range(generations):
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        new_pop = [pop[i] for i in range(min(elite, len(pop)))]

        while len(new_pop) < population:
            # Tournament selection - cheap and keeps diversity better than roulette.
            i, j = rng.choice(population, 2, replace=False)
            p1 = pop[i] if fitness[i] < fitness[j] else pop[j]
            i, j = rng.choice(population, 2, replace=False)
            p2 = pop[i] if fitness[i] < fitness[j] else pop[j]

            child = _ox(p1, p2)
            if rng.random() < mutation_rate:
                a, b = rng.choice(len(child), 2, replace=False)
                child[a], child[b] = child[b], child[a]
            new_pop.append(child)

        pop = np.array(new_pop)
        fitness = np.array([_tour_length(ind, D) for ind in pop])
        history.append(float(fitness.min()))

    best = pop[int(np.argmin(fitness))]
    route = two_opt([0, *best.tolist(), 0], D)     # polish the winner
    return MetaResult("genetic_algorithm", route_distance(route, D),
                      time.perf_counter() - t0, generations, history, route)


# --------------------------------------------------------------------------- #
# Simulated annealing
# --------------------------------------------------------------------------- #
def simulated_annealing(inst: RoutingInstance, iterations: int = 12000,
                        t_start: float = 120.0, t_end: float = 0.5,
                        seed: int = 0) -> MetaResult:
    """Geometric cooling with 2-opt segment-reversal moves."""
    rng = np.random.default_rng(seed)
    D = inst.distance_km
    t0 = time.perf_counter()

    current = np.array(nearest_neighbour_route(D)[1:-1])
    current_cost = _tour_length(current, D)
    best, best_cost = current.copy(), current_cost
    history = [current_cost]

    alpha = (t_end / t_start) ** (1.0 / max(iterations, 1))
    temp = t_start

    for step in range(iterations):
        i, j = sorted(rng.choice(len(current), 2, replace=False))
        if i == j:
            continue
        candidate = current.copy()
        candidate[i:j + 1] = candidate[i:j + 1][::-1]
        cost = _tour_length(candidate, D)
        delta = cost - current_cost

        # Accept improvements always; accept worse moves with Boltzmann probability.
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-9)):
            current, current_cost = candidate, cost
            if cost < best_cost:
                best, best_cost = candidate.copy(), cost
        temp *= alpha
        if step % 100 == 0:
            history.append(best_cost)

    route = two_opt([0, *best.tolist(), 0], D)
    return MetaResult("simulated_annealing", route_distance(route, D),
                      time.perf_counter() - t0, iterations, history, route)


# --------------------------------------------------------------------------- #
# Particle swarm
# --------------------------------------------------------------------------- #
def particle_swarm(inst: RoutingInstance, particles: int = 60, iterations: int = 300,
                   inertia: float = 0.72, c1: float = 1.5, c2: float = 1.5,
                   seed: int = 0) -> MetaResult:
    """Continuous PSO over random keys, decoded to permutations by argsort."""
    rng = np.random.default_rng(seed)
    D = inst.distance_km
    dim = inst.n_stops - 1
    t0 = time.perf_counter()

    pos = rng.random((particles, dim))
    vel = rng.uniform(-0.1, 0.1, (particles, dim))

    def decode(keys: np.ndarray) -> np.ndarray:
        return np.argsort(keys) + 1

    costs = np.array([_tour_length(decode(p), D) for p in pos])
    pbest, pbest_cost = pos.copy(), costs.copy()
    g = int(np.argmin(costs))
    gbest, gbest_cost = pos[g].copy(), float(costs[g])
    history = [gbest_cost]

    for _ in range(iterations):
        r1, r2 = rng.random((particles, dim)), rng.random((particles, dim))
        vel = inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -0.35, 0.35)
        pos = np.clip(pos + vel, 0.0, 1.0)

        costs = np.array([_tour_length(decode(p), D) for p in pos])
        better = costs < pbest_cost
        pbest[better], pbest_cost[better] = pos[better], costs[better]
        g = int(np.argmin(pbest_cost))
        if pbest_cost[g] < gbest_cost:
            gbest, gbest_cost = pbest[g].copy(), float(pbest_cost[g])
        history.append(gbest_cost)

    route = two_opt([0, *decode(gbest).tolist(), 0], D)
    return MetaResult("particle_swarm", route_distance(route, D),
                      time.perf_counter() - t0, iterations, history, route)
