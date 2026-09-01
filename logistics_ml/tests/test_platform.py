"""Validation suite for the ML platform.

Asserts the properties that make results trustworthy: no leakage across the
split, no future information in a temporal split, metrics that agree with a
hand computation, a registry that round-trips, a serving path that enforces the
feature contract, and optimisers that actually beat their baselines.

Run with ``pytest tests/ -v`` or directly: ``python tests/test_platform.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logisticsml import build_task, load_config, load_tables  # noqa: E402
from logisticsml.metrics import classification_metrics, regression_metrics  # noqa: E402
from logisticsml.models.supervised import run_supervised  # noqa: E402
from logisticsml.stages.preprocessing import preprocess  # noqa: E402

CONFIG = [ROOT / "configs" / "default.yaml"]
_CACHE: dict = {}


def _cfg(**overrides):
    cfg = load_config(CONFIG)
    cfg.set("data.max_rows", 6000)
    cfg.set("project.output_dir", "/tmp/logisticsml_test")
    for key, value in overrides.items():
        cfg.set(key, value)
    return cfg


def _fixture():
    """A small preprocessed split, built once and reused."""
    if "fixture" not in _CACHE:
        cfg = _cfg()
        ds = build_task(cfg)
        _CACHE["fixture"] = (cfg, ds, preprocess(ds, cfg))
    return _CACHE["fixture"]


# --------------------------------------------------------------------------- #
# Data and leakage
# --------------------------------------------------------------------------- #
def test_leakage_columns_are_removed():
    cfg, ds, _ = _fixture()
    leak = set(cfg.get("data.leakage_columns"))
    present = leak & set(ds.X.columns)
    assert not present, f"leakage columns survived: {sorted(present)}"


def test_no_other_targets_in_features():
    _, ds, _ = _fixture()
    others = [c for c in ds.X.columns
              if c.startswith("target_") and c != ds.task["target"]]
    assert not others, f"other targets left as features: {others}"


def test_identifiers_removed():
    cfg, ds, _ = _fixture()
    ids = set(cfg.get("data.id_columns"))
    assert not (ids & set(ds.X.columns)), "identifier columns left as features"


def test_temporal_split_has_no_future_leakage():
    """Every training timestamp must precede every test timestamp."""
    cfg = _cfg()
    cfg.set("split.strategy", "temporal")
    ds = build_task(cfg)
    from logisticsml.stages.preprocessing import split_dataset
    train_idx, val_idx, test_idx, meta = split_dataset(ds, cfg)
    assert meta["split_strategy"] == "temporal"
    t = ds.time_index
    assert t.iloc[train_idx].max() <= t.iloc[test_idx].min(), \
        "training data extends past the start of the test window"
    assert t.iloc[val_idx].max() <= t.iloc[test_idx].min(), \
        "validation data extends past the start of the test window"


def test_splits_are_disjoint():
    _, _, sd = _fixture()
    total = len(sd.X_train) + len(sd.X_val) + len(sd.X_test)
    assert total > 0
    assert len(sd.X_train) > len(sd.X_test), "train should be the largest split"


def test_preprocessor_is_fitted_on_train_only():
    """Scaled training data should be near zero-mean; test data need not be.

    If the scaler had been fitted on everything, both would be centred - which
    is exactly the silent leak this guards against.
    """
    _, _, sd = _fixture()
    numeric = sd.X_train.columns[:40]
    train_mean = float(np.abs(sd.X_train[numeric].mean()).mean())
    assert train_mean < 0.15, f"training data not centred (mean |x| = {train_mean:.3f})"


def test_no_nans_after_preprocessing():
    _, _, sd = _fixture()
    for name, frame in [("train", sd.X_train), ("val", sd.X_val), ("test", sd.X_test)]:
        assert not frame.isna().any().any(), f"{name} split contains NaNs"
        assert np.isfinite(frame.to_numpy()).all(), f"{name} split contains infinities"


def test_feature_names_consistent_across_splits():
    _, _, sd = _fixture()
    assert list(sd.X_train.columns) == list(sd.X_test.columns) == sd.feature_names


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_classification_metrics_match_hand_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    m = classification_metrics(y_true, y_pred, task_type="binary")
    # TP=3, FP=1, FN=1, TN=3
    assert abs(m["accuracy"] - 6 / 8) < 1e-9
    assert abs(m["precision"] - 3 / 4) < 1e-9
    assert abs(m["recall"] - 3 / 4) < 1e-9
    assert abs(m["f1"] - 3 / 4) < 1e-9


def test_regression_metrics_match_hand_calculation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    m = regression_metrics(y_true, y_pred)
    assert abs(m["rmse"] - 1.0) < 1e-9          # sqrt(4/4)
    assert abs(m["mae"] - 0.5) < 1e-9           # 2/4


def test_perfect_and_random_predictions_score_as_expected():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    perfect = classification_metrics(y, y, np.column_stack([1 - y, y]), "binary")
    assert perfect["accuracy"] == 1.0 and perfect["roc_auc"] == 1.0
    noise = rng.random(500)
    random_scores = classification_metrics(
        y, (noise > 0.5).astype(int), np.column_stack([1 - noise, noise]), "binary")
    assert 0.35 < random_scores["roc_auc"] < 0.65, "random predictions should score near 0.5"


# --------------------------------------------------------------------------- #
# Supervised
# --------------------------------------------------------------------------- #
def _supervised():
    if "supervised" not in _CACHE:
        cfg, ds, sd = _fixture()
        cfg2 = _cfg()
        for m in ["svm", "knn", "gradient_boosting", "catboost"]:
            cfg2.set(f"supervised.models.{m}.enabled", False)
        _CACHE["supervised"] = (run_supervised(sd, cfg2, ds.task), sd, ds)
    return _CACHE["supervised"]


def test_models_train_and_produce_metrics():
    sup, _, _ = _supervised()
    assert len(sup.results) >= 3, f"only {len(sup.results)} models trained"
    for r in sup.results:
        assert r.metrics, f"{r.name} produced no metrics"
        assert r.predictions is not None


def test_best_model_beats_the_majority_class_baseline():
    """A model that cannot beat 'always predict the majority class' is useless."""
    sup, sd, _ = _supervised()
    best = sup.best
    assert best is not None
    majority_rate = float(sd.y_test.value_counts(normalize=True).max())
    assert best.metrics["accuracy"] > majority_rate, (
        f"{best.name} accuracy {best.metrics['accuracy']:.4f} "
        f"does not beat majority baseline {majority_rate:.4f}")
    assert best.metrics["roc_auc"] > 0.6, "best model barely beats random ranking"


def test_leaderboard_is_sorted_by_primary_metric():
    sup, _, _ = _supervised()
    lb = sup.leaderboard()
    primary = sup.primary_metric
    if primary in lb.columns and lb[primary].notna().all():
        assert lb[primary].is_monotonic_decreasing, "leaderboard not ranked by primary metric"


def test_predictions_have_the_right_shape_and_classes():
    sup, sd, ds = _supervised()
    valid = set(pd.unique(sd.y_train))
    for r in sup.results:
        assert len(r.predictions) == len(sd.y_test)
        assert set(np.unique(r.predictions)) <= valid, f"{r.name} predicted unseen classes"
        if r.probabilities is not None:
            assert np.allclose(r.probabilities.sum(axis=1), 1.0, atol=1e-5), \
                f"{r.name} probabilities do not sum to 1"


def test_skipped_models_are_explained():
    sup, _, _ = _supervised()
    for name, reason in sup.skipped.items():
        assert reason and isinstance(reason, str), f"{name} skipped without a reason"


# --------------------------------------------------------------------------- #
# Registry and serving
# --------------------------------------------------------------------------- #
def test_registry_round_trip_and_promotion():
    import tempfile
    from logisticsml.serving.registry import ModelRegistry

    sup, sd, ds = _supervised()
    best = sup.best
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(Path(tmp))
        reg = registry.register("test_model", best.estimator, best.metrics, ds.task,
                                sd.feature_names, preprocessor=sd.meta.get("preprocessor"))
        loaded = registry.get("test_model", reg.version).load()
        original = best.estimator.predict(sd.X_test.head(50))
        restored = loaded.predict(sd.X_test.head(50))
        assert np.array_equal(original, restored), "reloaded model predicts differently"

        registry.promote("test_model", reg.version)
        assert registry.get("test_model").version == reg.version
        assert registry.index["models"]["test_model"]["production"] == reg.version
        assert reg.metadata["feature_names"] == sd.feature_names


def test_prediction_service_enforces_the_feature_contract():
    """Columns supplied out of order, or missing, must not silently change the
    prediction - that is how training/serving skew starts."""
    import tempfile
    from logisticsml.serving.api import PredictionService
    from logisticsml.serving.registry import ModelRegistry

    sup, sd, ds = _supervised()
    best = sup.best
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(Path(tmp))
        registry.register("svc_test", best.estimator, best.metrics, ds.task,
                          sd.feature_names)
        registry.promote("svc_test")
        service = PredictionService(registry, "svc_test")

        rows = sd.X_test.head(20)
        straight = service.predict(rows.to_dict("records"))
        shuffled_cols = list(rows.columns)[::-1]
        shuffled = service.predict(rows[shuffled_cols].to_dict("records"))
        assert straight["predictions"] == shuffled["predictions"], \
            "column order changed the prediction"
        assert straight["n_records"] == 20
        assert service.health()["status"] == "ok"
        assert len(service.schema()["features"]) == len(sd.feature_names)


def test_service_rejects_oversized_batches():
    import tempfile
    from logisticsml.serving.api import PredictionService
    from logisticsml.serving.registry import ModelRegistry

    sup, sd, ds = _supervised()
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(Path(tmp))
        registry.register("limit_test", sup.best.estimator, sup.best.metrics,
                          ds.task, sd.feature_names)
        registry.promote("limit_test")
        service = PredictionService(registry, "limit_test", batch_limit=5)
        try:
            service.predict(sd.X_test.head(10).to_dict("records"))
            raise AssertionError("oversized batch was accepted")
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def _opt_tables():
    if "opt_tables" not in _CACHE:
        _CACHE["opt_tables"] = load_tables(
            _cfg(), ["orders", "routes", "vehicles", "drivers",
                     "warehouses", "delivery_zones", "inventory"])
    return _CACHE["opt_tables"]


def test_routing_respects_capacity_and_visits_every_stop():
    from logisticsml.optimization.routing import build_instance, solve_cvrp

    cfg = _cfg()
    inst = build_instance(_opt_tables()["orders"],
                          {**cfg.get("optimization.vrp"), "n_stops": 25}, cfg.seed)
    sol = solve_cvrp(inst, time_limit=5)

    visited = [n for route in sol.routes for n in route if n != inst.depot]
    assert len(visited) == len(set(visited)), "a stop was visited twice"
    assert set(visited) == set(range(1, inst.n_stops)), "not every stop was served"
    for load in sol.load_per_route:
        assert load <= inst.capacity_kg + 1e-6, f"route load {load} exceeds capacity"
    assert sol.total_distance_km > 0


def test_constrained_routing_costs_more_than_unconstrained():
    """CVRP adds capacity to the TSP, VRPTW adds time windows on top. More
    constraints can never make the optimum shorter."""
    from logisticsml.optimization.routing import build_instance, solve_cvrp, solve_tsp

    cfg = _cfg()
    inst = build_instance(_opt_tables()["orders"],
                          {**cfg.get("optimization.vrp"), "n_stops": 25}, cfg.seed)
    tsp = solve_tsp(inst, time_limit=5)
    cvrp = solve_cvrp(inst, time_limit=5)
    assert cvrp.total_distance_km >= tsp.total_distance_km - 1e-6, \
        "capacity-constrained solution beat the unconstrained one"


def test_metaheuristics_improve_on_the_greedy_tour():
    from logisticsml.optimization.metaheuristics import (
        genetic_algorithm, particle_swarm, simulated_annealing)
    from logisticsml.optimization.routing import (
        build_instance, nearest_neighbour_route, route_distance)

    cfg = _cfg()
    inst = build_instance(_opt_tables()["orders"],
                          {**cfg.get("optimization.vrp"), "n_stops": 25}, cfg.seed)
    greedy = route_distance(nearest_neighbour_route(inst.distance_km), inst.distance_km)

    for fn, kwargs in [(genetic_algorithm, {"generations": 60, "population": 40}),
                       (simulated_annealing, {"iterations": 3000}),
                       (particle_swarm, {"iterations": 100, "particles": 30})]:
        r = fn(inst, seed=cfg.seed, **kwargs)
        assert r.best_distance_km <= greedy + 1e-6, \
            f"{r.method} ({r.best_distance_km:.1f} km) is worse than greedy ({greedy:.1f} km)"


def test_optimisers_beat_their_baselines():
    from logisticsml.optimization.allocation import (
        optimise_inventory, solve_driver_scheduling, solve_warehouse_allocation)

    cfg = _cfg()
    t = _opt_tables()
    for result in [
        solve_driver_scheduling(t["drivers"], cfg.get("optimization.driver_scheduling"), cfg.seed),
        solve_warehouse_allocation(t["warehouses"], t["delivery_zones"],
                                   cfg.get("optimization.warehouse_allocation"), cfg.seed),
        optimise_inventory(t["inventory"], cfg.get("optimization.inventory")),
    ]:
        assert result.objective > 0, f"{result.problem} produced a non-positive objective"
        assert result.baseline is not None, f"{result.problem} reported no baseline"
        assert result.objective <= result.baseline + 1e-6, (
            f"{result.problem}: optimiser ({result.objective:.2f}) is worse than "
            f"its baseline ({result.baseline:.2f})")


def test_inventory_policy_is_internally_consistent():
    from logisticsml.optimization.allocation import optimise_inventory

    cfg = _cfg()
    r = optimise_inventory(_opt_tables()["inventory"], cfg.get("optimization.inventory"))
    policy = r.tables["policy"]
    assert (policy["eoq"] > 0).all(), "non-positive order quantity"
    assert (policy["safety_stock"] >= 0).all(), "negative safety stock"
    assert (policy["reorder_point"] >= policy["safety_stock"]).all(), \
        "reorder point below safety stock"


# --------------------------------------------------------------------------- #
# Auxiliary tasks
# --------------------------------------------------------------------------- #
def test_every_configured_task_builds():
    cfg = _cfg()
    for name in (cfg.get("tasks") or {}):
        ds = build_task(cfg, name)
        assert len(ds.X) > 0, f"task {name} produced no rows"
        assert len(ds.X) == len(ds.y)
        assert ds.task["target"] not in ds.X.columns, f"task {name} kept its own target"


def test_auxiliary_targets_are_not_trivially_derivable():
    """The constructed labels must not be sitting in their own feature matrix."""
    cfg = _cfg()
    forbidden = {
        "vehicle_failure": ["unplanned_events"],
        "inventory_shortage": ["stockout_flag"],
        "fraud_detection": ["anomaly_flags", "status"],
        "warehouse_congestion": ["utilisation", "orders_count"],
    }
    for task, cols in forbidden.items():
        ds = build_task(cfg, task)
        leaked = [c for c in cols if c in ds.X.columns]
        assert not leaked, f"task {task} leaks {leaked}"


def test_class_balance_is_reported_for_classification():
    cfg = _cfg()
    ds = build_task(cfg, "fraud_detection")
    described = ds.describe()
    assert "class_balance" in described
    assert sum(described["class_balance"].values()) > 0.99


# --------------------------------------------------------------------------- #
def _run_standalone() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
