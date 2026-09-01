"""Validation suite.

Runs a tiny end-to-end generation and asserts the properties that matter:
reproducibility, referential integrity, physical plausibility, and that the
missingness / anomaly machinery hit its configured targets.

Run with ``pytest tests/ -v`` or directly: ``python tests/test_generator.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logisticsgen import generate_dataset, load_config  # noqa: E402

CONFIG = [ROOT / "configs" / "default.yaml"]


def _tiny(**overrides):
    cfg = load_config(CONFIG)
    cfg.set("project.scale", 0.003)
    cfg.set("project.output_dir", "/tmp/logisticsgen_test")
    cfg.set("child_tables.traffic.resolution_hours", 12)
    cfg.set("child_tables.weather.resolution_hours", 12)
    for key, value in overrides.items():
        cfg.set(key, value)
    return cfg


_RESULT = None


def result():
    global _RESULT
    if _RESULT is None:
        _RESULT = generate_dataset(_tiny(), export=False, reports=False)
    return _RESULT


# --------------------------------------------------------------------------- #
def test_all_tables_present():
    expected = {
        "warehouses", "pickup_locations", "delivery_zones", "customers", "vehicles",
        "drivers", "traffic", "weather", "fuel_prices", "regional_holidays", "orders",
        "routes", "gps_tracking", "delivery_history", "customer_feedback",
        "vehicle_maintenance", "inventory", "shift_planning", "operating_costs",
        "courier_performance",
    }
    missing = expected - set(result().tables)
    assert not missing, f"Missing tables: {missing}"


def test_no_table_is_empty():
    for name, df in result().tables.items():
        assert len(df) > 0, f"{name} is empty"


def test_primary_keys_unique():
    """First column of each table is its surrogate key and must be unique.

    ``orders`` is exempt because duplicate-order anomalies are injected on
    purpose - those rows carry a ``-DUP`` suffix and so stay unique anyway.
    """
    for name, df in result().tables.items():
        if name.startswith("ml_"):
            continue
        key = df.columns[0]
        if not key.endswith("_id"):
            continue
        assert df[key].is_unique, f"{name}.{key} has duplicates"


def test_referential_integrity():
    t = result().tables
    checks = [
        ("orders", "customer_id", "customers", "customer_id"),
        ("orders", "warehouse_id", "warehouses", "warehouse_id"),
        ("orders", "zone_id", "delivery_zones", "zone_id"),
        ("orders", "driver_id", "drivers", "driver_id"),
        ("orders", "vehicle_id", "vehicles", "vehicle_id"),
        ("routes", "vehicle_id", "vehicles", "vehicle_id"),
        ("gps_tracking", "order_id", "orders", "order_id"),
        ("delivery_history", "order_id", "orders", "order_id"),
        ("customer_feedback", "order_id", "orders", "order_id"),
        ("vehicle_maintenance", "vehicle_id", "vehicles", "vehicle_id"),
        ("inventory", "warehouse_id", "warehouses", "warehouse_id"),
        ("shift_planning", "driver_id", "drivers", "driver_id"),
        ("courier_performance", "driver_id", "drivers", "driver_id"),
    ]
    for child, fk, parent, pk in checks:
        c, p = t.get(child), t.get(parent)
        if c is None or p is None or c.empty or fk not in c.columns:
            continue
        values = c[fk].dropna()
        orphans = ~values.isin(set(p[pk]))
        assert not orphans.any(), f"{child}.{fk} has {orphans.sum()} orphan refs"


def test_reproducible_with_same_seed():
    a = generate_dataset(_tiny(), export=False, reports=False)
    b = generate_dataset(_tiny(), export=False, reports=False)
    pd.testing.assert_frame_equal(
        a.tables["orders"].head(200), b.tables["orders"].head(200))


def test_different_seed_gives_different_data():
    a = generate_dataset(_tiny(), export=False, reports=False)
    b = generate_dataset(_tiny(**{"project.seed": 999}), export=False, reports=False)
    assert not a.tables["orders"]["distance_km"].head(100).equals(
        b.tables["orders"]["distance_km"].head(100))


def test_physical_plausibility():
    o = result().tables["orders"]
    assert (o["distance_km"] > 0).all(), "non-positive distance"
    assert (o["package_weight_kg"].dropna() > 0).all(), "non-positive weight"
    assert (o["delivery_cost_usd"] >= 0).all(), "negative cost"
    # Ignore the deliberately-corrupted negative_duration anomaly rows.
    clean = o.loc[~o["anomaly_flags"].fillna("").str.contains("negative_duration")]
    assert (clean["actual_duration_min"].dropna() > 0).all(), "non-positive duration"
    speed = clean["distance_km"] / (clean["actual_duration_min"] / 60)
    assert speed.dropna().quantile(0.99) < 200, "implausible 99th-pct speed"


def test_timestamps_ordered():
    o = result().tables["orders"]
    assert (o["pickup_timestamp"] >= o["order_timestamp"]).all(), "pickup before order"
    delivered = o.dropna(subset=["actual_delivery_ts"])
    clean = delivered.loc[~delivered["anomaly_flags"].fillna("").str.contains("negative_duration")]
    assert (clean["actual_delivery_ts"] >= clean["pickup_timestamp"]).all(), "delivery before pickup"


def test_missingness_hits_target_rates():
    report = result().missingness_report
    assert not report.empty, "no missingness applied"
    # Rules are applied sequentially on independent columns, so achieved should
    # sit close to target. Allow generous tolerance for small-sample noise.
    for row in report.itertuples():
        assert abs(row.achieved_rate - row.target_rate) < 0.06, (
            f"{row.table}.{row.column}: target {row.target_rate}, got {row.achieved_rate}")


def test_all_three_mechanisms_present():
    mechanisms = set(result().missingness_report["mechanism"])
    assert {"MCAR", "MAR", "MNAR"} <= mechanisms, f"got only {mechanisms}"


def test_mnar_is_actually_biased():
    """MNAR on declared_value_usd should hide high values, shifting the
    observed mean *below* the true mean. This is the whole point of MNAR."""
    r = result()
    gt_key = "orders.declared_value_usd"
    o = r.tables["orders"]
    observed_mean = o["declared_value_usd"].mean()
    # Rebuild the true mean by comparing against a run with missingness off.
    clean = generate_dataset(_tiny(**{"missingness.enabled": False}),
                             export=False, reports=False)
    true_mean = clean.tables["orders"]["declared_value_usd"].mean()
    assert observed_mean < true_mean, (
        f"MNAR('high') should depress the observed mean: {observed_mean:.1f} vs {true_mean:.1f}")


def test_anomalies_injected_and_flagged():
    r = result()
    assert not r.anomaly_report.empty, "no anomalies reported"
    assert r.anomaly_report["rows_affected"].sum() > 0, "zero anomalous rows"
    flagged = r.tables["orders"]["anomaly_flags"].fillna("")
    assert (flagged != "").any(), "orders carry no anomaly flags"


def test_duplicate_orders_are_marked():
    o = result().tables["orders"]
    dups = o.loc[o["is_duplicate"]]
    assert len(dups) > 0, "no duplicate orders injected"
    assert dups["order_id"].str.endswith("-DUP").all(), "duplicates not suffixed"


def test_environment_actually_drives_delay():
    """Adverse weather and heavy traffic must slow deliveries down.

    Measured as minutes-per-km rather than raw ``delay_minutes``: delay is
    computed against the SLA, so it is dominated by how generous that order's
    promise was. Travel time per km isolates the actual speed penalty, which is
    what weather and traffic causally act on.
    """
    o = result().tables["orders"].dropna(subset=["actual_duration_min"])
    pace = o["actual_duration_min"] / o["distance_km"].clip(lower=0.1)

    adverse = o["weather_condition"].isin(["storm", "snow", "heavy_rain", "fog"])
    if adverse.any() and (~adverse).any():
        assert pace[adverse].median() > pace[~adverse].median(), \
            "adverse weather shows no slowdown"

    heavy = o["traffic_level"].isin(["heavy", "gridlock"])
    if heavy.any() and (~heavy).any():
        assert pace[heavy].median() > pace[~heavy].median(), \
            "heavy traffic shows no slowdown"


def test_late_rate_rises_with_adverse_conditions():
    """The business-facing signal: SLA breach rate must be higher in bad
    conditions than in good ones.

    Runs at a larger scale than the other checks - the weather effect is a few
    percentage points on a subgroup that is itself only ~10% of orders, so the
    default tiny fixture doesn't have the sample size to resolve it.
    """
    o = generate_dataset(_tiny(**{"project.scale": 0.02}),
                         export=False, reports=False).tables["orders"]
    adverse = o["weather_condition"].isin(["storm", "snow", "heavy_rain", "fog"])
    assert adverse.sum() > 300, "not enough adverse-weather orders to test"
    late_adverse = o.loc[adverse, "is_late"].mean()
    late_clear = o.loc[~adverse, "is_late"].mean()
    assert late_adverse > late_clear, (
        f"adverse weather does not raise the late rate ({late_adverse:.3f} vs {late_clear:.3f})")


def test_status_and_is_late_agree():
    """Without anomaly injection, ``is_late`` must be exactly equivalent to
    ``status == 'delivered_late'`` on completed orders."""
    r = generate_dataset(_tiny(**{"anomalies.enabled": False, "missingness.enabled": False}),
                         export=False, reports=False)
    o = r.tables["orders"]
    done = o.loc[o["status"].isin(["delivered", "delivered_late"])]
    assert (done["is_late"] == (done["status"] == "delivered_late")).all(), \
        "is_late contradicts status"


def test_on_time_rate_matches_config():
    """The SLA calibration should land the fleet on the configured on-time rate."""
    cfg = _tiny(**{"anomalies.enabled": False, "missingness.enabled": False})
    target = cfg.get("operations.on_time_base_rate")
    o = generate_dataset(cfg, export=False, reports=False).tables["orders"]
    achieved = 1 - o["is_late"].mean()
    assert abs(achieved - target) < 0.04, f"on-time {achieved:.3f} vs target {target}"


def test_distances_are_physically_sane():
    """Guards the regression where customers in unserved cities were fulfilled
    from another continent."""
    o = result().tables["orders"]
    assert o["distance_km"].median() < 80, "median distance implausible for last-mile"
    assert o["distance_km"].quantile(0.90) < 400, "90th-pct distance implausible"
    assert o["distance_km"].max() < 3000, "single delivery exceeds any road network"
    by_area = o.groupby("area_type")["distance_km"].median()
    if {"urban", "rural"} <= set(by_area.index):
        assert by_area["urban"] < by_area["rural"], "urban trips should be shorter than rural"


def test_traffic_levels_are_well_distributed():
    """A baseline that reads gridlock half the time carries no information."""
    share = result().tables["traffic"]["traffic_level"].value_counts(normalize=True)
    assert share.get("gridlock", 0) < 0.10, "gridlock is not an exception"
    assert share.get("free_flow", 0) + share.get("light", 0) > 0.25, "no quiet periods"


def test_routes_are_multistop():
    routes = result().tables["routes"]
    assert routes["stops"].mean() > 2.5, "routes collapsed to single stops"
    assert routes["capacity_utilisation_kg"].median() < 1.5, "fleet implausibly overloaded"


def test_targets_present_and_typed():
    f = result().features
    for col in ["target_delay_minutes", "target_is_late", "target_eta_minutes",
                "target_delivery_cost_usd", "target_risk_bucket", "target_will_be_returned"]:
        assert col in f.columns, f"missing target {col}"
    assert set(f["target_risk_bucket"].dropna().unique()) <= {"low", "medium", "high"}
    assert f["target_is_late"].dropna().isin([0.0, 1.0]).all()


def test_ml_table_has_no_leakage_and_no_nans():
    r = result()
    ml = r.ml_table
    from logisticsgen.pipeline import LEAKAGE_COLUMNS
    leaked = [c for c in LEAKAGE_COLUMNS if c in ml.columns]
    assert not leaked, f"leakage columns survived preprocessing: {leaked}"
    feature_cols = [c for c in ml.columns
                    if not c.startswith("target_")
                    and not pd.api.types.is_datetime64_any_dtype(ml[c])]
    n_nan = ml[feature_cols].isna().sum().sum()
    assert n_nan == 0, f"{n_nan} NaNs left in feature columns after imputation"


def test_features_have_no_future_leakage_in_rolling():
    """Rolling history features are shifted, so the very first order in each
    group must have a NaN prior-rate (nothing to look back on)."""
    f = result().features.sort_values("order_timestamp")
    first = f.groupby("zone_id", sort=False).head(1)
    assert first["zone_prior_late_rate"].isna().all(), "rolling feature sees its own row"


def test_scale_controls_volume():
    small = generate_dataset(_tiny(**{"project.scale": 0.002}), export=False, reports=False)
    big = generate_dataset(_tiny(**{"project.scale": 0.006}), export=False, reports=False)
    assert len(big.tables["orders"]) > len(small.tables["orders"]) * 2


def test_routes_aggregate_correctly():
    t = result().tables
    routes, orders = t["routes"], t["orders"]
    assert (routes["stops"] > 0).all()
    assert (routes["on_time_rate"].between(0, 1)).all()
    # Consolidation means actual route km should not exceed the naive sum.
    # Distance columns carry injected missingness, so compare complete pairs only.
    pairs = routes[["actual_distance_km", "planned_distance_km"]].dropna()
    assert (pairs["actual_distance_km"] <= pairs["planned_distance_km"] * 1.35).all(), \
        "route consolidation produced impossible distances"
    linked = orders["route_id"].notna().sum()
    assert linked > 0, "no orders linked to routes"


def test_business_domains_all_represented():
    o = result().tables["orders"]
    cfg = _tiny()
    assert set(o["business_domain"].unique()) == set(cfg.get("domains").keys())


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
            failures.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
