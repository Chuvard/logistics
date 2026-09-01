# Logistics Optimization — Synthetic Dataset Generator

Generates an enterprise-scale, internally consistent logistics dataset with
configurable missingness, anomalies and operational disruptions. Built for the
Logistics Optimization portfolio project (prompt 01); its output is the input
for the ML platform, dashboard and pitch deck that follow.

Covers transportation, courier delivery, e-commerce, warehouse distribution,
ride-sharing logistics, grocery delivery, pharmaceutical transport,
manufacturing supply chains and retail fulfilment.

---

## Quick start

```bash
pip install -r requirements.txt

# ~50,000 deliveries — runs in about a minute
python -m logisticsgen.cli --config configs/default.yaml --config configs/sample.yaml

# full enterprise spec: 1,000,000 deliveries
python -m logisticsgen.cli --config configs/default.yaml --config configs/full.yaml

# ad-hoc, no YAML editing
python -m logisticsgen.cli --scale 0.01 --seed 7 --formats parquet
```

As a library:

```python
from logisticsgen import load_config, generate_dataset

cfg = load_config(["configs/default.yaml", "configs/sample.yaml"])
result = generate_dataset(cfg)

orders = result.tables["orders"]
ml     = result.ml_table          # cleaned, encoded, leakage-free
```

---

## What it produces

20 related tables. At `scale: 1.0` the entity counts match the brief exactly:
1,000,000 deliveries · 100,000 customers · 10,000 vehicles · 5,000 drivers ·
500 warehouses · 2,000 pickup locations · 10,000 delivery zones.

| Group | Tables |
|---|---|
| Dimensions | `warehouses`, `pickup_locations`, `delivery_zones`, `customers`, `vehicles`, `drivers` |
| Environment | `traffic`, `weather`, `fuel_prices`, `regional_holidays` |
| Core facts | `orders`, `routes` |
| Operations | `gps_tracking`, `delivery_history`, `customer_feedback`, `vehicle_maintenance`, `inventory`, `shift_planning`, `operating_costs`, `courier_performance` |
| Derived | `ml_features` (165 cols), `ml_clean` (preprocessed, 177 cols) |

Everything scales together — `project.scale` multiplies all entity counts, so
the relationships between tables hold at any size.

---

## The part that matters: it's causally consistent

Most synthetic datasets sample every column independently, which makes them
useless for modelling — there is nothing to learn. Here the numbers are *built*,
not drawn:

```
effective_speed = base_speed(area_type) ÷ (traffic_factor × weather_factor)
travel_time     = distance ÷ effective_speed
duration        = travel_time + service_time + handling_time
delay           = (pickup_lag + duration) − promised_SLA
```

Traffic and weather come from exogenous city × time panels that orders join
against, so congestion and storms are genuine upstream causes of delay and cost.
Observed in the 50k sample:

| Condition | Late rate |
|---|---|
| Adverse weather (storm/snow/heavy rain/fog) | 13.0% |
| Clear conditions | 11.0% |
| Heavy traffic or gridlock | 12.8% |
| Free-flow or light traffic | 10.4% |

Other consistency properties that are enforced rather than hoped for:

- **SLA calibration.** Lateness is emergent. Promised SLAs are scaled by a
  single constant so the fleet lands on `operations.on_time_base_rate`
  (88.5% configured → 88.7% achieved), while preserving the relative SLA
  structure across priorities and domains. `is_late` and `status` cannot
  disagree.
- **Geography is real.** Zones only exist in cities that have a warehouse, and
  each city is served by its nearest sites. Median distance 21 km, splitting
  18.5 km urban / 27 km suburban / 51 km rural.
- **Domains behave differently.** Freight domains carry heavy loads with a
  long-haul tail (manufacturing: 211 kg median, 1,458 km at p95); courier and
  grocery stay last-mile (0.8–2.7 kg, ~50 km at p95).
- **Routes are multi-stop.** Orders are clustered by warehouse-day into runs of
  ~5 stops sharing one vehicle and driver, then rolled up into route facts with
  consolidation savings on distance.

---

## Missingness

Three mechanisms from Rubin's taxonomy, per-column, at configurable rates:

| Mechanism | Probability depends on | Example rule |
|---|---|---|
| **MCAR** | nothing | `orders.package_weight_kg` at 5% |
| **MAR** | another *observed* column | `orders.actual_delivery_ts` given `status` |
| **MNAR** | the *hidden value itself* | `orders.declared_value_usd`, high values hidden |

18 rules ship by default. Achieved rates track targets to within 0.6pp.

The MNAR bias is real and measurable — for `declared_value_usd`, masked values
average **$1,251** while the surviving observed values average **$726**. Naive
mean-imputation will be biased, which is exactly the point.

Every masked cell is written to `ground_truth/<table>__<column>.parquet` with
its true value, so imputation strategies can be scored rather than guessed at.
`missingness.rate_sweep` additionally reports what 1 / 5 / 10 / 20 / 40%
severity would look like without regenerating the data.

---

## Anomalies

15 classes, each tagged in the affected table's `anomaly_flags` column so
anomaly detection has labelled ground truth.

**Operational disruptions** — demand spikes, extreme weather events, traffic
gridlock, vehicle breakdowns (delay + cost spike + possible failure), driver
no-shows, warehouse stockouts, regional fuel price shocks, address errors,
fraudulent orders.

**Data-quality defects** — GPS jumps (position teleports with impossible implied
speed), GPS flatlines, sensor drift, duplicate orders (`-DUP` suffix), negative
durations, order-of-magnitude cost outliers.

---

## Features, targets and preprocessing

`ml_features` denormalises orders against every dimension and derives temporal
(including cyclical sin/cos), geospatial, package, resource, environmental,
rolling-history and efficiency features.

Rolling aggregates are **leakage-aware**: every historical feature is computed
with `.shift(1)`, so a row never sees its own outcome.

Six targets are provided:

| Target | Task |
|---|---|
| `target_delay_minutes` | regression |
| `target_eta_minutes` | regression |
| `target_delivery_cost_usd` | regression |
| `target_is_late` | binary (11.2% positive) |
| `target_will_be_returned` | binary |
| `target_risk_bucket` | multiclass (low / medium / high) |

`ml_clean` applies imputation, outlier clipping, ordinal encoding, missing
indicators and constant-column dropping — and drops 25 explicitly enumerated
leakage columns. Every decision is recorded in
`reports/preprocessing_metadata.json` so the transform can be replayed.

---

## Output

```
data/sample/
├── csv/                 22 files
├── parquet/             22 files
├── sqlite/logistics.db  indexed on all join keys
├── ground_truth/        true values behind every masked cell
├── reports/
│   ├── eda_report.html      styled, self-contained
│   ├── eda_report.md
│   ├── data_dictionary.csv  per-column stats for every table
│   ├── missingness_report.csv
│   ├── anomaly_report.csv
│   └── preprocessing_metadata.json
└── resolved_config.yaml   exact config used, for reproduction
```

Formats: `csv`, `parquet`, `sqlite`, `sql` (portable DDL + INSERT), `duckdb`
(needs `pip install duckdb`). A failure in one format never aborts the run.

---

## Configuration

Everything lives in `configs/default.yaml` — entity volumes, seasonality curves,
the nine business domains, operational physics, missingness rules, anomaly
rates, feature and target switches, preprocessing and export settings.

Config files merge left-to-right, so overrides stay small:

```bash
python -m logisticsgen.cli \
  --config configs/default.yaml \
  --config configs/sample.yaml \
  --set anomalies.gps_jump.rate=0.02 \
  --set missingness.default_rate=0.2
```

---

## Reproducibility

Every generator draws from its own named RNG stream derived from
`project.seed` via a keyed BLAKE2b hash. Adding, removing or reordering a table
never perturbs the values of the others, so runs stay diffable across code
changes. Same seed → byte-identical output.

---

## Validation

```bash
python tests/test_generator.py      # or: pytest tests/ -v
```

26 checks covering table presence, primary-key uniqueness, referential
integrity across 13 foreign keys, seed reproducibility, physical plausibility
(no negative durations outside injected anomalies, no impossible speeds),
timestamp ordering, missingness rates hitting their targets, all three
mechanisms present, MNAR bias actually materialising, anomaly flagging,
environmental causality, SLA calibration, distance sanity, route structure,
target validity and zero post-imputation NaNs with no leakage columns
surviving.

---

## Performance

| Scale | Deliveries | Total rows | Time |
|---|---|---|---|
| 0.05 (sample) | 50,150 | 888k | ~66s incl. all exports |
| 1.0 (full) | 1,000,000 | ~17M | Parquet + SQLite recommended; CSV is slow at this size |

Generation is vectorised NumPy/pandas throughout — no per-row Python loops, no
Faker dependency.

---

## Layout

```
logistics_datagen/
├── configs/            default.yaml, sample.yaml, full.yaml
├── logisticsgen/
│   ├── config.py       dotted-path YAML config with merge + CLI overrides
│   ├── rng.py          named reproducible random streams
│   ├── reference.py    geography, taxonomies, name pools
│   ├── utils.py        haversine, ID minting, seasonal timestamps
│   ├── generators/     entities · environment · orders · operations
│   ├── quality/        missingness · anomalies
│   ├── pipeline/       features · targets · preprocessing
│   ├── io/             csv · parquet · sql · sqlite · duckdb
│   ├── reports/        EDA, missingness, data dictionary
│   ├── generate.py     orchestration
│   └── cli.py          command line entry point
└── tests/
```
