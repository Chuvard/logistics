# Logistics Optimization — ML Platform

A reusable, config-driven Python platform covering the full pipeline: data
preprocessing, EDA, supervised learning, unsupervised learning, deep learning,
mathematical optimization and explainable AI. It consumes the dataset produced
by `logistics_datagen` (prompt 01).

Every heavy dependency is optional. Missing libraries downgrade a stage
gracefully and are reported — they never abort a run.

---

## Quick start

```bash
pip install -r requirements.txt

python -m logisticsml.cli run                      # full pipeline, default task
python -m logisticsml.cli tasks                    # list the 11 prediction tasks
python -m logisticsml.cli deps                     # what's installed, what it unlocks
python -m logisticsml.cli run --task fraud_detection
python -m logisticsml.cli run --max-rows 20000 --skip unsupervised,deep_learning
python -m logisticsml.cli registry                 # versions and production pointer
python -m logisticsml.cli serve --task late_delivery
```

As a library:

```python
from logisticsml import load_config, run_pipeline

cfg = load_config(["configs/default.yaml"])
result = run_pipeline(cfg, task_name="late_delivery")

print(result.supervised.leaderboard())
best = result.supervised.best.estimator
```

---

## Results from the shipped run

50,150 orders, temporal split (train ≤ 6 Oct, test from 6 Nov), 137 features.

**Supervised — late delivery at dispatch (ROC-AUC):**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | PR-AUC | Train |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.998 | 0.989 | 0.960 | 0.939 | 0.949 | 0.988 | 0.8s |
| LightGBM | 0.998 | 0.989 | 0.979 | 0.925 | 0.951 | 0.988 | 0.5s |
| Gradient Boosting | 0.997 | 0.984 | 0.934 | 0.921 | 0.927 | 0.974 | 59.5s |
| Logistic Regression | 0.990 | 0.912 | 0.562 | 0.990 | 0.717 | 0.928 | 1.6s |
| Random Forest | 0.990 | 0.967 | 0.878 | 0.821 | 0.849 | 0.920 | 4.9s |
| Decision Tree | 0.989 | 0.982 | 0.933 | 0.904 | 0.918 | 0.951 | 0.9s |
| SVM | 0.935 | 0.920 | 0.754 | 0.430 | 0.548 | 0.652 | 6.0s |
| KNN | 0.807 | 0.892 | 0.753 | 0.057 | 0.107 | 0.372 | 0.0s |

**Optimization — every result measured against a real baseline:**

| Problem | Solver | Objective | Baseline | Improvement |
|---|---|---|---|---|
| Driver scheduling | OR-Tools CP-SAT | $6,028 | $8,782 | **31.4%** |
| Fleet allocation | OR-Tools CP-SAT | $3,640 | $12,519 | **70.9%** |
| Warehouse allocation | OR-Tools CP-SAT | 48,935 | 49,073 | 0.3% |
| Inventory (EOQ) | closed form | $9.38M/yr | $10.45M/yr | **10.3%** |

**Routing — OR-Tools vs metaheuristics on the identical 40-stop instance:**

| Method | Distance | Time | vs OR-Tools |
|---|---|---|---|
| OR-Tools (TSP) | 232.6 km | 8.0s | — |
| Simulated annealing | 234.2 km | 0.09s | +0.7% |
| Genetic algorithm | 241.3 km | 1.2s | +3.8% |
| Particle swarm | 243.6 km | 0.07s | +4.7% |

Simulated annealing lands within 0.7% of the exact solver in **1/90th of the
time**. That trade-off is the point of running them head-to-head.

---

## Honest notes on the results

Numbers that look wrong usually are. These ones aren't, and here's why:

**ROC-AUC of 0.998 is not leakage — it was, and it got fixed.** The first run
scored 1.000 because `ml_features` still contained `is_late`, `delay_minutes`
and the realised costs. The platform now drops 25 enumerated outcome columns
(`data.leakage_columns`) after extracting the target. What remains is genuinely
high because the dataset's delay is a deterministic function of distance,
traffic, weather and SLA — all of which are features. Recovering that function
is the correct result. The model's top features (`sla_hours`,
`lead_time_hours`, `planned_duration_min`, `congestion_index`, `distance_km`)
are exactly the generator's causal drivers.

**Two framings of the same question.** `late_delivery` is scored at dispatch,
where `lead_time_hours` is known (AUC 0.998). `late_delivery_predispatch`
removes it and everything else known only after assignment, scoring at order
entry (**AUC 0.960**). The gap is the value of that one feature. Use the
pre-dispatch task if you want to quote a delivery promise up front.

**`warehouse_allocation` improves only 0.3%, and that's the honest number.**
The baseline is greedy facility location, which is provably near-optimal for
this problem. An earlier version compared against an open-every-site baseline
and reported a meaningless 66%.

**Not every task has signal, and the platform doesn't pretend otherwise:**

| Task | Result | Why |
|---|---|---|
| `late_delivery` | AUC 0.998 | strong causal structure |
| `route_failure` | AUC 0.976 | driven by zone difficulty |
| `risk_classification` | F1-macro 0.818 | 3-class, composite target |
| `fraud_detection` | PR-AUC 0.454 vs 0.0008 base rate | **567× lift** on a 0.08% positive class |
| `warehouse_congestion` | AUC 0.742 | lagged volume predicts busy days |
| `vehicle_failure` | AUC 0.558 | weak; only 500 vehicles, failure ~ Poisson(age) |
| `inventory_shortage` | AUC 0.515 | **effectively random by construction** — the generator injects stockouts independently per snapshot, so there is no temporal signal to learn |

`inventory_shortage` is reported rather than quietly dropped. A platform that
only shows its wins isn't measuring anything.

---

## The seven stages

**1. Preprocessing** — imputation, scaling, encoding and missing indicators,
all fitted on the training split only. The default split is **temporal**,
because predicting SLA breaches is forecasting and a random split lets the
model read the future. Group (by route) and random splits are also available.

**2. EDA** — numeric and categorical profiles, missingness, target balance,
correlation structure, redundant-pair detection, and univariate target
associations, with plots.

**3. Supervised** — nine families benchmarked on one split: Logistic
Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM,
CatBoost, KNN, SVM. Gradient boosters use native early stopping; classifiers get
optional isotonic calibration fitted on validation, never test. SVM and KNN are
subsampled with the cap printed in the results table, so a capped score is never
mistaken for a full-data one. Metrics: accuracy, precision, recall, F1, ROC-AUC,
PR-AUC, MCC, Brier, log-loss, plus confusion matrices and calibration curves.

**4. Unsupervised** — KMeans (swept over k, selected by silhouette), DBSCAN, GMM
(selected by BIC), PCA, t-SNE, UMAP, Isolation Forest. Clusters are profiled by
the features that most separate them.

**5. Deep learning** — MLP, Autoencoder, LSTM, Transformer (FT-Transformer
style, attention over feature tokens) and TabNet (sequential attention with
sparsemax feature selection), all implemented in PyTorch in `deep_torch.py`.
When torch is absent the stage runs scikit-learn stand-ins so it still produces
real metrics; approximations are labelled **APPROXIMATION** in every table and
the backend is stated in the report. PCA is used for the autoencoder — a linear
autoencoder is exactly PCA.

**6. Optimization** — TSP, CVRP and VRP with time windows via OR-Tools, with a
nearest-neighbour + 2-opt fallback. Driver scheduling, fleet allocation and
capacitated facility location as CP-SAT models. EOQ with safety stock and
reorder points. Genetic algorithm, simulated annealing and particle swarm
implemented from scratch and benchmarked on the same instance as OR-Tools.

**7. Explainable AI** — SHAP (global, beeswarm, local), permutation importance,
LIME, and partial dependence. The stage also reports the **rank correlation
between SHAP and permutation importance** (0.61 here); sharp disagreement
signals correlated features or an unstable model and is worth surfacing.

> SHAP note: the released `shap` package currently fails against XGBoost 3.x
> (it parses the JSON-encoded `base_score` as a float). The platform calls
> XGBoost's and LightGBM's **native TreeSHAP** instead — the same exact Shapley
> values, computed by the library that owns the tree structure.

---

## Model registry and serving

File-backed, versioned, no tracking server required.

```bash
python -m logisticsml.cli registry
python -m logisticsml.cli registry --promote late_delivery v20260807_151042
python -m logisticsml.cli registry --rollback late_delivery
python -m logisticsml.cli serve --task late_delivery --port 8000
```

Each version stores the estimator, the fitted preprocessor, the exact feature
order, metrics and provenance. Promotion is explicit and rollback is one
command; the production version is never pruned.

The prediction service enforces the **feature contract**: raw records go through
the same preprocessor used in training, missing columns are defaulted and
reported, and column order is restored — reordering the input cannot change a
prediction. That property is covered by a test.

```
GET  /health    model, version, metrics
GET  /schema    feature list and an example request
POST /predict   {"records": [{...}]} -> predictions + probabilities
```

FastAPI is used when installed; otherwise the same routes are served from the
standard library, so the API runs anywhere.

---

## Configuration

Everything lives in `configs/default.yaml`: data source, the leakage guard, 11
task definitions, split strategy, preprocessing, per-model hyperparameters,
optimization problem sizes and costs, explainability and reporting.

```bash
python -m logisticsml.cli run \
  --set supervised.models.xgboost.n_estimators=800 \
  --set split.strategy=random \
  --set optimization.vrp.n_stops=80
```

Configs merge left-to-right; relative paths resolve against the config file.

---

## Output

```
artifacts/
├── reports/report_<task>.html    self-contained, plots embedded as base64
├── reports/report_<task>.md
├── plots/                        eda · supervised · unsupervised · optimization · explainability
├── tables/                       every result table as CSV
├── registry/                     versioned models + index.json
├── run_summary.json
└── resolved_config.yaml          exact config used
```

---

## Validation

```bash
python tests/test_platform.py      # or: pytest tests/ -v
```

27 checks: leakage columns removed, no other targets in features, identifiers
stripped, **temporal split contains no future information**, preprocessor fitted
on train only, no NaNs post-preprocessing, metrics verified against hand
calculations, best model beats the majority-class baseline, probabilities sum to
1, registry round-trips and promotes, **serving is invariant to column order**,
oversized batches rejected, routing respects capacity and visits every stop,
constrained routing never beats unconstrained, all three metaheuristics beat
greedy, every optimizer beats its baseline, inventory policy is internally
consistent, all 11 tasks build, and auxiliary targets aren't derivable from
their own features.

---

## Dependencies

Required: pandas, numpy, scikit-learn, scipy, pyarrow, PyYAML, matplotlib, joblib.

| Optional | Unlocks | Absent → |
|---|---|---|
| `xgboost` | XGBoost | model skipped, reason reported |
| `lightgbm` | LightGBM | model skipped |
| `catboost` | CatBoost | model skipped |
| `ortools` | VRP/CVRP/VRPTW, CP-SAT | greedy + 2-opt, scipy Hungarian |
| `shap` | SHAP | native TreeSHAP for XGB/LGBM; else skipped |
| `lime` | LIME | in-house weighted-ridge surrogate |
| `umap-learn` | UMAP | skipped (PCA and t-SNE still run) |
| `torch` | MLP/AE/LSTM/Transformer/TabNet | scikit-learn stand-ins, labelled |
| `fastapi`+`uvicorn` | REST API | stdlib HTTP server |

`python -m logisticsml.cli deps` prints the current status.

---

## Layout

```
logistics_ml/
├── configs/default.yaml
├── logisticsml/
│   ├── config.py            dotted-path YAML config
│   ├── data.py              loading + task construction (incl. auxiliary targets)
│   ├── metrics.py           classification and regression metrics
│   ├── utils.py             optional imports, logging, plotting
│   ├── stages/              preprocessing · eda
│   ├── models/              supervised · unsupervised · deep · deep_torch
│   ├── optimization/        routing · metaheuristics · allocation · runner
│   ├── explain/             SHAP · permutation · LIME · PDP
│   ├── serving/             registry · prediction API
│   ├── reporting.py         HTML + Markdown reports
│   ├── pipeline.py          orchestration
│   └── cli.py
└── tests/test_platform.py
```
