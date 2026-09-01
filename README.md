# Logistics Optimization Portfolio

An end-to-end, industry-agnostic logistics optimization portfolio: a causally
consistent synthetic dataset, a config-driven ML platform, a self-contained
analytics dashboard and an executive pitch deck — each stage consuming the
real output of the one before it, so every number downstream traces back to
an artifact, not an assumption.

Built from four sequential prompts (see [`Prompts/`](Prompts)) and adaptable
to courier services, e-commerce, retail, manufacturing, healthcare logistics
and ride-hailing.

## Pipeline

```
01  logistics_datagen        → synthetic dataset (20 tables, causally consistent)
        │  data/*/parquet, sqlite, ground_truth, reports
        ▼
02  logistics_ml              → ML platform (11 tasks: supervised, unsupervised,
        │                        deep learning, optimization, explainable AI)
        │  artifacts/run_summary.json, artifacts/tables/*.csv, registry/
        ▼
03  logistics_dashboard       → interactive HTML dashboard (9 modules)
        │  dashboard.html (self-contained, double-click to open)
        ▼
04  logistics_presentation    → 39-slide executive pitch deck
           Logistics_Optimization_Presentation.pdf / .pptx
```

Each project's own README documents it in full — this file is the map.

## Projects

| Project | What it is | Deliverable |
|---|---|---|
| [`logistics_datagen`](logistics_datagen/README.md) | Synthetic logistics dataset generator — 20 related tables, configurable missingness (MCAR/MAR/MNAR), 15 anomaly classes, leakage-aware ML features | `data/{sample,full}/` (csv, parquet, sqlite, reports) |
| [`logistics_ml`](logistics_ml/README.md) | Config-driven ML platform — preprocessing, EDA, 9 supervised models, unsupervised, deep learning, OR-Tools optimization, SHAP/LIME explainability, model registry + serving API | `artifacts/` (reports, plots, tables, registry) |
| [`logistics_dashboard`](logistics_dashboard/README.md) | Single-file SaaS-style dashboard — executive KPIs, operations, fleet, warehouse, delivery map, forecast, optimization simulator, AI recommendations, explainable AI | `dashboard.html` |
| [`logistics_presentation`](logistics_presentation/README.md) | Enterprise pitch deck — business case, architecture, ROI, every figure sourced from the artifacts above | `Logistics_Optimization_Presentation.pdf` / `.pptx` |

`Prompts/` holds the four brief documents each project was built from.

## Quick start (full pipeline)

```bash
# 1. Generate the dataset
cd logistics_datagen
pip install -r requirements.txt
python -m logisticsgen.cli --config configs/default.yaml --config configs/sample.yaml

# 2. Run the ML platform against it
cd ../logistics_ml
pip install -r requirements.txt
python -m logisticsml.cli run

# 3. Rebuild the dashboard payload from fresh data + artifacts
cd ../logistics_dashboard
python scripts/build_payload.py \
  --data ../logistics_datagen/data/sample/parquet \
  --artifacts ../logistics_ml/artifacts
python scripts/build_dashboard.py
open dashboard.html

# 4. Rebuild the pitch deck (pulls figures from all three)
cd ../logistics_presentation
npm install
node build_deck.js
```

Each stage can also be opened and used independently — the dataset ships
pre-generated, `dashboard.html` opens with no server, and the PDF deck opens
and presents directly. Rebuilding is only needed after changing an upstream
input (e.g. regenerating the dataset at full scale, or adding a model).

## Headline results

Numbers below are from the shipped 50,150-order sample run; see the
[`logistics_ml`](logistics_ml/README.md) and
[`logistics_dashboard`](logistics_dashboard/README.md) READMEs for full
detail, caveats and honesty notes on what does and doesn't have signal.

- **Late-delivery prediction (at dispatch):** ROC-AUC 0.998 (XGBoost/LightGBM) —
  0.960 pre-dispatch, without the dispatch-time feature.
- **Fraud detection:** PR-AUC 0.454 against a 0.08% base rate — 567× lift.
- **Optimization vs. baseline:** driver scheduling −31.4% cost, fleet
  allocation −70.9% cost, inventory (EOQ) −10.3% cost.
- **Routing:** simulated annealing lands within 0.7% of the exact OR-Tools
  solution in 1/90th of the time.
- **Dataset causal consistency:** adverse weather and heavy traffic each
  measurably raise the late rate (13.0% / 12.8%) over clear/free-flow
  conditions (11.0% / 10.4%) — delay is a genuine downstream effect, not
  independently sampled noise.
- **Dashboard reconciliation:** displayed KPIs, the aggregation payload and
  the source Parquet agree exactly (50,150 orders, 88.73% on-time, $9.36M
  revenue) — verified by a 43-check test harness.

## Repository layout

```
Logistics/
├── README.md                    ← this file
├── Prompts/                      the four project briefs
│   ├── 01_Logistics_Optimization_Dataset_Generator.md
│   ├── 02_Logistics_Optimization_ML_Platform.md
│   ├── 03_Logistics_Optimization_Dashboard.md
│   └── 04_Logistics_Optimization_Presentation.md
├── logistics_datagen/            01 — dataset generator (see its README)
├── logistics_ml/                 02 — ML platform (see its README)
├── logistics_dashboard/          03 — HTML dashboard (see its README)
└── logistics_presentation/       04 — pitch deck (see its README)
```

## Validation

Every stage ships its own test suite, and each is designed to fail loudly
rather than paper over a gap:

```bash
python logistics_datagen/tests/test_generator.py     # 26 checks
python logistics_ml/tests/test_platform.py            # 27 checks
node logistics_dashboard/tests/test_dashboard.js       # 43 checks
```

Together they cover referential integrity and reproducibility in the data,
leakage-free and temporally sound modeling, and end-to-end reconciliation
between the dashboard's displayed figures and the source data.
