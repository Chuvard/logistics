# Logistics Optimization — HTML Dashboard

A SaaS-style analytics application covering nine modules over the dataset from
`logistics_datagen` (prompt 01) and the model artifacts from `logistics_ml`
(prompt 02).

**One self-contained file.** `dashboard.html` opens by double-click — no server,
no build step, no install. The data travels inside the page.

---

## Quick start

```bash
# already built and ready to open:
open dashboard.html

# to refresh after regenerating the dataset or re-running the ML platform:
python scripts/build_payload.py       # re-aggregate the data
python scripts/build_dashboard.py     # re-embed it into the page
node tests/test_dashboard.js          # 43 validation checks
```

---

## The nine modules

| Module | What it answers |
|---|---|
| **Executive Dashboard** | Revenue, cost, margin and on-time rate with month-over-month movement; domain and regional scorecards |
| **Operations Center** | Weekday × hour demand heatmap, hourly reliability, congestion profile, traffic/weather impact, logged exceptions |
| **Fleet Management** | Composition, age profile, energy mix, route capacity utilisation, maintenance cost vs downtime, courier league table |
| **Warehouse Analytics** | Peak-day throughput against rated capacity, automation levels, inventory position and stockout exposure |
| **Delivery Monitoring** | Interactive map with seven toggleable layers, SLA outcome by priority, distance mix, daily reliability |
| **Demand Forecast** | 30-day projection with a 95% interval, weekly and annual seasonality decomposition |
| **Optimization Simulator** | Eight what-if levers with live cost, service-level and margin recomputation |
| **AI Recommendations** | Nine ranked, quantified actions — each stating the evidence it came from |
| **Explainable AI** | SHAP global and per-instance attribution, model leaderboard, SHAP-vs-permutation agreement, partial dependence, model card |

---

## Things worth knowing

**The filters actually filter.** Pre-aggregated dashboards usually ship filter
controls that change a label but not a number. The payload carries a
region × domain × month fact cube (432 rows), so selecting a region genuinely
re-aggregates the executive figures in the browser. A banner states plainly
which modules respond and which are network-wide — better than a control that
silently does nothing.

**The simulator is calibrated, not invented.** Congestion and weather
elasticities are *measured from the dataset*: the gap in late rate between the
top and bottom congestion quartiles, and between adverse and clear conditions.
Cost shares come from the order book. The structural assumptions — square-root
warehouse-density law, 4% fixed overhead per site, +55% overtime rate once
demand outruns capacity, 18% cap on routing savings — are listed in the module
itself so they can be challenged rather than taken on trust.

Vehicles and drivers are modelled as near-complementary (Cobb-Douglas weighted
85/15 toward whichever is scarcer). Raising fleet alone while drivers bind gives
a small benefit, not zero and not full — and the panel names the binding
constraint, so a barely-moving slider is explained rather than mysterious.

**Every number reconciles.** The validation harness checks displayed KPIs
against the payload, and the payload against the source Parquet. 50,150 orders,
88.73% on-time, $9.36M revenue — identical at all three layers.

**Forecast accuracy is reported out-of-sample.** The decomposition is fitted
without the final 30 days and scored on them: **16.97% MAPE**, against 7.04%
in-sample. Quoting the in-sample figure would have been flattering and
meaningless. The recovered weekday factors match the generator's configuration
almost exactly (Sunday 0.619 against a configured 0.62), which is a useful
independent check that the pipeline is wired up correctly.

**Honest empty states.** If the ML artifacts are missing, the Explainable AI
module says so rather than rendering a plausible-looking placeholder.

---

## Technology

Tailwind for the shell, Plotly for charts, Leaflet with OpenStreetMap/CARTO
tiles for maps, D3 available for custom visuals, SheetJS for Excel export and
jsPDF for PDF generation — all from CDN.

**On Mapbox:** the brief lists it, but it requires a paid API token, so the map
would render blank for anyone opening the file without one. Leaflet with free
tiles works immediately for everyone. Swapping in a Mapbox basemap is a
one-line change in `initMap()` if you have a token.

**On Bootstrap + Tailwind:** the brief lists both. They are competing utility
systems and loading both produces specificity conflicts, so this uses Tailwind
alone.

---

## Exports

- **Excel** — 11-sheet workbook: filtered summary and cube slices, plus
  warehouses, fleet, forecast, recommendations, optimization results, the model
  leaderboard and SHAP values. Real `.xlsx` via SheetJS.
- **PDF** — formatted executive report with headline metrics and the top
  recommendations, generated client-side by jsPDF, with an on-screen preview.
- **PowerPoint** — 16:9 slide preview of the current view.

---

## Validation

```bash
node tests/test_dashboard.js
```

43 checks in a headless DOM with the charting libraries stubbed:

- all nine modules render, twice over, with zero console errors
- 36 charts drawn, none containing NaN or infinite values
- every `getElementById` in the app resolves against the markup (29 ids)
- every sized chart container is either filled or shows an explicit empty state
- displayed KPIs reconcile against the payload, and the payload against source
- filters re-aggregate to the correct totals; 12 region × domain combinations clean
- map builds 7 layers, 2,057 markers, 40 route polylines, all coordinates in range
- simulator baseline reproduces the real network (88.73% on-time), responds
  directionally correctly to all six levers, and stays bounded under extremes
- forecast intervals bracket the point estimate and widen with horizon
- SHAP values sorted, model metrics in [0,1], per-instance switching works
- all three export paths execute; the workbook has 11 populated sheets
- theme toggle and table sorting work

The harness found two genuine bugs during development: the simulator's original
pure-`min()` capacity model made the fleet slider completely dead whenever
drivers bound, and an earlier forecast omitted monthly seasonality (47% MAPE,
now 17%).

> **Not verified:** pixel-level visual rendering. The Chrome extension wasn't
> connected during the build, so charts were validated through a stubbed DOM
> rather than a real browser. Data, logic, interaction and error-freeness are
> covered; visual polish is worth a look on your machine.

---

## Layout

```
logistics_dashboard/
├── dashboard.html              ← the deliverable (0.72 MB, self-contained)
├── src/
│   ├── template.html           shell, styles, module markup
│   └── app.js                  application logic (~66 KB)
├── scripts/
│   ├── build_payload.py        dataset + artifacts → compact JSON
│   └── build_dashboard.py      template + js + payload → dashboard.html
├── data/dashboard_payload.json 0.63 MB aggregated payload
└── tests/test_dashboard.js     43-check validation harness
```

The payload is rebuilt from source data, so regenerating the dataset at a
different scale or re-running the ML pipeline flows straight through:

```bash
python scripts/build_payload.py \
  --data ../logistics_datagen/data/full/parquet \
  --artifacts ../logistics_ml/artifacts
python scripts/build_dashboard.py
```
