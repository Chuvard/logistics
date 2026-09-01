# Logistics Optimization — Pitch Deck

A 39-slide PDF/PPTX presentation (prompt 04) covering the full portfolio project:
`logistics_datagen` (01), `logistics_ml` (02) and `logistics_dashboard` (03).

**Deliverable:** `Logistics_Optimization_Presentation.pdf` — open and present directly.
`Logistics_Optimization_Presentation.pptx` is the editable source.

---

## What's in it

19 requested sections across six acts (Why This Matters → The Data Foundation →
The Intelligence Layer → The Product → Business Impact → Getting to Production),
plus a title, agenda and Q&A slide.

Every figure on every slide traces back to a real artifact from the other three
projects — the ML platform's `run_summary.json` and `artifacts/tables/*.csv`, the
dataset generator's `README.md` and `reports/`, and the dashboard's
`data/dashboard_payload.json`. Nothing is invented; where a number is an estimate
(market sizing, ROI projections), it is explicitly labelled illustrative and sourced.

The one required disclaimer from the brief — that performance and ROI figures are
illustrative estimates dependent on implementation and data quality — is stated
explicitly on its own slide (ROI, slide 33), not buried in a footnote.

## Design

Custom "Midnight Executive" palette (navy / amber / teal) built for this topic,
Cambria headlines over Calibri body text, a consistent icon-in-circle motif
(Lucide icons, rendered to PNG), dark divider slides between acts, light content
slides. Charts are a mix of real output from the ML pipeline
(`logistics_ml/artifacts/plots/`) and purpose-built matplotlib charts styled to
match the deck.

## Rebuilding

```bash
npm install                    # pptxgenjs, react-icons, react, sharp
python3 make_icons.js || node make_icons.js   # icon set → icons/
python3 make_charts.py                         # custom charts → charts/
python3 make_mockup.py                         # dashboard preview composite
node build_deck.js                             # → Logistics_Optimization_Presentation.pptx
python3 scripts/office/soffice.py --headless --convert-to pdf Logistics_Optimization_Presentation.pptx
```

`scripts/` (LibreOffice/validation helpers) comes from the pptx skill and isn't
duplicated here; any `soffice --headless --convert-to pdf` works the same way.

## Layout

```
logistics_presentation/
├── Logistics_Optimization_Presentation.pdf   ← the deliverable
├── Logistics_Optimization_Presentation.pptx  ← editable source
├── build_deck.js        slide-by-slide deck assembly (pptxgenjs)
├── make_icons.js         Lucide icon set → PNG, 45 icons × 6 colors
├── make_charts.py        8 custom matplotlib charts, deck-matched palette
├── make_mockup.py        dashboard "screenshot" composite from real payload data
├── icons/                 generated icon PNGs
└── charts/                generated chart PNGs
```
