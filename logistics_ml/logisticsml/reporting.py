"""Automatic report generation - self-contained HTML plus Markdown.

Plots are embedded as base64 so the HTML file can be emailed or opened from
anywhere without dragging an images folder along.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import dependency_report, get_logger

__all__ = ["build_report"]

logger = get_logger()

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:34px;
background:#f6f7fa;color:#1b2330;line-height:1.55;max-width:1500px}
h1{font-size:27px;margin:0 0 4px}
h2{font-size:20px;margin:36px 0 12px;padding-bottom:7px;border-bottom:2px solid #e2e6ee}
h3{font-size:15px;margin:22px 0 8px;color:#43506b}
.sub{color:#6c7788;font-size:13px;margin-bottom:26px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;
box-shadow:0 1px 3px rgba(20,30,50,.08);border-radius:6px;overflow:hidden;margin-bottom:18px}
th{background:#edf0f6;text-align:left;padding:8px 10px;font-weight:600;white-space:nowrap;
border-bottom:1px solid #dce1ea}
td{padding:6px 10px;border-bottom:1px solid #f0f2f6}
tbody tr:hover{background:#fafbfd}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px}
.card{background:#fff;border-radius:8px;padding:14px 18px;min-width:158px;
box-shadow:0 1px 3px rgba(20,30,50,.08)}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#78839a}
.card .v{font-size:22px;font-weight:600;margin-top:3px}
.best{background:#eef7f1;border-left:4px solid #3fa87a}
img{max-width:100%;border-radius:7px;box-shadow:0 1px 4px rgba(20,30,50,.12);
margin:10px 0 20px;background:#fff}
code{background:#edf0f6;padding:1px 5px;border-radius:3px;font-size:12px}
.note{background:#fff8e6;border-left:4px solid #e0a24c;padding:10px 14px;
border-radius:5px;margin:12px 0;font-size:13px}
.skip{color:#8a94a6;font-size:12.5px}
"""


def _cell(value) -> str:
    """Render any cell value as text.

    Cells are not always scalars - the run summary carries lists (dropped
    columns, feature names) and occasionally dicts. `pd.isna` returns an *array*
    for those, so testing it directly raises "truth value is ambiguous".
    """
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        shown = ", ".join(str(x) for x in items[:6])
        return f"{shown}{f' … (+{len(items) - 6})' if len(items) > 6 else ''}" if items else ""
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in list(value.items())[:6])
    if isinstance(value, np.ndarray):
        return _cell(value.tolist())
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _html_table(df: pd.DataFrame | None, limit: int = 120) -> str:
    if df is None or len(df) == 0:
        return "<p class='skip'><em>no rows</em></p>"
    view = df.head(limit)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell(v))}</td>" for v in row) + "</tr>"
        for row in view.itertuples(index=False))
    extra = (f"<p class='skip'>… {len(df) - limit} more rows</p>" if len(df) > limit else "")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{extra}"


def _md_table(df: pd.DataFrame | None, limit: int = 60) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_\n\n"
    view = df.head(limit)
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(_cell(v).replace("|", "\\|") for v in row) + " |"
        for row in view.itertuples(index=False))
    return f"{header}\n{sep}\n{body}\n\n"


def _embed(path: Path) -> str:
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        return f"<img src='data:image/png;base64,{data}' alt='{html.escape(path.stem)}'>"
    except Exception:
        return ""


def build_report(results: dict, cfg: Config, out_dir: Path) -> list[Path]:
    """Assemble the run report. ``results`` is the PipelineResult as a dict."""
    if not cfg.get("reports.enabled", True):
        return []

    report_dir = Path(out_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    formats = [str(f).lower() for f in cfg.get("reports.formats", ["html", "markdown"])]
    include_plots = bool(cfg.get("reports.include_plots", True))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    written: list[Path] = []

    task = results.get("task", {})
    dataset = results.get("dataset", {})
    split = results.get("split", {})
    supervised = results.get("supervised")
    unsup = results.get("unsupervised")
    deep = results.get("deep")
    opt = results.get("optimization")
    expl = results.get("explainability")
    eda = results.get("eda")

    # ---- headline cards ----------------------------------------------------
    kpis = {
        "Task": task.get("name", "-"),
        "Type": task.get("type", "-"),
        "Rows": f"{dataset.get('n_rows', 0):,}",
        "Features": dataset.get("n_features", "-"),
        "Split": split.get("split_strategy", "-"),
    }
    best_row = None
    if supervised is not None and not supervised["leaderboard"].empty:
        lb = supervised["leaderboard"]
        best_row = lb.iloc[0]
        primary = supervised["primary_metric"]
        kpis["Best model"] = str(best_row["model"])
        if primary in lb.columns:
            kpis[primary] = f"{best_row[primary]:.4f}" if pd.notna(best_row[primary]) else "-"
        kpis["Models trained"] = len(lb)

    # ---- HTML --------------------------------------------------------------
    if "html" in formats:
        cards = "".join(
            f"<div class='card{' best' if k == 'Best model' else ''}'>"
            f"<div class='k'>{html.escape(str(k))}</div>"
            f"<div class='v'>{html.escape(str(v))}</div></div>" for k, v in kpis.items())

        parts = [f"<h1>Logistics Optimization ML Platform</h1>",
                 f"<div class='sub'>Generated {generated} &middot; task "
                 f"<code>{html.escape(str(task.get('name')))}</code> &middot; seed "
                 f"<code>{cfg.seed}</code></div>",
                 f"<div class='cards'>{cards}</div>"]

        if task.get("description"):
            parts.append(f"<div class='note'><strong>Question:</strong> "
                         f"{html.escape(task['description'])}</div>")

        # --- data and EDA
        parts.append("<h2>1. Data and preprocessing</h2>")
        parts.append(_html_table(pd.DataFrame([{**dataset, **{
            k: v for k, v in split.items() if k != 'preprocessor'}}]).T
            .reset_index().rename(columns={"index": "property", 0: "value"})))

        if eda is not None:
            parts.append("<h2>2. Exploratory data analysis</h2>")
            for title, key in [("Target distribution", "target_distribution"),
                               ("Strongest univariate relationships", "target_relationships"),
                               ("Missingness", "missingness"),
                               ("Highly correlated pairs", "high_correlation_pairs")]:
                if key in eda["tables"]:
                    parts.append(f"<h3>{title}</h3>{_html_table(eda['tables'][key], 30)}")
            if include_plots:
                parts.extend(_embed(p) for p in eda["plots"])

        # --- supervised
        if supervised is not None:
            parts.append("<h2>3. Supervised learning</h2>")
            parts.append(_html_table(supervised["leaderboard"]))
            if supervised["skipped"]:
                parts.append("<p class='skip'>Skipped: " + "; ".join(
                    f"{html.escape(k)} ({html.escape(str(v))})"
                    for k, v in supervised["skipped"].items()) + "</p>")
            if supervised.get("confusion") is not None:
                parts.append(f"<h3>Confusion matrix - {best_row['model']}</h3>"
                             + _html_table(supervised["confusion"]))
            if supervised.get("calibration") is not None:
                parts.append(f"<h3>Calibration - {best_row['model']}</h3>"
                             + _html_table(supervised["calibration"]))
            if include_plots:
                parts.extend(_embed(p) for p in results.get("supervised_plots", []))

        # --- unsupervised
        if unsup is not None:
            parts.append("<h2>4. Unsupervised learning</h2>")
            for key in ["kmeans_sweep", "kmeans_sizes", "dbscan", "gmm_sweep",
                        "pca_variance", "pca_loadings", "anomaly_feature_gaps"]:
                if key in unsup["tables"]:
                    parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>"
                                 + _html_table(unsup["tables"][key], 30))
            if include_plots:
                parts.extend(_embed(p) for p in unsup["plots"])

        # --- deep learning
        if deep is not None:
            parts.append("<h2>5. Deep learning</h2>")
            parts.append(f"<div class='note'>Backend: <strong>{html.escape(deep['backend'])}"
                         f"</strong>. " + ("PyTorch models trained natively."
                                           if deep["backend"] == "torch" else
                                           "PyTorch is not installed - scikit-learn stand-ins "
                                           "were used. Rows marked APPROXIMATION are not "
                                           "architecturally equivalent.") + "</div>")
            parts.append(_html_table(deep["leaderboard"]))
            if include_plots:
                parts.extend(_embed(p) for p in deep["plots"])

        # --- optimization
        if opt is not None:
            parts.append("<h2>6. Mathematical optimization</h2>")
            for key in ["vrp_variants", "routing_comparison", "metaheuristics"]:
                if key in opt["tables"]:
                    parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>"
                                 + _html_table(opt["tables"][key], 20))
            if not opt["leaderboard"].empty:
                parts.append("<h3>Optimisation problems</h3>" + _html_table(opt["leaderboard"]))
            for key in ["driver_scheduling_coverage", "fleet_allocation_assignments",
                        "warehouse_allocation_opened_sites", "inventory_by_category"]:
                if key in opt["tables"]:
                    parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>"
                                 + _html_table(opt["tables"][key], 15))
            if include_plots:
                parts.extend(_embed(p) for p in opt["plots"])

        # --- explainability
        if expl is not None:
            parts.append("<h2>7. Explainable AI</h2>")
            for key in ["shap_global", "permutation_importance", "importance_agreement",
                        "shap_local", "lime_local"]:
                if key in expl["tables"]:
                    parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>"
                                 + _html_table(expl["tables"][key], 25))
            if expl.get("summary"):
                parts.append(_html_table(pd.DataFrame([expl["summary"]]).T.reset_index()
                                         .rename(columns={"index": "property", 0: "value"})))
            if include_plots:
                parts.extend(_embed(p) for p in expl["plots"])

        # --- registry and environment
        if results.get("registry") is not None:
            parts.append("<h2>8. Model registry</h2>")
            parts.append(_html_table(results["registry"]))

        parts.append("<h2>Environment</h2>")
        parts.append(_html_table(dependency_report()))

        doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>Logistics ML Platform report</title><style>{_CSS}</style>"
               f"</head><body>{''.join(parts)}</body></html>")
        path = report_dir / f"report_{task.get('name', 'run')}.html"
        path.write_text(doc, encoding="utf-8")
        written.append(path)

    # ---- Markdown ----------------------------------------------------------
    if "markdown" in formats:
        md = [f"# Logistics Optimization ML Platform\n\n",
              f"_Generated {generated} · task `{task.get('name')}` · seed `{cfg.seed}`_\n\n",
              "## Summary\n\n",
              "| Metric | Value |\n| --- | --- |\n"
              + "".join(f"| {k} | {v} |\n" for k, v in kpis.items()) + "\n"]

        if supervised is not None:
            md += ["## Supervised leaderboard\n\n", _md_table(supervised["leaderboard"])]
            if supervised["skipped"]:
                md.append("**Skipped:** " + "; ".join(
                    f"{k} ({v})" for k, v in supervised["skipped"].items()) + "\n\n")
        if deep is not None:
            md += [f"## Deep learning (backend: {deep['backend']})\n\n",
                   _md_table(deep["leaderboard"])]
        if unsup is not None:
            md.append("## Unsupervised\n\n")
            for key in ["kmeans_sweep", "pca_variance"]:
                if key in unsup["tables"]:
                    md += [f"### {key}\n\n", _md_table(unsup["tables"][key], 15)]
        if opt is not None:
            md.append("## Optimization\n\n")
            for key in ["vrp_variants", "routing_comparison", "metaheuristics"]:
                if key in opt["tables"]:
                    md += [f"### {key}\n\n", _md_table(opt["tables"][key], 15)]
            if not opt["leaderboard"].empty:
                md += ["### Problems solved\n\n", _md_table(opt["leaderboard"])]
        if expl is not None and "shap_global" in expl["tables"]:
            md += ["## Explainability - SHAP global\n\n",
                   _md_table(expl["tables"]["shap_global"], 20)]

        md += ["## Environment\n\n", _md_table(dependency_report())]
        path = report_dir / f"report_{task.get('name', 'run')}.md"
        path.write_text("".join(md), encoding="utf-8")
        written.append(path)

    logger.info("Reports written: %s", ", ".join(p.name for p in written))
    return written
