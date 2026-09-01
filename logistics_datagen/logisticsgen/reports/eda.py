"""Reporting: EDA summary, missingness report and data dictionary.

Written in Markdown and self-contained HTML - no plotting dependency, no
notebook required. The HTML uses inline CSS bar cells for the missingness
heatmap so the report opens anywhere.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger

__all__ = ["build_reports", "profile_tables", "data_dictionary"]

logger = get_logger()


# --------------------------------------------------------------------------- #
def profile_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per table: size, memory, missing share, duplicate keys."""
    rows = []
    for name, df in tables.items():
        if df is None:
            continue
        cells = df.size or 1
        key = df.columns[0] if len(df.columns) else None
        rows.append({
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "missing_cells_pct": round(float(df.isna().sum().sum()) / cells * 100, 3),
            "columns_with_missing": int((df.isna().sum() > 0).sum()),
            "duplicate_key_rows": int(df[key].duplicated().sum()) if key is not None else 0,
            "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
            "datetime_columns": int(df.select_dtypes(include=["datetime64[ns]"]).shape[1]),
        })
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def column_profile(df: pd.DataFrame, table: str, max_cols: int = 400) -> pd.DataFrame:
    """Per-column statistics: dtype, missing, cardinality, distribution."""
    rows = []
    for col in df.columns[:max_cols]:
        s = df[col]
        entry = {
            "table": table, "column": col, "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(float(s.isna().mean()) * 100, 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            entry.update({
                "mean": round(float(desc.get("mean", np.nan)), 4),
                "std": round(float(desc.get("std", np.nan)), 4),
                "min": round(float(desc.get("min", np.nan)), 4),
                "p25": round(float(desc.get("25%", np.nan)), 4),
                "median": round(float(desc.get("50%", np.nan)), 4),
                "p75": round(float(desc.get("75%", np.nan)), 4),
                "max": round(float(desc.get("max", np.nan)), 4),
                "skew": round(float(s.skew()), 4) if s.notna().sum() > 2 else np.nan,
            })
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry.update({"min": str(s.min()), "max": str(s.max())})
        else:
            top = s.value_counts(dropna=True).head(3)
            entry["top_values"] = "; ".join(f"{k}={v}" for k, v in top.items())
        rows.append(entry)
    return pd.DataFrame(rows)


def data_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [column_profile(df, name) for name, df in tables.items()
              if df is not None and not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, limit: int = 60) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    view = df.head(limit)
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in view.itertuples(index=False))
    extra = f"\n\n_… {len(df) - limit} more rows_\n" if len(df) > limit else "\n"
    return f"{header}\n{sep}\n{body}\n{extra}"


def _html_table(df: pd.DataFrame, limit: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>no rows</em></p>"
    view = df.head(limit)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{'' if pd.isna(v) else html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in view.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:32px;
background:#f7f8fa;color:#1c2430;line-height:1.55}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;
padding-bottom:6px;border-bottom:2px solid #e3e7ee}
h3{font-size:15px;margin:22px 0 8px;color:#44506b}
.sub{color:#6b7686;font-size:13px;margin-bottom:24px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;
box-shadow:0 1px 3px rgba(20,30,50,.08);border-radius:6px;overflow:hidden;margin-bottom:18px}
th{background:#eef1f6;text-align:left;padding:8px 10px;font-weight:600;
white-space:nowrap;border-bottom:1px solid #dde2ea}
td{padding:6px 10px;border-bottom:1px solid #f0f2f6}
tbody tr:hover{background:#fafbfd}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px}
.card{background:#fff;border-radius:8px;padding:14px 18px;min-width:150px;
box-shadow:0 1px 3px rgba(20,30,50,.08)}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#78839a}
.card .v{font-size:22px;font-weight:600;margin-top:3px}
.bar{height:9px;background:#e6eaf1;border-radius:5px;overflow:hidden;min-width:70px}
.bar>span{display:block;height:100%;background:linear-gradient(90deg,#4c7ef3,#7aa2f7)}
code{background:#eef1f6;padding:1px 5px;border-radius:3px;font-size:12px}
"""


def _missingness_bars(profile: pd.DataFrame) -> str:
    if profile.empty:
        return "<p><em>no missingness</em></p>"
    view = profile.loc[profile["pct_missing"] > 0].sort_values("pct_missing", ascending=False).head(60)
    if view.empty:
        return "<p><em>no missing values present</em></p>"
    rows = "".join(
        f"<tr><td>{html.escape(r.table)}</td><td>{html.escape(r.column)}</td>"
        f"<td>{r.pct_missing:.2f}%</td>"
        f"<td><div class='bar'><span style='width:{min(r.pct_missing, 100):.1f}%'></span></div></td></tr>"
        for r in view.itertuples())
    return ("<table><thead><tr><th>Table</th><th>Column</th><th>Missing</th>"
            f"<th style='width:40%'>&nbsp;</th></tr></thead><tbody>{rows}</tbody></table>")


# --------------------------------------------------------------------------- #
def build_reports(
    tables: dict[str, pd.DataFrame],
    cfg: Config,
    out_dir: Path,
    missingness_report: pd.DataFrame | None = None,
    anomaly_report: pd.DataFrame | None = None,
    sweep: dict[float, pd.DataFrame] | None = None,
) -> list[Path]:
    r = cfg.get("reports", {}) or {}
    report_dir = Path(out_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    formats = [str(f).lower() for f in r.get("format", ["markdown", "html"])]
    written: list[Path] = []

    overview = profile_tables(tables)
    dictionary = data_dictionary(tables) if r.get("data_dictionary", True) else pd.DataFrame()

    total_rows = int(overview["rows"].sum())
    total_cells = sum((df.size for df in tables.values() if df is not None))
    total_missing = sum((int(df.isna().sum().sum()) for df in tables.values() if df is not None))
    orders = tables.get("orders", pd.DataFrame())

    kpis = {
        "Tables": len(overview),
        "Total rows": f"{total_rows:,}",
        "Total columns": int(overview["columns"].sum()),
        "Memory (MB)": f"{overview['memory_mb'].sum():,.1f}",
        "Missing cells": f"{(total_missing / max(total_cells, 1) * 100):.2f}%",
    }
    if not orders.empty:
        kpis["Orders"] = f"{len(orders):,}"
        kpis["On-time rate"] = f"{(1 - orders['is_late'].mean()) * 100:.1f}%"
        kpis["Avg delivery cost"] = f"${orders['delivery_cost_usd'].mean():.2f}"
        kpis["Avg distance"] = f"{orders['distance_km'].mean():.1f} km"

    # ---- business slices ----------------------------------------------------
    slices: dict[str, pd.DataFrame] = {}
    if not orders.empty:
        slices["Volume and performance by business domain"] = (
            orders.groupby("business_domain")
            .agg(orders=("order_id", "count"),
                 on_time_pct=("is_late", lambda s: round((1 - s.mean()) * 100, 2)),
                 avg_delay_min=("delay_minutes", lambda s: round(s.mean(), 2)),
                 avg_cost_usd=("delivery_cost_usd", lambda s: round(s.mean(), 2)),
                 avg_distance_km=("distance_km", lambda s: round(s.mean(), 2)))
            .reset_index().sort_values("orders", ascending=False))
        slices["Performance by region"] = (
            orders.groupby("region")
            .agg(orders=("order_id", "count"),
                 on_time_pct=("is_late", lambda s: round((1 - s.mean()) * 100, 2)),
                 avg_cost_usd=("delivery_cost_usd", lambda s: round(s.mean(), 2)),
                 total_revenue_usd=("revenue_usd", lambda s: round(s.sum(), 2)))
            .reset_index().sort_values("orders", ascending=False))
        slices["Order status mix"] = (
            orders["status"].value_counts().rename_axis("status")
            .reset_index(name="orders")
            .assign(share_pct=lambda d: (d["orders"] / len(orders) * 100).round(2)))
        slices["Impact of weather on delay"] = (
            orders.groupby("weather_condition")
            .agg(orders=("order_id", "count"),
                 avg_delay_min=("delay_minutes", lambda s: round(s.mean(), 2)),
                 late_pct=("is_late", lambda s: round(s.mean() * 100, 2)))
            .reset_index().sort_values("avg_delay_min", ascending=False))
        slices["Impact of traffic on delay"] = (
            orders.groupby("traffic_level")
            .agg(orders=("order_id", "count"),
                 avg_delay_min=("delay_minutes", lambda s: round(s.mean(), 2)),
                 late_pct=("is_late", lambda s: round(s.mean() * 100, 2)))
            .reset_index().sort_values("avg_delay_min", ascending=False))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- markdown -----------------------------------------------------------
    if "markdown" in formats:
        parts = [
            f"# Logistics Dataset - EDA Report\n",
            f"_Generated {generated} · seed `{cfg.seed}` · scale `{cfg.scale}`_\n",
            "## Headline numbers\n",
            "| Metric | Value |\n| --- | --- |\n"
            + "".join(f"| {k} | {v} |\n" for k, v in kpis.items()),
            "\n## Table overview\n", _md_table(overview),
        ]
        for title, frame in slices.items():
            parts += [f"\n## {title}\n", _md_table(frame)]
        if missingness_report is not None and not missingness_report.empty:
            parts += ["\n## Missingness rules applied\n", _md_table(missingness_report)]
        if anomaly_report is not None and not anomaly_report.empty:
            parts += ["\n## Anomalies injected\n", _md_table(anomaly_report)]
        if sweep:
            parts.append("\n## Missingness rate sweep\n")
            for rate, frame in sorted(sweep.items()):
                parts += [f"\n### Target rate {rate:.0%}\n", _md_table(frame, limit=25)]
        if not dictionary.empty:
            parts += ["\n## Data dictionary\n", _md_table(dictionary, limit=1200)]

        path = report_dir / "eda_report.md"
        path.write_text("".join(parts), encoding="utf-8")
        written.append(path)

    # ---- html ---------------------------------------------------------------
    if "html" in formats:
        cards = "".join(
            f"<div class='card'><div class='k'>{html.escape(k)}</div>"
            f"<div class='v'>{html.escape(str(v))}</div></div>" for k, v in kpis.items())
        sections = [f"<h2>Table overview</h2>{_html_table(overview)}"]
        for title, frame in slices.items():
            sections.append(f"<h2>{html.escape(title)}</h2>{_html_table(frame)}")
        sections.append(f"<h2>Missingness by column</h2>{_missingness_bars(dictionary)}")
        if missingness_report is not None and not missingness_report.empty:
            sections.append(f"<h2>Missingness rules applied</h2>{_html_table(missingness_report)}")
        if anomaly_report is not None and not anomaly_report.empty:
            sections.append(f"<h2>Anomalies injected</h2>{_html_table(anomaly_report)}")
        if sweep:
            blocks = "".join(
                f"<h3>Target rate {rate:.0%}</h3>{_html_table(frame, limit=30)}"
                for rate, frame in sorted(sweep.items()))
            sections.append(f"<h2>Missingness rate sweep</h2>{blocks}")
        if not dictionary.empty:
            sections.append(f"<h2>Data dictionary</h2>{_html_table(dictionary, limit=1500)}")

        doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>Logistics Dataset - EDA Report</title><style>{_CSS}</style></head><body>"
               f"<h1>Logistics Dataset - EDA Report</h1>"
               f"<div class='sub'>Generated {generated} &middot; seed <code>{cfg.seed}</code>"
               f" &middot; scale <code>{cfg.scale}</code></div>"
               f"<div class='cards'>{cards}</div>{''.join(sections)}</body></html>")
        path = report_dir / "eda_report.html"
        path.write_text(doc, encoding="utf-8")
        written.append(path)

    # ---- machine-readable side artefacts ------------------------------------
    overview.to_csv(report_dir / "table_overview.csv", index=False)
    written.append(report_dir / "table_overview.csv")
    if not dictionary.empty:
        dictionary.to_csv(report_dir / "data_dictionary.csv", index=False)
        written.append(report_dir / "data_dictionary.csv")
    if missingness_report is not None and not missingness_report.empty:
        missingness_report.to_csv(report_dir / "missingness_report.csv", index=False)
        written.append(report_dir / "missingness_report.csv")
    if anomaly_report is not None and not anomaly_report.empty:
        anomaly_report.to_csv(report_dir / "anomaly_report.csv", index=False)
        written.append(report_dir / "anomaly_report.csv")
    if sweep:
        frames = [f.assign(sweep_rate=rate) for rate, f in sweep.items() if not f.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                report_dir / "missingness_sweep.csv", index=False)
            written.append(report_dir / "missingness_sweep.csv")

    logger.info("Reports: %d files → %s", len(written), report_dir)
    return written
