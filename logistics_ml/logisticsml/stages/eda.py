"""Stage 2 - Exploratory Data Analysis.

Produces the numbers and plots you would want before trusting any model:
missingness, distributions, target balance, correlation structure, and how each
candidate feature actually relates to the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import Dataset
from ..utils import get_logger, new_figure, save_figure, slugify

__all__ = ["run_eda", "EDAResult"]

logger = get_logger()


class EDAResult:
    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {}
        self.plots: list[Path] = []
        self.summary: dict = {}


def _numeric_profile(X: pd.DataFrame) -> pd.DataFrame:
    num = X.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    prof = num.describe().T
    prof["missing_pct"] = (num.isna().mean() * 100).round(3)
    prof["skew"] = num.skew().round(3)
    prof["kurtosis"] = num.kurtosis().round(3)
    prof["n_unique"] = num.nunique()
    return prof.round(4).reset_index().rename(columns={"index":"feature"})


def _categorical_profile(X: pd.DataFrame) -> pd.DataFrame:
    cat = X.select_dtypes(include=["object", "category", "bool"])
    rows = []
    for col in cat.columns:
        vc = cat[col].value_counts(dropna=True)
        rows.append({
            "feature": col,
            "n_unique": int(cat[col].nunique(dropna=True)),
            "missing_pct": round(float(cat[col].isna().mean() * 100), 3),
            "top_value": str(vc.index[0]) if len(vc) else "",
            "top_share_pct": round(float(vc.iloc[0] / len(cat) * 100), 2) if len(vc) else 0.0,
        })
    return pd.DataFrame(rows)


def _target_relationships(ds: Dataset, top_n: int) -> pd.DataFrame:
    """Rank features by univariate association with the target.

    Numeric features use Pearson correlation (regression) or point-biserial via
    standardised mean difference (classification). Categorical features use the
    spread of target means across levels. Crude but effective for a first look,
    and it needs no model.
    """
    y = ds.y
    is_clf = ds.task_type in {"binary", "multiclass"}
    y_num = pd.factorize(y)[0].astype(float) if y.dtype == object else y.astype(float)

    rows = []
    for col in ds.X.columns:
        s = ds.X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        try:
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                v = s.astype(float)
                if v.nunique(dropna=True) < 2:
                    continue
                corr = float(np.corrcoef(
                    v.fillna(v.median()), y_num)[0, 1])
                rows.append({"feature": col, "kind": "numeric",
                             "association": abs(corr), "signed": round(corr, 4)})
            else:
                if s.nunique(dropna=True) < 2 or s.nunique(dropna=True) > 60:
                    continue
                means = pd.DataFrame({"g": s.astype(str), "y": y_num}).groupby("g")["y"].mean()
                spread = float(means.max() - means.min())
                denom = float(np.nanstd(y_num)) or 1.0
                rows.append({"feature": col, "kind": "categorical",
                             "association": spread / denom, "signed": np.nan})
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("association", ascending=False).head(top_n).reset_index(drop=True)
    out["association"] = out["association"].round(4)
    return out


def run_eda(ds: Dataset, cfg: Config, out_dir: Path) -> EDAResult:
    result = EDAResult()
    if not cfg.get("eda.enabled", True):
        logger.info("EDA disabled by config")
        return result

    plot_dir = out_dir / "plots" / "eda"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("eda.dpi", 110))
    make_plots = bool(cfg.get("reports.include_plots", True))

    X, y = ds.X, ds.y

    # ---- profiles ----------------------------------------------------------
    result.tables["numeric_profile"] = _numeric_profile(X)
    result.tables["categorical_profile"] = _categorical_profile(X)

    missing = (X.isna().mean() * 100).sort_values(ascending=False)
    missing = missing.loc[missing > 0].round(3)
    result.tables["missingness"] = pd.DataFrame({
        "feature": missing.index.astype(str), "missing_pct": missing.to_numpy()})

    # ---- target ------------------------------------------------------------
    if ds.task_type in {"binary", "multiclass"}:
        counts = y.value_counts()
        target_tbl = pd.DataFrame({
            "class": counts.index.astype(str),
            "count": counts.to_numpy(),
            "share_pct": (counts / len(y) * 100).round(3).to_numpy(),
        })
        result.summary["class_balance"] = dict(zip(target_tbl["class"], target_tbl["share_pct"]))
        result.summary["imbalance_ratio"] = round(
            float(counts.max() / max(counts.min(), 1)), 2)
    else:
        target_tbl = y.describe().round(4).reset_index()
        target_tbl.columns = ["statistic", "value"]
        result.summary["target_mean"] = round(float(y.mean()), 4)
        result.summary["target_std"] = round(float(y.std()), 4)
    result.tables["target_distribution"] = target_tbl

    # ---- associations ------------------------------------------------------
    rel = _target_relationships(ds, int(cfg.get("eda.target_relationship_top_n", 12)))
    result.tables["target_relationships"] = rel

    num = X.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        top_cols = (num.std().sort_values(ascending=False)
                    .head(int(cfg.get("eda.correlation_top_n", 30))).index)
        corr = num[top_cols].corr().round(4)
        result.tables["correlation_matrix"] = corr.reset_index().rename(columns={"index":"feature"})

        # Flag redundant pairs - these bloat models and destabilise coefficients.
        cm = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = (cm.stack().rename("abs_corr").reset_index()
                 .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
        result.tables["high_correlation_pairs"] = (
            pairs.loc[pairs["abs_corr"] > 0.9].sort_values("abs_corr", ascending=False)
            .head(30).reset_index(drop=True))
    else:
        corr = pd.DataFrame()

    result.summary.update({
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric": int(num.shape[1]),
        "n_categorical": int(X.shape[1] - num.shape[1]),
        "total_missing_pct": round(float(X.isna().mean().mean() * 100), 3),
        "duplicate_rows": int(X.duplicated().sum()),
    })

    # ---- plots -------------------------------------------------------------
    if make_plots:
        try:
            result.plots.extend(_make_plots(ds, corr, rel, plot_dir, cfg, dpi))
        except Exception as exc:  # plotting must never break the pipeline
            logger.warning("EDA plotting failed: %s", exc)

    logger.info("EDA: %d tables, %d plots", len(result.tables), len(result.plots))
    return result


def _make_plots(ds, corr, rel, plot_dir: Path, cfg: Config, dpi: int) -> list[Path]:
    plots: list[Path] = []
    X, y = ds.X, ds.y

    # Target distribution
    fig, ax = new_figure(7, 4)
    if ds.task_type in {"binary", "multiclass"}:
        vc = y.value_counts().sort_index()
        ax.bar([str(i) for i in vc.index], vc.to_numpy(), color="#4c7ef3")
        ax.set_ylabel("count")
        ax.set_title(f"Target distribution - {ds.task['target']}")
    else:
        ax.hist(y.dropna(), bins=60, color="#4c7ef3")
        ax.set_xlabel(ds.task["target"])
        ax.set_ylabel("frequency")
        ax.set_title(f"Target distribution - {ds.task['target']}")
    plots.append(save_figure(fig, plot_dir / "target_distribution.png", dpi))

    # Missingness
    miss = (X.isna().mean() * 100).sort_values(ascending=False).head(25)
    miss = miss.loc[miss > 0]
    if not miss.empty:
        fig, ax = new_figure(8, max(3, 0.28 * len(miss)))
        ax.barh(miss.index[::-1], miss.to_numpy()[::-1], color="#e08a4c")
        ax.set_xlabel("% missing")
        ax.set_title("Missingness by feature")
        plots.append(save_figure(fig, plot_dir / "missingness.png", dpi))

    # Correlation heatmap
    if not corr.empty and corr.shape[0] > 2:
        fig, ax = new_figure(9, 7.5)
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index, fontsize=6)
        ax.set_title("Feature correlation")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plots.append(save_figure(fig, plot_dir / "correlation_matrix.png", dpi))

    # Top associations
    if not rel.empty:
        fig, ax = new_figure(8, max(3, 0.35 * len(rel)))
        ax.barh(rel["feature"][::-1], rel["association"][::-1], color="#3fa87a")
        ax.set_xlabel("|association| with target")
        ax.set_title("Strongest univariate relationships")
        plots.append(save_figure(fig, plot_dir / "target_relationships.png", dpi))

    # Distributions of the most-associated numeric features, split by target
    num_top = [f for f in rel.loc[rel["kind"] == "numeric", "feature"].head(6)
               if f in X.columns]
    if num_top:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = (len(num_top) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(13, 3.4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, feat in zip(axes, num_top):
            v = X[feat].astype(float)
            lo, hi = v.quantile(0.01), v.quantile(0.99)
            if ds.task_type in {"binary", "multiclass"}:
                for cls in sorted(y.unique())[:4]:
                    ax.hist(v[y == cls].clip(lo, hi), bins=40, alpha=0.55, label=str(cls))
                ax.legend(fontsize=7)
            else:
                ax.hist(v.clip(lo, hi), bins=40, color="#4c7ef3")
            ax.set_title(feat, fontsize=9)
        for ax in axes[len(num_top):]:
            ax.axis("off")
        fig.suptitle("Feature distributions by target", y=1.0)
        plots.append(save_figure(fig, plot_dir / "feature_distributions.png", dpi))

    # Volume over time - reveals seasonality and drift
    if ds.time_index is not None:
        ts = pd.DataFrame({"t": ds.time_index, "y": pd.to_numeric(
            pd.factorize(y)[0] if y.dtype == object else y, errors="coerce")})
        daily = ts.set_index("t").resample("W").agg(volume=("y", "size"), mean_target=("y", "mean"))
        fig, ax = new_figure(9, 4)
        ax.plot(daily.index, daily["volume"], color="#4c7ef3", label="weekly volume")
        ax.set_ylabel("orders")
        ax2 = ax.twinx()
        ax2.plot(daily.index, daily["mean_target"], color="#e0574c", label="mean target")
        ax2.set_ylabel("mean target")
        ax.set_title("Volume and target over time")
        fig.legend(loc="upper right", fontsize=8)
        plots.append(save_figure(fig, plot_dir / "time_series.png", dpi))

    return plots
