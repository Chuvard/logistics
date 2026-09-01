"""End-to-end pipeline orchestration.

Runs the seven stages in order and collects everything into one result object,
then registers the winning model and writes the report. Any stage can be
switched off in config; a stage that fails is recorded and the run continues,
because losing a 20-minute training run to a plotting error is unacceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import build_task, load_tables
from .explain.explainer import run_explainability
from .metrics import calibration_frame
from .models.deep import run_deep_learning
from .models.supervised import run_supervised
from .models.unsupervised import run_unsupervised
from .reporting import build_report
from .serving.registry import ModelRegistry
from .stages.eda import run_eda
from .stages.preprocessing import preprocess
from .utils import get_logger, new_figure, save_figure, save_json, set_seed, timed

__all__ = ["PipelineResult", "run_pipeline"]

logger = get_logger()


@dataclass
class PipelineResult:
    task: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    eda: object = None
    supervised: object = None
    unsupervised: object = None
    deep: object = None
    optimization: object = None
    explainability: object = None
    registered: object = None
    reports: list[Path] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    supervised_plots: list[Path] = field(default_factory=list)


def _plot_supervised(sup, sd, task, out_dir: Path, dpi: int) -> list[Path]:
    """ROC / PR / calibration curves and a metric bar chart for the leaderboard."""
    plot_dir = out_dir / "plots" / "supervised"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []
    lb = sup.leaderboard()
    if lb.empty:
        return plots

    primary = sup.primary_metric
    if primary in lb.columns:
        fig, ax = new_figure(8, max(3.5, 0.4 * len(lb)))
        vals = lb[primary].fillna(0)
        ax.barh(lb["model"][::-1], vals[::-1], color="#4c7ef3")
        ax.set_xlabel(primary)
        ax.set_title(f"Model comparison - {primary}")
        plots.append(save_figure(fig, plot_dir / "model_comparison.png", dpi))

    if task["type"] == "binary":
        from sklearn.metrics import precision_recall_curve, roc_curve
        y = sd.y_test
        fig, ax = new_figure(6.5, 5.5)
        for r in sup.results:
            if r.probabilities is None or r.probabilities.ndim != 2:
                continue
            fpr, tpr, _ = roc_curve(y, r.probabilities[:, 1])
            ax.plot(fpr, tpr, lw=1.4,
                    label=f"{r.name} ({r.metrics.get('roc_auc', float('nan')):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title("ROC curves"); ax.legend(fontsize=7)
        plots.append(save_figure(fig, plot_dir / "roc_curves.png", dpi))

        fig, ax = new_figure(6.5, 5.5)
        for r in sup.results:
            if r.probabilities is None or r.probabilities.ndim != 2:
                continue
            prec, rec, _ = precision_recall_curve(y, r.probabilities[:, 1])
            ax.plot(rec, prec, lw=1.4,
                    label=f"{r.name} ({r.metrics.get('average_precision', float('nan')):.3f})")
        ax.axhline(float(np.mean(y)), ls="--", color="#aaa", lw=1, label="base rate")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title("Precision-recall curves"); ax.legend(fontsize=7)
        plots.append(save_figure(fig, plot_dir / "pr_curves.png", dpi))

        best = sup.best
        if best is not None and best.probabilities is not None and best.probabilities.ndim == 2:
            cal = calibration_frame(y, best.probabilities[:, 1])
            if not cal.empty:
                fig, ax = new_figure(6, 5.5)
                ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1, label="perfect")
                ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-",
                        color="#e0574c", label=best.name)
                ax.set_xlabel("mean predicted probability")
                ax.set_ylabel("observed positive rate")
                ax.set_title(f"Calibration - {best.name}"
                             f"{' (isotonic)' if best.calibrated else ''}")
                ax.legend(fontsize=8)
                plots.append(save_figure(fig, plot_dir / "calibration.png", dpi))

    elif task["type"] == "regression":
        best = sup.best
        if best is not None and best.predictions is not None:
            resid = np.asarray(sd.y_test, dtype=float) - best.predictions
            fig, ax = new_figure(6.5, 5)
            ax.scatter(best.predictions, resid, s=5, alpha=0.4, color="#4c7ef3")
            ax.axhline(0, color="#e0574c", lw=1)
            ax.set_xlabel("predicted"); ax.set_ylabel("residual")
            ax.set_title(f"Residuals - {best.name}")
            plots.append(save_figure(fig, plot_dir / "residuals.png", dpi))

            fig, ax = new_figure(6, 5.5)
            ax.scatter(sd.y_test, best.predictions, s=5, alpha=0.4, color="#3fa87a")
            lo = float(min(np.min(sd.y_test), np.min(best.predictions)))
            hi = float(max(np.max(sd.y_test), np.max(best.predictions)))
            ax.plot([lo, hi], [lo, hi], "--", color="#aaa", lw=1)
            ax.set_xlabel("actual"); ax.set_ylabel("predicted")
            ax.set_title(f"Predicted vs actual - {best.name}")
            plots.append(save_figure(fig, plot_dir / "predicted_vs_actual.png", dpi))

    return plots


def run_pipeline(cfg: Config, task_name: str | None = None) -> PipelineResult:
    set_seed(cfg.seed)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("eda.dpi", 110))
    result = PipelineResult()

    # -- data ----------------------------------------------------------------
    with timed("Loading data and building task"):
        ds = build_task(cfg, task_name)
        result.task = ds.task
        result.dataset = ds.describe()

    # -- preprocessing -------------------------------------------------------
    with timed("Preprocessing and splitting"):
        sd = preprocess(ds, cfg)
        result.split = {k: v for k, v in sd.summary().items() if k != "preprocessor"}

    # -- EDA -----------------------------------------------------------------
    try:
        with timed("Exploratory data analysis"):
            result.eda = run_eda(ds, cfg, out_dir)
    except Exception as exc:
        result.errors["eda"] = str(exc)
        logger.error("EDA stage failed: %s", exc)

    # -- supervised ----------------------------------------------------------
    try:
        with timed("Supervised learning"):
            result.supervised = run_supervised(sd, cfg, ds.task)
            result.supervised_plots = _plot_supervised(result.supervised, sd, ds.task, out_dir, dpi)
    except Exception as exc:
        result.errors["supervised"] = str(exc)
        logger.error("Supervised stage failed: %s", exc)

    # -- unsupervised --------------------------------------------------------
    try:
        with timed("Unsupervised learning"):
            result.unsupervised = run_unsupervised(sd, cfg, out_dir, sd.y_train)
    except Exception as exc:
        result.errors["unsupervised"] = str(exc)
        logger.error("Unsupervised stage failed: %s", exc)

    # -- deep learning -------------------------------------------------------
    try:
        with timed("Deep learning"):
            result.deep = run_deep_learning(sd, cfg, ds.task, out_dir)
    except Exception as exc:
        result.errors["deep_learning"] = str(exc)
        logger.error("Deep learning stage failed: %s", exc)

    # -- optimization --------------------------------------------------------
    try:
        with timed("Mathematical optimization"):
            from .optimization import run_optimization
            needed = ["orders", "routes", "vehicles", "drivers", "warehouses",
                      "delivery_zones", "inventory"]
            result.optimization = run_optimization(load_tables(cfg, needed), cfg, out_dir)
    except Exception as exc:
        result.errors["optimization"] = str(exc)
        logger.error("Optimization stage failed: %s", exc)

    # -- explainability ------------------------------------------------------
    best = result.supervised.best if result.supervised else None
    if best is not None:
        try:
            with timed(f"Explainability ({best.name})"):
                result.explainability = run_explainability(
                    best.estimator, sd, cfg, ds.task, out_dir)
        except Exception as exc:
            result.errors["explainability"] = str(exc)
            logger.error("Explainability stage failed: %s", exc)

    # -- registry ------------------------------------------------------------
    registry_table = None
    if best is not None and cfg.get("registry.enabled", True):
        try:
            with timed("Model registry"):
                registry = ModelRegistry(cfg._resolve(cfg.get("registry.path", "./registry")))
                result.registered = registry.register(
                    name=ds.name, estimator=best.estimator, metrics=best.metrics,
                    task=ds.task, feature_names=sd.feature_names,
                    preprocessor=sd.meta.get("preprocessor"),
                    extra={"algorithm": best.name,
                           "n_train": int(len(sd.X_train)),
                           "split_strategy": result.split.get("split_strategy"),
                           "seed": cfg.seed},
                    keep_last_n=int(cfg.get("registry.keep_last_n_versions", 5)))
                registry.promote(ds.name, result.registered.version)
                registry_table = registry.list_models()
        except Exception as exc:
            result.errors["registry"] = str(exc)
            logger.error("Registry failed: %s", exc)

    # -- report --------------------------------------------------------------
    payload = {
        "task": result.task, "dataset": result.dataset, "split": result.split,
        "supervised_plots": result.supervised_plots,
        "registry": registry_table,
    }
    if result.eda is not None:
        payload["eda"] = {"tables": result.eda.tables, "plots": result.eda.plots,
                          "summary": result.eda.summary}
    if result.supervised is not None:
        payload["supervised"] = {
            "leaderboard": result.supervised.leaderboard(),
            "primary_metric": result.supervised.primary_metric,
            "skipped": result.supervised.skipped,
            "confusion": best.confusion if best else None,
            "calibration": best.calibration if best else None}
    if result.unsupervised is not None:
        payload["unsupervised"] = {"tables": result.unsupervised.tables,
                                   "plots": result.unsupervised.plots,
                                   "summary": result.unsupervised.summary}
    if result.deep is not None:
        payload["deep"] = {"backend": result.deep.backend,
                           "leaderboard": result.deep.leaderboard(),
                           "plots": result.deep.plots}
    if result.optimization is not None:
        payload["optimization"] = {"tables": result.optimization.tables,
                                   "leaderboard": result.optimization.leaderboard(),
                                   "plots": result.optimization.plots}
    if result.explainability is not None:
        payload["explainability"] = {"tables": result.explainability.tables,
                                     "plots": result.explainability.plots,
                                     "summary": result.explainability.summary}

    try:
        with timed("Report generation"):
            result.reports = build_report(payload, cfg, out_dir)
    except Exception as exc:
        result.errors["reporting"] = str(exc)
        logger.error("Reporting failed: %s", exc)

    # -- artefacts -----------------------------------------------------------
    _dump_tables(payload, out_dir)
    save_json({"task": result.task, "dataset": result.dataset, "split": result.split,
               "errors": result.errors,
               "best_model": best.name if best else None,
               "best_metrics": best.metrics if best else None},
              out_dir / "run_summary.json")
    cfg.dump(out_dir / "resolved_config.yaml")

    if result.errors:
        logger.warning("Completed with %d stage error(s): %s",
                       len(result.errors), ", ".join(result.errors))
    return result


def _dump_tables(payload: dict, out_dir: Path) -> None:
    """Persist every result table as CSV so nothing is trapped in the HTML."""
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for stage in ["eda", "unsupervised", "optimization", "explainability"]:
        block = payload.get(stage)
        if not block:
            continue
        for name, df in (block.get("tables") or {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(table_dir / f"{stage}__{name}.csv", index=False)
    for stage in ["supervised", "deep", "optimization"]:
        block = payload.get(stage)
        if block and isinstance(block.get("leaderboard"), pd.DataFrame) \
                and not block["leaderboard"].empty:
            block["leaderboard"].to_csv(table_dir / f"{stage}__leaderboard.csv", index=False)
    if isinstance(payload.get("registry"), pd.DataFrame) and not payload["registry"].empty:
        payload["registry"].to_csv(table_dir / "registry.csv", index=False)
