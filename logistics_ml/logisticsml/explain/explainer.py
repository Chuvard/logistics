"""Stage 7 - explainable AI.

Four complementary views of the same model, because no single one is
sufficient:

* **SHAP** - additive, theoretically grounded attributions; global ranking and
  per-prediction breakdowns. Uses TreeExplainer for tree models (exact and
  fast) and KernelExplainer otherwise.
* **Permutation importance** - measures the actual metric drop when a feature
  is shuffled. Model-agnostic and immune to the tree-importance bias toward
  high-cardinality features.
* **LIME** - local linear surrogate around individual predictions. If the
  `lime` package is absent, an equivalent surrogate is fitted in-house
  (perturb, weight by proximity, fit ridge), so the stage always runs.
* **Partial dependence** - the average shape of a feature's effect, which is
  what turns "this feature matters" into "and here is how".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..stages.preprocessing import SplitData
from ..utils import get_logger, new_figure, optional_import, save_figure

__all__ = ["ExplainResults", "run_explainability"]

logger = get_logger()


@dataclass
class ExplainResults:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: list[Path] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def _sample(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(X) <= n:
        return X
    return X.iloc[np.random.default_rng(seed).choice(len(X), n, replace=False)]


def _unwrap(estimator):
    """Reach the underlying model through a calibration wrapper."""
    for attr in ("estimator", "base_estimator"):
        inner = getattr(estimator, attr, None)
        if inner is not None:
            return inner
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated:
        return getattr(calibrated[0], "estimator", estimator)
    return estimator


def _is_tree(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in
               ("forest", "tree", "boost", "xgb", "lgbm", "catboost", "gbm"))


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _native_tree_shap(model, X: pd.DataFrame) -> np.ndarray | None:
    """Exact TreeSHAP from the booster itself.

    XGBoost and LightGBM both compute TreeSHAP internally, and the released
    ``shap`` package currently fails against XGBoost 3.x (it tries to parse the
    JSON-encoded ``base_score`` as a float). Going through the native API is not
    a workaround for a missing feature - these are the same exact Shapley values,
    computed by the library that owns the tree structure - and it removes a
    version-compatibility dependency from the critical path.

    Returns an ``(n_rows, n_features)`` array with the bias column stripped.
    """
    # XGBoost
    booster_fn = getattr(model, "get_booster", None)
    if callable(booster_fn):
        try:
            import xgboost as xgb
            contribs = booster_fn().predict(xgb.DMatrix(X), pred_contribs=True)
            arr = np.asarray(contribs)
            if arr.ndim == 3:            # multiclass: (n, classes, features+1)
                arr = arr[:, -1, :]
            return arr[:, :-1]           # drop the bias term
        except Exception:
            return None

    # LightGBM
    if hasattr(model, "booster_"):
        try:
            contribs = np.asarray(model.predict(X, pred_contrib=True))
            n_features = X.shape[1]
            if contribs.shape[1] > n_features + 1:      # multiclass, blocked
                contribs = contribs[:, -(n_features + 1):]
            return contribs[:, :-1]
        except Exception:
            return None
    return None


def _run_shap(estimator, sd: SplitData, cfg: Config, plot_dir: Path,
              res: ExplainResults, dpi: int) -> None:
    shap = optional_import("shap")
    n = int(cfg.get("explainability.shap.sample_size", 2000))
    max_display = int(cfg.get("explainability.shap.max_display", 20))
    X = _sample(sd.X_test, n, cfg.seed)
    model = _unwrap(estimator)

    try:
        arr = _native_tree_shap(model, X)
        if arr is not None:
            res.summary["shap_backend"] = "native booster TreeSHAP (exact)"
        elif shap is None:
            res.skipped["shap"] = "shap not installed and model has no native TreeSHAP"
            logger.warning("Skipping SHAP - shap not installed")
            return
        elif _is_tree(model):
            values = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
            arr = np.asarray(values)
            res.summary["shap_backend"] = "shap.TreeExplainer"
        else:
            # KernelExplainer is O(n * 2^features); keep the background tiny.
            background = shap.kmeans(_sample(sd.X_train, 200, cfg.seed), 20)
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                background)
            X = _sample(X, min(200, len(X)), cfg.seed)
            values = explainer.shap_values(X, nsamples=100)
            arr = np.asarray(values)
            res.summary["shap_backend"] = "shap.KernelExplainer"

        # Normalise the many shapes SHAP returns across model families.
        if arr.ndim == 3:
            arr = arr[:, :, -1] if arr.shape[2] <= arr.shape[1] else arr[-1]

        mean_abs = np.abs(arr).mean(axis=0)
        table = (pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
                 .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
        table["share_pct"] = (table["mean_abs_shap"] / table["mean_abs_shap"].sum() * 100).round(2)
        res.tables["shap_global"] = table.head(50).round(6)
        res.summary["shap_top_feature"] = table["feature"].iloc[0]

        top = table.head(max_display)
        fig, ax = new_figure(8, max(4, 0.32 * len(top)))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4c7ef3")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title("SHAP global feature importance")
        res.plots.append(save_figure(fig, plot_dir / "shap_global.png", dpi))

        # Beeswarm-style view: value vs impact for the strongest features.
        fig, ax = new_figure(8, max(4, 0.32 * min(12, len(top))))
        for i, feat in enumerate(top["feature"].head(12)[::-1]):
            col = list(X.columns).index(feat)
            sv = arr[:, col]
            xv = X.iloc[:, col].to_numpy()
            norm = (xv - xv.min()) / (np.ptp(xv) + 1e-9)
            jitter = np.random.default_rng(cfg.seed).normal(0, 0.08, len(sv))
            ax.scatter(sv, i + jitter, c=norm, cmap="coolwarm", s=6, alpha=0.6)
        ax.set_yticks(range(min(12, len(top))))
        ax.set_yticklabels(top["feature"].head(12)[::-1], fontsize=8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("SHAP value distribution (colour = feature value)")
        res.plots.append(save_figure(fig, plot_dir / "shap_beeswarm.png", dpi))

        # Local explanations for a few individual predictions.
        rows = []
        for i in range(min(5, len(X))):
            order = np.argsort(np.abs(arr[i]))[::-1][:8]
            for j in order:
                rows.append({"instance": i, "feature": X.columns[j],
                             "feature_value": round(float(X.iloc[i, j]), 4),
                             "shap_value": round(float(arr[i, j]), 6)})
        res.tables["shap_local"] = pd.DataFrame(rows)
        logger.info("  SHAP: top feature %r", table["feature"].iloc[0])

    except Exception as exc:
        res.skipped["shap"] = f"{type(exc).__name__}: {exc}"
        logger.error("SHAP failed: %s", exc)


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
def _run_permutation(estimator, sd: SplitData, cfg: Config, task: dict,
                     plot_dir: Path, res: ExplainResults, dpi: int) -> None:
    from sklearn.inspection import permutation_importance

    n = int(cfg.get("explainability.permutation_importance.sample_size", 5000))
    repeats = int(cfg.get("explainability.permutation_importance.n_repeats", 5))
    idx = np.random.default_rng(cfg.seed).choice(
        len(sd.X_test), min(n, len(sd.X_test)), replace=False)
    X, y = sd.X_test.iloc[idx], sd.y_test.iloc[idx]

    scoring = {"binary": "roc_auc", "multiclass": "f1_macro",
               "regression": "neg_root_mean_squared_error"}[task["type"]]
    try:
        r = permutation_importance(estimator, X, y, n_repeats=repeats,
                                   random_state=cfg.seed, scoring=scoring,
                                   n_jobs=cfg.get("project.n_jobs", -1))
        table = (pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))
        res.tables["permutation_importance"] = table.head(50).round(6)

        top = table.head(20)
        fig, ax = new_figure(8, max(4, 0.32 * len(top)))
        ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
                xerr=top["importance_std"][::-1], color="#3fa87a")
        ax.set_xlabel(f"drop in {scoring} when shuffled")
        ax.set_title("Permutation importance")
        res.plots.append(save_figure(fig, plot_dir / "permutation_importance.png", dpi))
        logger.info("  Permutation importance: top feature %r", table["feature"].iloc[0])
    except Exception as exc:
        res.skipped["permutation_importance"] = f"{type(exc).__name__}: {exc}"
        logger.error("Permutation importance failed: %s", exc)


# --------------------------------------------------------------------------- #
# LIME (library, or an in-house equivalent)
# --------------------------------------------------------------------------- #
def _local_surrogate(estimator, X_train: pd.DataFrame, instance: pd.Series,
                     n_samples: int, n_features: int, seed: int,
                     task_type: str) -> pd.DataFrame:
    """LIME from first principles.

    Perturb the instance in its neighbourhood, ask the model what it predicts,
    weight each perturbation by proximity to the original, and fit a weighted
    ridge regression. The coefficients are the local explanation.
    """
    rng = np.random.default_rng(seed)
    sd_vec = X_train.std().replace(0, 1.0).to_numpy()
    base = instance.to_numpy(dtype=float)

    noise = rng.normal(0, 1, (n_samples, len(base))) * sd_vec
    samples = base + noise
    frame = pd.DataFrame(samples, columns=X_train.columns)

    if task_type == "regression" or not hasattr(estimator, "predict_proba"):
        target = np.asarray(estimator.predict(frame), dtype=float)
    else:
        proba = estimator.predict_proba(frame)
        target = proba[:, -1]

    # Exponential kernel on scaled distance - closer points matter more.
    dist = np.sqrt((((samples - base) / sd_vec) ** 2).mean(axis=1))
    weights = np.exp(-(dist ** 2) / (0.75 ** 2))

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(frame, target, sample_weight=weights)

    coefs = pd.DataFrame({
        "feature": list(X_train.columns),
        "local_weight": ridge.coef_,
        "feature_value": base.round(4)})
    coefs["abs_weight"] = coefs["local_weight"].abs()
    return (coefs.sort_values("abs_weight", ascending=False)
            .head(n_features).drop(columns="abs_weight").reset_index(drop=True).round(6))


def _run_lime(estimator, sd: SplitData, cfg: Config, task: dict,
              res: ExplainResults) -> None:
    c = cfg.get("explainability.lime", {}) or {}
    n_instances = int(c.get("n_instances", 5))
    n_features = int(c.get("n_features", 12))
    n_samples = int(c.get("n_samples", 3000))

    lime_mod = optional_import("lime")
    frames = []
    try:
        if lime_mod is not None:
            from lime.lime_tabular import LimeTabularExplainer
            mode = "regression" if task["type"] == "regression" else "classification"
            explainer = LimeTabularExplainer(
                sd.X_train.to_numpy(), feature_names=list(sd.X_train.columns),
                mode=mode, random_state=cfg.seed, discretize_continuous=True)
            predict_fn = (estimator.predict if mode == "regression"
                          else estimator.predict_proba)
            for i in range(min(n_instances, len(sd.X_test))):
                exp = explainer.explain_instance(
                    sd.X_test.iloc[i].to_numpy(), predict_fn,
                    num_features=n_features, num_samples=n_samples)
                for feat, weight in exp.as_list():
                    frames.append({"instance": i, "condition": feat,
                                   "local_weight": round(float(weight), 6)})
            res.summary["lime_backend"] = "lime package"
        else:
            for i in range(min(n_instances, len(sd.X_test))):
                local = _local_surrogate(estimator, sd.X_train, sd.X_test.iloc[i],
                                         n_samples, n_features, cfg.seed + i, task["type"])
                local.insert(0, "instance", i)
                frames.append(local)
            res.summary["lime_backend"] = "in-house weighted ridge surrogate"
            logger.info("  LIME: lime package absent, used in-house surrogate")

        res.tables["lime_local"] = (pd.DataFrame(frames) if lime_mod is not None
                                    else pd.concat(frames, ignore_index=True))
    except Exception as exc:
        res.skipped["lime"] = f"{type(exc).__name__}: {exc}"
        logger.error("LIME failed: %s", exc)


# --------------------------------------------------------------------------- #
# Partial dependence
# --------------------------------------------------------------------------- #
def _run_partial_dependence(estimator, sd: SplitData, cfg: Config, task: dict,
                            plot_dir: Path, res: ExplainResults, dpi: int) -> None:
    from sklearn.inspection import partial_dependence

    top_n = int(cfg.get("explainability.partial_dependence.top_n_features", 6))
    ranking = res.tables.get("shap_global", res.tables.get("permutation_importance"))
    if ranking is None or ranking.empty:
        res.skipped["partial_dependence"] = "no feature ranking available"
        return

    features = [f for f in ranking["feature"].head(top_n) if f in sd.X_test.columns]
    idx = np.random.default_rng(cfg.seed).choice(
        len(sd.X_test), min(2000, len(sd.X_test)), replace=False)
    X = sd.X_test.iloc[idx]

    rows = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_rows = (len(features) + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, feat in zip(axes, features):
            col = list(X.columns).index(feat)
            pd_result = partial_dependence(estimator, X, [col], kind="average",
                                           grid_resolution=25)
            grid = pd_result["grid_values"][0]
            avg = np.asarray(pd_result["average"])
            avg = avg[0] if avg.ndim > 1 else avg
            ax.plot(grid, avg, color="#4c7ef3", lw=2)
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("feature value (scaled)", fontsize=8)
            ax.set_ylabel("partial dependence", fontsize=8)
            for g, a in zip(grid, avg):
                rows.append({"feature": feat, "grid_value": round(float(g), 4),
                             "partial_dependence": round(float(a), 6)})
        for ax in axes[len(features):]:
            ax.axis("off")
        fig.suptitle("Partial dependence of the strongest features", y=1.0)
        res.plots.append(save_figure(fig, plot_dir / "partial_dependence.png", dpi))
        res.tables["partial_dependence"] = pd.DataFrame(rows)
        logger.info("  Partial dependence: %d features", len(features))
    except Exception as exc:
        res.skipped["partial_dependence"] = f"{type(exc).__name__}: {exc}"
        logger.error("Partial dependence failed: %s", exc)


# --------------------------------------------------------------------------- #
def run_explainability(estimator, sd: SplitData, cfg: Config, task: dict,
                       out_dir: Path) -> ExplainResults:
    res = ExplainResults()
    if not cfg.get("explainability.enabled", True):
        logger.info("Explainability stage disabled by config")
        return res

    plot_dir = out_dir / "plots" / "explainability"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("eda.dpi", 110))

    if cfg.get("explainability.shap.enabled", True):
        _run_shap(estimator, sd, cfg, plot_dir, res, dpi)
    if cfg.get("explainability.permutation_importance.enabled", True):
        _run_permutation(estimator, sd, cfg, task, plot_dir, res, dpi)
    if cfg.get("explainability.lime.enabled", True):
        _run_lime(estimator, sd, cfg, task, res)
    if cfg.get("explainability.partial_dependence.enabled", True):
        _run_partial_dependence(estimator, sd, cfg, task, plot_dir, res, dpi)

    # Cross-check: do SHAP and permutation importance agree? Sharp disagreement
    # usually means correlated features or an unstable model, and is worth
    # surfacing rather than leaving for someone to notice later.
    shap_t = res.tables.get("shap_global")
    perm_t = res.tables.get("permutation_importance")
    if shap_t is not None and perm_t is not None and not shap_t.empty and not perm_t.empty:
        merged = (shap_t[["feature", "mean_abs_shap"]]
                  .merge(perm_t[["feature", "importance_mean"]], on="feature", how="inner"))
        if len(merged) > 3:
            merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False)
            merged["perm_rank"] = merged["importance_mean"].rank(ascending=False)
            corr = merged[["shap_rank", "perm_rank"]].corr(method="spearman").iloc[0, 1]
            res.summary["shap_vs_permutation_rank_corr"] = round(float(corr), 4)
            merged["rank_gap"] = (merged["shap_rank"] - merged["perm_rank"]).abs()
            res.tables["importance_agreement"] = (
                merged.sort_values("rank_gap", ascending=False).head(15).round(4))

    return res
