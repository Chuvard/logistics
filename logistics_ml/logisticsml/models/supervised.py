"""Stage 3 - supervised learning.

Nine model families benchmarked head-to-head on the same split with the same
preprocessing, so the comparison is fair. Each entry declares its own builder
so adding a tenth model is a one-function change.

Practical concessions that are made explicit rather than hidden:

* **SVM and KNN are subsampled.** SVC is O(n²)-O(n³) in training rows and KNN
  pays at predict time. Both get a configurable `max_train` cap; the cap is
  recorded in the results table so nobody mistakes a capped score for a
  full-data score.
* **Gradient boosters get early stopping** against the validation split where
  the library supports it.
* **Calibration is optional** and applied on the validation set, never on test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..config import Config
from ..metrics import (
    METRIC_DIRECTION, calibration_frame, classification_metrics, confusion_frame,
    primary_metric_value, regression_metrics,
)
from ..stages.preprocessing import SplitData
from ..utils import get_logger, optional_import

__all__ = ["ModelResult", "SupervisedResults", "run_supervised", "MODEL_BUILDERS"]

logger = get_logger()


@dataclass
class ModelResult:
    name: str
    estimator: Any
    metrics: dict
    train_seconds: float
    predict_seconds: float
    n_train_used: int
    calibrated: bool = False
    notes: str = ""
    confusion: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    feature_importance: pd.DataFrame | None = None


@dataclass
class SupervisedResults:
    task_type: str
    primary_metric: str
    results: list[ModelResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def best(self) -> ModelResult | None:
        scored = [r for r in self.results
                  if primary_metric_value(r.metrics, self.primary_metric) > float("-inf")]
        if not scored:
            return self.results[0] if self.results else None
        return max(scored, key=lambda r: primary_metric_value(r.metrics, self.primary_metric))

    def leaderboard(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                "model": r.name, **r.metrics,
                "train_seconds": round(r.train_seconds, 2),
                "predict_seconds": round(r.predict_seconds, 3),
                "n_train_used": r.n_train_used,
                "calibrated": r.calibrated,
                "notes": r.notes,
            })
        df = pd.DataFrame(rows)
        if self.primary_metric in df.columns:
            df = df.sort_values(
                self.primary_metric,
                ascending=METRIC_DIRECTION.get(self.primary_metric, 1) < 0)
        return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model builders. Each returns an estimator or None when the library is absent.
# --------------------------------------------------------------------------- #
def _build_logistic_regression(params, task_type, cfg, n_classes):
    if task_type == "regression":
        return LinearRegression()
    return LogisticRegression(
        max_iter=int(params.get("max_iter", 1500)),
        C=float(params.get("C", 1.0)),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=cfg.get("project.n_jobs", -1),
        random_state=cfg.seed)


def _build_decision_tree(params, task_type, cfg, n_classes):
    kw = dict(max_depth=params.get("max_depth", 12),
              min_samples_leaf=params.get("min_samples_leaf", 20),
              random_state=cfg.seed)
    if task_type == "regression":
        return DecisionTreeRegressor(**kw)
    return DecisionTreeClassifier(class_weight=params.get("class_weight", "balanced"), **kw)


def _build_random_forest(params, task_type, cfg, n_classes):
    kw = dict(n_estimators=int(params.get("n_estimators", 300)),
              max_depth=params.get("max_depth", 18),
              min_samples_leaf=params.get("min_samples_leaf", 5),
              n_jobs=cfg.get("project.n_jobs", -1),
              random_state=cfg.seed)
    if task_type == "regression":
        return RandomForestRegressor(**kw)
    return RandomForestClassifier(class_weight=params.get("class_weight", "balanced_subsample"), **kw)


def _build_gradient_boosting(params, task_type, cfg, n_classes):
    kw = dict(n_estimators=int(params.get("n_estimators", 200)),
              max_depth=int(params.get("max_depth", 3)),
              learning_rate=float(params.get("learning_rate", 0.1)),
              random_state=cfg.seed)
    return GradientBoostingRegressor(**kw) if task_type == "regression" \
        else GradientBoostingClassifier(**kw)


def _build_xgboost(params, task_type, cfg, n_classes):
    xgb = optional_import("xgboost")
    if xgb is None:
        return None
    kw = dict(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        n_jobs=cfg.get("project.n_jobs", -1),
        random_state=cfg.seed,
        tree_method="hist",
        verbosity=0)
    if task_type == "regression":
        return xgb.XGBRegressor(**kw)
    if task_type == "multiclass":
        return xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, **kw)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)


def _build_lightgbm(params, task_type, cfg, n_classes):
    lgb = optional_import("lightgbm")
    if lgb is None:
        return None
    kw = dict(
        n_estimators=int(params.get("n_estimators", 400)),
        num_leaves=int(params.get("num_leaves", 63)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        subsample=float(params.get("subsample", 0.9)),
        subsample_freq=1,
        n_jobs=cfg.get("project.n_jobs", -1),
        random_state=cfg.seed,
        verbosity=-1)
    return lgb.LGBMRegressor(**kw) if task_type == "regression" else lgb.LGBMClassifier(**kw)


def _build_catboost(params, task_type, cfg, n_classes):
    cb = optional_import("catboost")
    if cb is None:
        return None
    kw = dict(iterations=int(params.get("iterations", 400)),
              depth=int(params.get("depth", 6)),
              learning_rate=float(params.get("learning_rate", 0.08)),
              random_seed=cfg.seed,
              verbose=False,
              allow_writing_files=False)
    return cb.CatBoostRegressor(**kw) if task_type == "regression" else cb.CatBoostClassifier(**kw)


def _build_knn(params, task_type, cfg, n_classes):
    kw = dict(n_neighbors=int(params.get("n_neighbors", 25)),
              weights=params.get("weights", "distance"),
              n_jobs=cfg.get("project.n_jobs", -1))
    return KNeighborsRegressor(**kw) if task_type == "regression" else KNeighborsClassifier(**kw)


def _build_svm(params, task_type, cfg, n_classes):
    kw = dict(kernel=params.get("kernel", "rbf"), C=float(params.get("C", 1.0)))
    if task_type == "regression":
        return SVR(**kw)
    return SVC(probability=True, class_weight=params.get("class_weight", "balanced"),
               random_state=cfg.seed, **kw)


class LabelEncodedClassifier:
    """Adapts a classifier that only accepts integer classes to string labels.

    XGBoost raises ``Invalid classes inferred from unique values of y`` for
    string targets like ``low``/``medium``/``high``. Rather than forcing every
    task to pre-encode its target - which would leak encoding decisions into the
    data layer - the constraint is absorbed here. The wrapper is transparent:
    ``predict`` returns the original labels and ``classes_`` reports them, so
    metrics, calibration, SHAP and the registry all behave identically.
    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.classes_ = None

    def fit(self, X, y, **kwargs):
        self.classes_ = np.array(sorted(pd.unique(np.asarray(y))))
        lut = {c: i for i, c in enumerate(self.classes_)}
        encoded = np.array([lut[v] for v in np.asarray(y)])
        if "eval_set" in kwargs and kwargs["eval_set"]:
            kwargs["eval_set"] = [
                (ex, np.array([lut[v] for v in np.asarray(ey)]))
                for ex, ey in kwargs["eval_set"]]
        self.estimator.fit(X, encoded, **kwargs)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator.predict(X)).astype(int)]

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    def get_params(self, deep: bool = True):
        return {"estimator": self.estimator}

    def __getattr__(self, item):
        # Delegate anything else (feature_importances_, get_booster, ...).
        return getattr(self.__dict__["estimator"], item)


MODEL_BUILDERS: dict[str, Callable] = {
    "logistic_regression": _build_logistic_regression,
    "decision_tree": _build_decision_tree,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "catboost": _build_catboost,
    "knn": _build_knn,
    "svm": _build_svm,
}

# Libraries needed per model, for a clear "why was this skipped" message.
_REQUIRES = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


# --------------------------------------------------------------------------- #
def _feature_importance(estimator, feature_names: list[str]) -> pd.DataFrame | None:
    """Native importances where the model exposes them."""
    est = getattr(estimator, "base_estimator", estimator)
    est = getattr(est, "estimator", est)
    values = None
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if values is None or len(values) != len(feature_names):
        return None
    return (pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _subsample(X: pd.DataFrame, y: pd.Series, cap: int | None, seed: int):
    if not cap or len(X) <= int(cap):
        return X, y, len(X), ""
    idx = np.random.default_rng(seed).choice(len(X), size=int(cap), replace=False)
    note = f"trained on {int(cap):,} of {len(X):,} rows (cost cap)"
    return X.iloc[idx], y.iloc[idx], int(cap), note


def _fit_with_early_stopping(estimator, name, X_tr, y_tr, X_val, y_val) -> str:
    """Use the library's own early stopping when available."""
    try:
        if name == "lightgbm":
            lgb = optional_import("lightgbm")
            estimator.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(0)])
            return f"early stopping @ {getattr(estimator, 'best_iteration_', 'n/a')}"
        if name == "xgboost":
            estimator.set_params(early_stopping_rounds=30)
            estimator.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return f"early stopping @ {getattr(estimator, 'best_iteration', 'n/a')}"
        if name == "catboost":
            estimator.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                          early_stopping_rounds=30, verbose=False)
            return f"early stopping @ {getattr(estimator, 'best_iteration_', 'n/a')}"
    except Exception as exc:
        logger.debug("Early stopping unavailable for %s: %s", name, exc)
    estimator.fit(X_tr, y_tr)
    return ""


def run_supervised(sd: SplitData, cfg: Config, task: dict) -> SupervisedResults:
    task_type = task["type"]
    primary = task.get("primary_metric", "roc_auc" if task_type != "regression" else "rmse")
    out = SupervisedResults(task_type=task_type, primary_metric=primary)

    if not cfg.get("supervised.enabled", True):
        logger.info("Supervised stage disabled by config")
        return out

    models_cfg = cfg.get("supervised.models", {}) or {}
    global_cap = cfg.get("supervised.train_sample")
    calibrate = bool(cfg.get("supervised.calibrate", True)) and task_type in {"binary", "multiclass"}

    X_tr_full, y_tr_full = sd.X_train, sd.y_train
    labels = sorted(pd.unique(sd.y_train)) if task_type != "regression" else None
    n_classes = len(labels) if labels else 0

    # A degenerate split defeats every classifier in the same way. Detect it once
    # here rather than letting eight models fail with eight different errors.
    if task_type != "regression":
        if n_classes < 2:
            reason = (f"training split contains only class {labels[0]!r}. "
                      "The target is too rare, or the temporal split isolated it - "
                      "try split.strategy=random or a longer window.")
            logger.error("Cannot train: %s", reason)
            out.skipped["*all*"] = reason
            return out
        test_classes = set(pd.unique(sd.y_test))
        if len(test_classes) < 2:
            logger.warning("Test split has a single class (%s) - ranking metrics "
                           "such as ROC-AUC will be undefined", test_classes)
        counts = sd.y_train.value_counts()
        if counts.min() < 10:
            logger.warning("Rarest training class has only %d example(s); "
                           "metrics will be unstable", counts.min())

    for name, spec in models_cfg.items():
        if not (spec or {}).get("enabled", True):
            out.skipped[name] = "disabled in config"
            continue
        builder = MODEL_BUILDERS.get(name)
        if builder is None:
            out.skipped[name] = "no builder registered"
            continue

        estimator = builder(spec or {}, task_type, cfg, n_classes)

        # XGBoost only accepts integer classes; wrap it when labels are strings.
        if (estimator is not None and name == "xgboost" and task_type != "regression"
                and labels is not None and not all(isinstance(l, (int, np.integer))
                                                   for l in labels)):
            estimator = LabelEncodedClassifier(estimator)

        if estimator is None:
            pkg = _REQUIRES.get(name, name)
            out.skipped[name] = f"{pkg} not installed"
            logger.warning("Skipping %s - %s not installed", name, pkg)
            continue

        # Model-specific cap wins over the global one.
        cap = spec.get("max_train", global_cap)
        X_tr, y_tr, n_used, note = _subsample(X_tr_full, y_tr_full, cap, cfg.seed)

        try:
            t0 = time.perf_counter()
            es_note = _fit_with_early_stopping(estimator, name, X_tr, y_tr, sd.X_val, sd.y_val)
            train_seconds = time.perf_counter() - t0

            fitted, was_calibrated = estimator, False
            if calibrate and hasattr(estimator, "predict_proba") and name != "logistic_regression":
                try:
                    # Calibrate on validation data the base model never trained on.
                    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
                    calibrated.fit(sd.X_val, sd.y_val)
                    fitted, was_calibrated = calibrated, True
                except Exception as exc:
                    logger.debug("Calibration failed for %s: %s", name, exc)

            t0 = time.perf_counter()
            y_pred = fitted.predict(sd.X_test)
            y_proba = fitted.predict_proba(sd.X_test) if hasattr(fitted, "predict_proba") else None
            predict_seconds = time.perf_counter() - t0

            if task_type == "regression":
                m = regression_metrics(sd.y_test, y_pred)
                confusion = calib = None
            else:
                m = classification_metrics(sd.y_test, y_pred, y_proba, task_type, labels)
                confusion = confusion_frame(sd.y_test, y_pred, labels)
                calib = (calibration_frame(sd.y_test, y_proba[:, 1])
                         if (task_type == "binary" and y_proba is not None) else None)

            notes = "; ".join(x for x in (note, es_note) if x)
            out.results.append(ModelResult(
                name=name, estimator=fitted, metrics=m,
                train_seconds=train_seconds, predict_seconds=predict_seconds,
                n_train_used=n_used, calibrated=was_calibrated, notes=notes,
                confusion=confusion, calibration=calib,
                predictions=np.asarray(y_pred),
                probabilities=np.asarray(y_proba) if y_proba is not None else None,
                feature_importance=_feature_importance(fitted, sd.feature_names)))

            score = m.get(primary)
            logger.info("  %-20s %s=%s (%.1fs)", name, primary,
                        f"{score:.4f}" if isinstance(score, float) else score, train_seconds)

        except Exception as exc:
            out.skipped[name] = f"failed: {type(exc).__name__}: {exc}"
            logger.error("Model %s failed: %s", name, exc)

    if out.best is not None:
        logger.info("Best model: %s (%s=%s)", out.best.name, primary, out.best.metrics.get(primary))
    return out
