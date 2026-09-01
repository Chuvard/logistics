"""Stage 5 - deep learning.

Dispatches to PyTorch when it is installed, and to scikit-learn equivalents when
it is not, so the stage produces real trained models and real metrics in either
environment. The report always states which backend ran, so a fallback result is
never mistaken for a torch result.

Fallback mapping:

============  =========================================================
Architecture  scikit-learn stand-in
============  =========================================================
MLP           ``MLPClassifier`` / ``MLPRegressor`` (same idea, no GPU)
Autoencoder   PCA reconstruction - a linear autoencoder, mathematically
LSTM          MLP over the same chunked sequence, flattened
Transformer   MLP over feature-interaction terms (no attention)
TabNet        MLP with feature selection via ``SelectFromModel``
============  =========================================================

The stand-ins for LSTM/Transformer/TabNet are honest approximations, not
equivalents - they are labelled as such in the results table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..config import Config
from ..metrics import classification_metrics, confusion_frame, regression_metrics
from ..stages.preprocessing import SplitData
from ..utils import get_logger, new_figure, save_figure

from . import deep_torch

__all__ = ["DeepResults", "run_deep_learning"]

logger = get_logger()


@dataclass
class DeepResults:
    backend: str = "none"
    results: list[dict] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: list[Path] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(self.results) if self.results else pd.DataFrame()


def _cap(X: pd.DataFrame, y: pd.Series, cap, seed: int):
    if not cap or len(X) <= int(cap):
        return X.to_numpy(dtype=np.float32), np.asarray(y)
    idx = np.random.default_rng(seed).choice(len(X), int(cap), replace=False)
    return X.iloc[idx].to_numpy(dtype=np.float32), np.asarray(y.iloc[idx])


def _evaluate(y_true, y_pred, y_proba, task_type, labels):
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred, y_proba, task_type, labels)


# --------------------------------------------------------------------------- #
# Torch path
# --------------------------------------------------------------------------- #
def _run_torch(sd: SplitData, cfg: Config, task: dict, out_dir: Path) -> DeepResults:
    import torch

    res = DeepResults(backend="torch")
    dl = cfg.get("deep_learning", {}) or {}
    models_cfg = dl.get("models", {}) or {}
    task_type = task["type"]
    labels = sorted(pd.unique(sd.y_train)) if task_type != "regression" else None
    n_out = 1 if task_type == "regression" else len(labels)

    X_tr, y_tr = _cap(sd.X_train, sd.y_train, dl.get("train_sample"), cfg.seed)
    X_va, y_va = sd.X_val.to_numpy(np.float32), np.asarray(sd.y_val)
    X_te, y_te = sd.X_test.to_numpy(np.float32), np.asarray(sd.y_test)
    n_features = X_tr.shape[1]

    if task_type != "regression":
        lut = {l: i for i, l in enumerate(labels)}
        y_tr = np.array([lut[v] for v in y_tr])
        y_va = np.array([lut[v] for v in y_va])

    epochs = int(dl.get("epochs", 30))
    batch = int(dl.get("batch_size", 512))
    lr = float(dl.get("learning_rate", 1e-3))
    patience = int(dl.get("early_stopping_patience", 5))

    def _predict(model, X, seq=False):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
        if task_type == "regression":
            return logits.numpy().ravel(), None
        proba = torch.softmax(logits, dim=1).numpy()
        pred = np.array(labels)[proba.argmax(1)]
        return pred, proba

    specs = []
    if models_cfg.get("mlp", {}).get("enabled", True):
        m = models_cfg["mlp"]
        specs.append(("mlp", deep_torch.MLP(n_features, n_out,
                                            m.get("hidden", [256, 128, 64]),
                                            float(m.get("dropout", 0.2))), False))
    if models_cfg.get("transformer", {}).get("enabled", True):
        m = models_cfg["transformer"]
        specs.append(("transformer", deep_torch.TabTransformer(
            n_features, n_out, int(m.get("d_model", 64)), int(m.get("heads", 4)),
            int(m.get("layers", 2)), float(m.get("dropout", 0.1))), False))
    if models_cfg.get("tabnet", {}).get("enabled", True):
        m = models_cfg["tabnet"]
        specs.append(("tabnet", deep_torch.TabNet(
            n_features, n_out, int(m.get("n_d", 16)), int(m.get("n_a", 16)),
            int(m.get("n_steps", 3))), False))

    for name, model, seq in specs:
        try:
            t0 = time.perf_counter()
            info = deep_torch.train_supervised(
                model, X_tr, y_tr, X_va, y_va, task_type,
                epochs, batch, lr, patience)
            train_seconds = time.perf_counter() - t0
            pred, proba = _predict(model, X_te)
            metrics = _evaluate(y_te, pred, proba, task_type, labels)
            res.results.append({
                "model": name, "backend": "torch", **metrics,
                "train_seconds": round(train_seconds, 2),
                "epochs_run": info["epochs_run"],
                "best_val_loss": info["best_val_loss"]})
            res.models[name] = model
            res.tables[f"{name}_history"] = pd.DataFrame(info["history"])
            logger.info("  %-12s trained (%d epochs, %.1fs)", name, info["epochs_run"], train_seconds)
        except Exception as exc:
            res.skipped[name] = f"{type(exc).__name__}: {exc}"
            logger.error("Deep model %s failed: %s", name, exc)

    # LSTM operates on the chunked sequence view
    if models_cfg.get("lstm", {}).get("enabled", True):
        try:
            m = models_cfg["lstm"]
            seq_len = int(m.get("sequence_length", 12))
            S_tr = deep_torch.make_sequences(X_tr, seq_len)
            S_va = deep_torch.make_sequences(X_va, seq_len)
            S_te = deep_torch.make_sequences(X_te, seq_len)
            model = deep_torch.LSTMNet(S_tr.shape[2], n_out,
                                       int(m.get("hidden", 64)), int(m.get("layers", 2)))
            t0 = time.perf_counter()
            info = deep_torch.train_supervised(model, S_tr, y_tr, S_va, y_va,
                                               task_type, epochs, batch, lr, patience)
            train_seconds = time.perf_counter() - t0
            pred, proba = _predict(model, S_te)
            res.results.append({
                "model": "lstm", "backend": "torch",
                **_evaluate(y_te, pred, proba, task_type, labels),
                "train_seconds": round(train_seconds, 2),
                "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"]})
            res.models["lstm"] = model
            res.tables["lstm_history"] = pd.DataFrame(info["history"])
            logger.info("  %-12s trained (%d epochs, %.1fs)", "lstm", info["epochs_run"], train_seconds)
        except Exception as exc:
            res.skipped["lstm"] = f"{type(exc).__name__}: {exc}"

    # Autoencoder - unsupervised, scored by reconstruction error
    if models_cfg.get("autoencoder", {}).get("enabled", True):
        try:
            m = models_cfg["autoencoder"]
            ae = deep_torch.AutoEncoder(n_features, m.get("hidden", [128, 64]),
                                        int(m.get("latent", 16)))
            t0 = time.perf_counter()
            info = deep_torch.train_autoencoder(ae, X_tr, X_va, epochs, batch, lr, patience)
            train_seconds = time.perf_counter() - t0
            ae.eval()
            with torch.no_grad():
                recon, latent = ae(torch.tensor(X_te, dtype=torch.float32))
            err = ((recon.numpy() - X_te) ** 2).mean(axis=1)
            res.models["autoencoder"] = ae
            res.results.append({
                "model": "autoencoder", "backend": "torch",
                "reconstruction_mse": round(float(err.mean()), 6),
                "train_seconds": round(train_seconds, 2),
                "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"]})
            res.tables["autoencoder_history"] = pd.DataFrame(info["history"])
            res.tables["autoencoder_anomalies"] = _anomaly_table(err)
            res.plots.extend(_plot_recon(err, out_dir, int(cfg.get("eda.dpi", 110))))
        except Exception as exc:
            res.skipped["autoencoder"] = f"{type(exc).__name__}: {exc}"

    return res


# --------------------------------------------------------------------------- #
# scikit-learn fallback path
# --------------------------------------------------------------------------- #
def _run_sklearn(sd: SplitData, cfg: Config, task: dict, out_dir: Path) -> DeepResults:
    res = DeepResults(backend="sklearn")
    dl = cfg.get("deep_learning", {}) or {}
    models_cfg = dl.get("models", {}) or {}
    task_type = task["type"]
    labels = sorted(pd.unique(sd.y_train)) if task_type != "regression" else None

    X_tr, y_tr = _cap(sd.X_train, sd.y_train, dl.get("train_sample"), cfg.seed)
    X_te, y_te = sd.X_test.to_numpy(np.float32), np.asarray(sd.y_test)
    epochs = int(dl.get("epochs", 30))

    def _mlp(hidden, **kw):
        common = dict(hidden_layer_sizes=tuple(hidden), max_iter=max(epochs * 8, 200),
                      early_stopping=True, n_iter_no_change=int(dl.get("early_stopping_patience", 5)),
                      learning_rate_init=float(dl.get("learning_rate", 1e-3)),
                      batch_size=min(int(dl.get("batch_size", 512)), len(X_tr)),
                      random_state=cfg.seed, **kw)
            # noqa: E128
        return MLPRegressor(**common) if task_type == "regression" else MLPClassifier(**common)

    def _fit_eval(name, estimator, Xtr, ytr, Xte, note):
        t0 = time.perf_counter()
        estimator.fit(Xtr, ytr)
        train_seconds = time.perf_counter() - t0
        pred = estimator.predict(Xte)
        proba = estimator.predict_proba(Xte) if hasattr(estimator, "predict_proba") else None
        res.results.append({
            "model": name, "backend": "sklearn",
            **_evaluate(y_te, pred, proba, task_type, labels),
            "train_seconds": round(train_seconds, 2),
            "epochs_run": int(getattr(estimator, "n_iter_", 0)),
            "note": note})
        res.models[name] = estimator
        logger.info("  %-12s trained via sklearn (%.1fs)", name, train_seconds)

    # MLP - a true like-for-like stand-in
    if models_cfg.get("mlp", {}).get("enabled", True):
        try:
            _fit_eval("mlp", _mlp(models_cfg["mlp"].get("hidden", [256, 128, 64])),
                      X_tr, y_tr, X_te, "sklearn MLP (equivalent architecture)")
        except Exception as exc:
            res.skipped["mlp"] = str(exc)

    # LSTM stand-in - same chunked sequence view, flattened
    if models_cfg.get("lstm", {}).get("enabled", True):
        try:
            seq_len = int(models_cfg["lstm"].get("sequence_length", 12))
            S_tr = deep_torch.make_sequences(X_tr, seq_len).reshape(len(X_tr), -1)
            S_te = deep_torch.make_sequences(X_te, seq_len).reshape(len(X_te), -1)
            _fit_eval("lstm", _mlp([models_cfg["lstm"].get("hidden", 64), 32]),
                      S_tr, y_tr, S_te,
                      "APPROXIMATION: MLP over flattened sequence, no recurrence")
        except Exception as exc:
            res.skipped["lstm"] = str(exc)

    # Transformer stand-in - MLP over pairwise interactions of the top features
    if models_cfg.get("transformer", {}).get("enabled", True):
        try:
            k = min(24, X_tr.shape[1])
            var_idx = np.argsort(X_tr.var(axis=0))[::-1][:k]
            def _interact(A):
                base = A[:, var_idx]
                cross = (base[:, :8, None] * base[:, None, :8]).reshape(len(A), -1)
                return np.hstack([A, cross]).astype(np.float32)
            _fit_eval("transformer", _mlp([models_cfg["transformer"].get("d_model", 64), 32]),
                      _interact(X_tr), y_tr, _interact(X_te),
                      "APPROXIMATION: MLP over explicit interactions, no attention")
        except Exception as exc:
            res.skipped["transformer"] = str(exc)

    # TabNet stand-in - sparse feature selection then MLP
    if models_cfg.get("tabnet", {}).get("enabled", True):
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            selector_base = (RandomForestRegressor if task_type == "regression"
                             else RandomForestClassifier)(
                n_estimators=100, max_depth=10, random_state=cfg.seed, n_jobs=-1)
            selector = SelectFromModel(selector_base, threshold="median")
            selector.fit(X_tr, y_tr)
            _fit_eval("tabnet", _mlp([models_cfg["tabnet"].get("n_d", 16) * 4, 32]),
                      selector.transform(X_tr), y_tr, selector.transform(X_te),
                      "APPROXIMATION: sparse feature selection + MLP, no sequential attention")
        except Exception as exc:
            res.skipped["tabnet"] = str(exc)

    # Autoencoder stand-in - PCA is exactly a linear autoencoder
    if models_cfg.get("autoencoder", {}).get("enabled", True):
        try:
            latent = int(models_cfg["autoencoder"].get("latent", 16))
            t0 = time.perf_counter()
            pca = PCA(n_components=min(latent, X_tr.shape[1]), random_state=cfg.seed).fit(X_tr)
            train_seconds = time.perf_counter() - t0
            recon = pca.inverse_transform(pca.transform(X_te))
            err = ((recon - X_te) ** 2).mean(axis=1)
            res.models["autoencoder"] = pca
            res.results.append({
                "model": "autoencoder", "backend": "sklearn",
                "reconstruction_mse": round(float(err.mean()), 6),
                "train_seconds": round(train_seconds, 2),
                "note": "PCA reconstruction (a linear autoencoder)"})
            res.tables["autoencoder_anomalies"] = _anomaly_table(err)
            res.plots.extend(_plot_recon(err, out_dir, int(cfg.get("eda.dpi", 110))))
            logger.info("  %-12s via PCA reconstruction (%.2fs)", "autoencoder", train_seconds)
        except Exception as exc:
            res.skipped["autoencoder"] = str(exc)

    return res


# --------------------------------------------------------------------------- #
def _anomaly_table(err: np.ndarray) -> pd.DataFrame:
    thresh = float(np.percentile(err, 99))
    return pd.DataFrame([{
        "mean_reconstruction_error": round(float(err.mean()), 6),
        "p50": round(float(np.percentile(err, 50)), 6),
        "p95": round(float(np.percentile(err, 95)), 6),
        "p99_threshold": round(thresh, 6),
        "n_flagged_above_p99": int((err > thresh).sum()),
    }])


def _plot_recon(err: np.ndarray, out_dir: Path, dpi: int) -> list[Path]:
    plot_dir = out_dir / "plots" / "deep_learning"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = new_figure(7, 4)
    ax.hist(np.clip(err, 0, np.percentile(err, 99.5)), bins=60, color="#7a5cf0")
    ax.axvline(np.percentile(err, 99), color="#e0574c", ls="--", label="p99 threshold")
    ax.set_xlabel("reconstruction error"); ax.set_ylabel("count")
    ax.set_title("Autoencoder reconstruction error")
    ax.legend(fontsize=8)
    return [save_figure(fig, plot_dir / "reconstruction_error.png", dpi)]


def run_deep_learning(sd: SplitData, cfg: Config, task: dict, out_dir: Path) -> DeepResults:
    if not cfg.get("deep_learning.enabled", True):
        logger.info("Deep learning stage disabled by config")
        return DeepResults(backend="disabled")

    backend = str(cfg.get("deep_learning.backend", "auto")).lower()
    use_torch = deep_torch.available() if backend == "auto" else backend == "torch"

    if use_torch and not deep_torch.available():
        logger.warning("backend=torch requested but torch is not installed - using sklearn")
        use_torch = False

    if use_torch:
        logger.info("Deep learning backend: PyTorch")
        return _run_torch(sd, cfg, task, out_dir)

    logger.info("Deep learning backend: scikit-learn fallback "
                "(install torch for LSTM/Transformer/TabNet proper)")
    return _run_sklearn(sd, cfg, task, out_dir)
