"""Stage 4 - unsupervised learning.

Three jobs, each answering a real operational question:

* **Clustering** (KMeans, DBSCAN, GMM) - what natural delivery segments exist,
  and do they line up with how the business already thinks about its network?
* **Dimensionality reduction** (PCA, t-SNE, UMAP) - is the feature space
  genuinely high-dimensional, or is most variance in a handful of directions?
* **Anomaly detection** (Isolation Forest) - which deliveries look nothing like
  the rest? Because the generator tags injected anomalies, detection can be
  scored against ground truth instead of eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture

from ..config import Config
from ..stages.preprocessing import SplitData
from ..utils import get_logger, new_figure, optional_import, save_figure

__all__ = ["UnsupervisedResults", "run_unsupervised"]

logger = get_logger()


@dataclass
class UnsupervisedResults:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: list[Path] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def _sample(X: pd.DataFrame, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X.to_numpy(dtype=float), np.arange(len(X))
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X.iloc[idx].to_numpy(dtype=float), idx


def _cluster_quality(Z: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette / Calinski-Harabasz / Davies-Bouldin, guarded for degenerate
    labellings (DBSCAN happily returns a single cluster or all noise)."""
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    if len(uniq) < 2 or mask.sum() < 10:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    sub = Z[mask]
    sub_labels = labels[mask]
    # Silhouette is O(n^2); cap it.
    if len(sub) > 5000:
        idx = np.random.default_rng(0).choice(len(sub), 5000, replace=False)
        sil = silhouette_score(sub[idx], sub_labels[idx])
    else:
        sil = silhouette_score(sub, sub_labels)
    return {
        "silhouette": round(float(sil), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(sub, sub_labels)), 2),
        "davies_bouldin": round(float(davies_bouldin_score(sub, sub_labels)), 4),
    }


def run_unsupervised(sd: SplitData, cfg: Config, out_dir: Path,
                     y_reference: pd.Series | None = None) -> UnsupervisedResults:
    res = UnsupervisedResults()
    if not cfg.get("unsupervised.enabled", True):
        logger.info("Unsupervised stage disabled by config")
        return res

    seed = cfg.seed
    plot_dir = out_dir / "plots" / "unsupervised"
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_plots = bool(cfg.get("reports.include_plots", True))
    dpi = int(cfg.get("eda.dpi", 110))

    n_sample = int(cfg.get("unsupervised.sample_size", 12000))
    Z, idx = _sample(sd.X_train, n_sample, seed)
    y_ref = y_reference.iloc[idx].to_numpy() if y_reference is not None else None
    logger.info("Unsupervised on %d rows x %d features", Z.shape[0], Z.shape[1])

    # ---- KMeans: sweep k and pick by silhouette ---------------------------
    km_cfg = cfg.get("unsupervised.clustering.kmeans", {}) or {}
    if km_cfg.get("enabled", True):
        rows = []
        best = (None, -np.inf, None)
        for k in km_cfg.get("k_range", [2, 3, 4, 5, 6, 7, 8]):
            km = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
            labels = km.fit_predict(Z)
            q = _cluster_quality(Z, labels)
            rows.append({"k": int(k), "inertia": round(float(km.inertia_), 2), **q})
            if q["silhouette"] is not None and q["silhouette"] > best[1]:
                best = (int(k), q["silhouette"], labels)
        res.tables["kmeans_sweep"] = pd.DataFrame(rows)
        if best[2] is not None:
            res.labels["kmeans"] = best[2]
            res.summary["kmeans_best_k"] = best[0]
            res.summary["kmeans_best_silhouette"] = round(best[1], 4)
            logger.info("  KMeans: best k=%d (silhouette %.4f)", best[0], best[1])

            if make_plots:
                sweep = res.tables["kmeans_sweep"]
                fig, ax = new_figure(7, 4)
                ax.plot(sweep["k"], sweep["inertia"], "o-", color="#4c7ef3", label="inertia")
                ax.set_xlabel("k"); ax.set_ylabel("inertia")
                ax2 = ax.twinx()
                ax2.plot(sweep["k"], sweep["silhouette"], "s--", color="#e0574c", label="silhouette")
                ax2.set_ylabel("silhouette")
                ax.set_title("KMeans: elbow and silhouette")
                fig.legend(loc="upper right", fontsize=8)
                res.plots.append(save_figure(fig, plot_dir / "kmeans_sweep.png", dpi))

    # ---- DBSCAN ------------------------------------------------------------
    db_cfg = cfg.get("unsupervised.clustering.dbscan", {}) or {}
    if db_cfg.get("enabled", True):
        try:
            db = DBSCAN(eps=float(db_cfg.get("eps", 1.6)),
                        min_samples=int(db_cfg.get("min_samples", 25)),
                        n_jobs=cfg.get("project.n_jobs", -1))
            labels = db.fit_predict(Z)
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            noise = float((labels == -1).mean())
            res.labels["dbscan"] = labels
            res.tables["dbscan"] = pd.DataFrame([{
                "eps": db_cfg.get("eps"), "min_samples": db_cfg.get("min_samples"),
                "n_clusters": n_clusters, "noise_fraction": round(noise, 4),
                **_cluster_quality(Z, labels)}])
            logger.info("  DBSCAN: %d clusters, %.1f%% noise", n_clusters, noise * 100)
        except Exception as exc:
            res.skipped["dbscan"] = str(exc)

    # ---- Gaussian Mixture: select by BIC ----------------------------------
    gmm_cfg = cfg.get("unsupervised.clustering.gmm", {}) or {}
    if gmm_cfg.get("enabled", True):
        rows, best = [], (None, np.inf, None)
        for k in gmm_cfg.get("n_components", [2, 3, 4, 5, 6]):
            try:
                gmm = GaussianMixture(n_components=int(k), covariance_type="full",
                                      random_state=seed, max_iter=200)
                labels = gmm.fit_predict(Z)
                bic, aic = float(gmm.bic(Z)), float(gmm.aic(Z))
                rows.append({"n_components": int(k), "bic": round(bic, 1),
                             "aic": round(aic, 1), **_cluster_quality(Z, labels)})
                if bic < best[1]:
                    best = (int(k), bic, labels)
            except Exception as exc:
                logger.debug("GMM k=%s failed: %s", k, exc)
        if rows:
            res.tables["gmm_sweep"] = pd.DataFrame(rows)
            res.labels["gmm"] = best[2]
            res.summary["gmm_best_components"] = best[0]
            logger.info("  GMM: best n_components=%s (BIC %.0f)", best[0], best[1])

    # ---- PCA ---------------------------------------------------------------
    pca_cfg = cfg.get("unsupervised.dimensionality_reduction.pca", {}) or {}
    if pca_cfg.get("enabled", True):
        n_comp = min(int(pca_cfg.get("n_components", 10)), Z.shape[1], Z.shape[0])
        pca = PCA(n_components=n_comp, random_state=seed)
        emb = pca.fit_transform(Z)
        res.embeddings["pca"] = emb
        evr = pca.explained_variance_ratio_
        res.tables["pca_variance"] = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(n_comp)],
            "explained_variance_ratio": np.round(evr, 5),
            "cumulative": np.round(np.cumsum(evr), 5)})
        res.summary["pca_components_for_90pct"] = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
        logger.info("  PCA: %d components explain %.1f%% of variance",
                    n_comp, evr.sum() * 100)

        # Which original features drive PC1/PC2 - makes the embedding readable.
        load = pd.DataFrame(pca.components_[:2].T, columns=["PC1", "PC2"],
                            index=sd.feature_names[:Z.shape[1]])
        load["magnitude"] = load.abs().sum(axis=1)
        res.tables["pca_loadings"] = (load.sort_values("magnitude", ascending=False)
                                      .head(20).round(4).reset_index()
                                      .rename(columns={"index": "feature"}))

        if make_plots:
            fig, ax = new_figure(7, 4)
            ax.bar(range(1, n_comp + 1), evr, color="#4c7ef3")
            ax.plot(range(1, n_comp + 1), np.cumsum(evr), "o-", color="#e0574c")
            ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio")
            ax.set_title("PCA scree")
            res.plots.append(save_figure(fig, plot_dir / "pca_scree.png", dpi))

    # ---- t-SNE -------------------------------------------------------------
    tsne_cfg = cfg.get("unsupervised.dimensionality_reduction.tsne", {}) or {}
    if tsne_cfg.get("enabled", True):
        try:
            m = min(int(tsne_cfg.get("sample_size", 4000)), len(Z))
            sub_idx = np.random.default_rng(seed).choice(len(Z), m, replace=False)
            # PCA pre-reduction is standard practice: it denoises and makes
            # t-SNE tractable on wide tabular data.
            pre = PCA(n_components=min(30, Z.shape[1]), random_state=seed).fit_transform(Z[sub_idx])
            emb = TSNE(n_components=int(tsne_cfg.get("n_components", 2)),
                       perplexity=float(tsne_cfg.get("perplexity", 30)),
                       init="pca", random_state=seed).fit_transform(pre)
            res.embeddings["tsne"] = emb
            if make_plots:
                res.plots.append(_scatter(
                    emb, y_ref[sub_idx] if y_ref is not None else None,
                    "t-SNE projection", plot_dir / "tsne.png", dpi))
            logger.info("  t-SNE: embedded %d points", m)
        except Exception as exc:
            res.skipped["tsne"] = str(exc)

    # ---- UMAP (optional dependency) ---------------------------------------
    umap_cfg = cfg.get("unsupervised.dimensionality_reduction.umap", {}) or {}
    if umap_cfg.get("enabled", True):
        umap_mod = optional_import("umap")
        if umap_mod is None:
            res.skipped["umap"] = "umap-learn not installed"
            logger.warning("Skipping UMAP - umap-learn not installed")
        else:
            try:
                reducer = umap_mod.UMAP(
                    n_components=int(umap_cfg.get("n_components", 2)),
                    n_neighbors=int(umap_cfg.get("n_neighbors", 15)),
                    min_dist=float(umap_cfg.get("min_dist", 0.1)),
                    random_state=seed)
                emb = reducer.fit_transform(Z)
                res.embeddings["umap"] = emb
                if make_plots:
                    res.plots.append(_scatter(emb, y_ref, "UMAP projection",
                                              plot_dir / "umap.png", dpi))
                logger.info("  UMAP: embedded %d points", len(emb))
            except Exception as exc:
                res.skipped["umap"] = str(exc)

    # ---- Isolation Forest --------------------------------------------------
    iso_cfg = cfg.get("unsupervised.anomaly_detection.isolation_forest", {}) or {}
    if iso_cfg.get("enabled", True):
        iso = IsolationForest(
            n_estimators=int(iso_cfg.get("n_estimators", 200)),
            contamination=float(iso_cfg.get("contamination", 0.02)),
            random_state=seed, n_jobs=cfg.get("project.n_jobs", -1))
        flags = iso.fit_predict(Z)
        scores = iso.score_samples(Z)
        is_anom = flags == -1
        res.labels["isolation_forest"] = is_anom.astype(int)
        res.summary["anomaly_rate"] = round(float(is_anom.mean()), 4)
        res.tables["anomaly_scores"] = pd.DataFrame({
            "statistic": ["min", "p1", "p5", "median", "max"],
            "score": np.round([scores.min(), np.percentile(scores, 1),
                               np.percentile(scores, 5), np.median(scores),
                               scores.max()], 5)})

        # Which features are most extreme in flagged rows vs the rest?
        diff = (pd.DataFrame(Z, columns=sd.feature_names[:Z.shape[1]])
                .assign(_flag=is_anom).groupby("_flag").mean().T)
        if diff.shape[1] == 2:
            diff.columns = ["normal_mean", "anomaly_mean"]
            diff["abs_gap"] = (diff["anomaly_mean"] - diff["normal_mean"]).abs()
            res.tables["anomaly_feature_gaps"] = (
                diff.sort_values("abs_gap", ascending=False).head(15).round(4)
                .reset_index().rename(columns={"index": "feature"}))
        logger.info("  Isolation Forest: flagged %.2f%% of rows", is_anom.mean() * 100)

        if make_plots and "pca" in res.embeddings:
            res.plots.append(_scatter(
                res.embeddings["pca"][:, :2], is_anom.astype(int),
                "Isolation Forest flags on PCA projection",
                plot_dir / "isolation_forest_pca.png", dpi))

    # ---- Cluster profiling -------------------------------------------------
    if "kmeans" in res.labels:
        prof = (pd.DataFrame(Z, columns=sd.feature_names[:Z.shape[1]])
                .assign(cluster=res.labels["kmeans"]).groupby("cluster").mean())
        # Rank features by how much they separate clusters.
        spread = (prof.max() - prof.min()).sort_values(ascending=False)
        res.tables["kmeans_profile"] = (prof[spread.head(12).index].round(3)
                                        .reset_index())
        sizes = pd.Series(res.labels["kmeans"]).value_counts().sort_index()
        res.tables["kmeans_sizes"] = pd.DataFrame({
            "cluster": sizes.index, "n": sizes.to_numpy(),
            "share_pct": (sizes / sizes.sum() * 100).round(2).to_numpy()})

    return res


def _scatter(emb: np.ndarray, colour, title: str, path: Path, dpi: int) -> Path:
    fig, ax = new_figure(7, 6)
    if colour is not None:
        colour = np.asarray(colour)
        for value in np.unique(colour)[:6]:
            m = colour == value
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.55, label=str(value))
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.55, color="#4c7ef3")
    ax.set_title(title)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    return save_figure(fig, path, dpi)
