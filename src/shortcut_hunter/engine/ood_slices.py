"""Audit 3 — Covariate-Shift Slice Audit (PCA + K-Means clusters,
Leave-One-Cluster-Out OOD generalization vs. standard random i.i.d. K-Fold CV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import AuditConfig, ClusterShiftFeature, OodAuditResult, SliceResult
from .models import cv_score, make_model


def run_ood_audit(X: pd.DataFrame, y: pd.Series, config: AuditConfig,
                  task_type: str, scoring: str,
                  ) -> tuple[OodAuditResult, pd.DataFrame, pd.DataFrame]:
    """Cluster the feature space, compare i.i.d. CV against LOCO evaluation.

    Returns:
        result:     structured OodAuditResult
        slice_df:   per-cluster metrics + long-form centroid z-scores
        pred_df:    per-row cluster assignment, PCA coords, LOCO predictions/errors
    """
    features = list(X.columns)
    X_num = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(X_num, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_num = np.where(np.isfinite(X_num), X_num, med)

    scaler = StandardScaler()
    Z = scaler.fit_transform(X_num)  # z-scored originals; cluster means are z-deviations

    n_comp = int(min(8, max(2, min(Z.shape[1], 10)), Z.shape[0] - 1))
    pcs = PCA(n_components=n_comp, random_state=config.random_state).fit_transform(Z)

    km = KMeans(n_clusters=config.n_ood_clusters, n_init=10,
                random_state=config.random_state)
    labels = km.fit_predict(pcs)

    # --- i.i.d. baseline ------------------------------------------------------ #
    iid_mean, iid_std = cv_score(config.model_family, X, y, features, task_type,
                                 scoring, config.cv_folds, config.random_state)

    # --- Leave-One-Cluster-Out ------------------------------------------------ #
    y_arr = np.asarray(y)
    pred_rows = pd.DataFrame(index=X.index)
    pred_rows["cluster_id"] = labels
    pred_rows["pc1"] = pcs[:, 0]
    pred_rows["pc2"] = pcs[:, 1]

    cluster_metrics: list[tuple[int, int, float, str | None]] = []
    for c in range(config.n_ood_clusters):
        mask = labels == c
        n_rows = int(mask.sum())
        note = None
        if n_rows < 2:
            metric, note = float("nan"), "cluster too small to evaluate"
        elif task_type == "classification" and len(np.unique(y_arr[mask])) < 2 and scoring.startswith("roc_auc"):
            metric, note = float("nan"), "single-class cluster (AUC undefined)"
        else:
            model = make_model(config.model_family, task_type, config.random_state)
            X_tr = X.iloc[~mask]
            y_tr = y.iloc[~mask] if hasattr(y, "iloc") else y[~mask]
            model.fit(X_tr, y_tr)
            X_te = X.iloc[mask]
            y_te = y_arr[mask]
            metric = _safe_metric(model, X_te, y_te, scoring, task_type)
            if np.isnan(metric):
                note = "metric undefined for this split"
            # per-row predictions
            if task_type == "classification":
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_te)
                    pred_rows.loc[X_te.index, "y_prob"] = proba[:, -1]
                pred_rows.loc[X_te.index, "y_pred"] = model.predict(X_te)
                pred_rows.loc[X_te.index, "y_true"] = y_te
                pred_rows.loc[X_te.index, "row_error"] = (
                    np.asarray(model.predict(X_te)) != y_te).astype(int)
            else:
                yhat = np.asarray(model.predict(X_te), dtype=float)
                pred_rows.loc[X_te.index, "y_pred"] = yhat
                pred_rows.loc[X_te.index, "y_true"] = y_te
                pred_rows.loc[X_te.index, "row_error"] = np.abs(yhat - y_te.astype(float))

        cluster_metrics.append((c, n_rows, float(metric), note))

    valid = [(c, n, m, note) for c, n, m, note in cluster_metrics if np.isfinite(m)]
    if not valid:
        raise RuntimeError("LOCO audit produced no valid cluster metrics.")
    loco_mean = float(np.mean([m for _, _, m, _ in valid]))
    worst_cluster_id, worst_n, worst_metric, worst_note = min(valid, key=lambda t: t[2])
    iid_to_ood_drop = float(iid_mean - worst_metric)

    # --- centroid z-scores ------------------------------------------------------ #
    z_by_cluster = {c: Z[labels == c].mean(axis=0) for c in range(config.n_ood_clusters)}

    clusters: list[SliceResult] = []
    slice_rows: list[dict] = []
    for c, n, m, note in cluster_metrics:
        z = z_by_cluster[c]
        order = np.argsort(-np.abs(z))[:5]
        shift = [ClusterShiftFeature(feature=features[j], z_score=round(float(z[j]), 3))
                 for j in order]
        drop = float(iid_mean - m) if np.isfinite(m) else float("nan")
        clusters.append(SliceResult(
            cluster_id=int(c), n_rows=int(n),
            loco_metric=None if not np.isfinite(m) else round(m, 4),
            drop_from_iid=None if not np.isfinite(drop) else round(drop, 4),
            metric_note=note,
            top_shift_features=shift,
        ))
        for s in shift:
            slice_rows.append({
                "cluster_id": int(c), "n_rows": int(n),
                "loco_metric": None if not np.isfinite(m) else round(m, 4),
                "iid_metric": round(float(iid_mean), 4),
                "drop_from_iid": None if not np.isfinite(drop) else round(drop, 4),
                "feature": s.feature, "z_score": s.z_score,
            })

    result = OodAuditResult(
        metric=scoring,
        n_clusters=config.n_ood_clusters,
        iid_cv_metric=round(float(iid_mean), 4),
        iid_cv_std=round(float(iid_std), 4),
        loco_mean_metric=round(loco_mean, 4),
        loco_worst_metric=round(worst_metric, 4),
        iid_to_ood_drop=round(iid_to_ood_drop, 4),
        worst_cluster_id=int(worst_cluster_id),
        clusters=sorted(clusters, key=lambda s: s.cluster_id),
    )
    return result, pd.DataFrame(slice_rows), pred_rows


def _safe_metric(model, X_te, y_te, scoring: str, task_type: str) -> float:
    from sklearn.metrics import accuracy_score, get_scorer, r2_score
    try:
        return float(get_scorer(scoring)(model, X_te, y_te))
    except (ValueError, IndexError):
        try:
            if scoring.startswith("roc_auc"):
                return float(accuracy_score(y_te, model.predict(X_te)))
            return float(r2_score(y_te, model.predict(X_te)))
        except Exception:
            return float("nan")
