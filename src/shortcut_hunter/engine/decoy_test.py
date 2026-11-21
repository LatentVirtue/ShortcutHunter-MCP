"""Audit 2 — Synthetic Decoy / Confounder Stress Test ('Clever Hans' injection).

Injects a noisy target-proxy column that agrees with the target on ~92% of
training rows (flip probability 0.08) but is inverted on an out-of-distribution
evaluation split (flip probability 0.85), then measures:
  * ``decoy_reliance_ratio``  — normalized importance share captured by the decoy
  * ``metric_collapse``       — in-distribution metric minus inverted-OOD metric
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .config import AuditConfig, DecoyResult
from .models import make_model, normalized_importances

DECOY_COLUMN = "decoy_clever_hans"


def make_decoy(y: pd.Series, flip_prob: float, rng: np.random.RandomState,
               task_type: str) -> np.ndarray:
    """Noisy copy of the target: with probability ``flip_prob`` the copy is flipped."""
    y_arr = y.to_numpy()
    if task_type == "classification":
        classes = np.sort(np.unique(y_arr))
        flipped = np.roll(classes, 1)[np.searchsorted(classes, y_arr)]
        flip_mask = rng.random(len(y_arr)) < flip_prob
        return np.where(flip_mask, flipped, y_arr)
    # regression: "inverted" means sign-flipped around the training median
    med = float(np.median(y_arr.astype(float)))
    base = y_arr.astype(float) + rng.normal(0.0, 0.1 * (float(np.std(y_arr)) or 1.0), len(y_arr))
    inverted = 2.0 * med - base
    flip_mask = rng.random(len(y_arr)) < flip_prob
    return np.where(flip_mask, inverted, base)


def run_decoy_audit(X: pd.DataFrame, y: pd.Series, config: AuditConfig,
                   task_type: str, scoring: str,
                   n_eval_repeats: int = 3) -> DecoyResult:
    rng = np.random.RandomState(config.random_state)
    y_arr = np.asarray(y)

    if task_type == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                         random_state=config.random_state)
        train_idx, test_idx = next(iter(splitter.split(X, y_arr)))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.25, random_state=config.random_state)

    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    id_scores: list[float] = []
    ood_scores: list[float] = []
    shares: list[float] = []
    top_importances: dict[str, float] = {}

    for _ in range(max(n_eval_repeats, 1)):
        decoy_train = make_decoy(y_train, config.decoy_flip_prob_train, rng, task_type)
        decoy_id = make_decoy(y_test, config.decoy_flip_prob_train, rng, task_type)
        decoy_ood = make_decoy(y_test, config.decoy_flip_prob_ood, rng, task_type)

        X_train = X_train_raw.copy()
        X_train[DECOY_COLUMN] = decoy_train
        X_id = X_test_raw.copy(); X_id[DECOY_COLUMN] = decoy_id
        X_ood = X_test_raw.copy(); X_ood[DECOY_COLUMN] = decoy_ood

        model = make_model(config.model_family, task_type, config.random_state)
        model.fit(X_train, y_train)

        id_scores.append(float(_safe_score(model, X_id, y_test, scoring, task_type)))
        ood_scores.append(float(_safe_score(model, X_ood, y_test, scoring, task_type)))

        shares_dict = normalized_importances(
            config.model_family, X_train, y_train, list(X_train.columns),
            task_type, scoring, config.random_state,
        )
        shares.append(shares_dict.get(DECOY_COLUMN, 0.0))
        top_importances = shares_dict

    reliance = float(np.mean(shares))
    id_metric = float(np.mean(id_scores))
    ood_metric = float(np.mean(ood_scores))
    return DecoyResult(
        decoy_column=DECOY_COLUMN,
        flip_prob_train=config.decoy_flip_prob_train,
        flip_prob_ood=config.decoy_flip_prob_ood,
        in_distribution_metric=round(id_metric, 4),
        ood_metric=round(ood_metric, 4),
        metric_collapse=round(id_metric - ood_metric, 4),
        decoy_reliance_ratio=round(reliance, 4),
        top_importances={k: round(v, 4) for k, v in
                         sorted(top_importances.items(), key=lambda kv: -kv[1])[:6]},
    )


def _safe_score(model, X, y, scoring: str, task_type: str) -> float:
    """Score with a fallback so single-class folds don't blow up AUC."""
    from sklearn.metrics import get_scorer, accuracy_score, r2_score
    try:
        return float(get_scorer(scoring)(model, X, y))
    except (ValueError, IndexError):
        if scoring in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo"}:
            pred = model.predict(X)
            return float(accuracy_score(y, pred))
        pred = model.predict(X)
        return float(r2_score(y, pred))
