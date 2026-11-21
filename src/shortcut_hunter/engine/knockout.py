"""Audit 1 — Iterative Top-Feature Knockout ('Signal Depth Audit').

Repeatedly removes the #1 most predictive feature, retrains, and records the
performance decay curve plus which secondary proxy gets promoted to replace the
removed feature (with its correlation to the removed feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AuditConfig, KnockoutResult, KnockoutStep, ShortcutPair
from .models import cv_score, normalized_importances, pearson_abs


def resolve_knockout_steps(config: AuditConfig, n_features: int) -> int:
    if config.knockout_steps is not None:
        return min(config.knockout_steps, n_features - 1)
    return min(8, n_features - 1)


def run_knockout_audit(X: pd.DataFrame, y: pd.Series, config: AuditConfig,
                      task_type: str, scoring: str) -> tuple[KnockoutResult, pd.DataFrame]:
    """Iteratively knock out top features and trace the decay curve.

    Returns the structured result and a flat trace DataFrame
    (one row per evaluated state) for the parquet artifact and the report.
    """
    features = list(X.columns)
    n_removals = resolve_knockout_steps(config, len(features))

    steps: list[KnockoutStep] = []
    trace_rows: list[dict] = []
    removal_sequence: list[str] = []
    shortcut_pairs: list[ShortcutPair] = []
    patterns: set[str] = set()

    prev_shares: dict[str, float] | None = None
    prev_metric: float | None = None
    dropped_feature: str | None = None

    for i in range(n_removals + 1):
        metric, std = cv_score(config.model_family, X, y, features, task_type,
                               scoring, config.cv_folds, config.random_state)
        shares = normalized_importances(config.model_family, X, y, features,
                                       task_type, scoring, config.random_state)
        top_next = max(shares, key=shares.get) if features else None

        promoted_proxy: str | None = None
        proxy_corr: float | None = None
        drop_from_prev: float | None = None
        if prev_metric is not None:
            drop_from_prev = float(prev_metric - metric)
        if prev_shares is not None and dropped_feature is not None:
            deltas = {f: shares[f] - prev_shares.get(f, 0.0) for f in features}
            if deltas:
                promoted_proxy = max(deltas, key=deltas.get)
                proxy_corr = pearson_abs(X[dropped_feature], X[promoted_proxy])
                if (proxy_corr >= config.redundant_corr_threshold
                        and abs(drop_from_prev or 0.0) <= config.redundant_drop_threshold):
                    patterns.add("REDUNDANT_PROXY_CLUSTER")
                    shortcut_pairs.append(ShortcutPair(
                        dropped_feature=dropped_feature,
                        promoted_proxy=promoted_proxy,
                        correlation=round(float(proxy_corr), 4),
                        metric_drop=round(float(drop_from_prev or 0.0), 4),
                    ))
        if drop_from_prev is not None and drop_from_prev > config.cliff_threshold:
            patterns.add("SINGLE_FEATURE_SHORTCUT_DEPENDENCY")

        top_importances = dict(sorted(shares.items(), key=lambda kv: -kv[1])[:5])
        step = KnockoutStep(
            step=i,
            features_remaining=len(features),
            metric_mean=round(float(metric), 4),
            metric_std=round(float(std), 4),
            metric_drop_from_previous=None if drop_from_prev is None else round(drop_from_prev, 4),
            removed_feature=dropped_feature,
            next_top_feature=top_next,
            newly_promoted_proxy=promoted_proxy,
            proxy_correlation=None if proxy_corr is None else round(float(proxy_corr), 4),
            top_importances=top_importances,
        )
        steps.append(step)
        trace_rows.append({
            "step": i,
            "features_remaining": len(features),
            "metric_mean": float(metric),
            "metric_std": float(std),
            "metric_drop_from_previous": drop_from_prev,
            "removed_feature": dropped_feature,
            "next_top_feature": top_next,
            "newly_promoted_proxy": promoted_proxy,
            "proxy_correlation": proxy_corr,
            "top_importances": top_importances,
        })

        if top_next is None:
            break
        features.remove(top_next)
        removal_sequence.append(top_next)
        # Correlations must be computed against the ORIGINAL matrix; remember
        # the dropped column via X even after removing it from `features`.
        dropped_feature = top_next
        prev_shares = shares
        prev_metric = metric

    max_drop = max((s.metric_drop_from_previous for s in steps
                    if s.metric_drop_from_previous is not None), default=0.0)
    if not patterns:
        patterns.add("STABLE_DISTRIBUTED_SIGNAL")

    result = KnockoutResult(
        metric=scoring,
        iid_metric=steps[0].metric_mean,
        final_metric=steps[-1].metric_mean,
        total_decay=round(steps[0].metric_mean - steps[-1].metric_mean, 4),
        max_single_step_drop=round(float(max_drop), 4),
        removal_sequence=removal_sequence,
        patterns=sorted(patterns),
        shortcut_pairs=shortcut_pairs,
        steps=steps,
    )
    trace = pd.DataFrame(trace_rows)
    return result, trace
