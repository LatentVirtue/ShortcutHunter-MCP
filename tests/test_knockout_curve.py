"""Audit 1: iterative knockout decay curve + proxy-substitution detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcut_hunter.engine.config import AuditConfig
from shortcut_hunter.engine.knockout import resolve_knockout_steps, run_knockout_audit
from shortcut_hunter.engine.orchestrator import prepare_xy


@pytest.fixture(scope="module")
def xy(synthetic_df):
    X, y, task_type, _ = prepare_xy(synthetic_df, "y")
    return X, y, task_type


def _config(**kw) -> AuditConfig:
    cfg = AuditConfig(target_column="y", model_family="lightgbm", random_state=1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_knockout_flags_batch_id_proxy_first(xy):
    X, y, task_type = xy
    config = _config(knockout_steps=4)
    result, trace = run_knockout_audit(X, y, config, task_type, "roc_auc")

    assert result.removal_sequence[0] == "batch_id_proxy"
    assert trace.iloc[1]["removed_feature"] == "batch_id_proxy"
    # Removing the proxy must produce a measurable cliff -> single-feature shortcut.
    assert trace.iloc[1]["metric_drop_from_previous"] > 0.15
    assert "SINGLE_FEATURE_SHORTCUT_DEPENDENCY" in result.patterns
    assert result.max_single_step_drop > 0.15


def test_knockout_curve_is_monotonic_enough_and_trace_shape(xy):
    X, y, task_type = xy
    config = _config(knockout_steps=3)
    result, trace = run_knockout_audit(X, y, config, task_type, "roc_auc")
    assert len(trace) == 4                     # 3 removals + baseline
    assert list(trace["features_remaining"]) == [11, 10, 9, 8]
    assert len(result.steps) == 4
    assert result.iid_metric == pytest.approx(float(trace.iloc[0]["metric_mean"]), abs=1e-3)
    # After the proxy cliff, the genuine multi-feature signal decays gradually:
    # total decay across 3 removals < what one more proxy-like cliff would cost.
    assert result.total_decay < 0.6
    after_proxy = float(trace.iloc[2]["metric_mean"]) - float(trace.iloc[3]["metric_mean"])
    assert after_proxy < 0.15


def test_resolve_knockout_steps_defaults():
    cfg = AuditConfig(target_column="y")
    assert resolve_knockout_steps(cfg, 11) == 8      # min(8, n-1)
    assert resolve_knockout_steps(cfg, 5) == 4
    cfg2 = AuditConfig(target_column="y", knockout_steps=6)
    assert resolve_knockout_steps(cfg2, 11) == 6


def test_redundant_proxy_cluster_detected():
    """Two near-duplicate features carrying all signal: removing one must leave
    the metric flat and report the surviving partner as a promoted proxy."""
    rng = np.random.RandomState(3)
    n = 800
    f1 = rng.normal(0, 1, n)
    label = (f1 + rng.normal(0, 0.45, n) > 0).astype(int)
    f2 = f1 * 0.96 + rng.normal(0, 0.1, n)          # near-duplicate proxy
    noise = rng.normal(0, 1, (n, 3))
    X = pd.DataFrame(np.column_stack([f1, f2, noise]), columns=["f1", "f2", "z1", "z2", "z3"])
    y = pd.Series(label)

    config = AuditConfig(target_column="y", model_family="lightgbm",
                        knockout_steps=2, random_state=0, cv_folds=3)
    result, _ = run_knockout_audit(X, y, config, "classification", "roc_auc")

    assert "REDUNDANT_PROXY_CLUSTER" in result.patterns
    pair = result.shortcut_pairs[0]
    assert {pair.dropped_feature, pair.promoted_proxy} == {"f1", "f2"}
    assert pair.correlation > 0.9
    assert pair.metric_drop <= 0.03
