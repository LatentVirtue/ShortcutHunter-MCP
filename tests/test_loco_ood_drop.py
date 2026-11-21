"""Audit 3: PCA/K-Means covariate-shift slices + Leave-One-Cluster-Out drop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcut_hunter.engine.config import AuditConfig
from shortcut_hunter.engine.ood_slices import run_ood_audit
from shortcut_hunter.engine.orchestrator import prepare_xy


@pytest.fixture(scope="module")
def xy_full(synthetic_df):
    X, y, task_type, _ = prepare_xy(synthetic_df, "y")
    return X, y, task_type


@pytest.fixture(scope="module")
def loco(xy_full):
    X, y, task_type = xy_full
    config = AuditConfig(target_column="y", model_family="lightgbm",
                        n_ood_clusters=5, random_state=5)
    result, slice_df, pred_df = run_ood_audit(X, y, config, task_type, "roc_auc")
    return result, slice_df, pred_df, xy_full


def test_worst_loco_cluster_is_the_shifted_regime(loco, synthetic_df):
    result, slice_df, pred_df, _ = loco
    true_shift = synthetic_df["_true_cluster"] == 1
    worst_rows = pred_df.index[pred_df["cluster_id"] == result.worst_cluster_id]
    purity = float(true_shift.loc[worst_rows].mean())
    assert purity > 0.6, f"worst cluster {result.worst_cluster_id} purity={purity:.2f}"
    assert result.iid_to_ood_drop > 0.10
    assert result.loco_worst_metric < result.iid_cv_metric - 0.10


def test_slice_metrics_cover_all_clusters_with_z_scores(loco):
    result, slice_df, _, _ = loco
    assert set(slice_df["cluster_id"]) == set(range(result.n_clusters))
    for c in result.clusters:
        assert c.n_rows > 0
        assert len(c.top_shift_features) > 0
    worst = [c for c in result.clusters if c.cluster_id == result.worst_cluster_id][0]
    zmag = {s.feature: abs(s.z_score) for s in worst.top_shift_features}
    shifted = {"g1", "g2", "g5", "n2"}
    assert any(zmag.get(f, 0) > 0.8 for f in shifted), zmag


def test_predictions_table_complete_and_usable(loco):
    result, _, pred_df, (X, _, _) = loco
    assert len(pred_df) == len(X)
    assert {"cluster_id", "pc1", "pc2", "y_true", "y_pred", "row_error"} <= set(pred_df.columns)
    assert pred_df["cluster_id"].between(0, result.n_clusters - 1).all()
    worst_rows = pred_df[pred_df["cluster_id"] == result.worst_cluster_id]
    assert float(worst_rows["row_error"].mean()) > 0.2


def test_ood_result_is_self_consistent(loco):
    result, _, _, _ = loco
    valid = [c.loco_metric for c in result.clusters if c.loco_metric is not None]
    assert result.loco_worst_metric == pytest.approx(min(valid), abs=1e-3)
    assert result.iid_to_ood_drop == pytest.approx(
        result.iid_cv_metric - result.loco_worst_metric, abs=1e-3)
