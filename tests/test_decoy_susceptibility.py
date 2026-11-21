"""Audit 2: Clever-Hans decoy injection and shortcut-capture susceptibility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcut_hunter.engine.config import AuditConfig
from shortcut_hunter.engine.decoy_test import DECOY_COLUMN, make_decoy, run_decoy_audit
from shortcut_hunter.engine.orchestrator import prepare_xy


@pytest.fixture(scope="module")
def xy(genuine_only_df):
    X, y, task_type, _ = prepare_xy(genuine_only_df, "y")
    return X, y, task_type


def test_make_decoy_respects_flip_probabilities():
    rng = np.random.RandomState(0)
    y = (rng.random(5000) < 0.5).astype(int)
    train = make_decoy(pd.Series(y), 0.08, rng, "classification")
    ood = make_decoy(pd.Series(y), 0.85, rng, "classification")
    agree_train = float((train == y).mean())
    agree_ood = float((ood == y).mean())
    assert abs(agree_train - 0.92) < 0.02
    assert abs(agree_ood - 0.15) < 0.02


def test_lightgbm_captures_injected_decoy(xy):
    X, y, task_type = xy
    config = AuditConfig(target_column="y", model_family="lightgbm", random_state=2)
    result = run_decoy_audit(X, y, config, task_type, "roc_auc", n_eval_repeats=2)

    assert 0.0 < result.decoy_reliance_ratio < 1.0
    assert result.decoy_reliance_ratio > 0.15
    assert DECOY_COLUMN in result.top_importances
    assert result.in_distribution_metric > result.ood_metric
    assert result.metric_collapse > 0.1


def test_regularized_linear_surfaces_its_own_vulnerability(xy):
    """The test is comparative: whatever the L2-regularized linear baseline's
    collapse is, both runs must report a consistent, in-range susceptibility."""
    X, y, task_type = xy
    cfg_lin = AuditConfig(target_column="y", model_family="linear", random_state=2)
    lin = run_decoy_audit(X, y, cfg_lin, task_type, "roc_auc", n_eval_repeats=2)
    assert 0.0 < lin.decoy_reliance_ratio < 1.0
    assert lin.in_distribution_metric > lin.ood_metric
    assert lin.metric_collapse > 0.05

    cfg_lgbm = AuditConfig(target_column="y", model_family="lightgbm", random_state=2)
    tree = run_decoy_audit(X, y, cfg_lgbm, task_type, "roc_auc", n_eval_repeats=2)
    # On this dataset the tree ensemble relies on the shortcut at least as much
    # as the L2-penalized linear model (documented in the README walkthrough).
    assert tree.decoy_reliance_ratio >= lin.decoy_reliance_ratio - 0.02


def test_decoy_flip_config_is_respected():
    cfg = AuditConfig(target_column="y", decoy_flip_prob_train=0.10, decoy_flip_prob_ood=0.90)
    assert cfg.decoy_flip_prob_train == 0.10
    assert cfg.decoy_flip_prob_ood == 0.90
