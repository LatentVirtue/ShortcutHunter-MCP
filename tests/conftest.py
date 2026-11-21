"""Shared fixtures. The expensive full-audit run is done ONCE per test session."""
from __future__ import annotations

import pytest

from shortcut_hunter.engine.config import AuditConfig
from shortcut_hunter.engine.orchestrator import run_audit
from shortcut_hunter.engine.synthetic import generate_clever_hans_dataset


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_clever_hans_dataset(n_rows=1200, seed=13)


@pytest.fixture(scope="session")
def genuine_only_df(synthetic_df):
    """Same data with the built-in batch_id_proxy removed (genuine signal only)."""
    return synthetic_df.drop(columns=["batch_id_proxy"])


@pytest.fixture(scope="session")
def full_audit(tmp_path_factory, synthetic_df):
    """One full orchestrator run over the Clever-Hans dataset, shared read-only."""
    out = tmp_path_factory.mktemp("full_audit")
    config = AuditConfig(target_column="y", model_family="lightgbm")
    scorecard, run_dir = run_audit(synthetic_df, "y", config=config, output_dir=out)
    return scorecard, run_dir


@pytest.fixture(scope="session")
def audit_csv(tmp_path_factory, synthetic_df):
    out = tmp_path_factory.mktemp("data") / "clever_hans.csv"
    synthetic_df.to_csv(out, index=False)
    return out
