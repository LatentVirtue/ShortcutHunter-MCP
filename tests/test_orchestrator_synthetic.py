"""End-to-end orchestrator behavior on the synthetic Clever-Hans dataset.

Covers the spec's Definition-of-Done assertion: the orchestrator must flag
``batch_id_proxy`` and detect the OOD drop on the built-in benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shortcut_hunter.reporting.manifest import TOKEN_LIMIT, estimate_tokens


def test_orchestrator_flags_batch_id_proxy_and_ood_drop(full_audit):
    scorecard, run_dir = full_audit

    # --- the shortcut is flagged ------------------------------------------- #
    assert scorecard.knockout.removal_sequence[0] == "batch_id_proxy"
    assert "SINGLE_FEATURE_SHORTCUT_DEPENDENCY" in scorecard.patterns
    codes = {f.code for f in scorecard.findings}
    assert "SINGLE_FEATURE_SHORTCUT_DEPENDENCY" in codes
    assert "COVARIATE_SHIFT_FRAGILITY" in codes

    # --- the OOD drop is detected ------------------------------------------ #
    assert scorecard.ood_degradation > 0.10
    assert scorecard.loco_ood_score < scorecard.iid_cv_score - 0.10
    assert scorecard.risk_level == "HIGH_SHORTCUT_RISK"

    # --- metric used for a binary target is roc_auc ------------------------- #
    assert scorecard.metric == "roc_auc"
    assert scorecard.dataset.task_type == "classification"


def test_dual_artifacts_exist_and_are_valid(full_audit):
    scorecard, run_dir = full_audit
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.html"
    ko_path = run_dir / "knockout_trace.parquet"
    slice_path = run_dir / "slice_metrics.parquet"
    pred_path = run_dir / "ood_predictions.parquet"
    for p in (manifest_path, report_path, ko_path, slice_path, pred_path):
        assert p.exists(), f"missing artifact {p}"
        assert p.stat().st_size > 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["risk_level"] == scorecard.risk_level
    assert manifest["knockout"]["removal_sequence"][0] == "batch_id_proxy"
    assert manifest["_token_estimate"] < TOKEN_LIMIT

    html = report_path.read_text(encoding="utf-8")
    assert "<html" in html.lower() and "plotly" in html.lower()
    assert "Fragility Scorecard" in html
    assert "batch_id_proxy" in html  # knockout hover text carries the feature names

    for p in (ko_path, slice_path, pred_path):
        df = pd.read_parquet(p)
        assert len(df) > 0


def test_manifest_token_budget(full_audit):
    scorecard, run_dir = full_audit
    text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert estimate_tokens(text) < TOKEN_LIMIT, f"manifest too big: {len(text)} chars"


def test_engine_layer_has_zero_mcp_imports():
    engine_dir = Path(__file__).resolve().parents[1] / "src" / "shortcut_hunter" / "engine"
    offenders = []
    for py in engine_dir.glob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import mcp", "from mcp")) or "import mcp." in s:
                offenders.append(f"{py.name}:{i}: {s}")
    assert not offenders, "engine/ must not import mcp: " + "; ".join(offenders)
