"""MCP surface: FastMCP registration, run_shortcut_audit / inspect_ood_cluster
tools, and the scorecard resource."""
from __future__ import annotations

import asyncio
import json

import pytest

from shortcut_hunter.mcp_server import (
    inspect_ood_cluster,
    mcp,
    run_shortcut_audit,
)


@pytest.fixture(scope="module")
def mcp_run(tmp_path_factory, audit_csv):
    out_dir = tmp_path_factory.mktemp("mcp_runs")
    manifest = run_shortcut_audit(str(audit_csv), "y", model_family="lightgbm",
                                 knockout_steps=4, n_ood_clusters=5,
                                 output_dir=str(out_dir))
    return manifest, out_dir


def test_fastmcp_server_registers_tools_and_resources():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"run_shortcut_audit", "inspect_ood_cluster"} <= names
    resources = asyncio.run(mcp.list_resources())
    templates = asyncio.run(mcp.list_resource_templates())
    uris = {str(r.uri) for r in resources} | {str(t.uriTemplate) for t in templates}
    assert any("runs/{run_id}/scorecard" in u or "runs/{" in u for u in uris), uris


def test_run_shortcut_audit_tool_end_to_end(mcp_run):
    manifest, out_dir = mcp_run
    run_id = manifest["run_id"]
    run_dir = out_dir / run_id

    assert manifest["risk_level"] == "HIGH_SHORTCUT_RISK"
    assert manifest["knockout"]["removal_sequence"][0] == "batch_id_proxy"
    assert manifest["summary"]["ood_degradation"] > 0.10
    assert manifest["_token_estimate"] < 2000

    for name in ("manifest.json", "report.html", "knockout_trace.parquet",
                 "slice_metrics.parquet", "ood_predictions.parquet"):
        assert (run_dir / name).exists(), f"missing {name}"

    # every finding has an actionable message
    findings = {f["code"]: f for f in manifest["findings"]}
    assert "SINGLE_FEATURE_SHORTCUT_DEPENDENCY" in findings
    assert "COVARIATE_SHIFT_FRAGILITY" in findings


def test_inspect_ood_cluster_tool(mcp_run):
    manifest, out_dir = mcp_run
    run_id = manifest["run_id"]
    worst = manifest["ood"]["worst_cluster_id"]
    info = inspect_ood_cluster(run_id=run_id, cluster_id=worst, output_dir=str(out_dir))

    assert info["run_id"] == run_id
    assert info["n_rows"] > 0
    assert info["centroid_z_scores"], "expected centroid z-scores"
    assert all({"feature", "z_score"} == set(z) for z in info["centroid_z_scores"])
    assert info["worst_rows"], "expected failing example rows"
    assert info["loco_metric"] < info["iid_metric"]

    with pytest.raises(KeyError):
        inspect_ood_cluster(run_id=run_id, cluster_id=999, output_dir=str(out_dir))


def test_scorecard_resource(mcp_run, monkeypatch):
    manifest, out_dir = mcp_run
    monkeypatch.setenv("SHORTCUT_HUNTER_RUNS", str(out_dir))
    contents = asyncio.run(mcp.read_resource(
        f"shortcut-hunter://runs/{manifest['run_id']}/scorecard"))
    text = "".join(
        getattr(c, "text", None) or getattr(c, "content", None) or ""
        for c in contents)
    assert text, "resource returned no text content"
    parsed = json.loads(text)
    assert parsed["run_id"] == manifest["run_id"]
    assert parsed["risk_level"] == "HIGH_SHORTCUT_RISK"

    with pytest.raises(Exception):
        asyncio.run(mcp.read_resource("shortcut-hunter://runs/nope-000/scorecard"))
