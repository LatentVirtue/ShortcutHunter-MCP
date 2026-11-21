"""FastMCP server exposing ShortcutHunter audits to LLM agents.

Tools:
    run_shortcut_audit    — full three-audit robustness scan, returns the manifest.
    inspect_ood_cluster   — z-scores + worst rows of one OOD slice from a stored run.
Resource:
    shortcut-hunter://runs/{run_id}/scorecard — the stored manifest.json.

Run standalone: ``shortcut-hunter-mcp`` (stdio), or
``shortcut-hunter-mcp --transport streamable-http --port 8000``.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .engine.orchestrator import (
    MANIFEST_NAME,
    inspect_ood_cluster as _inspect_ood_cluster,
    load_run,
    run_audit,
)
from .reporting.manifest import build_manifest, estimate_tokens

mcp = FastMCP("shortcut-hunter")


@mcp.tool()
def run_shortcut_audit(
    data_path: str,
    target_column: str,
    model_family: str = "lightgbm",
    knockout_steps: int = 6,
    n_ood_clusters: int = 5,
    output_dir: str = "runs",
) -> dict:
    """Run the full ShortcutHunter robustness audit on a tabular dataset
    (.csv/.tsv/.parquet): iterative top-feature knockout with proxy-substitution
    tracking, a Clever-Hans decoy injection stress test, and a PCA/K-Means
    Leave-One-Cluster-Out covariate-shift audit.

    Writes manifest.json, report.html and raw parquet logs under
    ``<output_dir>/<run_id>/`` and returns the compact Fragility Scorecard
    manifest (risk_level, iid_cv vs loco_ood scores, shortcut patterns,
    governance findings, artifact paths).
    """
    scorecard, run_dir = run_audit(
        data_path, target_column, model_family=model_family,
        knockout_steps=knockout_steps, n_ood_clusters=n_ood_clusters,
        output_dir=output_dir,
    )
    manifest = build_manifest(scorecard)
    manifest["artifacts"] = {k: str(Path(v).resolve())
                             for k, v in scorecard.artifacts.items()}
    manifest["_token_estimate"] = estimate_tokens(json.dumps(manifest, default=str))
    return manifest


@mcp.tool()
def inspect_ood_cluster(
    run_id: str,
    cluster_id: int,
    output_dir: str = "runs",
) -> dict:
    """Return the centroid z-score shifts and the worst-failing rows of one
    out-of-distribution cluster from a previous run_shortcut_audit, so the
    calling agent can explain which subpopulation breaks the model and why."""
    return _inspect_ood_cluster(output_dir, run_id, cluster_id)


@mcp.resource("shortcut-hunter://runs/{run_id}/scorecard",
              mime_type="application/json",
              title="Fragility Scorecard")
def scorecard_resource(run_id: str) -> str:
    """Stored manifest.json (Fragility Scorecard) for a completed audit run."""
    import os
    runs_root = Path(os.environ.get("SHORTCUT_HUNTER_RUNS", "runs"))
    manifest_path = runs_root / run_id / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No scorecard for run {run_id!r} at {manifest_path}.")
    return manifest_path.read_text(encoding="utf-8")


@mcp.resource("shortcut-hunter://help")
def help_resource() -> str:
    """How to use the shortcut-hunter tools: workflow and parameter guidance."""
    return (
        "ShortcutHunter workflow:\n"
        "1. run_shortcut_audit(data_path, target_column, model_family="
        "'lightgbm'|'linear'|'pytorch_mlp', knockout_steps=6, n_ood_clusters=5, "
        "output_dir='runs') -> compact Fragility Scorecard manifest with risk_level "
        "LOW|MODERATE|HIGH_SHORTCUT_RISK.\n"
        "2. For the worst OOD slice, call inspect_ood_cluster(run_id, cluster_id) to get "
        "centroid z-scores and failing example rows, then explain the subpopulation.\n"
        "3. The full scorecard is also readable as the resource "
        "shortcut-hunter://runs/{run_id}/scorecard; report.html has the interactive charts.\n"
        "Interpretation: knockout cliffs = single-feature shortcut dependency; flat decay with "
        "high promoted-proxy correlation = redundant proxy cluster; decoy_reliance_ratio = "
        "Clever-Hans capture share; iid_to_ood_drop = covariate-shift fragility."
    )


def main() -> None:  # console_script entrypoint
    import typer
    runner = typer.Typer(add_completion=False, help="Run the shortcut-hunter MCP server.")

    @runner.command()
    def serve(
        transport: str = typer.Option("stdio", "--transport",
                                      help="stdio | sse | streamable-http"),
        host: str = typer.Option("127.0.0.1", "--host"),
        port: int = typer.Option(8000, "--port"),
    ) -> None:
        if transport != "stdio":
            mcp.settings.host = host
            mcp.settings.port = port
        mcp.run(transport=transport)

    runner()


if __name__ == "__main__":
    main()
