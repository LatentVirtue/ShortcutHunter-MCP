"""Typer CLI: ``shortcut-hunter audit | generate-demo | inspect-slice``."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__
from .engine.orchestrator import inspect_ood_cluster, run_audit
from .engine.synthetic import generate_clever_hans_dataset

app = typer.Typer(
    help="ShortcutHunter — automated ML robustness audits: feature knockout, "
         "Clever-Hans decoy capture and covariate-shift (LOCO) slice analysis.",
    no_args_is_help=True,
    add_completion=False,
)


class Family(str, Enum):
    lightgbm = "lightgbm"
    linear = "linear"
    pytorch_mlp = "pytorch_mlp"
    logistic_or_ridge = "logistic_or_ridge"  # alias accepted by AuditConfig


def _version_cb(value: bool) -> None:
    if value:
        typer.echo(f"shortcut-hunter {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", callback=_version_cb, is_eager=True,
                                          help="Print version and exit.")] = False,
) -> None:
    """ShortcutHunter CLI."""


@app.command()
def audit(
    data: Annotated[Path, typer.Argument(help="Dataset path (.csv/.tsv/.parquet).")],
    target: Annotated[str, typer.Argument(help="Target column name.")],
    model_family: Annotated[Family, typer.Option("--model-family", "-m",
                                                help="Model family to audit.")] = Family.lightgbm,
    knockout_steps: Annotated[int, typer.Option("--knockout-steps", min=1, max=30,
                                               help="Number of top-feature removals.")] = 6,
    n_ood_clusters: Annotated[int, typer.Option("--n-ood-clusters", min=2, max=20)] = 5,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("runs"),
    cv_folds: Annotated[int, typer.Option("--cv-folds", min=2, max=10)] = 5,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Run all three robustness audits and write manifest.json + report.html + parquet logs."""
    from .engine.config import AuditConfig
    config = AuditConfig(
        target_column=target, model_family=model_family.value,
        knockout_steps=knockout_steps, n_ood_clusters=n_ood_clusters,
        cv_folds=cv_folds, random_state=seed,
    )
    scorecard, run_dir = run_audit(data, target, config=config, output_dir=output_dir)
    typer.echo("")
    typer.secho(f"RISK LEVEL: {scorecard.risk_level}", fg=typer.colors.RED
                if scorecard.risk_level == "HIGH_SHORTCUT_RISK"
                else typer.colors.YELLOW if scorecard.risk_level == "MODERATE"
                else typer.colors.GREEN, bold=True)
    typer.echo(f"  iid_cv={scorecard.iid_cv_score:.4f}  loco_worst={scorecard.loco_ood_score:.4f}  "
               f"iid_to_ood_drop={scorecard.ood_degradation:.4f}  metric={scorecard.metric}")
    typer.echo(f"  knockout removal order: {' > '.join(scorecard.knockout.removal_sequence)}")
    typer.echo(f"  patterns: {', '.join(scorecard.patterns)}")
    typer.echo(f"  decoy: reliance={scorecard.decoy.decoy_reliance_ratio:.2f} "
               f"collapse={scorecard.decoy.metric_collapse:.3f}")
    typer.echo("  findings:")
    for f in scorecard.findings:
        typer.echo(f"    [{f.severity}] {f.code}: {f.message}")
    typer.echo(f"\nArtifacts in: {run_dir}")
    for name in sorted(p.name for p in run_dir.iterdir()):
        typer.echo(f"  - {name}")


@app.command("generate-demo")
def generate_demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output dataset path "
                                     "(.parquet or .csv).")] = Path("demo_dataset.parquet"),
    rows: Annotated[int, typer.Option("--rows", min=200)] = 1200,
    seed: Annotated[int, typer.Option("--seed")] = 13,
) -> None:
    """Write the synthetic Clever-Hans benchmark (genuine signal + batch_id_proxy shortcut)."""
    df = generate_clever_hans_dataset(n_rows=rows, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    typer.echo(f"Wrote {len(df)} rows x {len(df.columns)} cols to {out}")
    typer.echo("Contains: genuine multi-feature nonlinear signal (g1..g6), noise (n1..n4), "
               "and the brittle 'batch_id_proxy' shortcut that breaks in one cluster.")


@app.command("inspect-slice")
def inspect_slice(
    run_id: Annotated[str, typer.Argument(help="Run id from a previous audit.")],
    cluster_id: Annotated[int, typer.Argument(help="OOD cluster id to inspect.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("runs"),
    top: Annotated[int, typer.Option("--top", min=1, max=50)] = 10,
    json_out: Annotated[bool, typer.Option("--json", help="Dump full JSON instead of summary.")] = False,
) -> None:
    """Show centroid z-scores and the worst-failing rows inside one OOD slice."""
    info = inspect_ood_cluster(output_dir, run_id, cluster_id)
    if json_out:
        typer.echo(json.dumps(info, indent=2))
        return
    typer.echo(f"run={run_id} cluster={cluster_id} rows={info['n_rows']}")
    typer.echo(f"iid_cv={info['iid_metric']:.4f}  loco={float(info['loco_metric'] or float('nan')):.4f} "
               f" drop={float(info['drop_from_iid'] or float('nan')):.4f}")
    typer.echo("\nCentroid z-score shifts (how this slice differs from the bulk population):")
    for z in info["centroid_z_scores"][:12]:
        typer.echo(f"  {z['feature']:>18}: {z['z_score']:+.2f} sd")
    typer.echo(f"\nWorst {min(top, len(info['worst_rows']))} rows in this slice:")
    for row in info["worst_rows"][:top]:
        typer.echo(f"  {json.dumps(row)}")


def main() -> None:  # console_script entrypoint
    app()


if __name__ == "__main__":
    main()
