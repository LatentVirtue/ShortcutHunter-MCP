"""Runs the full audit (all three tests), computes the composite Fragility
Risk Level with actionable governance findings, and writes the dual artifacts.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AuditConfig,
    DatasetProfile,
    FragilityScorecard,
    GovernanceFinding,
    RiskLevel,
)
from .decoy_test import run_decoy_audit
from .knockout import run_knockout_audit, resolve_knockout_steps
from .models import default_metric, detect_task_type
from .ood_slices import run_ood_audit

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.html"
KNOCKOUT_TRACE_NAME = "knockout_trace.parquet"
SLICE_METRICS_NAME = "slice_metrics.parquet"
PREDICTIONS_NAME = "ood_predictions.parquet"
RUN_META_NAME = "run_meta.json"


# --------------------------------------------------------------------------- #
# Data loading & preparation                                                   #
# --------------------------------------------------------------------------- #

def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    raise ValueError(f"Unsupported dataset format: {p.suffix!r} (use .csv/.tsv/.parquet)")


def prepare_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, str, int]:
    """Split target/features. Drops ``_``-prefixed ground-truth columns and
    encodes categoricals as integer codes; classification labels are encoded
    0..k-1 (original label order preserved in the class mapping).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} not in dataset columns: {list(df.columns)}")
    work = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    y_raw = work[target_column]
    X = work.drop(columns=[target_column])

    n_categorical = 0
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col].astype(str).fillna("<NA>"))[0]
            n_categorical += 1
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    task_type = detect_task_type(y_raw)
    if task_type == "classification":
        codes, classes = pd.factorize(pd.Series(y_raw).astype(str).fillna("<NA>"), sort=True)
        y = pd.Series(codes, index=df.index)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        classes = np.array([])
    return X, y, task_type, int(len(classes))


# --------------------------------------------------------------------------- #
# Composite risk                                                               #
# --------------------------------------------------------------------------- #
# Risk rule (documented in README):
#   HIGH_SHORTCUT_RISK — at least one strong trigger (knockout cliff,
#                        decoy capture, material LOCO drop).
#   MODERATE           — at least one soft trigger (redundant proxy clusters,
#                        sub-alert decoy/shift suspicion).
#   LOW                — no trigger: signal looks deep and shift-robust.

def assess_risk(knockout, decoy, ood, config: AuditConfig) -> tuple[RiskLevel, list[GovernanceFinding]]:
    strong: list[GovernanceFinding] = []
    soft: list[GovernanceFinding] = []

    if "SINGLE_FEATURE_SHORTCUT_DEPENDENCY" in knockout.patterns:
        cliff_at = next((s for s in knockout.steps
                         if (s.metric_drop_from_previous or 0) > config.cliff_threshold), None)
        who = cliff_at.removed_feature if cliff_at else "?"
        strong.append(GovernanceFinding(
            severity="critical", code="SINGLE_FEATURE_SHORTCUT_DEPENDENCY",
            message=(f"Removing one feature ({who!r}) costs {knockout.max_single_step_drop:.3f} "
                     f"{knockout.metric} (> cliff {config.cliff_threshold}). The model leans on a "
                     "brittle single-feature shortcut. Retrain with that feature constrained/removed "
                     "or add adversarial data before deployment.")))
    if "REDUNDANT_PROXY_CLUSTER" in knockout.patterns and knockout.shortcut_pairs:
        pairs = ", ".join(f"{p.dropped_feature!r}<->{p.promoted_proxy!r} (r={p.correlation:.2f})"
                          for p in knockout.shortcut_pairs[:3])
        soft.append(GovernanceFinding(
            severity="warning", code="REDUNDANT_PROXY_CLUSTER",
            message=(f"Metric stays flat across removals of highly-correlated proxies: {pairs}. "
                     "Importance is smeared across a redundant cluster; feature-importance reports "
                     "here are not causal evidence.")))

    if (decoy.decoy_reliance_ratio >= config.decoy_reliance_alert
            or decoy.metric_collapse >= config.decoy_collapse_alert):
        strong.append(GovernanceFinding(
            severity="critical", code="DECOY_CAPTURE",
            message=(f"Injected Clever-Hans decoy captured {decoy.decoy_reliance_ratio:.0%} of "
                     f"normalized importance and {config.model_family} lost {decoy.metric_collapse:.3f} "
                     f"{knockout.metric} when the shortcut inverted "
                     f"({decoy.in_distribution_metric:.3f} -> {decoy.ood_metric:.3f}). The current "
                     "regularization setup is shortcut-prone; strengthen regularization or decorrelate "
                     "engineered proxies from the label.")))
    elif decoy.decoy_reliance_ratio >= config.decoy_reliance_alert / 2:
        soft.append(GovernanceFinding(
            severity="warning", code="DECOY_PARTIAL_CAPTURE",
            message=f"Decoy captured {decoy.decoy_reliance_ratio:.0%} of importance (below alert)."))

    if ood.iid_to_ood_drop >= config.ood_drop_alert:
        worst = next(c for c in ood.clusters if c.cluster_id == ood.worst_cluster_id)
        shifts = ", ".join(f"{s.feature} (z={s.z_score:+.1f})" for s in worst.top_shift_features[:3])
        strong.append(GovernanceFinding(
            severity="critical", code="COVARIATE_SHIFT_FRAGILITY",
            message=(f"Leave-One-Cluster-Out on cluster {ood.worst_cluster_id} ({worst.n_rows} rows) "
                     f"drops {knockout.metric} from {ood.iid_cv_metric:.3f} to {ood.loco_worst_metric:.3f} "
                     f"(|iid_to_ood_drop|={ood.iid_to_ood_drop:.3f}). That subpopulation is shifted on "
                     f"{shifts}. Inspect with inspect-ood-cluster and add coverage or gate predictions.")))
    elif ood.iid_to_ood_drop >= config.ood_drop_alert / 2:
        soft.append(GovernanceFinding(
            severity="warning", code="COVARIATE_SHIFT_SUSPECTED",
            message=(f"Worst-cluster LOCO drop {ood.iid_to_ood_drop:.3f} is under alert but not "
                     "negligible; monitor slice performance before deployment.")))

    findings = strong + soft
    if strong:
        return "HIGH_SHORTCUT_RISK", findings
    if soft:
        return "MODERATE", findings
    return "LOW", [GovernanceFinding(
        severity="info", code="NO_SHORTCUT_TRIGGERS",
        message=("No knockout cliff, decoy capture or material LOCO degradation detected across "
                 "the iterative knockout and "
                 f"{config.n_ood_clusters} OOD slices; signal looks distributed and shift-robust. "
                 "Re-run on a production sample to confirm."))]


# --------------------------------------------------------------------------- #
# Run orchestration + artifact writing                                         #
# --------------------------------------------------------------------------- #

def _run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"audit-{ts}-{uuid.uuid4().hex[:6]}"


def run_audit(data: str | Path | pd.DataFrame, target_column: str,
              config: AuditConfig | None = None, *,
              model_family: str = "lightgbm",
              knockout_steps: int | None = None,
              n_ood_clusters: int = 5,
              output_dir: str | Path = "runs",
              ) -> tuple[FragilityScorecard, Path]:
    """Full audit pipeline. Returns (scorecard, run_dir).

    ``data`` is a path (csv/tsv/parquet) or an in-memory DataFrame.
    """
    if config is None:
        config = AuditConfig(target_column=target_column, model_family=model_family,
                             knockout_steps=knockout_steps, n_ood_clusters=n_ood_clusters)
    target_column = config.target_column

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        dataset_path = "<in-memory>"
    else:
        df = load_table(data)
        dataset_path = str(Path(data).resolve())

    X, y, task_type, n_classes = prepare_xy(df, target_column)
    if len(X.columns) < 3:
        raise ValueError(f"Need at least 3 features, found {len(X.columns)}.")
    scoring = default_metric(task_type, y)

    run_id = _run_id()
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Audit 1: knockout ----------------------------------------------------- #
    knockout, ko_trace = run_knockout_audit(X, y, config, task_type, scoring)
    # --- Audit 2: decoy -------------------------------------------------------- #
    decoy = run_decoy_audit(X, y, config, task_type, scoring)
    # --- Audit 3: LOCO slices --------------------------------------------------- #
    ood, slice_df, pred_df = run_ood_audit(X, y, config, task_type, scoring)

    risk_level, findings = assess_risk(knockout, decoy, ood, config)
    patterns = sorted(set(knockout.patterns))

    n_categorical = 0  # categorical handling re-checked from original frame
    for col in X.columns:
        orig = df[col]
        if not pd.api.types.is_numeric_dtype(orig):
            n_categorical += 1

    scorecard = FragilityScorecard(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        risk_level=risk_level,
        dataset=DatasetProfile(
            path=dataset_path, n_rows=int(len(df)), n_features=int(len(X.columns)),
            n_categorical_features=n_categorical, target=target_column,
            task_type=task_type, n_classes=n_classes or None,
        ),
        model_family=config.model_family,
        metric=scoring,
        iid_cv_score=ood.iid_cv_metric,
        iid_cv_std=ood.iid_cv_std,
        loco_ood_score=ood.loco_worst_metric,
        ood_degradation=ood.iid_to_ood_drop,
        knockout=knockout,
        decoy=decoy,
        ood=ood,
        patterns=patterns,
        findings=findings,
        config=config.model_dump(),
    )

    # --- artifacts -------------------------------------------------------------- #
    ko_trace.to_parquet(run_dir / KNOCKOUT_TRACE_NAME, index=False)
    slice_df.to_parquet(run_dir / SLICE_METRICS_NAME, index=False)
    pred_out = X.copy()
    pred_out.insert(0, "cluster_id", pred_df["cluster_id"].to_numpy())
    pred_out["pc1"] = pred_df["pc1"].to_numpy()
    pred_out["pc2"] = pred_df["pc2"].to_numpy()
    for c in ("y_true", "y_pred", "y_prob", "row_error"):
        if c in pred_df.columns:
            pred_out[c] = pred_df[c].to_numpy()
    pred_out.to_parquet(run_dir / PREDICTIONS_NAME, index=False)

    run_meta: dict[str, Any] = {
        "run_id": run_id, "risk_level": risk_level, "worst_cluster_id": ood.worst_cluster_id,
        "config": config.model_dump(), "artifacts": {
            "manifest": MANIFEST_NAME, "report": REPORT_NAME,
            "knockout_trace": KNOCKOUT_TRACE_NAME, "slice_metrics": SLICE_METRICS_NAME,
            "ood_predictions": PREDICTIONS_NAME,
        },
    }
    (run_dir / RUN_META_NAME).write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # Deferred import keeps engine importable even without reporting deps,
    # and avoids any import cycle (reporting only imports engine.config).
    from ..reporting.html_report import render_report
    from ..reporting.manifest import write_manifest

    manifest = write_manifest(run_dir / MANIFEST_NAME, scorecard)
    render_report(run_dir / REPORT_NAME, scorecard, manifest, ko_trace, slice_df, pred_out)
    scorecard.artifacts = {k: str((run_dir / v)) for k, v in run_meta["artifacts"].items()}

    return scorecard, run_dir


# --------------------------------------------------------------------------- #
# Run inspection (used by CLI + MCP inspect_ood_cluster)                       #
# --------------------------------------------------------------------------- #

def load_run(output_dir: str | Path, run_id: str) -> tuple[dict, Path]:
    run_dir = Path(output_dir) / run_id
    meta_path = run_dir / RUN_META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"No run {run_id!r} under {output_dir!r} (missing {RUN_META_NAME}).")
    return json.loads(meta_path.read_text(encoding="utf-8")), run_dir


def inspect_ood_cluster(output_dir: str | Path, run_id: str, cluster_id: int) -> dict:
    """Centroid z-scores + worst rows of one OOD cluster from a stored run."""
    meta, run_dir = load_run(output_dir, run_id)
    slice_df = pd.read_parquet(run_dir / meta["artifacts"]["slice_metrics"])
    pred_df = pd.read_parquet(run_dir / meta["artifacts"]["ood_predictions"])
    manifest = json.loads((run_dir / meta["artifacts"]["manifest"]).read_text(encoding="utf-8"))

    z_rows = slice_df[slice_df["cluster_id"] == cluster_id].sort_values(
        "z_score", key=lambda s: s.abs(), ascending=False)
    if z_rows.empty:
        raise KeyError(f"Cluster {cluster_id} not found in run {run_id!r}. "
                       f"Available: {sorted(slice_df['cluster_id'].unique().tolist())}")
    cluster_rows = pred_df[pred_df["cluster_id"] == cluster_id].copy()
    if cluster_rows.empty:
        raise KeyError(f"No prediction rows for cluster {cluster_id}.")

    n_features = len([c for c in pred_df.columns
                      if c not in {"cluster_id", "pc1", "pc2", "y_true", "y_pred", "y_prob", "row_error"}])
    worst = cluster_rows.sort_values("row_error", ascending=False).head(10)
    return {
        "run_id": run_id,
        "cluster_id": int(cluster_id),
        "n_rows": int(len(cluster_rows)),
        "iid_metric": float(z_rows["iid_metric"].iloc[0]),
        "loco_metric": z_rows["loco_metric"].iloc[0],
        "drop_from_iid": z_rows["drop_from_iid"].iloc[0],
        "n_model_features": int(n_features),
        "centroid_z_scores": [
            {"feature": r["feature"], "z_score": float(r["z_score"])}
            for _, r in z_rows.iterrows()
        ],
        "worst_rows": json.loads(worst.drop(columns=["pc1", "pc2"]).to_json(orient="records")),
        "scorecard_summary": manifest.get("summary", {}),
        "report": manifest.get("artifacts", {}).get("report"),
    }
