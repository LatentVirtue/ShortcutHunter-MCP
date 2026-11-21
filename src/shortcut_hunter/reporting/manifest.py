"""Compact JSON 'Fragility Scorecard' manifest for LLM agents (<2,000 tokens)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..engine.config import FragilityScorecard

TOKEN_LIMIT = 2000


def _r(v: float | None, nd: int = 4) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else round(f, nd)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for this kind of JSON."""
    return math.ceil(len(text) / 4)


def build_manifest(scorecard: FragilityScorecard) -> dict[str, Any]:
    ko = scorecard.knockout
    dec = scorecard.decoy
    ood = scorecard.ood

    steps_compact = [
        {
            "i": s.step,
            "removed": s.removed_feature,
            "metric": _r(s.metric_mean),
            "drop": _r(s.metric_drop_from_previous),
            "promoted": s.newly_promoted_proxy,
            "proxy_r": _r(s.proxy_correlation, 2),
        }
        for s in ko.steps
    ]

    clusters_compact = [
        {
            "id": c.cluster_id,
            "n": c.n_rows,
            "loco": _r(c.loco_metric),
            "drop": _r(c.drop_from_iid),
            "shift": [[s.feature, _r(s.z_score, 2)] for s in c.top_shift_features[:3]],
        }
        for c in ood.clusters
    ]

    manifest: dict[str, Any] = {
        "run_id": scorecard.run_id,
        "created_at": scorecard.created_at,
        "risk_level": scorecard.risk_level,
        "summary": {
            "iid_cv": _r(scorecard.iid_cv_score),
            "loco_worst_ood": _r(scorecard.loco_ood_score),
            "ood_degradation": _r(scorecard.ood_degradation),
            "metric": scorecard.metric,
            "model_family": scorecard.model_family,
            "patterns": scorecard.patterns,
        },
        "dataset": {
            "path": scorecard.dataset.path,
            "rows": scorecard.dataset.n_rows,
            "features": scorecard.dataset.n_features,
            "target": scorecard.dataset.target,
            "task": scorecard.dataset.task_type,
        },
        "knockout": {
            "iid": _r(ko.iid_metric),
            "final": _r(ko.final_metric),
            "total_decay": _r(ko.total_decay),
            "max_step_drop": _r(ko.max_single_step_drop),
            "removal_sequence": ko.removal_sequence[:8],
            "shortcut_pairs": [
                {"dropped": p.dropped_feature, "proxy": p.promoted_proxy,
                 "r": _r(p.correlation, 2)} for p in ko.shortcut_pairs[:5]
            ],
            "steps": steps_compact[:10],
        },
        "decoy": {
            "reliance_ratio": _r(dec.decoy_reliance_ratio),
            "in_distribution": _r(dec.in_distribution_metric),
            "inverted_ood": _r(dec.ood_metric),
            "collapse": _r(dec.metric_collapse),
        },
        "ood": {
            "n_clusters": ood.n_clusters,
            "iid_cv": _r(ood.iid_cv_metric),
            "loco_mean": _r(ood.loco_mean_metric),
            "loco_worst": _r(ood.loco_worst_metric),
            "iid_to_ood_drop": _r(ood.iid_to_ood_drop),
            "worst_cluster_id": ood.worst_cluster_id,
            "clusters": clusters_compact[:10],
        },
        "findings": [
            {"sev": f.severity, "code": f.code, "msg": f.message[:400]}
            for f in scorecard.findings[:6]
        ],
        "artifacts": {
            name: Path(p).name for name, p in scorecard.artifacts.items()
        } or {
            "manifest": "manifest.json", "report": "report.html",
            "knockout_trace": "knockout_trace.parquet",
            "slice_metrics": "slice_metrics.parquet",
            "ood_predictions": "ood_predictions.parquet",
        },
    }
    return manifest


def serialize_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=1, ensure_ascii=False, default=str)


def write_manifest(path: str | Path, scorecard: FragilityScorecard) -> dict[str, Any]:
    """Build, size-check and write manifest.json. Returns the manifest dict."""
    manifest = build_manifest(scorecard)
    text = serialize_manifest(manifest)
    tokens = estimate_tokens(text)
    if tokens > TOKEN_LIMIT:  # shrink progressively, then give up loudly
        manifest["knockout"]["steps"] = manifest["knockout"]["steps"][:5]
        for c in manifest["ood"]["clusters"]:
            c["shift"] = c["shift"][:2]
        text = serialize_manifest(manifest)
        tokens = estimate_tokens(text)
        if tokens > TOKEN_LIMIT:
            raise ValueError(f"Manifest would exceed {TOKEN_LIMIT} tokens ({tokens}).")
    manifest["_token_estimate"] = tokens
    Path(path).write_text(serialize_manifest(manifest), encoding="utf-8")
    return manifest
