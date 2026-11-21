"""Pydantic v2 configuration and result schemas for ShortcutHunter audits.

Layering rule: nothing under ``engine/`` may ever import ``mcp``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ModelFamily = Literal["lightgbm", "linear", "pytorch_mlp"]
TaskType = Literal["classification", "regression"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH_SHORTCUT_RISK"]
PatternName = Literal[
    "SINGLE_FEATURE_SHORTCUT_DEPENDENCY",
    "REDUNDANT_PROXY_CLUSTER",
    "STABLE_DISTRIBUTED_SIGNAL",
]
Severity = Literal["critical", "warning", "info"]


class AuditConfig(BaseModel):
    """Full audit configuration; thresholds double as risk-trigger constants."""

    target_column: str
    model_family: ModelFamily = "lightgbm"
    knockout_steps: Optional[int] = Field(
        default=None, ge=1, le=30,
        description="Number of top-feature removals. None -> min(8, n_features - 1).",
    )
    n_ood_clusters: int = Field(default=5, ge=2, le=20)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_state: int = 42

    # Decoy injection calibration.
    decoy_flip_prob_train: float = Field(default=0.08, ge=0, le=0.5,
                                         description="P(decoy disagrees with y) on train/ID rows.")
    decoy_flip_prob_ood: float = Field(default=0.85, ge=0.5, le=1.0,
                                       description="P(decoy disagrees with y) on the inverted OOD split.")

    # Risk-trigger thresholds (documented in README).
    cliff_threshold: float = Field(default=0.15, gt=0, le=1,
                                  description="Single-step metric drop that counts as a cliff.")
    redundant_corr_threshold: float = Field(default=0.50, gt=0, le=1)
    redundant_drop_threshold: float = Field(default=0.03, ge=0, le=1)
    decoy_reliance_alert: float = Field(default=0.30, gt=0, le=1)
    decoy_collapse_alert: float = Field(default=0.15, gt=0, le=1)
    ood_drop_alert: float = Field(default=0.15, gt=0, le=1)

    @field_validator("model_family", mode="before")
    @classmethod
    def _normalize_family(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower()
            if v in {"logistic_or_ridge", "linear", "linear_logistic_ridge"}:
                return "linear"
        return v


# --------------------------------------------------------------------------- #
# Audit 1 — iterative knockout                                                 #
# --------------------------------------------------------------------------- #

class KnockoutStep(BaseModel):
    step: int
    features_remaining: int
    metric_mean: float
    metric_std: float
    metric_drop_from_previous: Optional[float] = None
    removed_feature: Optional[str] = None
    next_top_feature: Optional[str] = None
    newly_promoted_proxy: Optional[str] = None
    proxy_correlation: Optional[float] = None
    top_importances: dict[str, float] = Field(default_factory=dict)


class ShortcutPair(BaseModel):
    dropped_feature: str
    promoted_proxy: str
    correlation: float
    metric_drop: float


class KnockoutResult(BaseModel):
    metric: str
    iid_metric: float
    final_metric: float
    total_decay: float
    max_single_step_drop: float
    removal_sequence: list[str] = Field(default_factory=list)
    patterns: list[PatternName] = Field(default_factory=list)
    shortcut_pairs: list[ShortcutPair] = Field(default_factory=list)
    steps: list[KnockoutStep] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Audit 2 — decoy / Clever-Hans capture                                        #
# --------------------------------------------------------------------------- #

class DecoyResult(BaseModel):
    decoy_column: str
    flip_prob_train: float
    flip_prob_ood: float
    in_distribution_metric: float
    ood_metric: float
    metric_collapse: float
    decoy_reliance_ratio: float
    top_importances: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Audit 3 — covariate-shift LOCO                                               #
# --------------------------------------------------------------------------- #

class ClusterShiftFeature(BaseModel):
    feature: str
    z_score: float


class SliceResult(BaseModel):
    cluster_id: int
    n_rows: int
    loco_metric: Optional[float] = None
    drop_from_iid: Optional[float] = None
    metric_note: Optional[str] = None
    top_shift_features: list[ClusterShiftFeature] = Field(default_factory=list)


class OodAuditResult(BaseModel):
    metric: str
    n_clusters: int
    iid_cv_metric: float
    iid_cv_std: float
    loco_mean_metric: float
    loco_worst_metric: float
    iid_to_ood_drop: float
    worst_cluster_id: int
    clusters: list[SliceResult] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Composite scorecard                                                          #
# --------------------------------------------------------------------------- #

class GovernanceFinding(BaseModel):
    severity: Severity
    code: str
    message: str


class DatasetProfile(BaseModel):
    path: str
    n_rows: int
    n_features: int
    n_categorical_features: int = 0
    target: str
    task_type: TaskType
    n_classes: Optional[int] = None


class FragilityScorecard(BaseModel):
    """Composite 'Fragility Scorecard' — also serialized into manifest.json."""

    run_id: str
    created_at: str
    risk_level: RiskLevel
    dataset: DatasetProfile
    model_family: ModelFamily
    metric: str
    iid_cv_score: float
    iid_cv_std: float
    loco_ood_score: float
    ood_degradation: float
    knockout: KnockoutResult
    decoy: DecoyResult
    ood: OodAuditResult
    patterns: list[PatternName] = Field(default_factory=list)
    findings: list[GovernanceFinding] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _normalize_risk(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().upper()
        return v
