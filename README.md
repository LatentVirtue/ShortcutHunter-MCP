# ShortcutHunter-MCP

Automated ML diagnostic & stress-testing suite for tabular models, exposed as a CLI and an
MCP server. Models that hit 0.95+ cross-validation scores often cheat: they ride a brittle
single-feature shortcut, a bundle of redundant proxies, or an in-distribution artifact — and
collapse silently under covariate shift. ShortcutHunter runs three automated robustness
audits against the model family you actually deploy, and reports what it finds.

Three audits, one composite verdict:

1. **Iterative Top-Feature Knockout** ("Signal Depth Audit") — repeatedly removes the #1
   most predictive feature, retrains via cross-validation, and plots the performance decay
   curve. Distinguishes a dangerously brittle 1-feature shortcut (a single-step cliff) from
   deep, distributed signal (gradual decay), and tracks which secondary proxy jumps up to
   replace each knocked-out feature (with its correlation to the removed one).
2. **Synthetic Decoy / Confounder Stress Test** — injects a calibrated "Clever Hans"
   target-proxy that agrees with the label on ~92% of training rows but *inverts* on an
   out-of-distribution split, then measures how much normalized importance the decoy
   captures and how far the metric collapses when the shortcut inverts.
3. **Covariate-Shift Slice Audit** — standardizes features, projects via PCA, clusters into
   geometric slices (K-Means), and compares standard random i.i.d. K-Fold CV against
   Leave-One-Cluster-Out (LOCO) evaluation. Reports `iid_to_ood_drop`, the worst-degrading
   slice, and centroid z-score shifts that explain *how* that slice differs from the bulk.

Every run writes **dual artifacts**:

- `manifest.json` — a compact (< 2,000-token) **Fragility Scorecard** built for LLM agents,
- `report.html` — a standalone interactive Plotly dashboard (scorecard banner, decay curve
  with per-point knockout/proxy hover, i.i.d. vs LOCO comparison, PCA scatter colored by
  slice error, decoy stress chart). Single self-contained HTML; plotly.js loads from CDN.
- `knockout_trace.parquet`, `slice_metrics.parquet`, `ood_predictions.parquet` — raw audit
  tables for humans and downstream tooling.

## Install

```bash
python -m venv .venv && source .venv/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.11. With `uv`, the torch CPU wheel is picked automatically
(`[tool.uv]` config); plain pip pulls the default torch wheel.

## Quickstart (CLI)

```bash
# 1. generate the built-in Clever-Hans benchmark: genuine multi-feature nonlinear
#    signal + a brittle `batch_id_proxy` shortcut that breaks in one cluster
shortcut-hunter generate-demo --out examples/demo_run/demo_dataset.parquet

# 2. audit it
shortcut-hunter audit examples/demo_run/demo_dataset.parquet y \
    --model-family lightgbm --knockout-steps 6 --n-ood-clusters 5 \
    --output-dir examples/demo_run
```

Example output (verbatim from the shipped demo run):

```
RISK LEVEL: HIGH_SHORTCUT_RISK
  iid_cv=0.9561  loco_worst=0.6379  iid_to_ood_drop=0.3182  metric=roc_auc
  knockout removal order: batch_id_proxy > g1 > g3 > g5 > g2 > g4 > n3
  patterns: SINGLE_FEATURE_SHORTCUT_DEPENDENCY
  decoy: reliance=0.59 collapse=0.277
  findings:
    [critical] SINGLE_FEATURE_SHORTCUT_DEPENDENCY: Removing one feature ('batch_id_proxy')
      costs 0.208 roc_auc (> cliff 0.15). The model leans on a brittle single-feature
      shortcut. Retrain with that feature constrained/removed or add adversarial data...
    [critical] COVARIATE_SHIFT_FRAGILITY: Leave-One-Cluster-Out on cluster 0 (237 rows)
      drops roc_auc from 0.956 to 0.638. That subpopulation is shifted on g1 (z=+1.4),
      g2 (z=-1.3), g5 (z=+1.1). Inspect with inspect-ood-cluster...
```

Inspect the worst slice the run found (what breaks the model):

```bash
shortcut-hunter inspect-slice <run_id> <cluster_id> --output-dir examples/demo_run
```

```
run=audit-... cluster=0 rows=237
iid_cv=0.9561  loco=0.6379  drop=0.3182
Centroid z-score shifts (how this slice differs from the bulk population):
                  g1: +1.40 sd
                  g2: -1.32 sd
                  g5: +1.13 sd
Worst 10 rows in this slice:
  {"g1": 3.26, ..., "batch_id_proxy": 1, "y_true": 0.0, "y_pred": 1.0, "y_prob": 0.9998, ...}
```

Any CSV/TSV/Parquet with a target column works; model families: `lightgbm`,
`linear` (logistic/ridge), `pytorch_mlp` (alias: `logistic_or_ridge`). Numeric features
are used as-is, categoricals are factorized, columns starting with `_` are ignored
(the demo dataset carries ground-truth `_true_cluster` / `_score` under that convention).

## MCP server

```bash
shortcut-hunter-mcp                    # stdio (the default)
shortcut-hunter-mcp --transport streamable-http --port 8000
```

Client config (Claude Desktop / any MCP client):

```json
{
  "mcpServers": {
    "shortcut-hunter": {
      "command": "shortcut-hunter-mcp"
    }
  }
}
```

Tools:

- `run_shortcut_audit(data_path, target_column, model_family="lightgbm", knockout_steps=6,
  n_ood_clusters=5, output_dir="runs") -> dict` — full audit, returns the manifest.
- `inspect_ood_cluster(run_id, cluster_id, output_dir="runs") -> dict` — centroid
  z-score shifts + worst failing rows of one OOD slice, so the agent can explain the
  subpopulation that breaks the model.

Resource:

- `shortcut-hunter://runs/{run_id}/scorecard` — the stored `manifest.json`.
  (Set `SHORTCUT_HUNTER_RUNS` to point the server at a non-default runs root.)

## Interpreting the scorecard

`risk_level` is derived from the configured alert thresholds (`AuditConfig`):

| Trigger | Condition | Effect |
|---|---|---|
| `SINGLE_FEATURE_SHORTCUT_DEPENDENCY` | a single knockout step drops the metric > `cliff_threshold` (0.15) | HIGH_SHORTCUT_RISK |
| `DECOY_CAPTURE` | decoy importance share ≥ `decoy_reliance_alert` (0.30) or collapse ≥ `decoy_collapse_alert` (0.15) | HIGH_SHORTCUT_RISK |
| `COVARIATE_SHIFT_FRAGILITY` | worst-cluster LOCO drop ≥ `ood_drop_alert` (0.15) | HIGH_SHORTCUT_RISK |
| `REDUNDANT_PROXY_CLUSTER` | flat decay while a removed feature's near-duplicate (r ≥ 0.5) absorbs the signal | MODERATE |
| sub-alert decoy/shift suspicion | ≥ half of the respective alert | MODERATE |
| no triggers | — | LOW |

Metrics are chosen automatically: `roc_auc` for binary targets, `accuracy` for multiclass,
`r2` for regression. All findings are actionable strings ("retrain with the feature
constrained", "add coverage for the shifted slice", …).

## Architecture

```
src/shortcut_hunter/
├── engine/            # pure ML — zero imports from `mcp`, enforced by a unit test
│   ├── config.py      # Pydantic v2: AuditConfig, KnockoutStep, SliceResult, FragilityScorecard
│   ├── models.py      # unified factory: lightgbm / linear / pytorch_mlp (clf & reg)
│   ├── knockout.py    # audit 1 + proxy-substitution tracker
│   ├── decoy_test.py  # audit 2: Clever-Hans injection, inverted-OOD susceptibility
│   ├── ood_slices.py  # audit 3: PCA+K-Means, LOCO vs i.i.d. CV
│   ├── orchestrator.py# full run, composite risk level, governance findings, artifacts
│   └── synthetic.py   # benchmark with genuine signal + injected batch_id_proxy
├── reporting/
│   ├── manifest.py    # compact <2k-token Fragility Scorecard JSON
│   └── html_report.py # standalone Plotly dashboard
├── cli.py             # typer: audit | generate-demo | inspect-slice
└── mcp_server.py      # FastMCP tools + resources
```

## Tests

```bash
pytest -v
```

The suite builds the synthetic Clever-Hans dataset once per session and asserts, end-to-end,
that the orchestrator flags `batch_id_proxy` first in the knockout, detects the LOCO drop on
the shifted cluster, writes all five artifacts (manifest < 2,000 tokens), serves the MCP
tools/resource, and that `engine/` stays free of any `mcp` import.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
