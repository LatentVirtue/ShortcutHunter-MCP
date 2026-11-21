"""ShortcutHunter — automated ML diagnostic & stress-testing suite.

Public API:
    run_audit(...)              -> (FragilityScorecard, run_dir)
    AuditConfig                 -> pydantic audit configuration
    generate_clever_hans_dataset -> synthetic benchmark with a built-in shortcut

CLI entrypoint: ``shortcut-hunter``; MCP server entrypoint: ``shortcut-hunter-mcp``.
"""
from .engine.config import AuditConfig, FragilityScorecard
from .engine.orchestrator import inspect_ood_cluster, run_audit
from .engine.synthetic import generate_clever_hans_dataset

__version__ = "0.1.0"

__all__ = [
    "AuditConfig",
    "FragilityScorecard",
    "generate_clever_hans_dataset",
    "inspect_ood_cluster",
    "run_audit",
    "__version__",
]
