"""Benchmark dataset generator: genuine multi-feature nonlinear signal PLUS an
injected brittle ``batch_id_proxy`` shortcut that breaks in one geometric cluster.

This is the 'Clever Hans' dataset used by the tests and the CLI demo:
  * ``g1..g6`` genuine predictive features (linear + interaction signal, moderate SNR)
  * ``n1..n4`` pure-noise features
  * ``batch_id_proxy`` matches the label on 97% of bulk rows but is coin-flip
    uncorrelated inside the shifted cluster -> a brittle shortcut any
    greedy tree/linear learner will capture.
  * Ground-truth columns are prefixed with ``_`` and are ignored by the audits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENUINE = [f"g{i}" for i in range(1, 7)]
NOISE = [f"n{i}" for i in range(1, 5)]
PROXY = "batch_id_proxy"

# Shift applied to a few genuine features inside the shifted cluster, so the
# cluster is geometrically separable (PCA + K-Means finds it).
_SHIFT = {"g1": 2.4, "g2": -2.2, "g5": 1.7, "n2": 1.5}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_clever_hans_dataset(n_rows: int = 1200, seed: int = 13,
                                shift_frac: float = 0.2) -> pd.DataFrame:
    """Mixed two-regime binary classification dataset (see module docstring)."""
    rng = np.random.RandomState(seed)
    n_shift = int(round(n_rows * shift_frac))
    n_bulk = n_rows - n_shift

    cols: dict[str, np.ndarray] = {}
    for name in GENUINE + NOISE:
        cols[name] = rng.normal(0.0, 1.0, n_rows)

    in_shift = np.zeros(n_rows, dtype=int)
    in_shift[rng.permutation(n_rows)[:n_shift]] = 1

    for name, shift in _SHIFT.items():
        cols[name] = cols[name] + shift * in_shift

    # Genuine (multi-feature, nonlinear) signal — deliberately moderate SNR so
    # a model forced off the proxy tops out near ~0.8 AUC, well below the
    # proxy-inflated ~0.95.
    g = {k: cols[k] for k in GENUINE}
    score = (1.1 * g["g1"] + 0.9 * g["g2"] - 0.8 * g["g3"]
             + 0.7 * g["g4"] * g["g1"] + 0.6 * g["g5"] - 0.5 * g["g6"]
             + rng.normal(0.0, 1.6, n_rows))
    y = (rng.random(n_rows) < _sigmoid(score)).astype(int)

    # Brittle shortcut: near-perfect in the bulk regime, coin flip in the
    # shifted cluster.
    agree = np.where(in_shift == 0, 0.97, 0.5)
    flip = rng.random(n_rows) < agree
    proxy = np.where(flip, y, 1 - y)

    df = pd.DataFrame({**cols, PROXY: proxy.astype(int), "y": y,
                      "_true_cluster": in_shift, "_score": score})
    ordered = GENUINE + NOISE + [PROXY, "y", "_true_cluster", "_score"]
    return df[ordered]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model features (excludes target and ground-truth ``_``-prefixed columns)."""
    return [c for c in df.columns if c != "y" and not c.startswith("_")]
