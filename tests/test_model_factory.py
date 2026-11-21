"""Smoke tests for the unified model factory: all three families train and
predict on classification and regression; importance shares sum to 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcut_hunter.engine.models import (
    detect_task_type,
    make_model,
    normalized_importances,
)


def _classification_data():
    rng = np.random.RandomState(1)
    n = 300
    f1 = rng.normal(size=n)
    y = (f1 + rng.normal(0, 0.6, n) > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": y * 0.5 + rng.normal(0, 1, n),
                      "f3": rng.normal(size=n)})
    return X, pd.Series(y)


def _regression_data():
    rng = np.random.RandomState(2)
    n = 300
    f1 = rng.normal(size=n)
    y = 2.0 * f1 + rng.normal(0, 1.0, n)
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(size=n)})
    return X, pd.Series(y)


@pytest.mark.parametrize("family", ["lightgbm", "linear", "pytorch_mlp"])
def test_classification_families_fit_predict(family):
    X, y = _classification_data()
    model = make_model(family, "classification", random_state=0)
    model.fit(X, y)
    pred = np.asarray(model.predict(X))
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, pred) > 0.7


@pytest.mark.parametrize("family", ["lightgbm", "linear", "pytorch_mlp"])
def test_regression_families_fit_predict(family):
    X, y = _regression_data()
    model = make_model(family, "regression", random_state=0)
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float)
    assert pred.shape == (len(X),)
    from sklearn.metrics import r2_score
    assert r2_score(y, pred) > 0.5


@pytest.mark.parametrize("family", ["lightgbm", "linear", "pytorch_mlp"])
def test_importance_shares_sum_to_one(family):
    X, y = _classification_data()
    shares = normalized_importances(family, X, y, list(X.columns),
                                   "classification", "roc_auc", 0,
                                   permutation_repeats=2)
    assert set(shares) == set(X.columns)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in shares.values())
    assert shares["f1"] == max(shares.values())


def test_detect_task_type():
    assert detect_task_type(pd.Series([0, 1, 0, 0])) == "classification"
    assert detect_task_type(pd.Series([0.0, 1.0, 1.0])) == "classification"
    assert detect_task_type(pd.Series(["a", "b", "a"])) == "classification"
    assert detect_task_type(pd.Series(np.random.RandomState(0).normal(size=50))) == "regression"
    assert detect_task_type(pd.Series([1, 2, 3, 4, 5] * 4)) == "classification"
