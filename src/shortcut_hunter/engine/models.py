"""Unified model factory for ``lightgbm``, ``linear`` and ``pytorch_mlp`` families,
plus task/metric detection, CV helpers and feature-importance extraction.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelFamily, TaskType

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Task & metric detection                                                      #
# --------------------------------------------------------------------------- #

def detect_task_type(y: pd.Series) -> TaskType:
    """Heuristic: binary/small-integer labels -> classification, else regression."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    try:
        uniq = set(pd.unique(y.dropna()))
    except TypeError:  # unhashable (e.g. floats are hashable; defensive)
        return "regression"
    if len(uniq) <= 2 and all(float(u) in (0.0, 1.0) for u in uniq):
        return "classification"
    if len(uniq) <= 10 and all(float(u).is_integer() for u in uniq):
        return "classification"
    return "regression"


def default_metric(task_type: TaskType, y: pd.Series) -> str:
    if task_type == "regression":
        return "r2"
    n_classes = y.nunique()
    return "roc_auc" if n_classes == 2 else "accuracy"


# --------------------------------------------------------------------------- #
# Torch MLP (scikit-learn compatible)                                          #
# --------------------------------------------------------------------------- #

def _torch():
    import torch  # imported lazily so lightgbm-only flows never pay the cost
    return torch


class _BaseTorchMLP(BaseEstimator):
    """Small deterministic MLP trained with Adam (CPU, float32)."""

    def __init__(self, hidden=(64, 32), epochs=150, lr=0.01, batch_size=256,
                 dropout=0.2, weight_decay=1e-4, random_state=RANDOM_STATE):
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.random_state = random_state

    # -- shared fit plumbing ------------------------------------------------- #

    def _prepare(self, X, y):
        torch = _torch()
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        self._center_ = X.mean(axis=0)
        self._scale_ = X.std(axis=0)
        self._scale_[self._scale_ == 0] = 1.0
        Xs = (X - self._center_) / self._scale_
        Xt = torch.tensor(Xs, dtype=torch.float32)
        return X, Xs, Xt

    def _build_net(self, torch, n_features: int, n_out: int):
        layers: list = []
        d = n_features
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        layers.append(torch.nn.Linear(d, n_out))
        torch.manual_seed(self.random_state)
        net = torch.nn.Sequential(*layers)
        return net

    def _train_loop(self, torch, net, Xt, yt_tensor, loss_fn):
        optim = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = Xt.shape[0]
        g = np.random.RandomState(self.random_state)
        net.train()
        for _ in range(self.epochs):
            perm = g.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = torch.tensor(perm[start:start + self.batch_size], dtype=torch.long)
                optim.zero_grad()
                loss = loss_fn(net(Xt[idx]), yt_tensor[idx])
                loss.backward()
                optim.step()
        net.eval()
        return net


class TorchMLPClassifier(ClassifierMixin, _BaseTorchMLP):
    def fit(self, X, y):
        torch = _torch()
        _, Xs, Xt = self._prepare(X, y)
        self.classes_ = np.sort(np.unique(np.asarray(y)))
        y_int = np.searchsorted(self.classes_, np.asarray(y))
        n_out = 1 if len(self.classes_) == 2 else len(self.classes_)
        net = self._build_net(torch, Xt.shape[1], n_out)
        yt = torch.tensor(y_int, dtype=torch.float32).unsqueeze(1)
        loss_fn = (torch.nn.BCEWithLogitsLoss() if n_out == 1
                   else torch.nn.CrossEntropyLoss())
        ytr = yt if n_out == 1 else yt.squeeze(1).to(torch.int64)
        self.net_ = self._train_loop(torch, net, Xt, ytr, loss_fn)
        return self

    def decision_function(self, X):
        torch = _torch()
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, self._center_)
        Xs = (X - self._center_) / self._scale_
        with torch.no_grad():
            out = self.net_(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return out

    def predict_proba(self, X):
        out = self.decision_function(X)
        if out.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-out[:, 0]))
            return np.column_stack([1 - p, p])
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _BaseTorchMLP):
    def fit(self, X, y):
        torch = _torch()
        _, Xs, Xt = self._prepare(X, y)
        y = np.asarray(y, dtype=np.float64)
        self._y_center_ = float(y.mean())
        self._y_scale_ = float(y.std()) or 1.0
        net = self._build_net(torch, Xt.shape[1], 1)
        yt = torch.tensor((y - self._y_center_) / self._y_scale_,
                         dtype=torch.float32).unsqueeze(1)
        self.net_ = self._train_loop(torch, net, Xt, yt, torch.nn.MSELoss())
        return self

    def predict(self, X):
        torch = _torch()
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, self._center_)
        Xs = (X - self._center_) / self._scale_
        with torch.no_grad():
            out = self.net_(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return out[:, 0] * self._y_scale_ + self._y_center_


# --------------------------------------------------------------------------- #
# Model factory                                                                #
# --------------------------------------------------------------------------- #

def make_model(model_family: ModelFamily, task_type: TaskType,
               random_state: int = RANDOM_STATE) -> BaseEstimator:
    """Fresh (unfitted) scikit-learn-compatible estimator for the given family."""
    if model_family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        params = dict(
            n_estimators=200, learning_rate=0.08, num_leaves=15,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, random_state=random_state,
            n_jobs=2, verbose=-1,
        )
        return LGBMClassifier(**params) if task_type == "classification" else LGBMRegressor(**params)

    if model_family == "linear":
        if task_type == "classification":
            final = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
        else:
            final = Ridge(alpha=1.0)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", final),
        ])

    if model_family == "pytorch_mlp":
        cls = TorchMLPClassifier if task_type == "classification" else TorchMLPRegressor
        return cls(random_state=random_state)

    raise ValueError(f"Unknown model family: {model_family!r}")


# --------------------------------------------------------------------------- #
# CV helpers                                                                   #
# --------------------------------------------------------------------------- #

def make_cv(task_type: TaskType, y, folds: int, random_state: int):
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def cv_score(model_family: ModelFamily, X: pd.DataFrame, y, features: list[str],
             task_type: TaskType, scoring: str, folds: int,
             random_state: int = RANDOM_STATE) -> tuple[float, float]:
    """Cross-validated metric mean/std for ``features`` of ``X``."""
    model = make_model(model_family, task_type, random_state)
    cv = make_cv(task_type, y, folds, random_state)
    scores = cross_validate(
        model, X[features], y, cv=cv, scoring=scoring,
        return_train_score=False, error_score=np.nan,
    )["test_score"]
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), float("nan")
    return float(scores.mean()), float(scores.std())


# --------------------------------------------------------------------------- #
# Feature importance                                                           #
# --------------------------------------------------------------------------- #

def _normalize_share(imp: np.ndarray) -> np.ndarray:
    imp = np.clip(np.asarray(imp, dtype=float), 0.0, None)
    total = imp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(imp, 1.0 / max(len(imp), 1))
    return imp / total


def normalized_importances(model_family: ModelFamily, X: pd.DataFrame, y,
                           features: list[str], task_type: TaskType,
                           scoring: str, random_state: int = RANDOM_STATE,
                           permutation_repeats: int = 3) -> dict[str, float]:
    """Fit once on the full data and return normalized importance shares
    (LightGBM gain / |coef| for linear / permutation importance for MLP).
    Shares sum to 1 so they are comparable across families and steps.
    """
    model = make_model(model_family, task_type, random_state)
    model.fit(X[features], y)

    if model_family == "lightgbm":
        booster = model.booster_
        imp = booster.feature_importance(importance_type="gain")
    elif model_family == "linear":
        final = model["model"] if isinstance(model, Pipeline) else model
        coef = np.asarray(final.coef_, dtype=float)
        if coef.ndim > 1:  # multiclass: max abs coefficient per feature
            imp = np.abs(coef).max(axis=0)
        else:
            imp = np.abs(coef)
    else:  # pytorch_mlp
        r = permutation_importance(
            model, X[features], y, scoring=scoring,
            n_repeats=permutation_repeats, random_state=random_state, n_jobs=1,
        )
        imp = r.importances_mean
    share = _normalize_share(imp)
    return {f: float(s) for f, s in zip(features, share)}


def pearson_abs(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return 0.0
    return float(abs(np.corrcoef(a.astype(float), b.astype(float))[0, 1]))
