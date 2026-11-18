"""Learning-to-rank models for candidate selection.

A classifier scores every (observation, candidate) row independently. A ranker is trained
on the candidates of one observation as a group and learns to put the true arc on top,
which is exactly what Top-1 accuracy measures (LambdaRank objective, optimising NDCG@1).

Ranker scores are not probabilities, so `predict_proba` turns them into a probability per
candidate with a softmax inside each observation's group: the values of one observation sum
to 1, which is also the form the HMM emission uses.

The harness passes the group of every row (one group per GPS observation) through the
`groups=` argument when an estimator sets `needs_groups = True`.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def _group_sizes(groups: np.ndarray) -> np.ndarray:
    """Rows must be contiguous per group (the candidate tables are sorted by trip, pt)."""
    change = np.r_[True, groups[1:] != groups[:-1]]
    starts = np.flatnonzero(change)
    return np.diff(np.r_[starts, len(groups)])


def group_softmax(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    sizes = _group_sizes(groups)
    out = np.empty_like(scores, dtype=float)
    i = 0
    for n in sizes:
        s = scores[i:i + n] - scores[i:i + n].max()
        e = np.exp(s)
        out[i:i + n] = e / e.sum()
        i += n
    return out


class GroupRanker(BaseEstimator, ClassifierMixin):
    needs_groups = True

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, groups):
        self.ranker_ = clone(self.estimator)
        self.ranker_.fit(X, y, group=_group_sizes(np.asarray(groups)))
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.ranker_.predict(X)

    def predict_proba(self, X, groups):
        p = group_softmax(self.decision_function(X), np.asarray(groups))
        return np.c_[1 - p, p]

    def predict(self, X, groups):
        return (self.predict_proba(X, groups)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.ranker_.feature_importances_


def point_groups(df) -> np.ndarray:
    """One integer id per GPS observation (trip_id, pt), in row order."""
    key = df["trip_id"].astype(str) + ":" + df["pt"].astype(str)
    return key.factorize()[0]


def fit_model(model, df):
    """Fit any model (classifier or GroupRanker) on candidate rows with a `label` column."""
    from pathsnap.features.candidates import FEATURE_COLUMNS
    X, y = df[FEATURE_COLUMNS].to_numpy(), df["label"].to_numpy()
    if getattr(model, "needs_groups", False):
        return model.fit(X, y, groups=point_groups(df))
    return model.fit(X, y)


def predict_p(model, df, X=None) -> np.ndarray:
    """Probability of 'true arc' per row for any model (classifier or GroupRanker)."""
    from pathsnap.features.candidates import FEATURE_COLUMNS
    X = df[FEATURE_COLUMNS].to_numpy() if X is None else X
    if getattr(model, "needs_groups", False):
        return model.predict_proba(X, groups=point_groups(df))[:, 1]
    return model.predict_proba(X)[:, 1]
