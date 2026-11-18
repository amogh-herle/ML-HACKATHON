"""Feature-set and model-type experiments, decided on TRAIN cross-validation only.

Usage:
    python -m pathsnap.models.ablation

Each experiment = (feature set, model) evaluated with the same 5-fold GroupKFold as the
model comparison. Student-test data is not touched here, so these choices cannot leak it.
Writes results/models/ablation.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.config import RESULTS_DIR
from pathsnap.features.candidates import BASE_COLUMNS, CONTEXT_COLUMNS
from pathsnap.models.evaluate import N_FOLDS, evaluate, load_candidates
from pathsnap.models.ranker import point_groups
from pathsnap.models.registry import REGISTRY

FEATURE_SETS = {
    "base (18)": BASE_COLUMNS,
    "base + ranks (22)": BASE_COLUMNS + CONTEXT_COLUMNS[:4],
    "base + support (20)": BASE_COLUMNS + CONTEXT_COLUMNS[4:],
    "base + context (24)": BASE_COLUMNS + CONTEXT_COLUMNS,
}
EXPERIMENTS = [
    ("base (18)", "xgb"),
    ("base + ranks (22)", "xgb"),
    ("base + support (20)", "xgb"),
    ("base + context (24)", "xgb"),
    ("base + context (24)", "lgbm"),
    ("base + context (24)", "xgb_rank"),
    ("base + context (24)", "lgbm_rank"),
]


def cv(train: pd.DataFrame, cols: list[str], est) -> dict:
    X, y = train[cols].to_numpy(), train["label"].to_numpy()
    groups = point_groups(train)
    rows = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(train, groups=train["trip_id"]):
        m = clone(est)
        t0 = time.perf_counter()
        if getattr(m, "needs_groups", False):
            m.fit(X[tr], y[tr], groups=groups[tr])
            p = m.predict_proba(X[va], groups=groups[va])[:, 1]
        else:
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
        met = evaluate(train.iloc[va], p)
        met["fit_seconds"] = time.perf_counter() - t0
        rows.append(met)
    f = pd.DataFrame(rows)
    return {"cv_top1": f.top1_accuracy.mean(), "cv_top1_std": f.top1_accuracy.std(),
            "cv_macro_f1": f.macro_f1.mean(), "cv_log_loss": f.log_loss.mean(),
            "fit_seconds_per_fold": f.fit_seconds.mean()}


def main() -> None:
    train = load_candidates("train")
    out = []
    for fs, model in EXPERIMENTS:
        r = cv(train, FEATURE_SETS[fs], REGISTRY[model].estimator)
        out.append({"features": fs, "model": model, **r})
        print(f"{fs:22s} {model:10s} CV top-1 {r['cv_top1']:.4f} ± {r['cv_top1_std']:.4f} | "
              f"macro F1 {r['cv_macro_f1']:.4f} | log loss {r['cv_log_loss']:.4f} | "
              f"{r['fit_seconds_per_fold']:.0f} s/fold", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_DIR / "models" / "ablation.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
