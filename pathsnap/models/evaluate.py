"""Shared training / evaluation harness. Every model goes through exactly this
code so the comparison is fair: same data, same folds, same metrics.

Protocol
  * Data: data/processed/<partition>_candidates.parquet, rows with label_reliable.
  * X = FEATURE_COLUMNS, y = label (1 = candidate is the true arc).
  * Cross-validation: GroupKFold(5) on the 80 train trips, grouped by trip_id,
    so no trip is in both the fitting and the validation part of a fold.
  * Student test: refit on all train rows, evaluate on the 10 student_test trips.
  * Evaluator-test data is never read here.

Metrics
  * row level (threshold 0.5): accuracy, precision, recall, macro F1, log loss
  * observation level: Top-1 accuracy = for each GPS point, the highest-scoring
    candidate is the true arc (points whose true arc is not a candidate count as wrong)
  * route continuity (student test): share of consecutive observations whose
    predicted arcs are the same arc or directly connected (u of next == v of previous)
  * training time, inference time
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

from pathsnap.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from pathsnap.data_processing.loader import load_arcs, trip_dir

N_FOLDS = 5
KEY = ["trip_id", "pt"]


def load_candidates(partition: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{partition}_candidates.parquet")
    df = df[df["label_reliable"]].reset_index(drop=True)
    return df


def subsample_points(df: pd.DataFrame, fraction: float, seed: int = SEED) -> pd.DataFrame:
    """Keep a random fraction of GPS points WITH all their candidates (for slow models)."""
    if fraction >= 1.0:
        return df
    pts = df[KEY].drop_duplicates()
    keep = pts.sample(frac=fraction, random_state=seed)
    return df.merge(keep, on=KEY, how="inner")


# ---------------------------------------------------------------- metrics
def top1_predictions(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per GPS point."""
    d = df[KEY + ["obs_idx", "segment_id", "arc_id", "label_arc"]].copy()
    d["proba"] = proba
    d = d.sort_values(KEY + ["proba"], ascending=[True, True, False], kind="stable")
    return d.groupby(KEY, sort=True).head(1).reset_index(drop=True)


def row_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def evaluate(df: pd.DataFrame, proba: np.ndarray) -> dict:
    m = row_metrics(df["label"].to_numpy(), proba)
    top = top1_predictions(df, proba)
    m["top1_accuracy"] = float((top["arc_id"] == top["label_arc"]).mean())
    m["n_points"] = len(top)
    m["n_rows"] = len(df)
    return m


def route_continuity(top: pd.DataFrame, partition: str) -> float:
    """Share of consecutive observation pairs (same trip, same segment) whose
    predicted arcs are identical or connected head-to-tail in the network."""
    ok, total = 0, 0
    for tid, g in top.groupby("trip_id"):
        arcs = load_arcs(trip_dir(tid, partition), tid)
        u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
        g = g.sort_values("pt")
        a = g["arc_id"].to_numpy()
        same_seg = g["segment_id"].to_numpy()[1:] == g["segment_id"].to_numpy()[:-1]
        prev, nxt = a[:-1][same_seg], a[1:][same_seg]
        ok += int(((prev == nxt) | (u[nxt] == v[prev])).sum())
        total += int(same_seg.sum())
    return ok / total if total else float("nan")


# ---------------------------------------------------------------- runs
@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: object
    description: str
    train_fraction: float = 1.0          # <1 = subsample GPS points for slow models
    notes: list[str] = field(default_factory=list)


def _fit_predict(est, train: pd.DataFrame, test: pd.DataFrame):
    from pathsnap.models.ranker import fit_model, predict_p
    t0 = time.perf_counter()
    fit_model(est, train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = predict_p(est, test)
    pred_s = time.perf_counter() - t0
    return est, proba, fit_s, pred_s


def cross_validate(spec: ModelSpec, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, va) in enumerate(gkf.split(train, groups=train["trip_id"])):
        fit_part = subsample_points(train.iloc[tr], spec.train_fraction)
        val_part = train.iloc[va]
        _, proba, fit_s, pred_s = _fit_predict(clone(spec.estimator), fit_part, val_part)
        m = evaluate(val_part, proba)
        m.update(fold=fold, fit_seconds=fit_s, predict_seconds=pred_s,
                 n_val_trips=val_part["trip_id"].nunique())
        rows.append(m)
        print(f"    fold {fold}: macro F1 {m['macro_f1']:.4f}  top-1 {m['top1_accuracy']:.4f}  "
              f"log loss {m['log_loss']:.4f}  ({fit_s:.1f} s fit)")
    return pd.DataFrame(rows)


def run(spec: ModelSpec, save_model: bool = True) -> dict:
    print(f"\n=== {spec.name} ({spec.family}) ===")
    train = load_candidates("train")
    test = load_candidates("student_test")

    cv = cross_validate(spec, train)

    fit_part = subsample_points(train, spec.train_fraction)
    est, proba, fit_s, pred_s = _fit_predict(clone(spec.estimator), fit_part, test)
    tm = evaluate(test, proba)
    top = top1_predictions(test, proba)
    tm["route_continuity"] = route_continuity(top, "student_test")
    tm["fit_seconds"] = fit_s
    tm["predict_seconds"] = pred_s
    tm["inference_ms_per_1000_points"] = 1000 * pred_s / tm["n_points"] * 1000

    out = RESULTS_DIR / "models" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "cv_folds.csv", index=False)
    cm = confusion_matrix(test["label"], (proba >= 0.5).astype(int), labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out / "confusion_matrix_student_test.csv")

    final = est[-1] if hasattr(est, "steps") else est
    if hasattr(final, "loss_curve_"):
        curve = pd.DataFrame({"epoch": np.arange(1, len(final.loss_curve_) + 1),
                              "train_loss": final.loss_curve_})
        if getattr(final, "validation_scores_", None) is not None:
            curve["val_accuracy"] = final.validation_scores_
        curve.to_csv(out / "loss_curve.csv", index=False)

    summary = {
        "model": spec.name,
        "family": spec.family,
        "description": spec.description,
        "config": repr(spec.estimator),
        "train_fraction": spec.train_fraction,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "notes": spec.notes,
        "cv_mean": cv.drop(columns=["fold"]).mean().to_dict(),
        "cv_std": cv.drop(columns=["fold"]).std().to_dict(),
        "student_test": tm,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    _update_comparison(summary)

    if save_model:
        (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(est, ARTIFACTS_DIR / "models" / f"{spec.name}.joblib", compress=3)

    print(f"  CV macro F1 {summary['cv_mean']['macro_f1']:.4f} ± {summary['cv_std']['macro_f1']:.4f} | "
          f"test top-1 {tm['top1_accuracy']:.4f} | test macro F1 {tm['macro_f1']:.4f} | "
          f"log loss {tm['log_loss']:.4f} | continuity {tm['route_continuity']:.4f}")
    return summary


def _comparison_row(s: dict) -> dict:
    t = s["student_test"]
    return {
        "model": s["model"], "family": s["family"], "config": s["description"],
        "train_fraction": s["train_fraction"],
        "cv_macro_f1": s["cv_mean"]["macro_f1"], "cv_macro_f1_std": s["cv_std"]["macro_f1"],
        "cv_top1": s["cv_mean"]["top1_accuracy"], "cv_top1_std": s["cv_std"]["top1_accuracy"],
        "cv_row_accuracy": s["cv_mean"]["accuracy"],
        "cv_log_loss": s["cv_mean"]["log_loss"],
        "test_top1": t["top1_accuracy"], "test_row_accuracy": t["accuracy"],
        "test_macro_f1": t["macro_f1"],
        "test_precision_macro": t["precision_macro"], "test_recall_macro": t["recall_macro"],
        "test_log_loss": t["log_loss"], "test_route_continuity": t["route_continuity"],
        "fit_seconds": t["fit_seconds"],
        "inference_ms_per_1000_points": t["inference_ms_per_1000_points"],
    }


def _update_comparison(s: dict) -> None:
    """Insert/replace this model's row in results/models/comparison.csv."""
    path = RESULTS_DIR / "models" / "comparison.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df["model"] != s["model"]]
    df = pd.concat([df, pd.DataFrame([_comparison_row(s)])], ignore_index=True)
    df.sort_values("cv_macro_f1", ascending=False).to_csv(path, index=False)


def rebuild_comparison() -> pd.DataFrame:
    """Recreate comparison.csv from every saved results/models/<model>/metrics.json."""
    rows = [_comparison_row(json.loads(p.read_text()))
            for p in sorted((RESULTS_DIR / "models").glob("*/metrics.json"))]
    df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "models" / "comparison.csv", index=False)
    return df
