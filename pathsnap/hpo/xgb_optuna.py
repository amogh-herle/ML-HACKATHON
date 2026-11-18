"""Bayesian hyperparameter optimisation of XGBoost with Optuna (TPE sampler).

Usage:
    python -m pathsnap.hpo.xgb_optuna --trials 40      # run / resume the study
    python -m pathsnap.hpo.xgb_optuna --refit          # refit best params, evaluate, save final model

Design
  * Objective: mean observation-level Top-1 accuracy over 5-fold GroupKFold on the
    80 train trips (grouped by trip_id) - the same folds as the model comparison.
    Macro F1 and log loss are recorded per trial as well.
  * Sampler: TPE (tree-structured Parzen estimator), seeded -> Bayesian search that
    models which regions of the space give good scores and samples there next.
  * Pruner: MedianPruner on the running fold mean - a trial whose first folds are
    below the median of earlier trials at the same fold is stopped early.
  * Trial 0 is the untuned configuration, so the search record contains the baseline.
  * Student-test and evaluator data are never used inside the search.
  * The study is stored in results/hpo/xgb_study.db (SQLite) and exported to
    results/hpo/xgb_trials.csv, so the full search record is kept and resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR, SEED
from pathsnap.features.candidates import FEATURE_COLUMNS
from pathsnap.models.evaluate import N_FOLDS, ModelSpec, evaluate, load_candidates, run

OUT = RESULTS_DIR / "hpo"
# Overridable from the environment, e.g. on Colab:
#   PATHSNAP_XGB_DEVICE=cuda   train on the GPU (needs the full `xgboost` wheel, not xgboost-cpu)
#   PATHSNAP_STUDY=xgb_top1_gpu  keep GPU trials in their own study (GPU and CPU results are
#                                close but not bit-identical, so they are not mixed)
STUDY_NAME = os.environ.get("PATHSNAP_STUDY", "xgb_top1")
STORAGE = f"sqlite:///{OUT / 'xgb_study.db'}"
JOBS = int(os.environ.get("PATHSNAP_JOBS", 16))  # threads per XGBoost fit
DEVICE = os.environ.get("PATHSNAP_XGB_DEVICE", "cpu")

# untuned configuration used in the model comparison (registry "xgb")
BASELINE = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0,
            "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.0, "reg_lambda": 1.0,
            "reg_alpha": 1e-3}

# search space: (why) ---------------------------------------------------------
#   n_estimators     100-1000   number of boosting rounds; trades fit vs over-fit, linked to lr
#   learning_rate    0.01-0.3   (log) shrinkage per round; small lr needs more rounds
#   max_depth        3-10       interaction depth; 16 features, deep trees overfit trips
#   min_child_weight 1-50 (log) minimum hessian per leaf; larger = smoother, fewer tiny leaves
#   subsample        0.5-1.0    row sampling per tree (bagging-style regularisation)
#   colsample_bytree 0.5-1.0    feature sampling per tree; decorrelates trees
#   gamma            0-5        minimum loss reduction to split; prunes weak splits
#   reg_lambda       1e-3-10    (log) L2 penalty on leaf weights
#   reg_alpha        1e-3-10    (log) L1 penalty on leaf weights


def suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def make_model(params: dict, jobs: int = JOBS) -> XGBClassifier:
    return XGBClassifier(**params, tree_method="hist", device=DEVICE, n_jobs=jobs,
                         random_state=SEED, eval_metric="logloss")


def objective_factory(train: pd.DataFrame):
    X = train[FEATURE_COLUMNS].to_numpy()
    y = train["label"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, groups=train["trip_id"]))

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        top1, f1, ll = [], [], []
        t0 = time.perf_counter()
        for k, (tr, va) in enumerate(folds):
            m = make_model(params).fit(X[tr], y[tr])
            met = evaluate(train.iloc[va], m.predict_proba(X[va])[:, 1])
            top1.append(met["top1_accuracy"]); f1.append(met["macro_f1"]); ll.append(met["log_loss"])
            trial.report(float(np.mean(top1)), step=k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1)))
        trial.set_user_attr("cv_log_loss", float(np.mean(ll)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1, ddof=1)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        trial.set_user_attr("folds_done", N_FOLDS)
        return float(np.mean(top1))

    return objective


def load_study() -> optuna.Study:
    OUT.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))


def export(study: optuna.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs",
                                       "duration"))
    df["duration_s"] = df["duration"].dt.total_seconds()
    df = df.drop(columns=["duration"])
    df.to_csv(OUT / "xgb_trials.csv", index=False)
    return df


def search(n_trials: int) -> None:
    study = load_study()
    if not study.trials:
        study.enqueue_trial(BASELINE, user_attrs={"note": "untuned baseline"})
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, n_trials - done)
    print(f"study {STUDY_NAME}: {done} trials finished, running {remaining} more")
    train = load_candidates("train")
    t0 = time.perf_counter()
    study.optimize(objective_factory(train), n_trials=remaining, gc_after_trial=True)
    df = export(study)
    states = df["state"].value_counts().to_dict()
    print(f"search time {time.perf_counter() - t0:.0f} s | trials {states}")
    print(f"best trial #{study.best_trial.number}: CV top-1 {study.best_value:.4f}")
    print(json.dumps(study.best_params, indent=2))


def refit() -> None:
    """Refit the best configuration through the shared harness (same CV + student test
    as every other model), save it as the final classifier, and write the comparison."""
    study = load_study()
    best = study.best_params
    spec = ModelSpec("xgb_tuned", "Boosting", make_model(best, jobs=min(8, JOBS)),
                     "XGBoost tuned by Optuna TPE: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                                                 else f"{k}={v}" for k, v in best.items()),
                     notes=[f"best of {len(study.trials)} Optuna trials (objective: CV top-1)"])
    summary = run(spec, save_model=True)

    # final model for the HMM / API: always saved for CPU inference (the API laptop has no GPU)
    import joblib
    est = joblib.load(ARTIFACTS_DIR / "models" / "xgb_tuned.joblib")
    est.set_params(device="cpu", n_jobs=8)
    joblib.dump(est, ARTIFACTS_DIR / "models" / "xgb_tuned.joblib", compress=3)
    joblib.dump(est, ARTIFACTS_DIR / "final_model.joblib", compress=3)
    (ARTIFACTS_DIR / "final_model.json").write_text(json.dumps({
        "model": "xgb_tuned", "params": best, "features": FEATURE_COLUMNS, "seed": SEED,
        "study": STUDY_NAME, "trained_on": DEVICE,
        "selected_by": "Optuna TPE, objective = mean 5-fold GroupKFold top-1 accuracy on train",
        "student_test": summary["student_test"]}, indent=2, default=float))

    # tuned vs untuned
    base = json.loads((RESULTS_DIR / "models" / "xgb" / "metrics.json").read_text())
    rows = []
    for label, s in [("untuned", base), ("tuned", summary)]:
        t = s["student_test"]
        rows.append({"model": label, "cv_macro_f1": s["cv_mean"]["macro_f1"],
                     "cv_top1": s["cv_mean"]["top1_accuracy"], "cv_log_loss": s["cv_mean"]["log_loss"],
                     "test_top1": t["top1_accuracy"], "test_accuracy": t["accuracy"],
                     "test_macro_f1": t["macro_f1"], "test_log_loss": t["log_loss"],
                     "test_route_continuity": t["route_continuity"], "fit_seconds": t["fit_seconds"],
                     "inference_ms_per_1000_points": t["inference_ms_per_1000_points"]})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "xgb_tuned_vs_untuned.csv", index=False)
    print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40, help="total trials in the study")
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    if a.refit:
        refit()
    else:
        search(a.trials)
