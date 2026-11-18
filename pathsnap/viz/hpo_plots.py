"""Plots of the XGBoost Optuna search, drawn from the saved study (results/hpo/).

Usage:
    python -m pathsnap.viz.hpo_plots
"""
from __future__ import annotations

import matplotlib.pyplot as plt  # figures are only saved and closed, never shown
import numpy as np
import optuna
import pandas as pd

from pathsnap.config import RESULTS_DIR
from pathsnap.hpo.xgb_optuna import load_study

OUT = RESULTS_DIR / "hpo" / "plots"
ACCENT, GREY = "#1f77b4", "#9e9e9e"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    study = load_study()
    trials = [t for t in study.trials if t.state.is_finished()]
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

    # 1. optimisation history
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter([t.number for t in done], [t.value for t in done], color=GREY, s=20, label="completed trial")
    if pruned:
        ax.scatter([t.number for t in pruned], [t.intermediate_values[max(t.intermediate_values)]
                                                for t in pruned],
                   marker="x", color="#d62728", s=20, label="pruned (running mean when stopped)")
    nums = [t.number for t in done]
    best = np.maximum.accumulate([t.value for t in done])
    ax.step(nums, best, where="post", color=ACCENT, lw=2, label="best so far")
    base = study.trials[0].value
    ax.axhline(base, color="#555555", ls="--", lw=1, label=f"untuned baseline ({base:.4f})")
    ax.set_xlabel("trial"); ax.set_ylabel("CV top-1 accuracy (5-fold GroupKFold)")
    ax.set_title("Optuna TPE search, XGBoost"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "optimisation_history.png", dpi=150); plt.close(fig)

    # 2. hyperparameter importance (fANOVA over completed trials)
    imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color=ACCENT)
    ax.set_xlabel("importance (fANOVA, share of objective variance)")
    ax.set_title("Which hyperparameters mattered")
    fig.tight_layout(); fig.savefig(OUT / "param_importance.png", dpi=150); plt.close(fig)
    s.sort_values(ascending=False).to_csv(RESULTS_DIR / "hpo" / "xgb_param_importance.csv",
                                          header=["importance"])

    # 3. objective vs each hyperparameter
    params = list(study.best_params)
    log_scale = {"learning_rate", "min_child_weight", "reg_lambda", "reg_alpha"}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, p in zip(axes.ravel(), params):
        x = [t.params[p] for t in done]; y = [t.value for t in done]
        ax.scatter(x, y, c=[t.number for t in done], cmap="Blues", s=18, edgecolors="#555555", linewidths=0.3)
        ax.scatter([study.best_params[p]], [study.best_value], color="#d62728", s=40, label="best")
        if p in log_scale:
            ax.set_xscale("log")
        ax.set_title(p, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("CV top-1 against each hyperparameter (darker = later trial)")
    fig.tight_layout(); fig.savefig(OUT / "param_slices.png", dpi=150); plt.close(fig)

    # 4. tuned vs untuned (written by --refit)
    p = RESULTS_DIR / "hpo" / "xgb_tuned_vs_untuned.csv"
    if p.exists():
        c = pd.read_csv(p).set_index("model")
        metrics = ["cv_top1", "cv_macro_f1", "test_top1", "test_macro_f1", "test_accuracy"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
        x = np.arange(len(metrics)); w = 0.38
        for j, (lab, col) in enumerate([("untuned", GREY), ("tuned", ACCENT)]):
            vals = c.loc[lab, metrics].to_numpy(dtype=float)
            axes[0].bar(x + (j - 0.5) * w, vals, w, color=col, label=lab)
            for i, v in enumerate(vals):
                axes[0].text(i + (j - 0.5) * w, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        axes[0].set_xticks(x, metrics); axes[0].set_ylim(0.8, 1.0); axes[0].legend(frameon=False)
        axes[0].set_title("Tuned vs untuned XGBoost (higher is better)")
        ll = c["test_log_loss"]
        axes[1].bar(ll.index, ll.values, color=[GREY, ACCENT])
        for i, v in enumerate(ll.values):
            axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        axes[1].set_title("Student-test log loss (lower is better)")
        fig.tight_layout(); fig.savefig(OUT / "tuned_vs_untuned.png", dpi=150); plt.close(fig)

    print(f"plots -> {OUT}")


if __name__ == "__main__":
    main()
