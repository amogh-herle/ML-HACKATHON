"""Model-comparison plots from the saved experiment results (never from typed numbers).

Usage:
    python -m pathsnap.viz.model_plots

Reads results/models/comparison.csv and results/models/<model>/*.csv,
writes PNGs to results/models/plots/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR

MODELS_DIR = RESULTS_DIR / "models"
OUT = MODELS_DIR / "plots"
GREY, ACCENT = "#9e9e9e", "#1f77b4"


def _bar(ax, names, values, errors=None, best="max", fmt="{:.3f}"):
    idx = int(np.argmax(values) if best == "max" else np.argmin(values))
    colors = [ACCENT if i == idx else GREY for i in range(len(values))]
    ax.bar(names, values, yerr=errors, color=colors, capsize=4)
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(MODELS_DIR / "comparison.csv").sort_values("cv_macro_f1", ascending=False)
    names = comp["model"].tolist()

    # 1. cross-validation macro F1 with fold std
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar(ax, names, comp["cv_macro_f1"].to_numpy(), comp["cv_macro_f1_std"].to_numpy())
    ax.set_ylabel("Macro F1 (5-fold GroupKFold, train)")
    ax.set_title("Cross-validation macro F1 by model (error bar = std over folds)")
    ax.set_ylim(min(0.5, comp["cv_macro_f1"].min() - 0.05), 1.0)
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=150); plt.close(fig)

    # 2. student-test top-1 and macro F1 side by side
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, comp["test_top1"], w, label="Top-1 accuracy (per GPS point)", color=ACCENT)
    ax.bar(x + w / 2, comp["test_macro_f1"], w, label="Macro F1 (per candidate row)", color=GREY)
    for i, (a, b) in enumerate(zip(comp["test_top1"], comp["test_macro_f1"])):
        ax.text(i - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names); ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right")
    ax.set_title("Student-test performance (10 trips)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    # 3. log loss (lower is better)
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar(ax, names, comp["test_log_loss"].to_numpy(), best="min")
    ax.set_ylabel("Log loss (student test)"); ax.set_title("Probability quality: log loss (lower is better)")
    fig.tight_layout(); fig.savefig(OUT / "log_loss.png", dpi=150); plt.close(fig)

    # 4. route continuity and inference time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar(axes[0], names, comp["test_route_continuity"].to_numpy())
    axes[0].set_ylim(0.8, 1.0); axes[0].set_title("Route continuity, classifier only (student test)")
    _bar(axes[1], names, comp["inference_ms_per_1000_points"].to_numpy(), best="min", fmt="{:.0f}")
    axes[1].set_yscale("log"); axes[1].set_title("Inference time, ms per 1000 GPS points (log)")
    fig.tight_layout(); fig.savefig(OUT / "continuity_inference.png", dpi=150); plt.close(fig)

    # 5. confusion matrices (student test, threshold 0.5)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = pd.read_csv(MODELS_DIR / n / "confusion_matrix_student_test.csv", index_col=0).to_numpy()
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.2f})", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(n)
    fig.suptitle("Confusion matrices, student test (row-normalised)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    # 6. epoch-wise curves, only for models that produce them
    for n in names:
        p = MODELS_DIR / n / "loss_curve.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(c["epoch"], c["train_loss"], color=ACCENT, label="training loss (log loss)")
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
        if "val_accuracy" in c:
            ax2 = ax.twinx()
            ax2.plot(c["epoch"], c["val_accuracy"], color=GREY, label="validation accuracy")
            ax2.set_ylabel("validation accuracy (held-out 10%)")
            ax2.legend(loc="center right")
        ax.legend(loc="upper right"); ax.set_title(f"{n}: training history (final fit on all train)")
        fig.tight_layout(); fig.savefig(OUT / f"{n}_training_curve.png", dpi=150); plt.close(fig)

    print(f"plots -> {OUT}")


if __name__ == "__main__":
    main()
