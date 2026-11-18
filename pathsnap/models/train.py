"""Train + evaluate models from the registry.

Usage:
    python -m pathsnap.models.train --list            # show available models
    python -m pathsnap.models.train lgbm rf           # train specific models
    python -m pathsnap.models.train                   # train the recommended five

Writes results/models/<name>/{metrics.json, cv_folds.csv, confusion_matrix_student_test.csv,
loss_curve.csv*}, upserts results/models/comparison.csv, saves artifacts/models/<name>.joblib.
"""
from __future__ import annotations

import argparse

from pathsnap.models.evaluate import run
from pathsnap.models.registry import RECOMMENDED, REGISTRY

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=RECOMMENDED)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="do not write the .joblib model")
    a = ap.parse_args()
    if a.list:
        for s in REGISTRY.values():
            flag = "*" if s.name in RECOMMENDED else " "
            print(f"{flag} {s.name:11s} {s.family:24s} {s.description}")
    else:
        for name in a.models:
            run(REGISTRY[name], save_model=not a.no_save)
