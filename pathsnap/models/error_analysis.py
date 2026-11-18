"""Where do top-1 errors come from? Classifies every student-test GPS point by how the
predicted arc relates to the reference (label) arc in the road network.

Usage:
    python -m pathsnap.models.error_analysis              # final model (artifacts/final_model.joblib)
    python -m pathsnap.models.error_analysis xgb lgbm     # saved registry models

Categories
  correct                    predicted arc == label arc
  reverse twin               same street, opposite direction (u,v swapped): identical geometry
  next/previous arc          shares the junction node in driving order (boundary ambiguity)
  sibling at same junction   leaves/enters the same node as the label arc
  other road                 none of the above
  label not a candidate      the label arc was not in the candidate set (unrecoverable)

For reverse-twin errors it also reports whether the GPS heading agreed with the prediction
or with the label: if the GPS motion agrees with the prediction, the reference label (not the
model) is the likely culprit.

Writes results/models/error_breakdown_<model>.csv
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR
from pathsnap.data_processing.loader import load_arcs, trip_dir
from pathsnap.features.candidates import FEATURE_COLUMNS
from pathsnap.models.evaluate import KEY, load_candidates

ORDER = ["correct", "reverse twin", "next/previous arc", "sibling at same junction", "other road",
         "label not a candidate"]


def breakdown(model_path, name: str, partition: str = "student_test") -> pd.DataFrame:
    df = load_candidates(partition)
    df["p"] = joblib.load(model_path).predict_proba(df[FEATURE_COLUMNS].to_numpy())[:, 1]
    top = df.sort_values(KEY + ["p"], ascending=[True, True, False]).groupby(KEY).head(1)
    true = df[df["label"] == 1].set_index(KEY)[["heading_diff_deg"]].rename(
        columns={"heading_diff_deg": "true_heading_diff"})
    top = top.set_index(KEY).join(true, how="left").reset_index()

    cats = []
    for tid, g in top.groupby("trip_id"):
        a = load_arcs(trip_dir(tid, partition), tid)
        u, v = a["u"].to_numpy(), a["v"].to_numpy()
        p, t = g["arc_id"].to_numpy(), g["label_arc"].to_numpy()
        in_set = g["true_heading_diff"].notna().to_numpy()
        cats.append(pd.Series(np.select(
            [p == t, ~in_set, (u[p] == v[t]) & (v[p] == u[t]), (u[p] == v[t]) | (v[p] == u[t]),
             (u[p] == u[t]) | (v[p] == v[t])],
            ["correct", "label not a candidate", "reverse twin", "next/previous arc",
             "sibling at same junction"], "other road"), index=g.index))
    top["category"] = pd.concat(cats)

    out = top["category"].value_counts().reindex(ORDER, fill_value=0).to_frame("points")
    out["share_pct"] = 100 * out["points"] / len(top)
    twin = top[top["category"] == "reverse twin"]
    agree = float((twin["heading_diff_deg"] < twin["true_heading_diff"]).mean()) if len(twin) else np.nan
    out.loc["reverse twin", "gps_heading_agrees_with_prediction_pct"] = 100 * agree
    out.index.name = "category"
    path = RESULTS_DIR / "models" / f"error_breakdown_{name}.csv"
    out.round(3).to_csv(path)
    print(f"\n[{name}] {len(top)} {partition} points -> {path}")
    print(out.round(2).to_string())
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        breakdown(ARTIFACTS_DIR / "final_model.joblib", "final_model")
    for n in names:
        breakdown(ARTIFACTS_DIR / "models" / f"{n}.joblib", n)
