"""Run cleaning + labelling for whole partitions.

Usage:
    python -m pathsnap.data_processing.preprocess                 # train + student_test
    python -m pathsnap.data_processing.preprocess student_test

Outputs:
    data/processed/<partition>_points.parquet       one row per GPS observation
    results/eda/preprocess_summary_<partition>.csv  one row per trip
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from pathsnap.config import PARTITIONS, PROCESSED_DIR, RESULTS_DIR, TRIP_SUSPECT_MEDIAN_M
from pathsnap.data_processing.clean import clean_track
from pathsnap.data_processing.labels import label_track
from pathsnap.data_processing.loader import list_trips, load_trip


def preprocess_trip(trip_id: str, partition: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    trip = load_trip(trip_id, partition, with_route=True)
    df = clean_track(trip.track)
    if trip.route is not None:
        df = label_track(df, trip.nodes, trip.arcs, trip.route)
    df.insert(0, "trip_id", trip_id)

    lab = df[df["label_arc"] >= 0] if "label_arc" in df else df.iloc[:0]
    summary = {
        "trip_id": trip_id,
        "crs": trip.crs,
        "n_points": len(df),
        "n_invalid": int(df["is_invalid"].sum()),
        "n_outliers": int(df["is_outlier"].sum()),
        "n_segments": int(df["segment_id"].nunique()),
        "n_labelled": len(lab),
        "n_reliable": int(df.get("label_reliable", pd.Series(False)).sum()),
        "median_label_dist_m": float(lab["label_dist_m"].median()) if len(lab) else np.nan,
        "p95_label_dist_m": float(lab["label_dist_m"].quantile(0.95)) if len(lab) else np.nan,
        "route_arcs_used": int(lab["route_idx"].nunique()) if len(lab) else 0,
        "n_route_arcs": 0 if trip.route is None else len(trip.route),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    # Trip-level flag: GPS and reference route disagree badly (e.g. train 54, 33)
    summary["suspect_reference"] = bool(summary["median_label_dist_m"] > TRIP_SUSPECT_MEDIAN_M)
    return df, summary


def run(partition: str) -> None:
    frames, rows = [], []
    for tid in list_trips(partition):
        df, s = preprocess_trip(tid, partition)
        frames.append(df)
        rows.append(s)
        print(f"  {partition}/{tid}: {s['n_points']} pts, {s['n_outliers']} outliers, "
              f"median label dist {s['median_label_dist_m']:.1f} m ({s['seconds']} s)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{partition}_points.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)

    summ = pd.DataFrame(rows)
    summ_path = RESULTS_DIR / "eda" / f"preprocess_summary_{partition}.csv"
    summ.to_csv(summ_path, index=False)

    tot = summ[["n_points", "n_invalid", "n_outliers", "n_labelled", "n_reliable"]].sum()
    print(f"[{partition}] {len(summ)} trips, {tot.n_points} points | invalid {tot.n_invalid} | "
          f"outliers {tot.n_outliers} ({tot.n_outliers / tot.n_points:.2%}) | "
          f"reliable labels {tot.n_reliable / tot.n_points:.2%}")
    print(f"  -> {out}\n  -> {summ_path}")


if __name__ == "__main__":
    for part in (sys.argv[1:] or PARTITIONS):
        run(part)
