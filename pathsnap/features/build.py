"""Build labelled candidate tables, or sweep candidate settings.

Usage:
    python -m pathsnap.features.build --sweep        # miss rate vs radius/k (train + student_test)
    python -m pathsnap.features.build                # candidate tables with config settings

Outputs:
    data/processed/<partition>_candidates.parquet   one row per (observation, candidate arc)
    results/eda/candidate_sweep.csv                 miss rate per (partition, radius, k)
    results/eda/candidate_summary_<partition>.csv   per-trip miss rate / set size

Training rows = observations with label_reliable; label = (arc_id == label_arc).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pathsnap.config import CAND_K, CAND_RADIUS_M, PARTITIONS, PROCESSED_DIR, RESULTS_DIR
from pathsnap.data_processing.clean import clean_track
from pathsnap.data_processing.labels import label_track
from pathsnap.data_processing.loader import list_trips, load_trip
from pathsnap.features.candidates import build_candidates

SWEEP_RADII = [15, 25, 50, 75, 100]
SWEEP_K = [1, 2, 4, 8, 12, 16, 20]


def labelled_candidates(trip_id: str, partition: str, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    trip = load_trip(trip_id, partition, with_route=True)
    pts = label_track(clean_track(trip.track), trip.nodes, trip.arcs, trip.route)
    cand = build_candidates(pts, trip.nodes, trip.arcs, **kw)
    p = cand["pt"].to_numpy()
    cand["label_arc"] = pts["label_arc"].to_numpy()[p]
    cand["label_reliable"] = pts["label_reliable"].to_numpy()[p]
    cand["label"] = (cand["arc_id"] == cand["label_arc"]).astype(int)
    cand.insert(0, "trip_id", trip_id)
    pts.insert(0, "trip_id", trip_id)
    return cand, pts


def sweep(partition: str) -> pd.DataFrame:
    rows = []
    for tid in list_trips(partition):
        cand, pts = labelled_candidates(tid, partition, radius=max(SWEEP_RADII),
                                        k=max(SWEEP_K), fallback_radius=None)
        rel = pts.index[pts["label_reliable"]].to_numpy()  # pt positions with reliable labels
        hits = cand[(cand["label"] == 1) & cand["label_reliable"]]
        # position of the true arc in distance order, and its distance
        order = cand.groupby("pt").cumcount()
        hit_rank = pd.Series(order[hits.index].to_numpy(), index=hits["pt"].to_numpy())
        hit_dist = pd.Series(hits["dist_m"].to_numpy(), index=hits["pt"].to_numpy())
        n_within = cand.assign(o=order)
        for r in SWEEP_RADII:
            within = n_within[n_within["dist_m"] <= r].groupby("pt").size()
            for k in SWEEP_K:
                ok = (hit_rank < k) & (hit_dist <= r)
                rows.append({"partition": partition, "trip_id": tid, "radius_m": r, "k": k,
                             "n_points": len(rel),
                             "n_hit": int(ok.reindex(rel, fill_value=False).sum()),
                             "sum_cand": int(np.minimum(within.reindex(rel, fill_value=0), k).sum())})
        print(f"  sweep {partition}/{tid}", end="\r")
    df = pd.DataFrame(rows).groupby(["partition", "radius_m", "k"], as_index=False)[
        ["n_points", "n_hit", "sum_cand"]].sum()
    df["miss_rate"] = 1 - df["n_hit"] / df["n_points"]
    df["mean_cand"] = df["sum_cand"] / df["n_points"]
    return df


def build(partition: str) -> None:
    frames, rows = [], []
    for tid in list_trips(partition):
        t0 = time.perf_counter()
        cand, pts = labelled_candidates(tid, partition)
        frames.append(cand)
        rel = cand[cand["label_reliable"]]
        n_rel = rel["pt"].nunique()
        n_hit = rel.loc[rel["label"] == 1, "pt"].nunique()
        top1 = rel[(rel["dist_rank"] == 0)].groupby("pt")["label"].max().mean()
        rows.append({"trip_id": tid, "n_points": len(pts), "n_with_cand": cand["pt"].nunique(),
                     "n_rows": len(cand), "n_reliable": n_rel,
                     "miss_rate": 1 - n_hit / n_rel if n_rel else np.nan,
                     "nearest_arc_top1": top1,
                     "n_fallback_pts": int(cand.loc[cand["fallback"], "pt"].nunique()),
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(f"  {partition}/{tid}: {len(cand)} rows, miss {rows[-1]['miss_rate']:.3f}, "
              f"nearest-arc top1 {top1:.3f} ({rows[-1]['seconds']} s)")

    out = PROCESSED_DIR / f"{partition}_candidates.parquet"
    allc = pd.concat(frames, ignore_index=True)
    allc.to_parquet(out, index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS_DIR / "eda" / f"candidate_summary_{partition}.csv", index=False)

    rel = allc[allc["label_reliable"]]
    key = ["trip_id", "pt"]
    n_rel = len(rel[key].drop_duplicates())
    n_hit = len(rel.loc[rel["label"] == 1, key].drop_duplicates())
    print(f"[{partition}] {len(allc)} rows, {n_rel} reliable points, "
          f"miss rate {1 - n_hit / n_rel:.4f}, positives {rel['label'].mean():.3f} of rows\n  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("partitions", nargs="*", default=list(PARTITIONS))
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        res = pd.concat([sweep(p) for p in a.partitions], ignore_index=True)
        path = RESULTS_DIR / "eda" / "candidate_sweep.csv"
        res.to_csv(path, index=False)
        piv = res.pivot_table(index=["partition", "radius_m"], columns="k", values="miss_rate")
        print("\nmiss rate (rows: partition, radius m; cols: k)\n", piv.round(4).to_string())
        print("\nmean candidates per point\n",
              res.pivot_table(index=["partition", "radius_m"], columns="k", values="mean_cand").round(1).to_string())
        print(f"  -> {path}")
    else:
        print(f"radius={CAND_RADIUS_M} m, k={CAND_K}")
        for p in a.partitions:
            build(p)
