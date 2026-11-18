"""Dataset inventory + data-quality checks for every trip.

Usage:
    python -m pathsnap.data_processing.inventory            # train + student_test
    python -m pathsnap.data_processing.inventory train

Writes results/eda/inventory_<partition>.csv (one row per trip) and prints a summary.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import PARTITIONS, RESULTS_DIR
from pathsnap.data_processing.loader import list_trips, load_trip, trip_dir

GAP_S = 10.0          # time gap (s) flagged as a GPS dropout
MAX_SPEED_MS = 50.0   # ~180 km/h; faster implied speed = likely GPS jump


def trip_stats(trip) -> dict:
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    n_nodes = len(nodes)
    s: dict = {"trip_id": trip.trip_id, "partition": trip.partition}

    # file sizes
    d = trip_dir(trip.trip_id, trip.partition)
    for ext in ("track", "nodes", "arcs", "route"):
        p = d / f"{trip.trip_id}.{ext}"
        s[f"{ext}_bytes"] = p.stat().st_size if p.exists() else 0

    # --- track ---
    t = tr["t"].to_numpy()
    dt = np.diff(t)
    step = np.hypot(np.diff(tr["x"].to_numpy()), np.diff(tr["y"].to_numpy()))
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.inf)
    s.update(
        n_points=len(tr),
        n_nan=int(tr[["lon", "lat", "t"]].isna().sum().sum()),
        duration_s=float(t[-1] - t[0]),
        n_non_increasing_t=int((dt <= 0).sum()),
        frac_dt_1s=float(np.mean(np.isclose(dt, 1.0))),
        median_dt=float(np.median(dt)),
        max_dt=float(dt.max()),
        n_gaps_gt_10s=int((dt > GAP_S).sum()),
        n_duplicate_pos=int((step == 0).sum()),
        path_length_m=float(step.sum()),
        median_speed_ms=float(np.median(speed[np.isfinite(speed)])),
        n_speed_gt_50ms=int((speed > MAX_SPEED_MS).sum()),
        lon_min=tr["lon"].min(), lon_max=tr["lon"].max(),
        lat_min=tr["lat"].min(), lat_max=tr["lat"].max(),
    )

    # --- road network ---
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    bad_ref = (u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes)
    ok = ~bad_ref
    g = coo_matrix((np.ones(ok.sum()), (u[ok], v[ok])), shape=(n_nodes, n_nodes))
    n_wcc, labels = connected_components(g, directed=True, connection="weak")
    n_scc, _ = connected_components(g, directed=True, connection="strong")
    degree = np.bincount(u[ok], minlength=n_nodes) + np.bincount(v[ok], minlength=n_nodes)
    seg_len = np.hypot(nodes["x"].to_numpy()[u[ok]] - nodes["x"].to_numpy()[v[ok]],
                       nodes["y"].to_numpy()[u[ok]] - nodes["y"].to_numpy()[v[ok]])
    s.update(
        n_nodes=n_nodes,
        n_arcs=len(arcs),
        n_arcs_bad_node_ref=int(bad_ref.sum()),
        n_self_loops=int((u == v).sum()),
        n_isolated_nodes=int((degree == 0).sum()),
        n_weak_components=int(n_wcc),
        largest_wcc_frac=float(np.bincount(labels).max() / n_nodes),
        n_strong_components=int(n_scc),
        median_arc_len_m=float(np.median(seg_len)),
        net_lon_min=nodes["lon"].min(), net_lon_max=nodes["lon"].max(),
        net_lat_min=nodes["lat"].min(), net_lat_max=nodes["lat"].max(),
    )
    s["track_inside_net_bbox"] = bool(
        s["lon_min"] >= s["net_lon_min"] and s["lon_max"] <= s["net_lon_max"]
        and s["lat_min"] >= s["net_lat_min"] and s["lat_max"] <= s["net_lat_max"])

    # --- reference route ---
    if route is None:
        s.update(n_route_arcs=0)
        return s
    valid = (route >= 0) & (route < len(arcs))
    r = route[valid]
    connected = v[r[:-1]] == u[r[1:]]
    s.update(
        n_route_arcs=len(route),
        n_route_bad_arc_ref=int((~valid).sum()),
        n_route_unique_arcs=int(len(np.unique(r))),
        n_route_repeat_consecutive=int((r[1:] == r[:-1]).sum()),
        route_continuity=float(connected.mean()) if len(connected) else np.nan,
        n_route_breaks=int((~connected).sum()),
        route_length_m=float(np.hypot(
            nodes["x"].to_numpy()[u[r]] - nodes["x"].to_numpy()[v[r]],
            nodes["y"].to_numpy()[u[r]] - nodes["y"].to_numpy()[v[r]]).sum()),
    )
    return s


def run(partition: str) -> pd.DataFrame:
    rows = []
    for tid in list_trips(partition):
        rows.append(trip_stats(load_trip(tid, partition, with_route=True)))
        print(f"  {partition}/{tid} done", end="\r")
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "eda" / f"inventory_{partition}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[{partition}] {len(df)} trips -> {out}")
    return df


def summarise(df: pd.DataFrame) -> None:
    cols = ["n_points", "duration_s", "frac_dt_1s", "max_dt", "n_gaps_gt_10s",
            "n_speed_gt_50ms", "n_nodes", "n_arcs", "n_weak_components",
            "largest_wcc_frac", "n_route_arcs", "route_continuity", "n_route_breaks"]
    print(df[cols].describe().T[["min", "50%", "mean", "max"]].round(3).to_string())
    flags = {
        "NaN values": df["n_nan"].gt(0),
        "non-increasing timestamps": df["n_non_increasing_t"].gt(0),
        "arcs with bad node refs": df["n_arcs_bad_node_ref"].gt(0),
        "route arcs with bad refs": df.get("n_route_bad_arc_ref", pd.Series(0, df.index)).gt(0),
        "route not fully connected": df["route_continuity"].lt(1),
        "track outside network bbox": ~df["track_inside_net_bbox"],
        "gaps > 10 s": df["n_gaps_gt_10s"].gt(0),
    }
    for name, mask in flags.items():
        ids = df.loc[mask, "trip_id"].tolist()
        print(f"  {name}: {len(ids)} trips {ids[:10]}{' ...' if len(ids) > 10 else ''}")


if __name__ == "__main__":
    for part in (sys.argv[1:] or PARTITIONS):
        summarise(run(part))
