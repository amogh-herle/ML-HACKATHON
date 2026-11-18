"""Candidate generation + feature engineering.

For every valid GPS observation, the candidate set is the k nearest road arcs
within a search radius. Each (observation, candidate arc) pair becomes one row
with features describing how well that arc explains the observation.
The classifier scores each row; the HMM later uses those scores as emissions.

Uses only the GPS track and the road network - never the reference route - so
the API calls exactly this function at inference time.

Road arcs are directed: a two-way street appears as u->v and v->u with the same
geometry. Distance cannot separate them; heading difference can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pathsnap.config import (CAND_FALLBACK_RADIUS_M, CAND_K, CAND_RADIUS_M, HEADING_PENALTY_M,
                             HEADING_WIDE_WINDOW)
from pathsnap.data_processing.clean import gps_heading as _gps_heading
from pathsnap.data_processing.geometry import angle_diff_deg, point_segment_distance

FEATURE_COLUMNS = [
    "dist_m",              # perpendicular distance GPS point -> arc
    "dist_rank",           # 0 = nearest candidate of this point
    "dist_gap_to_best",    # dist_m - nearest candidate's dist_m
    "n_cand",              # size of this point's candidate set
    "heading_diff_deg",    # |GPS heading - arc direction|, 0..180 (90 if unknown)
    "heading_known",       # 1 if GPS moved enough to estimate heading
    "heading_diff_wide_deg",  # same over a wider window (3rd good point before/after): steadier
    "along_track_m",       # wide-window displacement projected on the arc direction (signed):
                           #   > 0 travelling u->v, < 0 against it; separates two-way twins
    "arc_len_m",
    "proj_frac",           # where on the arc the point projects, 0..1
    "speed_ms",
    "dt_prev",
    "same_as_prev_best",   # arc == previous point's best arc (distance + direction penalty)
    "follows_prev_best",   # arc starts where previous point's best arc ends
    "same_as_next_best",
    "precedes_next_best",  # arc ends where next point's best arc starts
    "out_deg_v",           # branching at arc end (intersection complexity)
    "in_deg_u",
]


def direction_penalty(heading_diff_deg, known):
    """Metres added for pointing against travel: HEADING_PENALTY_M * (1 - cos) / 2, 0 if unknown."""
    return np.where(known, HEADING_PENALTY_M * (1 - np.cos(np.radians(heading_diff_deg))) / 2, 0.0)


@dataclass
class ArcIndex:
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    u: np.ndarray
    v: np.ndarray
    length: np.ndarray
    heading: np.ndarray
    out_deg: np.ndarray  # per node
    in_deg: np.ndarray   # per node
    tree: STRtree


def build_arc_index(nodes: pd.DataFrame, arcs: pd.DataFrame) -> ArcIndex:
    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    lines = shapely.linestrings(np.stack([np.c_[ax, ay], np.c_[bx, by]], axis=1))
    n = len(nodes)
    return ArcIndex(
        ax=ax, ay=ay, bx=bx, by=by, u=u, v=v,
        length=np.hypot(bx - ax, by - ay),
        heading=np.arctan2(by - ay, bx - ax),
        out_deg=np.bincount(u, minlength=n),
        in_deg=np.bincount(v, minlength=n),
        tree=STRtree(lines),
    )


def gps_heading(clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Heading (radians) per observation from the central difference of the
    neighbouring good points in the same segment. Unknown when stationary."""
    heading, known, _, _ = _gps_heading(clean, window=1)
    return heading, known


def _query_pairs(idx: ArcIndex, px, py, pts, radius: float):
    """All (point, arc) pairs within radius, with distance and projection."""
    pi, ai = idx.tree.query(pts, predicate="dwithin", distance=radius)
    d, t = point_segment_distance(px[pi], py[pi], idx.ax[ai], idx.ay[ai], idx.bx[ai], idx.by[ai])
    return pi, ai, d, t


def build_candidates(clean: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                     radius: float = CAND_RADIUS_M, k: int = CAND_K,
                     fallback_radius: float | None = CAND_FALLBACK_RADIUS_M,
                     arc_index: ArcIndex | None = None) -> pd.DataFrame:
    """One row per (observation, candidate arc). `clean` comes from clean_track()."""
    idx = arc_index or build_arc_index(nodes, arcs)
    valid = np.flatnonzero(~clean["is_invalid"].to_numpy())
    px_all, py_all = clean["x"].to_numpy(), clean["y"].to_numpy()
    px, py = px_all[valid], py_all[valid]
    pts = shapely.points(px, py)

    pi, ai, d, t = _query_pairs(idx, px, py, pts, radius)
    fb = np.zeros(len(valid), dtype=bool)
    if fallback_radius is not None:
        missing = np.setdiff1d(np.arange(len(valid)), pi)
        if len(missing):
            fpi, fai, fd, ft = _query_pairs(idx, px[missing], py[missing], pts[missing], fallback_radius)
            fpi = missing[fpi]
            pi, ai, d, t = (np.r_[pi, fpi], np.r_[ai, fai], np.r_[d, fd], np.r_[t, ft])
            fb[missing] = True

    c = pd.DataFrame({"pt": valid[pi], "arc_id": ai, "dist_m": d, "proj_frac": t})
    c = c.sort_values(["pt", "dist_m", "arc_id"], kind="stable")
    c["order"] = c.groupby("pt").cumcount()
    c = c[c["order"] < k].reset_index(drop=True)

    # --- per-point context ---
    g = c.groupby("pt")["dist_m"]
    c["dist_rank"] = g.rank(method="min").astype(int) - 1
    c["dist_gap_to_best"] = c["dist_m"] - g.transform("min")
    c["n_cand"] = g.transform("size")
    c["fallback"] = fb[np.searchsorted(valid, c["pt"].to_numpy())]

    heading, known = gps_heading(clean)
    w_head, w_known, w_dx, w_dy = _gps_heading(clean, window=HEADING_WIDE_WINDOW)
    arc = c["arc_id"].to_numpy()
    pt = c["pt"].to_numpy()
    c["heading_known"] = known[pt].astype(int)
    c["heading_diff_deg"] = np.where(known[pt], angle_diff_deg(heading[pt], idx.heading[arc]), 90.0)
    c["heading_diff_wide_deg"] = np.where(w_known[pt], angle_diff_deg(w_head[pt], idx.heading[arc]), 90.0)
    length = idx.length[arc]
    ux = np.where(length > 0, (idx.bx[arc] - idx.ax[arc]) / np.where(length > 0, length, 1), 0.0)
    uy = np.where(length > 0, (idx.by[arc] - idx.ay[arc]) / np.where(length > 0, length, 1), 0.0)
    c["along_track_m"] = np.where(w_known[pt], w_dx[pt] * ux + w_dy[pt] * uy, 0.0)
    c["arc_len_m"] = length
    c["speed_ms"] = np.nan_to_num(clean["speed_ms"].to_numpy()[pt], nan=0.0, posinf=0.0)
    c["dt_prev"] = np.nan_to_num(clean["dt_prev"].to_numpy()[pt], nan=0.0)
    c["out_deg_v"] = idx.out_deg[idx.v[arc]]
    c["in_deg_u"] = idx.in_deg[idx.u[arc]]

    # --- link to neighbouring good observations' best arc (same segment) ---
    # "best" = lowest distance + direction penalty, so of a two-way street's twins the one
    # pointing along the travel direction is chosen (not whichever has the lower arc id)
    score = c["dist_m"].to_numpy() + direction_penalty(c["heading_diff_deg"].to_numpy(), known[pt])
    best = (c.assign(_s=score).sort_values(["pt", "_s", "arc_id"], kind="stable")
            .groupby("pt").head(1).set_index("pt")["arc_id"])
    good = ~(clean["is_invalid"] | clean["is_outlier"]).to_numpy()
    seg = clean["segment_id"].to_numpy()
    gp = np.array([p for p in best.index if good[p]])
    best_arc = np.full(len(clean), -1)
    best_arc[best.index.to_numpy()] = best.to_numpy()
    prev_p = np.full(len(clean), -1)
    next_p = np.full(len(clean), -1)
    if len(gp) > 1:
        same_prev = seg[gp[1:]] == seg[gp[:-1]]
        prev_p[gp[1:][same_prev]] = gp[:-1][same_prev]
        next_p[gp[:-1][same_prev]] = gp[1:][same_prev]
    pb = np.where(prev_p[pt] >= 0, best_arc[np.maximum(prev_p[pt], 0)], -1)
    nb = np.where(next_p[pt] >= 0, best_arc[np.maximum(next_p[pt], 0)], -1)
    c["same_as_prev_best"] = ((pb >= 0) & (arc == pb)).astype(int)
    c["follows_prev_best"] = ((pb >= 0) & (idx.u[arc] == idx.v[np.maximum(pb, 0)])).astype(int)
    c["same_as_next_best"] = ((nb >= 0) & (arc == nb)).astype(int)
    c["precedes_next_best"] = ((nb >= 0) & (idx.v[arc] == idx.u[np.maximum(nb, 0)])).astype(int)

    # carry observation identifiers / flags
    c["obs_idx"] = clean["obs_idx"].to_numpy()[pt]
    c["segment_id"] = seg[pt]
    c["is_outlier"] = clean["is_outlier"].to_numpy()[pt]
    return c.drop(columns=["order"])
