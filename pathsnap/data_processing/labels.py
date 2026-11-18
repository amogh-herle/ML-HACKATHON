"""Per-observation reference labels from the reference route.

The .route file lists arcs in driving order but does not say which GPS point
belongs to which arc. We assign each (non-outlier) GPS point to one route
position j with a monotonic alignment:

    minimise  sum_i dist(point_i, route_arc[j_i])   subject to  j_0 <= j_1 <= ...

Monotonicity matters when the route revisits a street (loops, U-turns): a plain
nearest-route-arc lookup would jump back and forth between visits.
Solved exactly with DP in O(N*M):  D[i, j] = c[i, j] + min_{k<=j} D[i-1, k].

TRAINING / EVALUATION ONLY - the prediction pipeline must never call this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.config import LABEL_MAX_DIST_M
from pathsnap.data_processing.geometry import point_segment_distance


def monotone_align(px: np.ndarray, py: np.ndarray, ax, ay, bx, by) -> np.ndarray:
    """Return route position j for every point (non-decreasing)."""
    n, m = len(px), len(ax)
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    idx = np.arange(m)
    dp, _ = point_segment_distance(px[0], py[0], ax, ay, bx, by)
    back[0] = idx
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp == pm, idx, 0))  # argmin of prefix
        c, _ = point_segment_distance(px[i], py[i], ax, ay, bx, by)
        dp = c + pm
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j


def label_track(clean: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                route: np.ndarray) -> pd.DataFrame:
    """Add route_idx, label_arc, label_dist_m, label_frac, label_reliable.
    Invalid/outlier points get label_arc = -1 and are excluded from training."""
    df = clean.copy()
    u = arcs["u"].to_numpy()[route]
    v = arcs["v"].to_numpy()[route]
    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]

    use = ~(df["is_invalid"] | df["is_outlier"]).to_numpy()
    px, py = df.loc[use, "x"].to_numpy(), df.loc[use, "y"].to_numpy()

    j = monotone_align(px, py, ax, ay, bx, by)
    dist = np.empty(len(j))
    frac = np.empty(len(j))
    for k, (x, y, jj) in enumerate(zip(px, py, j)):
        d, t = point_segment_distance(x, y, ax[jj], ay[jj], bx[jj], by[jj])
        dist[k], frac[k] = d, t

    df["route_idx"] = -1
    df["label_arc"] = -1
    df["label_dist_m"] = np.nan
    df["label_frac"] = np.nan
    df.loc[use, "route_idx"] = j
    df.loc[use, "label_arc"] = route[j]
    df.loc[use, "label_dist_m"] = dist
    df.loc[use, "label_frac"] = frac
    df["label_reliable"] = use & (df["label_dist_m"] <= LABEL_MAX_DIST_M).to_numpy()
    return df
