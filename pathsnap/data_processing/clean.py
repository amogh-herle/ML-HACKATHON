"""GPS track cleaning. Uses only the track itself (never the route), so the
same function runs at training time and inside the API at inference time.

Nothing is dropped: every observation keeps its original obs_idx so the API can
return a prediction per observation. Cleaning adds columns instead:

  dt_prev      seconds since previous observation (NaN for first)
  step_m       metres moved since previous observation
  speed_ms     step_m / dt_prev
  is_outlier   implied speed from last good point > MAX_SPEED_MS
  segment_id   increments at every gap > GAP_BREAK_S (HMM may restart there)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.config import GAP_BREAK_S, HEADING_MIN_MOVE_M, MAX_OUTLIER_RUN, MAX_SPEED_MS


def gps_heading(clean: pd.DataFrame, window: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Travel direction per observation from the displacement between the `window`-th good
    point before and the `window`-th good point after it (central difference), never across a
    segment break. Returns (heading radians, known, dx, dy); heading is unknown when the
    displacement is shorter than HEADING_MIN_MOVE_M (vehicle stopped)."""
    n = len(clean)
    heading = np.zeros(n)
    known = np.zeros(n, dtype=bool)
    dx_all, dy_all = np.zeros(n), np.zeros(n)
    good = ~(clean["is_invalid"] | clean["is_outlier"]).to_numpy()
    x, y, seg = clean["x"].to_numpy(), clean["y"].to_numpy(), clean["segment_id"].to_numpy()
    gi = np.flatnonzero(good)
    if len(gi) < 2:
        return heading, known, dx_all, dy_all
    k = np.arange(len(gi))
    s = seg[gi]
    first = pd.Series(k).groupby(s).transform("min").to_numpy()   # first position of the segment
    last = pd.Series(k).groupby(s).transform("max").to_numpy()
    prev_i = gi[np.maximum(k - window, first)]
    next_i = gi[np.minimum(k + window, last)]
    dx, dy = x[next_i] - x[prev_i], y[next_i] - y[prev_i]
    heading[gi] = np.arctan2(dy, dx)
    known[gi] = np.hypot(dx, dy) >= HEADING_MIN_MOVE_M
    dx_all[gi], dy_all[gi] = dx, dy
    return heading, known, dx_all, dy_all


def flag_outliers(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                  max_speed: float = MAX_SPEED_MS, max_run: int = MAX_OUTLIER_RUN) -> np.ndarray:
    """Greedy forward filter: a point is an outlier if reaching it from the last
    accepted point needs speed > max_speed. If max_run points in a row are
    rejected, the anchor itself was probably wrong, so re-anchor on the current point."""
    n = len(x)
    out = np.zeros(n, dtype=bool)
    anchor, run = 0, 0
    for i in range(1, n):
        dt = t[i] - t[anchor]
        d = np.hypot(x[i] - x[anchor], y[i] - y[anchor])
        if dt > 0 and d / dt <= max_speed:
            anchor, run = i, 0
            continue
        run += 1
        if run > max_run:
            anchor, run = i, 0           # accept as new anchor
        else:
            out[i] = True
    return out


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    df = track.copy()
    df.insert(0, "obs_idx", np.arange(len(df)))
    df = df.sort_values("t", kind="stable").reset_index(drop=True)

    x, y, t = df["x"].to_numpy(), df["y"].to_numpy(), df["t"].to_numpy()
    invalid = (~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(t)
               | (df["lat"].abs() > 90).to_numpy() | (df["lon"].abs() > 180).to_numpy())
    dup_t = np.r_[False, np.diff(t) <= 0]

    dt = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)

    df["dt_prev"] = dt
    df["step_m"] = step
    df["speed_ms"] = speed
    df["is_invalid"] = invalid | dup_t
    df["is_outlier"] = False
    good = ~df["is_invalid"].to_numpy()
    df.loc[good, "is_outlier"] = flag_outliers(x[good], y[good], t[good])
    df["segment_id"] = np.cumsum(np.nan_to_num(dt, nan=0.0) > GAP_BREAK_S)
    return df
