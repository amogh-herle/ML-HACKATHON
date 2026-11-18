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

from pathsnap.config import GAP_BREAK_S, MAX_OUTLIER_RUN, MAX_SPEED_MS


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
