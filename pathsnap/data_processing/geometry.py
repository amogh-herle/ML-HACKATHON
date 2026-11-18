"""Planar geometry helpers (inputs in projected metres)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def point_segment_distance(
    px: ArrayLike, py: ArrayLike, ax: ArrayLike, ay: ArrayLike, bx: ArrayLike, by: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distance from point(s) p to segment(s) a->b (broadcasts), and the
    projection position t in [0, 1] along each segment.

    All coordinates are projected metres (see ``config.WGS84`` for why each trip uses its own
    UTM zone). A zero-length segment (a == b) is treated as a point: t = 0 and the distance is
    the distance to a. Returns ``(distance_m, t)``, both broadcast to a common shape.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def angle_diff_deg(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Smallest absolute difference between two headings (radians) in degrees, 0..180.

    Wrap-around safe: headings of 359 deg and 1 deg differ by 2 deg, not 358.
    """
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)
