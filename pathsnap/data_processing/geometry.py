"""Planar geometry helpers (inputs in projected metres)."""
from __future__ import annotations

import numpy as np


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point(s) p to segment(s) a->b (broadcasts), and the
    projection position t in [0, 1] along each segment."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def angle_diff_deg(a, b):
    """Smallest absolute difference between two headings (radians) in degrees, 0..180."""
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)
