"""Read raw PathSnap trip files.

Each trip folder <partition>/<trip_id>/ holds:
  <trip_id>.track  lon lat t          (GPS observations, t in seconds)
  <trip_id>.nodes  lon lat            (row index = node_id)
  <trip_id>.arcs   from_node to_node  (row index = arc_id)
  <trip_id>.route  arc_id             (reference route, chronological)

Road networks are per-trip: arc/node ids are only meaningful inside their own trip.

The route is loaded separately (load_route) so prediction code can be written
to never touch it; load_trip(with_route=False) is what inference should use.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from pathsnap.config import DATA_ROOT, WGS84


def utm_crs_for(lon: float, lat: float) -> str:
    """EPSG code of the UTM zone containing (lon, lat): 326zz north, 327zz south."""
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@lru_cache(maxsize=64)
def _transformer(crs: str) -> Transformer:
    # always_xy=True -> input/output order is (lon, lat) / (x, y)
    return Transformer.from_crs(WGS84, crs, always_xy=True)


def project(lon: np.ndarray, lat: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 lon/lat (degrees) -> projected x/y (metres) in `crs`."""
    return _transformer(crs).transform(lon, lat)


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame  # columns: lon, lat, t, x, y   (row = observation index)
    nodes: pd.DataFrame  # columns: lon, lat, x, y      (index = node_id)
    arcs: pd.DataFrame   # columns: u, v                (index = arc_id)
    crs: str = WGS84                 # projected CRS used for x/y
    route: np.ndarray | None = None  # arc_ids, only when with_route=True

    def __repr__(self) -> str:
        r = "hidden" if self.route is None else len(self.route)
        return (f"Trip({self.partition}/{self.trip_id}: {len(self.track)} pts, "
                f"{len(self.nodes)} nodes, {len(self.arcs)} arcs, route={r}, {self.crs})")


def trip_dir(trip_id: str, partition: str, data_root: Path = DATA_ROOT) -> Path:
    return Path(data_root) / partition / trip_id


def list_trips(partition: str, data_root: Path = DATA_ROOT) -> list[str]:
    root = Path(data_root) / partition
    if not root.is_dir():
        raise FileNotFoundError(f"Partition folder not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _add_xy(df: pd.DataFrame, crs: str) -> pd.DataFrame:
    df["x"], df["y"] = project(df["lon"].to_numpy(), df["lat"].to_numpy(), crs)
    return df


def load_track(d: Path, trip_id: str) -> pd.DataFrame:
    return _read(d / f"{trip_id}.track", ["lon", "lat", "t"], float)


def load_nodes(d: Path, trip_id: str) -> pd.DataFrame:
    df = _read(d / f"{trip_id}.nodes", ["lon", "lat"], float)
    df.index.name = "node_id"
    return df


def load_arcs(d: Path, trip_id: str) -> pd.DataFrame:
    df = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64)
    df.index.name = "arc_id"
    return df


def load_route(trip_id: str, partition: str, data_root: Path = DATA_ROOT) -> np.ndarray | None:
    """Reference route arc_ids, or None if the file is absent."""
    p = trip_dir(trip_id, partition, data_root) / f"{trip_id}.route"
    if not p.exists():
        return None
    return _read(p, ["arc_id"], np.int64)["arc_id"].to_numpy()


def load_trip(trip_id: str, partition: str, with_route: bool = True,
              data_root: Path = DATA_ROOT) -> Trip:
    d = trip_dir(trip_id, partition, data_root)
    track = load_track(d, trip_id)
    crs = utm_crs_for(track["lon"].median(), track["lat"].median())
    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=_add_xy(track, crs),
        nodes=_add_xy(load_nodes(d, trip_id), crs),
        arcs=load_arcs(d, trip_id),
        crs=crs,
        route=load_route(trip_id, partition, data_root) if with_route else None,
    )


def iter_trips(partition: str, with_route: bool = True, data_root: Path = DATA_ROOT):
    for tid in list_trips(partition, data_root):
        yield load_trip(tid, partition, with_route, data_root)
