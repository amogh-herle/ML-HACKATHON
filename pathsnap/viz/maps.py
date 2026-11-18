"""Interactive Folium maps of a trip: GPS observations, nearby road network,
reference route and (optionally) a predicted route, with layer control + legend.

Usage:
    python -m pathsnap.viz.maps train 00000001 student_test 00000000 ...
    python -m pathsnap.viz.maps                       # default showcase trips

Writes results/maps/<partition>_<trip_id>.html

trip_map() is reusable: pass predicted_arcs (arc_ids in order) to add the
predicted-route layer, e.g. from the classifier or the HMM decoder.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

import folium
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR
from pathsnap.data_processing.clean import clean_track
from pathsnap.data_processing.loader import Trip, load_trip

COLORS = {
    "network": "#9e9e9e",
    "gps": "#1f77b4",
    "outlier": "#d62728",
    "reference": "#2ca02c",
    "predicted": "#ff7f0e",
}
NETWORK_MARGIN_M = 300.0  # draw road arcs within this distance of the track bbox


def _lines(lon_a, lat_a, lon_b, lat_b) -> list:
    return [[[round(a, 6), round(b, 6)], [round(c, 6), round(d, 6)]]
            for a, b, c, d in zip(lon_a, lat_a, lon_b, lat_b)]


def _arc_multiline(trip: Trip, arc_ids: np.ndarray) -> dict:
    u = trip.arcs["u"].to_numpy()[arc_ids]
    v = trip.arcs["v"].to_numpy()[arc_ids]
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    return {"type": "MultiLineString", "coordinates": _lines(lon[u], lat[u], lon[v], lat[v])}


def _nearby_network_arcs(trip: Trip) -> np.ndarray:
    """Arcs with an endpoint inside the track bbox (+margin); two-way twins deduplicated."""
    x, y = trip.track["x"], trip.track["y"]
    x0, x1 = x.min() - NETWORK_MARGIN_M, x.max() + NETWORK_MARGIN_M
    y0, y1 = y.min() - NETWORK_MARGIN_M, y.max() + NETWORK_MARGIN_M
    nx, ny = trip.nodes["x"].to_numpy(), trip.nodes["y"].to_numpy()
    inside = (nx >= x0) & (nx <= x1) & (ny >= y0) & (ny <= y1)
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    keep = inside[u] | inside[v]
    pairs = pd.DataFrame({"a": np.minimum(u, v), "b": np.maximum(u, v)})[keep]
    return pairs.drop_duplicates().index.to_numpy()


def _legend(items: dict[str, str]) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items.items())
    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px sans-serif;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.3)">{rows}</div>')


def trip_map(trip: Trip, clean: pd.DataFrame | None = None,
             predicted_arcs: Sequence[int] | None = None,
             title: str | None = None) -> folium.Map:
    clean = clean_track(trip.track) if clean is None else clean
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    legend = {}

    # road network near the trip
    net = _nearby_network_arcs(trip)
    folium.GeoJson(_arc_multiline(trip, net), name=f"Road network ({len(net)} segments)",
                   style_function=lambda _: {"color": COLORS["network"], "weight": 1.5,
                                             "opacity": 0.7}).add_to(m)
    legend["Road network"] = COLORS["network"]

    # reference route (ground truth; NEVER pass this in the evaluator UI)
    if trip.route is not None:
        folium.GeoJson(_arc_multiline(trip, trip.route),
                       name=f"Reference route ({len(trip.route)} arcs)",
                       style_function=lambda _: {"color": COLORS["reference"], "weight": 6,
                                                 "opacity": 0.6}).add_to(m)
        legend["Reference route"] = COLORS["reference"]

    if predicted_arcs is not None:
        # a list -> one "Predicted route" layer; a dict {name: arcs} -> one layer per entry
        layers = predicted_arcs if isinstance(predicted_arcs, dict) else {"Predicted route": predicted_arcs}
        palette = [COLORS["predicted"], "#9467bd", "#8c564b"]
        for (name, arcs_), color in zip(layers.items(), palette):
            pa = np.asarray(arcs_, dtype=int)
            folium.GeoJson(_arc_multiline(trip, pa), name=f"{name} ({len(pa)} arcs)",
                           style_function=lambda _, c=color: {"color": c, "weight": 3,
                                                              "opacity": 0.9, "dashArray": "6 4"}).add_to(m)
            legend[name] = color

    # GPS observations (good vs flagged outliers)
    bad = (clean["is_outlier"] | clean["is_invalid"]).to_numpy()
    for name, mask, color in [("GPS observations", ~bad, COLORS["gps"]),
                              ("GPS outliers", bad, COLORS["outlier"])]:
        sub = clean[mask]
        if not len(sub):
            continue
        feats = [{"type": "Feature",
                  "geometry": {"type": "Point", "coordinates": [round(lo, 6), round(la, 6)]},
                  "properties": {"obs": int(o), "t": float(t), "speed_kmh": round(float(s) * 3.6, 1)
                                 if np.isfinite(s) else None}}
                 for lo, la, o, t, s in zip(sub["lon"], sub["lat"], sub["obs_idx"], sub["t"],
                                            sub["speed_ms"])]
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       name=f"{name} ({len(sub)})",
                       marker=folium.CircleMarker(radius=2.5, fill=True, fill_opacity=0.8, weight=0),
                       style_function=lambda _, c=color: {"color": c, "fillColor": c},
                       tooltip=folium.GeoJsonTooltip(["obs", "t", "speed_kmh"])).add_to(m)
        legend[name] = color

    # start / end markers
    first, last = clean.iloc[0], clean.iloc[-1]
    folium.Marker([first.lat, first.lon], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([last.lat, last.lon], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[clean.lat.min(), clean.lon.min()], [clean.lat.max(), clean.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    if title:
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
            f'padding:4px 8px;border:1px solid #999;font:bold 13px sans-serif">{title}</div>'))
    return m


DEFAULT_TRIPS = [("train", "00000001"),        # autobahn trip, Germany
                 ("train", "00000054"),        # suspect reference (GPS ~140 m off route)
                 ("student_test", "00000000")]


def main(args: list[str]) -> None:
    pairs = list(zip(args[::2], args[1::2])) if args else DEFAULT_TRIPS
    out_dir = RESULTS_DIR / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for partition, tid in pairs:
        trip = load_trip(tid, partition, with_route=True)
        m = trip_map(trip, title=f"{partition} / {tid} - GPS vs reference route")
        path = out_dir / f"{partition}_{tid}.html"
        m.save(str(path))
        print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
