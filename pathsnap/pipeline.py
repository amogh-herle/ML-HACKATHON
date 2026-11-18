"""End-to-end inference for one trip: GPS track + road network in, matched route out.
This is what the API runs. It never reads the reference route; scoring against the
reference is a separate, server-side function (`score_trip`).

    from pathsnap.pipeline import Matcher
    m = Matcher()
    result = m.match("00000000", "student_test")
    metrics = score_trip("00000000", "student_test", result)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACTS_DIR, DATA_ROOT
from pathsnap.data_processing.clean import clean_track
from pathsnap.data_processing.labels import label_track
from pathsnap.data_processing.loader import load_route, load_trip
from pathsnap.features.candidates import FEATURE_COLUMNS, build_candidates
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.viterbi import HMMParams, assign_all, build_lattice, decode, reconstruct_route


@dataclass
class MatchResult:
    trip_id: str
    partition: str
    points: pd.DataFrame          # obs_idx, lon, lat, t, is_outlier, cls_arc, hmm_arc
    route_arcs: list[int]         # connected predicted route (HMM)
    route_breaks: int
    arc_coords: dict[int, list]   # arc_id -> [[lon, lat], [lon, lat]] for every arc used
    timings_s: dict[str, float]


class Matcher:
    def __init__(self, model_path=None, hmm_config_path=None, data_root=DATA_ROOT):
        self.model = joblib.load(model_path or ARTIFACTS_DIR / "final_model.joblib")
        cfg = json.loads((hmm_config_path or ARTIFACTS_DIR / "hmm_config.json").read_text())
        self.params = HMMParams(beta=cfg["beta"], emission_weight=cfg["emission_weight"],
                                max_mismatch_m=cfg["max_mismatch_m"])
        self.data_root = data_root

    def match(self, trip_id: str, partition: str) -> MatchResult:
        t = {}
        t0 = time.perf_counter()
        trip = load_trip(trip_id, partition, with_route=False, data_root=self.data_root)
        clean = clean_track(trip.track)
        t["load_clean"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cand = build_candidates(clean, trip.nodes, trip.arcs)
        cand["p"] = self.model.predict_proba(cand[FEATURE_COLUMNS].to_numpy())[:, 1]
        t["candidates_classifier"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        graph = RoadGraph(trip.nodes, trip.arcs)
        res = assign_all(decode(build_lattice(cand, clean, graph, self.params.max_mismatch_m),
                                self.params), cand, clean).sort_values("pt")
        route, breaks = reconstruct_route(res["hmm_arc"].tolist(), graph)
        t["hmm_viterbi"] = time.perf_counter() - t0

        pts = clean.loc[res["pt"], ["obs_idx", "lon", "lat", "t", "is_outlier"]].reset_index(drop=True)
        pts["cls_arc"] = res["cls_arc"].to_numpy()
        pts["hmm_arc"] = res["hmm_arc"].to_numpy()
        used = set(route) | set(pts["cls_arc"]) | set(pts["hmm_arc"])
        lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
        coords = {int(a): [[float(lon[graph.u[a]]), float(lat[graph.u[a]])],
                           [float(lon[graph.v[a]]), float(lat[graph.v[a]])]] for a in used}
        return MatchResult(trip_id, partition, pts, route, breaks, coords, t)


def score_trip(trip_id: str, partition: str, result: MatchResult, data_root=DATA_ROOT) -> dict:
    """SERVER-SIDE ONLY: compare a match with the reference route (never sent to the model)."""
    trip = load_trip(trip_id, partition, with_route=True, data_root=data_root)
    lab = label_track(clean_track(trip.track), trip.nodes, trip.arcs, trip.route)
    lab = lab[lab["label_reliable"]].set_index("obs_idx")["label_arc"]
    p = result.points.set_index("obs_idx").reindex(lab.index)
    ref = set(load_route(trip_id, partition, data_root).tolist())
    pred = set(result.route_arcs)
    inter = len(ref & pred)
    prec, rec = inter / max(len(pred), 1), inter / max(len(ref), 1)
    return {
        "n_scored_points": int(len(lab)),
        "top1_accuracy_hmm": float((p["hmm_arc"] == lab).mean()),
        "top1_accuracy_classifier": float((p["cls_arc"] == lab).mean()),
        "route_precision": prec, "route_recall": rec,
        "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }
