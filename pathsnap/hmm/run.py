"""HMM / Viterbi experiments: tune the decoder on train, compare classifier-only vs
classifier + HMM on student test.

Usage:
    python -m pathsnap.hmm.run

Protocol
  * Emissions come from the final classifier (artifacts/final_model.joblib + .json).
  * Train trips use OUT-OF-FOLD probabilities (5-fold GroupKFold with the final model's
    configuration): probabilities a model gives on its own training data are over-confident
    and would make the HMM look less necessary than it is.
  * Decoder parameters (beta, emission weight) are chosen on train only, by observation-level
    Top-1 accuracy; student test is used once, for the final comparison.
Outputs (results/hmm/): tuning_grid.csv, student_test_by_trip.csv, summary.csv,
predictions_student_test.parquet; artifacts/hmm_config.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR
from pathsnap.data_processing.clean import clean_track
from pathsnap.data_processing.loader import list_trips, load_route, load_trip
from pathsnap.features.candidates import FEATURE_COLUMNS
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.viterbi import HMMParams, assign_all, build_lattice, decode, reconstruct_route
from pathsnap.models.evaluate import N_FOLDS
from pathsnap.models.ranker import fit_model, predict_p

OUT = RESULTS_DIR / "hmm"
BETAS = [2, 5, 10, 20, 40, 80, 160]
WEIGHTS = [0.5, 1.0, 2.0]


def load_all_candidates(partition: str) -> pd.DataFrame:
    """ALL candidate rows (not only reliable points): the decoder needs every observation."""
    return pd.read_parquet(PROCESSED_DIR / f"{partition}_candidates.parquet")


def probabilities(partition: str, cand: pd.DataFrame) -> np.ndarray:
    model = joblib.load(ARTIFACTS_DIR / "final_model.joblib")
    if partition != "train":
        return predict_p(model, cand)
    # out-of-fold: each train trip scored by a model that never saw it
    p = np.zeros(len(cand))
    fit_rows = cand["label_reliable"].to_numpy()
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(cand, groups=cand["trip_id"]):
        tr = tr[fit_rows[tr]]
        m = fit_model(clone(model), cand.iloc[tr])
        p[va] = predict_p(m, cand.iloc[va])
    return p


def build(partition: str) -> dict:
    """Per trip: candidates (with p), cleaned track, graph, lattice."""
    cand = load_all_candidates(partition)
    t0 = time.perf_counter()
    cand["p"] = probabilities(partition, cand)
    print(f"[{partition}] probabilities in {time.perf_counter() - t0:.0f} s")
    trips = {}
    t0 = time.perf_counter()
    for tid in list_trips(partition):
        trip = load_trip(tid, partition, with_route=False)
        clean = clean_track(trip.track)
        g = RoadGraph(trip.nodes, trip.arcs)
        c = cand[cand["trip_id"] == tid]
        trips[tid] = {"cand": c, "clean": clean, "graph": g, "lattice": build_lattice(c, clean, g)}
    print(f"[{partition}] lattices for {len(trips)} trips in {time.perf_counter() - t0:.0f} s")
    return trips


def labels_of(trip: dict) -> pd.DataFrame:
    """Reliable reference label per observation (from the candidate table)."""
    c = trip["cand"]
    return c[c["label_reliable"]].groupby("pt")["label_arc"].first()


def score(trips: dict, params: HMMParams) -> dict:
    hit_cls = hit_hmm = n = 0
    for t in trips.values():
        res = assign_all(decode(t["lattice"], params), t["cand"], t["clean"])
        lab = labels_of(t)
        r = res.set_index("pt").loc[lab.index]
        hit_cls += int((r["cls_arc"] == lab).sum()); hit_hmm += int((r["hmm_arc"] == lab).sum()); n += len(lab)
    return {"top1_classifier": hit_cls / n, "top1_hmm": hit_hmm / n, "n_points": n}


def continuity(arcs: np.ndarray, graph: RoadGraph, seg: np.ndarray) -> float:
    same_seg = seg[1:] == seg[:-1]
    a, b = arcs[:-1][same_seg], arcs[1:][same_seg]
    return float(((a == b) | (graph.u[b] == graph.v[a])).mean()) if len(a) else float("nan")


def route_prf(pred: list[int], ref: np.ndarray) -> tuple[float, float, float]:
    p, r = set(pred), set(ref.tolist())
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def evaluate_test(trips: dict, params: HMMParams) -> pd.DataFrame:
    rows, preds = [], []
    for tid, t in trips.items():
        res = assign_all(decode(t["lattice"], params), t["cand"], t["clean"]).sort_values("pt")
        lab = labels_of(t)
        r = res.set_index("pt").loc[lab.index]
        seg = t["clean"]["segment_id"].to_numpy()[res["pt"].to_numpy()]
        ref = load_route(tid, "student_test")
        row = {"trip_id": tid, "n_points": len(lab),
               "top1_classifier": float((r["cls_arc"] == lab).mean()),
               "top1_hmm": float((r["hmm_arc"] == lab).mean())}
        for kind in ["cls", "hmm"]:
            arcs = res[f"{kind}_arc"].to_numpy()
            route, breaks = reconstruct_route(arcs.tolist(), t["graph"])
            prec, rec, f1 = route_prf(route, ref)
            row.update({f"continuity_{kind}": continuity(arcs, t["graph"], seg),
                        f"route_breaks_{kind}": breaks, f"route_precision_{kind}": prec,
                        f"route_recall_{kind}": rec, f"route_f1_{kind}": f1})
        row["steps_without_any_path"] = t["lattice"].n_steps_no_path
        rows.append(row)
        preds.append(res.assign(trip_id=tid, label_arc=res["pt"].map(lab)))
    pd.concat(preds).to_parquet(OUT / "predictions_student_test.parquet", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = build("train")
    t0 = time.perf_counter()
    grid = []
    for beta in BETAS:
        for w in WEIGHTS:
            s = score(train, HMMParams(beta=beta, emission_weight=w))
            grid.append({"beta": beta, "emission_weight": w, **s})
            print(f"  beta={beta:>4} w={w}: train top-1 classifier {s['top1_classifier']:.4f} -> HMM {s['top1_hmm']:.4f}")
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "tuning_grid.csv", index=False)
    best = grid.loc[grid["top1_hmm"].idxmax()]
    params = HMMParams(beta=float(best["beta"]), emission_weight=float(best["emission_weight"]))
    print(f"grid search {time.perf_counter() - t0:.0f} s -> best on train: {params}")

    test = build("student_test")
    by_trip = evaluate_test(test, params)
    by_trip.to_csv(OUT / "student_test_by_trip.csv", index=False)
    w = by_trip["n_points"]
    summary = {k: float(np.average(by_trip[k], weights=w)) for k in by_trip.columns
               if k not in ("trip_id", "n_points", "steps_without_any_path") and not k.startswith("route_breaks")}
    summary.update({"route_breaks_cls": int(by_trip["route_breaks_cls"].sum()),
                    "route_breaks_hmm": int(by_trip["route_breaks_hmm"].sum()),
                    "n_points": int(w.sum()), **{f"param_{k}": v for k, v in asdict(params).items()},
                    "train_top1_classifier_oof": float(best["top1_classifier"]),
                    "train_top1_hmm_oof": float(best["top1_hmm"])})
    pd.Series(summary).to_csv(OUT / "summary.csv", header=["value"])
    (ARTIFACTS_DIR / "hmm_config.json").write_text(json.dumps(
        {**asdict(params), "chosen_by": "train out-of-fold top-1 grid search",
         "grid": {"beta": BETAS, "emission_weight": WEIGHTS}}, indent=2))
    print(pd.Series(summary).round(4).to_string())


if __name__ == "__main__":
    main()
