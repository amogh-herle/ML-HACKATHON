"""HMM map matching with Viterbi decoding.

Model (one HMM per track segment; segments are split at time gaps > GAP_BREAK_S)
  * Time steps: the GPS observations of the segment (outliers excluded, filled afterwards).
  * Hidden state at step i: which candidate arc the vehicle is on (<= CAND_K states).
    Each state keeps its arc_id and projection fraction, so the candidate-to-road-segment
    correspondence is preserved.
  * Emission: log of the classifier probability, normalised over the step's candidates,
        e_i(s) = w * log( p_s / sum_s' p_s' )
  * Transition (Newson & Krumm 2009): a real move between two fixes covers about the same
    distance on the road as in a straight line, so
        t_i(s', s) = - min(|route_dist(s' -> s) - gps_dist(i-1, i)|, M) / beta
    route_dist = driving distance through the network from the previous candidate's
    position to this one (bounded Dijkstra). No path within reach (disconnected arcs,
    wrong direction, other component) gets the capped mismatch M: a strong penalty but
    never probability zero, so the decoder can always continue.
  * Viterbi recurrence:  delta_i(s) = e_i(s) + max_s' [ delta_{i-1}(s') + t_i(s', s) ],
    with back-pointers; computed in log space.

The expensive part (route distances) does not depend on beta / w, so it is computed once
per trip (`build_lattice`) and `decode` can be rerun cheaply while tuning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pathsnap.hmm.graph import RoadGraph

MIN_PROB = 1e-6


@dataclass
class HMMParams:
    beta: float = 20.0             # metres of distance mismatch per unit of log-probability
    emission_weight: float = 1.0   # w: trust in the classifier relative to transitions
    max_mismatch_m: float = 300.0  # M: cap on mismatch and the penalty for "no path"


@dataclass
class Step:
    pt: int                        # row in the cleaned track
    arcs: np.ndarray               # candidate arc ids (states)
    fracs: np.ndarray              # projection position on each arc
    logp: np.ndarray               # log normalised classifier probability per state
    mismatch: np.ndarray | None    # (prev states x states) metres, None for the first step


@dataclass
class Lattice:
    segments: list[list[Step]] = field(default_factory=list)
    n_no_path: int = 0             # state pairs with no route within reach
    n_steps_no_path: int = 0       # steps where NO state pair had a route


def build_lattice(cand: pd.DataFrame, clean: pd.DataFrame, graph: RoadGraph,
                  max_mismatch_m: float = HMMParams.max_mismatch_m) -> Lattice:
    """cand: one trip's candidate rows with columns pt, arc_id, proj_frac, p (classifier)."""
    good = ~(clean["is_invalid"] | clean["is_outlier"]).to_numpy()
    x, y, seg = clean["x"].to_numpy(), clean["y"].to_numpy(), clean["segment_id"].to_numpy()
    lat = Lattice()
    groups = {pt: g for pt, g in cand.groupby("pt", sort=True)}
    chain = [pt for pt in sorted(groups) if good[pt]]
    prev, current = None, []
    for pt in chain:
        g = groups[pt]
        p = np.clip(g["p"].to_numpy(dtype=float), MIN_PROB, None)
        step = Step(pt=pt, arcs=g["arc_id"].to_numpy(), fracs=g["proj_frac"].to_numpy(),
                    logp=np.log(p / p.sum()), mismatch=None)
        if prev is not None and seg[pt] == seg[prev.pt]:
            gps = math.hypot(x[pt] - x[prev.pt], y[pt] - y[prev.pt])
            cutoff = gps + max_mismatch_m
            mm = np.empty((len(prev.arcs), len(step.arcs)))
            for i, (a, ta) in enumerate(zip(prev.arcs, prev.fracs)):
                for j, (b, tb) in enumerate(zip(step.arcs, step.fracs)):
                    rd = graph.route_distance(int(a), float(ta), int(b), float(tb), cutoff)
                    mm[i, j] = min(abs(rd - gps), max_mismatch_m) if math.isfinite(rd) else max_mismatch_m
            no_path = mm >= max_mismatch_m
            lat.n_no_path += int(no_path.sum())
            lat.n_steps_no_path += int(no_path.all())
            step.mismatch = mm
            current.append(step)
        else:
            if current:
                lat.segments.append(current)
            current = [step]
        prev = step
    if current:
        lat.segments.append(current)
    return lat


def decode(lat: Lattice, params: HMMParams) -> dict[int, int]:
    """Viterbi over every segment; returns {pt: decoded arc_id}."""
    out = {}
    w, beta, cap = params.emission_weight, params.beta, params.max_mismatch_m
    for steps in lat.segments:
        delta = w * steps[0].logp
        back = []
        for st in steps[1:]:
            trans = -np.minimum(st.mismatch, cap) / beta           # (prev, cur)
            scores = delta[:, None] + trans
            arg = scores.argmax(axis=0)
            delta = scores[arg, np.arange(scores.shape[1])] + w * st.logp
            back.append(arg)
        s = int(delta.argmax())
        out[steps[-1].pt] = int(steps[-1].arcs[s])
        for st, arg in zip(reversed(steps[:-1]), reversed(back)):
            s = int(arg[s])
            out[st.pt] = int(st.arcs[s])
    return out


def assign_all(decoded: dict[int, int], cand: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Prediction for every observation with candidates: decoded arc for chain points; for
    GPS outliers (not in the chain) the decoded arc of the nearest chain point in time in the
    same segment, else the classifier's top candidate."""
    top = cand.sort_values(["pt", "p"], ascending=[True, False]).groupby("pt").head(1)
    res = top[["pt", "arc_id"]].rename(columns={"arc_id": "cls_arc"}).reset_index(drop=True)
    seg = clean["segment_id"].to_numpy()
    chain = np.array(sorted(decoded))
    hmm = []
    for pt, cls in zip(res["pt"], res["cls_arc"]):
        if pt in decoded:
            hmm.append(decoded[pt]); continue
        same = chain[seg[chain] == seg[pt]] if len(chain) else chain
        if len(same):
            hmm.append(decoded[int(same[np.argmin(np.abs(same - pt))])])
        else:
            hmm.append(int(cls))
    res["hmm_arc"] = hmm
    return res


def reconstruct_route(arcs_in_order: list[int], graph: RoadGraph, cutoff: float = 2000.0
                      ) -> tuple[list[int], int]:
    """Connected route from per-observation arcs: drop repeats, fill each gap with the
    shortest network path. Returns (route arcs, number of breaks where no path was found)."""
    seq = [a for i, a in enumerate(arcs_in_order) if i == 0 or a != arcs_in_order[i - 1]]
    route, breaks = [], 0
    for i, a in enumerate(seq):
        if i:
            fill = graph.path_arcs(seq[i - 1], a, cutoff)
            if fill is None:
                breaks += 1
            else:
                route.extend(fill)
        route.append(a)
    return route, breaks
