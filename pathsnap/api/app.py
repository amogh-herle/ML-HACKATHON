"""PathSnap FastAPI backend (Laptop 1).

Runs the full prediction pipeline for a selected evaluation trip and scores it against the
reference route that stays on this server. The reference route is never sent to the model
or to the UI: responses contain the GPS observations, the predicted arcs/route geometry and
summary accuracy numbers only.

Endpoints
    GET  /health               model + configuration status + uptime
    GET  /trips                trip ids of the evaluation partition
    GET  /trips/{trip_id}/gps  raw GPS observations of a trip (for drawing before prediction)
    POST /predict              {"trip_id": "..."} -> predicted route + accuracy

Configuration (environment variables)
    PATHSNAP_DATA        folder holding the partition folders (default: ../Student folder)
    PATHSNAP_PARTITION   partition served to the UI (default: evaluator_test if present,
                         otherwise student_test)

Run:  python -m pathsnap.api --partition evaluator_test   (host 0.0.0.0, port 8000; see --help)
      --partition overrides PATHSNAP_PARTITION.
"""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap.config import ARTIFACTS_DIR, DATA_ROOT, REPO_ROOT
from pathsnap.data_processing.loader import list_trips, load_track, trip_dir
from pathsnap.pipeline import Matcher, score_trip

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log")],
)
log = logging.getLogger("pathsnap.api")

TRIP_ID = re.compile(r"^\d{1,12}$")


def _partition() -> str:
    env = os.environ.get("PATHSNAP_PARTITION")
    if env:
        return env
    return "evaluator_test" if (DATA_ROOT / "evaluator_test").is_dir() else "student_test"


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    STATE["started"] = time.time()
    STATE["matcher"] = Matcher()
    STATE["partition"] = _partition()
    STATE["trips"] = list_trips(STATE["partition"])
    log.info("model loaded: %s | HMM beta=%.1f w=%.1f | partition '%s' with %d trips | data %s | %.1f s",
             ARTIFACTS_DIR / "final_model.joblib", STATE["matcher"].params.beta,
             STATE["matcher"].params.emission_weight, STATE["partition"], len(STATE["trips"]),
             DATA_ROOT, time.perf_counter() - t0)
    log.info(_banner(STATE["partition"], STATE["trips"]))
    yield


def _banner(partition: str, trips: list[str]) -> str:
    """Start-up summary for the evaluator. ASCII only: the Windows console may not be UTF-8."""
    url = os.environ.get("PATHSNAP_PUBLIC_URL", "http://<this laptop's IP>:8000")
    bar = "=" * 66
    lines = ["", bar, "  PATHSNAP API READY",
             f"  Partition : {partition}   ({len(trips)} trips)"]
    lines += [f"  Trips     : {' '.join(trips[i:i + 5])}" if i == 0 else f"              {' '.join(trips[i:i + 5])}"
              for i in range(0, len(trips), 5)]
    lines.append(f"  UI address: {url}")
    if partition != "evaluator_test":
        lines.append(f"  PRACTICE MODE: serving '{partition}', NOT the evaluator trips.")
        lines.append("  For the demo restart with:  python -m pathsnap.api --partition evaluator_test")
    lines.append(bar)
    return "\n".join(lines)


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
# the UI runs on another laptop, so cross-origin requests must be allowed
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000000"])


def _check_trip(trip_id: str) -> str:
    if not TRIP_ID.match(trip_id) or trip_id not in STATE["trips"]:
        raise HTTPException(404, f"unknown trip '{trip_id}' in partition '{STATE['partition']}'")
    return trip_id


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %s -> %d in %.0f ms", request.client.host if request.client else "-",
             request.method, request.url.path, response.status_code, 1000 * (time.perf_counter() - t0))
    return response


@app.get("/health")
def health():
    m = STATE["matcher"]
    return {"status": "ok", "partition": STATE["partition"], "n_trips": len(STATE["trips"]),
            "uptime_s": round(time.time() - STATE["started"]),
            "model": "final_model.joblib (tuned XGBoost)",
            "hmm": {"beta": m.params.beta, "emission_weight": m.params.emission_weight,
                    "max_mismatch_m": m.params.max_mismatch_m}}


@app.get("/trips")
def trips():
    return {"partition": STATE["partition"], "trips": STATE["trips"]}


@app.get("/trips/{trip_id}/gps")
def gps(trip_id: str):
    tid = _check_trip(trip_id)
    tr = load_track(trip_dir(tid, STATE["partition"]), tid)
    return {"trip_id": tid, "n_points": len(tr),
            "points": [[round(lo, 6), round(la, 6)] for lo, la in zip(tr["lon"], tr["lat"])]}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = _check_trip(req.trip_id)
    part = STATE["partition"]
    client = request.client.host if request.client else "-"
    log.info("predict request from %s: trip %s (%s)", client, tid, part)

    t0 = time.perf_counter()
    result = STATE["matcher"].match(tid, part)            # no reference route used here
    t_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics = score_trip(tid, part, result)                # server-side, reference stays here
    t_score = time.perf_counter() - t0

    pts = result.points
    log.info("trip %s: %d GPS points, %d route arcs, %d breaks | top-1 HMM %.4f (classifier %.4f) | "
             "route F1 %.4f | predict %.2f s, score %.2f s", tid, len(pts), len(result.route_arcs),
             result.route_breaks, metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier"],
             metrics["route_f1"], t_pred, t_score)

    route_geom = {"type": "MultiLineString",
                  "coordinates": [result.arc_coords[a] for a in result.route_arcs]}
    return {
        "trip_id": tid,
        "partition": part,
        "n_points": int(len(pts)),
        "points": {                                       # predictions per GPS observation
            "obs_idx": pts["obs_idx"].astype(int).tolist(),
            "lon": pts["lon"].round(6).tolist(),
            "lat": pts["lat"].round(6).tolist(),
            "is_outlier": pts["is_outlier"].astype(bool).tolist(),
            "predicted_arc": pts["hmm_arc"].astype(int).tolist(),
        },
        "predicted_route": {"n_arcs": len(result.route_arcs), "breaks": int(result.route_breaks),
                            "geometry": route_geom},
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "timings_s": {**{k: round(v, 3) for k, v in result.timings_s.items()},
                      "scoring": round(t_score, 3), "total": round(t_pred + t_score, 3)},
    }
