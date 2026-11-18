"""Project-wide settings. Override paths with environment variables, not by editing code."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Raw data root containing partition folders (train/, student_test/, ...).
# Default assumes the extracted "Student folder" sits next to the repo.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT.parent / "Student folder"))

PARTITIONS = ("train", "student_test")

RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Trips are spread worldwide (23+ UTM zones), so each trip is projected to the
# UTM zone of its own GPS centroid -> metres, low distortion at city scale.
WGS84 = "EPSG:4326"

# --- preprocessing parameters (fit/tuned on train + student_test only) ---
# ~306 km/h. Train has autobahn trips at a real 40-72 m/s (median 41 m/s on trip 1),
# so 50 m/s flagged 31% of those trips. Only 34 of 208k train steps exceed 90 m/s.
MAX_SPEED_MS = 85.0      # faster implied jump = GPS outlier
MAX_OUTLIER_RUN = 5      # after this many rejected points in a row, re-anchor
GAP_BREAK_S = 30.0       # time gap that starts a new track segment
LABEL_MAX_DIST_M = 50.0  # label farther than this from its route arc = unreliable
TRIP_SUSPECT_MEDIAN_M = 30.0  # trip median label distance above this = suspect reference route

# --- candidate generation (chosen by miss-rate sweep on train only) ---
# Sweep (results/eda/candidate_sweep.csv, train): R=50,k=8 miss 0.84% / 5.7 cands;
# R=50,k=12 miss 0.56% / 7.1 cands; larger R adds nothing (labels capped at 50 m).
CAND_RADIUS_M = 50.0           # search radius around each GPS point
CAND_K = 12                    # keep at most k nearest arcs per point
CAND_FALLBACK_RADIUS_M = 200.0 # if nothing within radius, search this far
HEADING_MIN_MOVE_M = 2.0       # below this displacement GPS heading is unknown
HEADING_WIDE_WINDOW = 3        # wide heading: from the 3rd good point before to the 3rd after

# Direction penalty (metres) for an arc pointing against the GPS travel direction, scaled by
# (1 - cos(angle)) / 2: 0 when aligned, full when opposite. Used (a) in label alignment, so a
# street the route drives both ways gets the copy matching the travel direction, and (b) to pick
# each point's "best" arc for the neighbour features. 20 m exceeds typical GPS error (5-15 m),
# so a wrong-direction copy at the same position loses, but a correct-direction arc up to
# ~20 m farther still wins. Fixed by reasoning, not tuned on any test data.
HEADING_PENALTY_M = 20.0
CONTEXT_WINDOW = 5             # neighbouring observations on each side for the support features

SEED = 42
