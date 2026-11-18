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

SEED = 42
