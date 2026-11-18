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

# Data is in Germany (~8.6E, 49.9N), inside UTM zone 32N -> metres, low distortion at city scale.
WGS84 = "EPSG:4326"
PROJECTED_CRS = "EPSG:32632"

SEED = 42
