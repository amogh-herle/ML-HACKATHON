"""One-command reproduction of every result in the repo, plus the checks the
notebooks run first so a teammate gets a clear message instead of a traceback.

Usage:
    python -m pathsnap.reproduce              # whole pipeline + execute notebooks
    python -m pathsnap.reproduce --check      # only check data / outputs exist
    python -m pathsnap.reproduce --skip-models  # everything except model training

Order: inventory -> preprocess -> candidate sweep -> candidates -> maps ->
models (recommended five) -> comparison plots -> notebooks 01-03.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from pathsnap.config import ARTIFACTS_DIR, DATA_ROOT, PARTITIONS, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

# 04 reads the committed Optuna study (results/hpo/xgb_study.db); rerunning the search itself is
# a separate ~40 min step: python -m pathsnap.hpo.xgb_optuna --trials 50 && ... --refit
NOTEBOOKS = ["01_eda.ipynb", "02_preprocessing.ipynb", "03_model_comparison.ipynb", "04_hpo.ipynb",
             "05_hmm.ipynb"]


class SetupError(RuntimeError):
    pass


def check_data() -> None:
    """Raw data present? Raises SetupError with instructions if not."""
    missing = [p for p in PARTITIONS if not (DATA_ROOT / p).is_dir()]
    if missing:
        raise SetupError(
            f"Raw data not found: {DATA_ROOT} is missing {missing}.\n"
            "Put the extracted 'Student folder' (with train/ and student_test/) next to the repo, "
            "or set the PATHSNAP_DATA environment variable to its path, e.g.\n"
            '  export PATHSNAP_DATA="/path/to/Student folder"   (then restart the kernel)')
    counts = {p: sum(1 for d in (DATA_ROOT / p).iterdir() if d.is_dir()) for p in PARTITIONS}
    print(f"raw data OK: {DATA_ROOT}  {counts}")


def processed_ready() -> bool:
    need = [PROCESSED_DIR / f"{p}_{kind}.parquet" for p in PARTITIONS for kind in ("points", "candidates")]
    need += [RESULTS_DIR / "eda" / f"inventory_{p}.csv" for p in PARTITIONS]
    need += [RESULTS_DIR / "eda" / "candidate_sweep.csv"]
    return all(f.exists() for f in need)


def ensure_processed() -> None:
    """Build processed tables if they are missing (about 1 minute)."""
    check_data()
    if processed_ready():
        print("processed data OK:", PROCESSED_DIR)
        return
    print("processed data missing -> building it now (about 1 minute)")
    run_preprocessing()


def models_ready(names: list[str]) -> list[str]:
    """Names of models whose results or weights are missing."""
    return [n for n in names
            if not (RESULTS_DIR / "models" / n / "metrics.json").exists()
            or not (ARTIFACTS_DIR / "models" / f"{n}.joblib").exists()]


def _step(title: str, module: str, *args: str) -> None:
    t0 = time.perf_counter()
    print(f"\n>>> {title}: python -m {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=REPO_ROOT, check=True)
    print(f"<<< {title} done in {time.perf_counter() - t0:.0f} s", flush=True)


def run_preprocessing() -> None:
    _step("inventory", "pathsnap.data_processing.inventory")
    _step("cleaning + labels", "pathsnap.data_processing.preprocess")
    _step("candidate sweep", "pathsnap.features.build", "--sweep")
    _step("candidates + features", "pathsnap.features.build")
    _step("maps", "pathsnap.viz.maps")


def run_notebooks() -> None:
    for nb in NOTEBOOKS:
        path = REPO_ROOT / "notebooks" / nb
        if not path.exists():
            print(f"skip {nb} (not found)")
            continue
        t0 = time.perf_counter()
        print(f"\n>>> executing notebooks/{nb}", flush=True)
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=1800",
                        "--ExecutePreprocessor.kernel_name=python3", str(path)],
                       cwd=path.parent, check=True)
        print(f"<<< {nb} done in {time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what is present")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-notebooks", action="store_true")
    a = ap.parse_args()

    from pathsnap.models.registry import RECOMMENDED

    if a.check:
        check_data()
        print("processed data:", "OK" if processed_ready() else "MISSING")
        miss = models_ready(RECOMMENDED)
        print("models:", "OK" if not miss else f"MISSING {miss}")
        return

    check_data()
    run_preprocessing()
    if not a.skip_models:
        _step("models (recommended five, ~30-40 min)", "pathsnap.models.train", *RECOMMENDED)
        _step("comparison plots", "pathsnap.viz.model_plots")
    if not a.skip_notebooks:
        run_notebooks()
    print("\nAll done.")


if __name__ == "__main__":
    main()
