"""Build one zip with the code and the raw data, for manual upload to Google Colab.

Usage:
    python -m pathsnap.colab_bundle                      # -> ../colab/pathsnap_colab.zip
    python -m pathsnap.colab_bundle --out some/path.zip

Contents (top-level folders inside the zip):
    pathsnap/          the repo code: pathsnap/ package, requirements.txt, README.md,
                       results/models (untuned results, needed for tuned-vs-untuned),
                       results/hpo (current Optuna study, so a search can be resumed)
    Student folder/    raw train/ and student_test/ data
Not included: .venv, data/processed (rebuilt on Colab), model weights, maps, notebooks.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from pathsnap.config import DATA_ROOT, PARTITIONS, REPO_ROOT

CODE_ITEMS = ["pathsnap", "requirements.txt", "README.md", "results/models", "results/hpo"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def _files(base: Path, item: str):
    p = base / item
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not SKIP_PARTS & set(f.parts):
                yield f


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_code = n_data = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in CODE_ITEMS:
            for f in _files(REPO_ROOT, item):
                z.write(f, Path("pathsnap") / f.relative_to(REPO_ROOT))
                n_code += 1
        for part in PARTITIONS:
            for f in _files(DATA_ROOT, part):
                z.write(f, Path("Student folder") / f.relative_to(DATA_ROOT))
                n_data += 1
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB): {n_code} code/result files, {n_data} data files")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT.parent / "colab" / "pathsnap_colab.zip")
    build(ap.parse_args().out)
