# PathSnap — Map-Matching & Route Reconstruction

Matches noisy GPS trajectories to a road network: per-observation candidate
classification followed by HMM/Viterbi route decoding, served via FastAPI
(Laptop 1) to an evaluator UI (Laptop 2).

## Repository layout

```
pathsnap/
  config.py            paths, CRS, seed (override data path with PATHSNAP_DATA)
  data_processing/     raw-file loading, inventory / data-quality checks, preprocessing
  features/            candidate generation + feature engineering
  models/              training / evaluation scripts for every model tried
  hpo/                 Bayesian hyperparameter optimisation (Optuna)
  hmm/                 HMM + Viterbi decoder
  api/                 FastAPI backend (model loading, /predict)
  viz/                 Folium maps and plots
ui/                    evaluator-facing UI
notebooks/             EDA and exploration notebooks
results/               CSVs, logs, plots, maps (eda/ models/ hpo/ hmm/ maps/)
artifacts/             exported model, scaler, HMM config
report/                technical report sources + final PDF
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw data is **not** committed. Place the extracted `Student folder/`
(containing `train/` and `student_test/`) next to this repo, or point to it:

```bash
export PATHSNAP_DATA="/path/to/Student folder"
```

## Data

| File | Columns (no header, whitespace-separated) |
|---|---|
| `.track` | lon, lat, t (seconds) |
| `.nodes` | lon, lat — row index = node_id |
| `.arcs`  | from_node, to_node — row index = arc_id |
| `.route` | arc_id sequence (reference route) |

Road networks are **per trip**; ids are only valid within their own trip.
Trips are spread worldwide, so each trip is projected to the UTM zone of its
own GPS centroid (`Trip.crs`) for metric distances.

## Reproducing everything (teammates start here)

```bash
git clone https://github.com/amogh-herle/ML-HACKATHON.git pathsnap
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# raw data: put the extracted "Student folder" next to the repo, or point to it
export PATHSNAP_DATA="/path/to/Student folder"

python -m pathsnap.reproduce --check        # what is present / missing
python -m pathsnap.reproduce                # full pipeline + models + notebooks (~40 min)
python -m pathsnap.reproduce --skip-models  # data, maps and notebooks 01-02 only (~2 min)
```

Notebooks (`notebooks/01_eda`, `02_preprocessing`, `03_model_comparison`) are committed
with their outputs. To rerun them yourself, use the repo's `.venv` as the kernel:

```bash
python -m ipykernel install --user --name pathsnap --display-name "PathSnap (.venv)"
```

then pick **PathSnap (.venv)** as the kernel (VS Code: *Select Kernel* > *Jupyter Kernel*).
Each notebook's first cell finds the repo, checks the raw data, and builds
`data/processed/` if it is missing; notebook 03 also trains any missing model
(`TRAIN_IF_MISSING`). Large files (`data/processed/*.parquet`,
`artifacts/models/*.joblib`) are not in git and are regenerated this way.
Preprocessing is deterministic; models use `SEED = 42`.

## Running individual steps

```bash
# dataset inventory + data-quality checks -> results/eda/inventory_*.csv
python -m pathsnap.data_processing.inventory

# cleaning + per-observation labels -> data/processed/*_points.parquet,
#                                       results/eda/preprocess_summary_*.csv
python -m pathsnap.data_processing.preprocess
```

## Preprocessing

1. **Projection** - per-trip UTM zone (`loader.utm_crs_for`).
2. **Cleaning** (`clean.py`, route-free, reused by the API): flags invalid
   coordinates / non-increasing timestamps, GPS jumps (implied speed > 85 m/s
   from last good point), and splits segments at gaps > 30 s. Nothing is
   dropped; every observation keeps its `obs_idx`.
3. **Labels** (`labels.py`, training/evaluation only): each GPS point is
   aligned to one arc of the reference route with a monotonic DP so the route
   order is respected even when streets are revisited. Points > 50 m from their
   label arc get `label_reliable = False`; trips whose median distance > 30 m
   are flagged `suspect_reference`.

Parameters live in `pathsnap/config.py`.

## Maps

```bash
python -m pathsnap.viz.maps                                  # default showcase trips
python -m pathsnap.viz.maps train 00000001 student_test 00000000
```

Writes `results/maps/<partition>_<trip_id>.html`: GPS observations (with
tooltips), flagged outliers, nearby road network, reference route, start/end
markers, legend and layer control. `viz.maps.trip_map(trip, predicted_arcs=...)`
adds a predicted-route layer.

## Candidates and features

```bash
python -m pathsnap.features.build --sweep   # miss rate vs radius/k -> results/eda/candidate_sweep.csv
python -m pathsnap.features.build           # -> data/processed/<partition>_candidates.parquet
```

`features/candidates.py::build_candidates` (route-free, reused by the API)
takes the k = 12 nearest arcs within 50 m of each valid GPS point (fallback
search to 200 m if none). Settings were chosen from the train sweep: candidate
miss rate 0.56 % on train, 0.07 % on student_test, ~7 candidates per point.

One row per (observation, candidate arc); feature list in `FEATURE_COLUMNS`:
distance, distance rank / gap to best, candidate-set size, heading difference
(separates the two directions of a two-way street), arc length, projection
position, speed, time gap, connectivity to the previous / next observation's
nearest arc, and node degrees. Training rows: `label_reliable`; target: `label`.

```python
from pathsnap.data_processing.loader import load_trip, list_trips
trip = load_trip(list_trips("train")[0], "train")
trip.track.head(); trip.nodes.head(); trip.arcs.head(); trip.route[:10]
```

_Training, HPO, HMM, API and UI commands: TODO as components are added._

## Models

```bash
python -m pathsnap.models.train --list        # every registered model (* = recommended five)
python -m pathsnap.models.train               # train + evaluate the recommended five
python -m pathsnap.models.train xgb hgb       # any others
python -m pathsnap.viz.model_plots            # comparison plots -> results/models/plots/
```

All models share one harness (`pathsnap/models/evaluate.py`): reliable rows of
the candidate tables, `FEATURE_COLUMNS` -> `label`, 5-fold `GroupKFold` by
`trip_id` on the 80 train trips, then a refit on all train and evaluation on the
10 student-test trips. Metrics: macro F1, precision, recall, log loss (rows,
threshold 0.5), observation-level Top-1 accuracy, route continuity, training and
inference time. Results: `results/models/comparison.csv` and
`results/models/<model>/`. Add a model = one entry in `models/registry.py`.

## Bayesian hyperparameter optimisation (XGBoost)

```bash
python -m pathsnap.hpo.xgb_optuna --trials 50   # Optuna TPE search (resumable), ~40 min
python -m pathsnap.hpo.xgb_optuna --refit       # refit best, tuned vs untuned, final model
python -m pathsnap.viz.hpo_plots                # history, importance, slices, comparison
```

Objective: mean 5-fold `GroupKFold` Top-1 accuracy on the train trips (same folds as
the model comparison). TPE sampler (seed 42) with a median pruner; trial 0 is the
untuned configuration. The full search record is `results/hpo/xgb_study.db`
(SQLite) and `results/hpo/xgb_trials.csv`. The tuned model becomes
`artifacts/final_model.joblib` (+ `final_model.json`). See `notebooks/04_hpo.ipynb`.

## HMM / Viterbi route reconstruction

```bash
python -m pathsnap.hmm.run     # tune beta / emission weight on train (out-of-fold), evaluate on student test
```

States = candidate arcs per GPS observation; emission = normalised classifier
probability; transition = `-|route distance - GPS distance| / beta` with route
distances from a bounded Dijkstra on the trip's network (Newson & Krumm);
no-path pairs get a capped penalty; chains restart at gaps > 30 s; Viterbi in
log space. Config in `artifacts/hmm_config.json`, results in `results/hmm/`,
analysis in `notebooks/05_hmm.ipynb`. `pathsnap/pipeline.py` (`Matcher`) runs
the whole inference for one trip (what the API calls); `score_trip` scores it
against the reference server-side.

## Google Colab (GPU)

No GitHub access is needed on Colab; everything is uploaded by hand.

```bash
python -m pathsnap.colab_bundle     # -> ../colab/pathsnap_colab.zip (code + raw data, ~30 MB)
```

In Colab: *File › Upload notebook* → `notebooks/colab_gpu.ipynb`, *Runtime › Change
runtime type › T4 GPU*, upload `pathsnap_colab.zip` in the Files panel (or let the
notebook's upload dialog ask for it), then run all cells. The notebook installs
the packages (full `xgboost` wheel with CUDA), preprocesses, runs a GPU-vs-CPU
speed check, the XGBoost Optuna search on the GPU (`PATHSNAP_XGB_DEVICE=cuda`,
own study `xgb_top1_gpu`), refits, and downloads an outputs zip to unzip inside
the repo.

Environment settings: `PATHSNAP_DATA` (data path), `PATHSNAP_JOBS` (threads),
`PATHSNAP_XGB_DEVICE` (`cpu`/`cuda`), `PATHSNAP_STUDY` (Optuna study name).

## Leakage rules

- Evaluator-test trips are never used for training, tuning or selection.
- Reference routes stay server-side; inference code loads trips with
  `load_trip(..., with_route=False)`.
- No hardcoded predictions.
