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

## Running

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

## Leakage rules

- Evaluator-test trips are never used for training, tuning or selection.
- Reference routes stay server-side; inference code loads trips with
  `load_trip(..., with_route=False)`.
- No hardcoded predictions.
