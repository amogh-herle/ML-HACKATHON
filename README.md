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
Coordinates are projected to UTM zone 32N (EPSG:32632) for metric distances.

## Running

```bash
# dataset inventory + data-quality checks -> results/eda/inventory_*.csv
python -m pathsnap.data_processing.inventory
```

```python
from pathsnap.data_processing.loader import load_trip, list_trips
trip = load_trip(list_trips("train")[0], "train")
trip.track.head(); trip.nodes.head(); trip.arcs.head(); trip.route[:10]
```

_Training, HPO, HMM, API and UI commands: TODO as components are added._

## Leakage rules

- Evaluator-test trips are never used for training, tuning or selection.
- Reference routes stay server-side; inference code loads trips with
  `load_trip(..., with_route=False)`.
- No hardcoded predictions.
