# PathSnap — Map-Matching & Route Reconstruction

Matches noisy GPS trajectories to a road network: every GPS observation gets candidate
road arcs, a classifier scores them, and an HMM/Viterbi decoder turns the scores into a
connected route. A FastAPI backend (Laptop 1) serves the pipeline to an evaluator UI (Laptop 2).

## Team
Team member 1: Amogh Herle (SRN : PES1UGCS653)  
Team member 2: Anjana Vijaysankar (SRN :PES1UG24CS654)  
Team member 3: Sinchana Urs   (SRN : PES1UG24CS626)    
Team member 4: Siri S S    (SRN : PES1UG24CS627)  


## Results (student test, 10 trips, 19,133 reliably labelled GPS observations)

| | Top-1 accuracy | Route F1 | Route continuity |
|---|---|---|---|
| Tuned XGBoost (classifier only) | 0.9086 | 0.857 | 0.959 |
| **Tuned XGBoost + HMM (final system)** | **0.9152** | **0.933** | **0.976** |

Final classifier: XGBoost, 24 features, tuned with 50 Optuna trials (CV top-1 0.9019 -> 0.9040).
All model / feature / decoder choices were made on train cross-validation; student-test data
was only used to report results; evaluator-test data is never used. Full model comparison:
`results/models/comparison.csv` and `notebooks/03_model_comparison.ipynb`.

## Repository layout

```
pathsnap/
  config.py            every parameter (paths, CRS, cleaning, labels, candidates, seed)
  data_processing/     loading, per-trip UTM projection, inventory, cleaning, labels
  features/            candidate generation (STRtree) + 24 features, ablation inputs
  models/              shared CV harness, model registry, training, ablation, error analysis, rankers
  hpo/                 Bayesian hyperparameter optimisation (Optuna TPE)
  hmm/                 road graph (bounded Dijkstra), HMM + Viterbi, tuning/evaluation
  pipeline.py          end-to-end inference for one trip (used by the API) + server-side scoring
  api/                 FastAPI backend
  viz/                 Folium maps, model and HPO plots
  reproduce.py         one-command reproduction + checks used by the notebooks
  colab_bundle.py      zip of code + data for manual upload to Google Colab
notebooks/             01_eda, 02_preprocessing, 03_model_comparison, 04_hpo, 05_hmm (executed)
  colab/               Colab GPU notebooks
results/
  eda/                 inventory, preprocessing, candidate sweep and summaries (CSV)
  models/              per-model metrics, CV folds, confusion matrices, comparison, ablation, plots
  hpo/                 Optuna study (SQLite), trials, importance, tuned vs untuned, plots
    archive_v1_old_features/   first study, before the label/feature changes (history only)
  hmm/                 decoder tuning grid, per-trip results, summary
  maps/                Folium HTML maps
  logs/                training / ablation / HPO / HMM run logs
artifacts/             final_model.joblib + .json, hmm_config.json (per-model weights not in git)
ui/                    evaluator UI (Laptop 2): index.html + bundled Leaflet in vendor/
deploy/                one launcher per laptop: laptop1_api.bat (Windows), laptop2_ui.sh (macOS / Linux)
report/                technical report
```

## Setup

Linux / macOS:

```bash
git clone https://github.com/amogh-herle/ML-HACKATHON.git pathsnap
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"   # folder with train/ student_test/ (evaluator_test/)
```

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp` (only for running models
or the API; Laptop 2 does not need it). `requirements.txt` installs `xgboost` on macOS and
`xgboost-cpu` elsewhere, same version.

Windows (PowerShell):

```powershell
git clone https://github.com/amogh-herle/ML-HACKATHON.git pathsnap
cd pathsnap
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PATHSNAP_DATA = "C:\path\to\Student folder"
```

If PowerShell refuses to run the activation script, run this once in that window:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Windows (Command Prompt):

```bat
git clone https://github.com/amogh-herle/ML-HACKATHON.git pathsnap
cd pathsnap
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
set PATHSNAP_DATA=C:\path\to\Student folder
```

Every `python -m pathsnap...` command below is the same on Windows once the venv is active.
Environment variables set this way last only for that terminal window; set them again in a new
one (or skip `PATHSNAP_DATA` entirely by putting `Student folder` next to the repo folder).
The table maps the Linux forms used below to Windows:

| Linux / macOS | PowerShell | Command Prompt |
|---|---|---|
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` |
| `python3 ...` | `py ...` (or `python ...`) | `py ...` (or `python ...`) |
| `export NAME="value"` | `$env:NAME = "value"` | `set NAME=value` |
| `NAME=value python -m ...` (one command) | `$env:NAME = "value"; python -m ...` | `set NAME=value && python -m ...` |
| `cmd1 && cmd2` | `cmd1; cmd2` (PowerShell 5) or `cmd1 && cmd2` (PowerShell 7) | `cmd1 && cmd2` |

Raw data and large intermediate files (`data/processed/*.parquet`, `artifacts/models/`) are not
in git; they are regenerated by the commands below. Preprocessing is deterministic; models use
`SEED = 42`.

## Reproducing everything

```bash
python -m pathsnap.reproduce --check        # what is present / missing
python -m pathsnap.reproduce --skip-models  # data, candidates, maps, notebooks 01-02 (~2 min)
python -m pathsnap.reproduce                # + the five recommended models + plots + notebooks
```

Step by step:

```bash
python -m pathsnap.data_processing.inventory        # inventory + data-quality checks
python -m pathsnap.data_processing.preprocess       # cleaning + per-observation labels
python -m pathsnap.features.build --sweep           # candidate radius / k sweep
python -m pathsnap.features.build                   # candidates + 24 features
python -m pathsnap.models.ablation                  # feature-set / model-type experiments (train CV)
python -m pathsnap.models.train logreg knn rf lgbm mlp xgb dtree xgb_rank lgbm_rank
python -m pathsnap.hpo.xgb_optuna --trials 50       # Bayesian search (resumable)
python -m pathsnap.hpo.xgb_optuna --refit           # final model + tuned vs untuned
python -m pathsnap.hmm.run                          # HMM tuning (train) + evaluation (student test)
python -m pathsnap.models.error_analysis            # where the remaining errors are
python -m pathsnap.viz.model_plots && python -m pathsnap.viz.hpo_plots
```

Notebooks run with the repo's `.venv` as kernel
(`python -m ipykernel install --user --name pathsnap --display-name "PathSnap (.venv)"`, the same
command on Windows with the venv active).
Each first cell finds the repo, checks the data and builds `data/processed/` if missing.

## Method

**Data.** Each trip: `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from, to),
`.route` (reference arc ids). Road networks are per trip; ids are only valid within a trip.
Trips span 25+ UTM zones, so each is projected to its own zone.

**Preprocessing** (`data_processing/`). Cleaning flags invalid points and GPS jumps
(> 85 m/s from the last good point; real autobahn speeds reach ~72 m/s) and splits segments at
gaps > 30 s; nothing is dropped. Labels: each GPS point is aligned to one arc of the reference
route by a monotonic dynamic programme whose cost is distance + a direction penalty, so loops
and out-and-back drives get the right copy of a street. Labels > 50 m away are unreliable.

**Candidates and features** (`features/`). The 12 nearest arcs within 50 m (0.57 % miss rate on
train). 24 features: distance and its ranks, heading differences (1-step and wide window),
signed movement along the arc, arc length, projection position, speed, time gap, links to the
neighbouring observations' best arcs, node degrees, ranks within the candidate set, and support
from the +-5 neighbouring observations. `results/models/ablation.csv` shows each group's effect.

**Models** (`models/`). One harness for all: 5-fold `GroupKFold` by trip on train, refit, then
student test. Metrics: Top-1 accuracy per observation, macro F1, precision, recall, accuracy,
log loss, route continuity, fit and inference time. Nine models from six families, including
two learning-to-rank models (`models/ranker.py`).

**HPO** (`hpo/`). Optuna TPE (seed 42), median pruner, objective = CV Top-1, trial 0 = untuned.

**HMM** (`hmm/`). States = candidate arcs; emission = classifier probability normalised per
observation; transition = `-|route distance - GPS distance| / beta` with route distances from a
bounded Dijkstra; no-path pairs get a capped penalty (never zero); chains restart at gaps; log-
space Viterbi; gaps between decoded arcs are filled with shortest paths. beta and the emission
weight are tuned on train out-of-fold probabilities (`artifacts/hmm_config.json`).

## Two-laptop deployment

| Laptop | Role | Uses from this repo | Needs |
|---|---|---|---|
| Laptop 1 | Model + API | `pathsnap/`, `artifacts/`, `requirements.txt` | `.venv` with the requirements, `Student folder/` (with `evaluator_test/`) |
| Laptop 2 | Evaluator UI | `ui/` only | a browser (python3 only for the launcher script) |

Laptop 2 never has the data, the model or the reference routes: it only sends a trip id to
Laptop 1 and draws what comes back. Both laptops clone this same repository.

**Laptop 1 (Windows)**, after the one-time [Setup](#setup) and firewall rule (see [API](#api-laptop-1)):

```bat
deploy\laptop1_api.bat                 :: demo: serves the 10 evaluator_test trips
deploy\laptop1_api.bat student_test    :: practice run
```

It works from Command Prompt, PowerShell or a double-click (no activation or execution-policy
change needed) and prints a banner with the served partition, its trip ids and the address to
give Laptop 2. On macOS / Linux: `python -m pathsnap.api --partition evaluator_test`.

**Laptop 2 (macOS / Linux)**:

```bash
deploy/laptop2_ui.sh 192.168.1.20:8000   # the address from Laptop 1's banner; Ctrl+C to stop
```

On Windows, double-click `ui\index.html` and type the address instead.

**Before the demo (rehearsal):** both laptops on the same network (a phone hotspot avoids
campus Wi-Fi blocking device-to-device traffic) -> Laptop 1 `deploy\laptop1_api.bat student_test`
-> Laptop 2 connects and runs all 10 trips -> Laptop 1 restarts with `deploy\laptop1_api.bat`.
The UI shows a red *PRACTICE MODE* status whenever it is not connected to `evaluator_test`.

**Backup** if Laptop 2 cannot reach Laptop 1: open `ui\index.html` on Laptop 1 and connect to
`http://localhost:8000`.

## API (Laptop 1)

```bash
python -m pathsnap.api --partition evaluator_test   # demo: 0.0.0.0:8000, prints the LAN address + trips
python -m pathsnap.api --partition student_test     # practice runs
```

Windows (Laptop 1 on Windows):

```powershell
python -m pathsnap.api --partition evaluator_test   # demo (same command, venv active)
python -m pathsnap.api --partition student_test     # practice run
ipconfig                                            # "IPv4 Address" = address for the UI

# allow incoming connections on port 8000 (run once, in an Administrator PowerShell)
New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Command Prompt equivalent of the firewall rule (as Administrator):
`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`.
If Windows shows a "Windows Defender Firewall has blocked some features" pop-up the first time
the API starts, click **Allow access**, and make sure the Wi-Fi network is set to **Private**
(Settings > Network & internet > Wi-Fi > the network > Network profile type); on a Public
network Windows blocks incoming connections.

On Linux the firewall is usually off; if `ufw` is enabled: `sudo ufw allow 8000/tcp`.
Find the address with `hostname -I` (Linux) or `ipconfig getifaddr en0` (macOS); the API also
prints it at start-up.

| Endpoint | Returns |
|---|---|
| `GET /health` | status, served partition, number of trips, HMM settings |
| `GET /trips` | trip ids of the served partition |
| `GET /trips/{trip_id}/gps` | raw GPS observations `[lon, lat]` |
| `POST /predict` `{"trip_id": "..."}` | predicted arc per observation, predicted route (GeoJSON), accuracy, timings |

`/predict` runs `pipeline.Matcher` without the reference route, then scores the result
server-side; the reference route is never in a response. Requests are logged to the terminal
and `logs/api.log`. Served partition: `--partition`, else `PATHSNAP_PARTITION`, else `evaluator_test` if that
folder exists, else `student_test`. Allow incoming TCP 8000 in Laptop 1's firewall.

## UI (Laptop 2)

`ui/index.html` is a single static page (Leaflet map, bundled in `ui/vendor/`); Laptop 2 needs
only a browser, no Python and no data. Internet is needed only for the background map tiles;
without it the GPS track and predicted route still draw. Either open the file directly, or serve it:

```bash
python3 -m http.server 5500 --directory ui      # Linux / macOS, then open http://localhost:5500
```

```powershell
py -m http.server 5500 --directory ui           # Windows, then open http://localhost:5500
```

On Windows you can also just double-click `ui\index.html`; it opens in the default browser and
works the same way.

Enter the address printed by the API on Laptop 1 (e.g. `http://192.168.1.20:8000`) and press
Connect (or open `index.html?api=192.168.1.20:8000`; the last address is remembered). The page
lists the served trips; selecting one draws its GPS track, *Run prediction* sends the trip id to
`/predict` and shows the predicted route (orange) with the GPS observations, Top-1 accuracy
(classifier + HMM and classifier only), route precision / recall / F1, number of scored points,
route arcs / breaks and the server-side timing breakdown. A request log shows every call.
The reference route is never sent to the UI.

**Demo checklist:** see [Two-laptop deployment](#two-laptop-deployment). Keep Laptop 1's
terminal visible during the demo: it shows every request and its accuracy.

## Google Colab (optional)

`python -m pathsnap.colab_bundle` (or `--fresh`) builds a zip of code + data in `../colab/`;
upload it with `notebooks/colab/colab_gpu.ipynb` (GPU HPO) or `colab_rerun.ipynb` (full rerun).
Settings: `PATHSNAP_DATA`, `PATHSNAP_JOBS`, `PATHSNAP_XGB_DEVICE`, `PATHSNAP_STUDY`,
`PATHSNAP_HPO_MODEL`.

## Leakage rules

- Evaluator-test trips are never used for training, tuning, feature or model selection.
- Reference routes stay server-side; inference loads trips with `load_trip(..., with_route=False)`.
- No hardcoded predictions.
