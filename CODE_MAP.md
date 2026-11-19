# Code map: "if someone asks to change X, open Y"

Line numbers are for the submitted version. If you edit a file they will drift, so search for the function or class name instead.

## AIML-02: Medical imaging

### The portal (React): `AIML-02/frontend/src/`

| What you see on screen | File and component | Line |
|---|---|---|
| Whole page layout, top bar, orange warning banner, upload box, big "AI prediction" card | `App.jsx` → `export default function App` | 122 |
| Left list of uploaded studies | `App.jsx` → `Worklist` | 11 |
| "Needs review" / "Reviewed" pill | `App.jsx` → `Badge` | 6 |
| Scan image, heatmap overlay, show/hide checkbox, opacity slider | `App.jsx` → `Viewer` | 31 |
| Agree / Disagree / Uncertain, notes, reviewer name, Sign off | `App.jsx` → `ReviewPanel` | 58 |
| Percent formatting (e.g. 90.1%) | `App.jsx` → `pct` | 4 |
| Calls to the backend (list, upload, review, health) | `api.js` | whole file |
| Static demo mode (GitHub Pages): sample buttons, in-memory review | `api.js` (`isDemo`, `demoLoad`), `App.jsx` → `SamplePicker` | 1, 31 |
| Precomputed demo results | `scripts/make_demo_assets.py` writes `frontend/public/demo/` | whole file |
| Publishing the demo | `.github/workflows/pages.yml` | whole file |
| Colours and fonts | `styles.css` → `:root` (the `--brand`, `--warn`, `--bg` variables) | 1 |
| Spacing and layout of the two columns | `styles.css` → `main`, `.split`, `.summary` | 8, 23, 20 |
| Mobile layout | `styles.css` → `@media (max-width:820px)` | 9 |
| Dev server and `/api` proxy to port 8000 | `vite.config.js` | whole file |

Quick edits: change the app name in `App.jsx` (`<h1>`) and `index.html` (`<title>`); change the accent colour by editing `--brand` in `styles.css`.

### The backend API: `AIML-02/backend/app.py`

| Question | Where |
|---|---|
| What happens when a scan is uploaded (decode, predict, heatmap, save, Orthanc push) | `upload` (line 70) |
| How DICOM vs PNG/JPG is detected and read | `decode_image` (49) |
| Loading the model at startup, which weights file is used | `lifespan` (33) |
| Doctor review rules (what "disagree" does to the label) | `Review` (120) and `review` (128) |
| List studies / get one / get image or heatmap | `list_studies` (99), `get_study` (104), `study_image` (112) |
| Health check (model loaded, Orthanc up) | `health` (63) |
| Upload size limit | `MAX_BYTES` near the top |
| Database table and columns | `backend/db.py` → `SCHEMA`; reads/writes in class `DB` (22) |
| Orthanc (DICOM server) upload | `backend/storage.py` → `Orthanc.upload` (24), `available` (18) |

### The AI: `AIML-02/imaging/`

| Question | Where |
|---|---|
| The neural network | `model.py` → `build_model` (11), a MONAI ResNet |
| Pixel preprocessing (same code in training and serving) | `model.py` → `normalize` (21), `preprocess` (33) |
| Training loop, augmentations, optimiser, saving best model | `train.py` → `main` (43), `make_loader` (24) |
| Test metrics (AUC, sensitivity, specificity) and the heatmap accuracy check | `evaluate.py` → `main` (17) |
| Grad-CAM heatmap computation | `gradcam.py` → `heatmap` (21) |
| Heatmap colours and overlay on the image | `gradcam.py` → `_jet` (41), `overlay` (48) |
| "Does the heatmap point at the nodule?" measure | `gradcam.py` → `pointing_hit` (66) |
| DICOM reading / writing | `dicom_utils.py` → `read_dicom` (42), `array_to_dicom` (19) |
| Fake training images | `synth.py` → `make_image` (46), `make_split` (64) |
| Training settings (epochs, learning rate, data size) | `params.yaml` |
| The reproducible pipeline (generate → train → evaluate) | `dvc.yaml` |
| Containers | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| Tests | `tests/test_imaging.py` |

## AIML-03: Recommendation engine

### Stage 1: find candidates

| Question | Where |
|---|---|
| The two-tower model (user tower, item tower) | `recsys/models.py` → `UserTower` (12), `ItemTower` (26), `TwoTower` (40) |
| How it is trained | `recsys/train.py` → `train_retrieval` (34) |
| FAISS index (flat, HNSW, IVF) and search | `recsys/index.py` → `build_index` (10), `search` (30) |
| How many candidates (200) and final list size (10) | `recsys/config.py` → `n_candidates`, `top_k` |
| Which FAISS index type is used | `recsys/config.py` → `index_kind` |

### Stage 2: rank

| Question | Where |
|---|---|
| The ranking network | `recsys/models.py` → `RankerNet` (61), `RankingModel` (91) |
| How ranker training rows are built (negatives, validation split) | `recsys/train.py` → `build_ranker_examples` (62), `train_ranker` (95) |
| Exporting the ranker for serving | `recsys/models.py` → `RankerServing` (112) |
| Scoring locally or through Triton | `recsys/pipeline.py` → `LocalRanker` (16), `TritonRanker` (31) |

### Serving, cache and data

| Question | Where |
|---|---|
| The full online flow (retrieve → rank → top K) | `recsys/pipeline.py` → `Recommender.retrieve` (84), `rank` (97), `recommend` (109) |
| Redis keys and result caching with expiry | `recsys/cache.py` → `FeatureCache` (88), `RedisStore` (45) |
| What happens when Redis is not running | `recsys/cache.py` → `make_store` (75), `DictStore` (21) |
| REST endpoints | `serving/api.py` → `recommend` (34), `health` (28), `stats` (44) |
| Spark job (split, popularity and activity features) | `recsys/spark_etl.py` → `run` (19) |
| Fake dataset | `recsys/data_gen.py` → `generate` (20) |
| MovieLens loader | `recsys/movielens.py` → `load` (25) |
| Evaluation metrics and latency numbers | `recsys/evaluate.py` → `evaluate` (22), `_metrics` (15) |
| Nightly pipeline definition | `airflow/dags/recsys_pipeline.py` |
| Triton model repo and config | `serving/triton/export.py` → `export` (31), `config_pbtxt` (13) |
| Command-line steps (`generate`, `etl`, `train`, `evaluate`...) | `recsys/cli.py` → `main` (11) |
| Containers | `Dockerfile`, `docker-compose.yml` |
| Tests | `tests/test_recsys.py` |

## Likely questions, short answers

- **Why two stages?** Scoring all items with a big model is slow. Stage 1 is cheap (a vector search) and keeps 200 of 2,000 items; stage 2 spends more compute on only those 200.
- **Why does the ranker not beat retrieval?** The synthetic data comes from a latent model that stage 1 already learns. Explained in `AIML-03/README.md`.
- **Why a hand-written Grad-CAM?** MONAI's version returned a flat, useless map on this model. Explained in `AIML-02/imaging/gradcam.py` (top docstring) and `AIML-02/README.md`.
- **Is it clinically valid?** No. Synthetic images, research prototype, stated on the portal banner and in the README.
