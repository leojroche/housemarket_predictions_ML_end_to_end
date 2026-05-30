# 🏠 House Market Price Prediction — End-to-End ML Project

> 🔗 **Live UI available here:** [Link](http://18.132.2.159:6969/)

An end-to-end machine learning (ML) system for predicting US housing market prices across major metropolitan areas. The project covers the full ML lifecycle — from raw data ingestion and temporal splitting, through leakage-safe feature engineering and Bayesian hyperparameter tuning, to a production-ready REST API and an interactive geospatial frontend — with automated CI/CD deployment to AWS.

![UI_screenshot](img/ui_screenshot.svg)

---

## ✨ Highlights

- 📐 **Three independent, composable ML pipelines** — feature, training, inference — each runnable standalone or chained via `make`
<!-- - 🔒 **Strict data leakage prevention** — all encoders are fitted exclusively on training data and serialized for inference reuse -->
- 🎛️ **Bayesian hyperparameter optimization** with Optuna tracked in MLflow
- 🌐 **Full serving stack** — FastAPI backend + Streamlit frontend with interactive US map and prediction metrics
- ☁️ **Cloud-native deployment** — Dockerized microservices on AWS ECS, data on S3
- 🔄 **Automated CI/CD** — GitHub Actions builds, pushes to AWS ECR, and deploys to ECS on every push to `main`
- 🧪 **Comprehensive test suite** — 15 pytest tests covering all three pipelines and the API

---

## 🛠️ Tech Stack

**ML & Data**
`xgboost` · `optuna` · `mlflow` · `scikit-learn` · `category_encoders` · `pandas` · `numpy`

**API & UI**
`fastapi` · `uvicorn` · `streamlit` · `plotly`

**Cloud & Infrastructure**
`boto3` · AWS S3 · AWS ECR · AWS ECS · Docker · GitHub Actions

**Dev Tooling**
`uv` · `pytest` · `argparse`

---

## 🏗️ Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FEATURE PIPELINE                          │
│  load_and_split.py  ──►  preprocess.py  ──►  feature_engineering.py  │
└─────────────────────────────┬────────────────────────────────────┘
                              │  data/feature_engineered/
┌─────────────────────────────▼────────────────────────────────────┐
│                       TRAINING PIPELINE                          │
│              tune.py  ──►  eval.py  ──►  train.py               │
└─────────────────────────────┬────────────────────────────────────┘
                              │  models/xgb_model.pkl
┌─────────────────────────────▼────────────────────────────────────┐
│                      INFERENCE PIPELINE                          │
│                          predict.py                              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
          ┌───────────────────▼────────────────────┐
          │             SERVING LAYER               │
          │  FastAPI Backend  ◄──►  Streamlit UI    │
          └───────────────────┬────────────────────┘
                              │
          ┌───────────────────▼────────────────────┐
          │        CI/CD  (GitHub Actions)          │
          │   build → ECR push → ECS deploy         │
          └─────────────────────────────────────────┘
```

---

## 🔬 Feature Pipeline

### 1. Temporal Data Splitting — `load_and_split.py`

The dataset is split **chronologically** into three non-overlapping sets, mirroring real production conditions where future data is never seen during training:

<!-- | Split | Date Range | Purpose |
|---|---|---|
| **Train** | start → Jan 2020 | Hyperparameter tuning training — training set |
| **Eval** | Jan 2020 → Jan 2022 | Hyperparameter tuning — training set |
| **Holdout** | Jan 2022 → end | Final unseen test set / production inference | -->

| Split | Date Range | Hyperparameter Tuning | Final Model
|:---:|:---:|:---:|:---:|
| **Train** | March 2012 → Jan 2020 | training set | training set |
| **Eval** | Jan 2020 → Jan 2022 | evaluation set | training set |
| **Holdout** | Jan 2022 → December 2023 |  -  | prediction set |

### 2. Preprocessing — `preprocess.py`

- **Geospatial enrichment** — cities are merged with the [SimpleMaps US Metros dataset](https://simplemaps.com/data/us-metros) to attach `lat`/`lng` (also useful for the map UI display)
- **Deduplication** — duplicate rows are removed while deliberately ignoring time-dependent columns (`date`, `year`) to avoid discarding valid temporal entries
- **Outlier removal** — properties above a $19M listing-price threshold are filtered out

### 3. Feature Engineering — `feature_engineering.py`

All encoders are fitted on the **training dataset only** and applied to evaluation/holdout set to avoid data leakage.

- `year`, `quart`, `month` : Extracted from `date` timestamp 
-  `zipcode_freq` : Frequency encoding on the zipcode
- `city_encoded` : Target encoding (→ `price`) 

---

## 🎛️ Training Pipeline

### Hyperparameter Tuning — `tune.py`

**Bayesian optimization** via Optuna minimizes RMSE on the eval set across a structured 9-parameter search space. Every trial is a nested MLflow run, capturing the full parameter set alongside MAE, RMSE, and R² — enabling reproducible experiment comparison and full audit trails.

```
Search space: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda        
```

The best model and its hyperparameters are saved as `.pkl` file for the training of the final model.

### Evaluation — `eval.py`

A standalone evaluation script — decoupled from training — reporting MAE, RMSE, and R² on the held-out eval set. Accepts either a model path or a live `XGBRegressor` instance, making it composable with the tuning loop or usable independently.

### Final Training — `train.py`

The production model is trained on the **combined train + eval dataset** using the optimal hyperparameters selected during tuning — maximizing available training data while keeping the holdout completely untouched for final evaluation.

---

## 🚀 Inference Pipeline — `predict.py`

Loads the serialized XGBoost model and runs batch predictions on any feature-engineered dataset. Designed as both a **library module** (imported by the FastAPI backend) and a **CLI tool**:

```bash
python src/inference_pipeline/predict.py data/feature_engineered/fe_holdout.csv \
    --output predictions.csv \
    --xgb_model models/xgb_model.pkl
```

---

## 🌐 Serving Layer

### Backend API — `backend.py`

A **FastAPI** REST API exposing the inference pipeline over HTTP:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Welcome message |
| `/health` | `GET` | Verifies model artifact exists on disk |
| `/predict` | `POST` | Accepts `list[dict]` of feature rows → returns `list[float]` predictions |

### Frontend UI — `frontend.py`

An interactive **Streamlit** dashboard featuring:

- **Clickable US map** (Plotly `Scattergeo`, Albers USA projection) — click any city to select it
- **City / Year / Month selectors** — dynamic filter controls synchronized with map state
- **Actual vs. predicted time-series chart** — monthly average prices with a green highlight on the selected month
- **Prediction table** — full breakdown for the selected city and year

> 🔗 **Live UI available here:** [Link](http://18.132.2.159:6969/)

---

## ☁️ Cloud & CI/CD

### Infrastructure

| Component | Service |
|---|---|
| Model artifact + holdout data | AWS S3 (`eu-west-2`) |
| Container registry | AWS ECR |
| Backend container | AWS ECS (`housemarket-cluster`) |
| Frontend container | AWS ECS (`housemarket-cluster`) |

### CI/CD Pipeline 

A **GitHub Actions** workflow triggers automatically on every push to `main`.
Both images are tagged with the git commit SHA for traceability, then promoted to `:latest` for ECS rolling updates. This means every merge to `main` is live in production within minutes, with zero manual steps.

---

## 🧪 Testing

A **pytest** suite covers all three pipelines.

| File | Tests |
|:---:|---:|
| `test_features_pipeline.py` | 9 |
| `test_training_pipeline.py` | 3 |
| `test_inference_pipeline.py` | 1 |

```bash
# Run all pipeline tests
make run_pipeline_tests

# Or individually using
PYTHONPATH=. pytest tests/test_features_pipeline.py
PYTHONPATH=. pytest tests/test_training_pipeline.py
PYTHONPATH=. pytest tests/test_inference_pipeline.py
```

---

## ⚡ Quickstart

### Prerequisites

- Python ≥ 3.11
- [`uv`](https://github.com/astral-sh/uv) — `pip install uv`
- AWS credentials configured (for S3 data access)
- Original datasets should be downloaded:
    - HouseTS Dataset ([link](https://www.kaggle.com/datasets/shengkunwang/housets-dataset)) to be saved as `/data/raw/untouched_raw_original.csv`
    - United States Metro Areas Database ([link](https://simplemaps.com/data/us-metros), click on Basic) to be saved as `/data/raw/usmetros.csv`

### Build and run locally

```bash
# 1. Install dependencies
uv sync

# 2. Run the full ML pipeline (feature engineering → tuning → training → inference)
make build

# 3. Start the backend API  →  http://localhost:8000
make sb

# 4. Start the frontend UI  →  http://localhost:8501
make sf
```

### Run individual pipeline stages

```bash
make feature_pipeline    # Load → split → preprocess → feature-engineer
make training_pipeline   # Tune hyperparameters → evaluate → train final model
make inference_pipeline  # Batch predict on holdout set
```


---

## 📁 Project Structure

```
.
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions CI/CD → AWS ECR + ECS
├── data/
│   ├── raw/                         # Original dataset + train / eval / holdout splits
│   ├── processed/                   # Preprocessed splits
│   └── feature_engineered/          # Model-ready datasets (tuning + final)
├── models/
│   ├── tuning/                      # Best tuning model + hyperparameters (.pkl)
│   ├── xgb_model.pkl                # Final production model
│   ├── freq_encoder.pkl             # Serialized frequency encoder
│   └── target_encoder.pkl           # Serialized target encoder
├── src/
│   ├── feature_pipeline/
│   │   ├── load_and_split.py        # Temporal train / eval / holdout split
│   │   ├── preprocess.py            # Geo enrichment, dedup, outlier removal
│   │   └── feature_engineering.py   # Temporal features, freq & target encoding
│   ├── training_pipeline/
│   │   ├── tune.py                  # Optuna + MLflow hyperparameter tuning
│   │   ├── eval.py                  # Standalone model evaluation
│   │   └── train.py                 # Final model training on train+eval
│   ├── inference_pipeline/
│   │   └── predict.py               # Batch inference (library + CLI)
│   └── api/
│       ├── backend.py               # FastAPI REST API
│       └── frontend.py              # Streamlit interactive dashboard
├── tests/
│   ├── test_features_pipeline.py
│   ├── test_training_pipeline.py
│   ├── test_inference_pipeline.py
│   └── test_backend.py
├── backend.dockerfile               # backend docker image
├── frontend.dockerfile              # frontend docker image
├── makefile
└── pyproject.toml
```

