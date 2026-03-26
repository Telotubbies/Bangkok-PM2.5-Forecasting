# Bangkok PM2.5 Forecasting — ST-UNN Pipeline

Production-grade PM2.5 forecasting system for Bangkok, Thailand using **ST-UNN** (Spatio-Temporal Unified Neural Network) + baseline models (LSTM, GRU, MLP).

## Pipeline Overview

```
 ┌────────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
 │  Step 1: Ingestion     │     │  Step 2: Preprocessing  │     │  Step 3: Training    │
 │  bangkok_environmental │────▶│  preprocessing_pipeline  │────▶│  model_training      │
 │  _ingestion.ipynb      │     │  .ipynb                  │     │  .ipynb              │
 │                        │     │                          │     │                      │
 │  API → Bronze → Silver │     │  Silver → Gold/Tensors   │     │  Train + Eval + SHAP │
 └────────────────────────┘     └─────────────────────────┘     └──────────┬───────────┘
                                                                           │
                                                                ┌──────────▼───────────┐
                                                                │  visualization.ipynb  │
                                                                │  EDA + Data Quality   │
                                                                └──────────────────────┘
```

## Project Structure

```
bkk-pm25-data-ingestion/
├── bangkok_environmental_ingestion.ipynb   # Step 1: API ingestion (Bronze → Silver)
├── preprocessing_pipeline.ipynb            # Step 2: Feature engineering (Silver → Gold)
├── model_training.ipynb                    # Step 3: ST-UNN + baselines training & forecasting
├── visualization.ipynb                     # EDA, data quality, spatial/temporal analysis
├── requirements.txt                        # Dependencies (PyTorch ROCm separate)
├── README.md
├── data/
│   ├── bronze/openmeteo_weather/           # Raw JSON.gz (immutable)
│   ├── silver/openmeteo_weather/           # Parquet (year=YYYY/month=MM/)
│   ├── silver/openmeteo_airquality/        # ⚠ Empty — AQ API bug
│   ├── stations/bangkok_stations.parquet   # 79 station metadata
│   └── gold/model_ready/                   # train/val/test Parquet + manifest
│       ├── train.parquet                   #   156,435 rows (2010 → 2015)
│       ├── val.parquet                     #    33,549 rows (2015 → 2016)
│       ├── test.parquet                    #    29,067 rows (2016 → 2017)
│       ├── normalization_stats.parquet     #   Z-score stats (train set only)
│       └── pipeline_manifest.json          #   61 features, seq=30, horizons=[1,3]
├── models/                                 # Saved checkpoints + forecasts (after training)
├── plots/                                  # Visualization outputs
├── checkpoints/                            # Ingestion progress tracking
└── logs/                                   # Ingestion run logs
```

---

## Notebooks

### 1. `bangkok_environmental_ingestion.ipynb` — Data Ingestion

Ingests raw environmental data from APIs into Bronze (JSON.gz) and Silver (Parquet) layers.

| Source | API | Variables | Status |
|--------|-----|-----------|--------|
| **Weather** | Open-Meteo Archive | temp, humidity, pressure, precip, wind, radiation, cloud | 2010-01 → 2017-09 (79 stations) |
| **Air Quality** | Open-Meteo AQ | PM2.5, PM10, NO2, O3, SO2, CO | **MISSING** — API parameter bug |
| **Stations** | Air4Thai | stationID, name, lat/lon, area | 79 Bangkok stations |
| **Hotspot** | NASA FIRMS VIIRS | FRP, lat/lon, confidence | **NOT INTEGRATED** |

**Current volume**: 5,255,304 hourly records → 7,350 Parquet files (~374 MB)

---

### 2. `preprocessing_pipeline.ipynb` — Feature Engineering (52 cells)

Transforms Silver layer data into model-ready Gold tensors.

| Stage | What It Does | Output |
|-------|-------------|--------|
| **Load Silver** | Read Hive-partitioned Parquet (filters .md5 sidecars) | 5.2M weather rows, 79 stations |
| **Quality Check** | Per-column dtype, nulls, min/max/mean report | 0% missing in weather |
| **Temporal Align** | Hourly → daily aggregation (solar-noon aligned, vector-mean wind) | 219,051 daily rows |
| **Merge** | Outer join weather + AQ on (stationID, date) | 219,051 × 18 cols |
| **Wind Vectors** | Speed+direction → u10 (east), v10 (north) | wind_u10, wind_v10 |
| **Lag Features** | PM2.5 lag1–3, weather lag1–2 per station | 13 lag columns |
| **Rolling Stats** | Mean & std over 3/7/14-day windows | 24 rolling features |
| **Temporal Encoding** | Cyclical sin/cos for day-of-year & month | 4 columns |
| **Hotspot Placeholders** | NaN columns for count_th/mm/la, frp_sum, TBI | 5 columns |
| **Missing Data** | Linear interpolation (max 3-day gap) | Applied to weather |
| **Outlier Clipping** | Physical bounds (temp: -10–55°C, PM2.5: 0–1000) | Clipped |
| **Chrono Split** | 70% train / 15% val / 15% test by date (no shuffle) | 3 splits |
| **Normalization** | Z-score using training set statistics only | Stats saved |
| **Tensor Prep** | Sliding window: 30 days × 61 features → PM2.5 at +1d, +3d | 0 sequences* |

*\*0 sequences because PM2.5 target is all NaN — see [Missing Data](#missing-data) below.*

---

### 3. `model_training.ipynb` — Model Training & Forecasting (56 cells)

Trains ST-UNN + baseline models, evaluates, and generates forecasts.

| Section | What It Does |
|---------|-------------|
| **Config** | `TrainConfig` dataclass — hyperparams, paths, seed=42 |
| **Data Load** | Reads Gold Parquet + manifest → `PM25SequenceDataset` (sliding window) |
| **Loss** | `CombinedLoss` = 0.7×MAE + 0.3×RMSE |
| **Models** | 5 architectures (see below) |
| **Training** | AdamW + ReduceLROnPlateau + early stopping (patience=15) + gradient clipping |
| **Evaluation** | MAE, RMSE, R² per horizon + seasonal breakdown + extreme day detection |
| **SHAP** | GradientExplainer feature importance for ST-UNN |
| **Forecasting** | Time series plots, error distributions, rolling MAE, AQI band analysis, multi-horizon scatter |
| **Export** | Deployment bundle (`stunn_deployment.pt`) + reusable `forecast()` function |

#### Model Architectures

| Model | Type | Architecture |
|-------|------|-------------|
| **Persistence** | Naive baseline | PM2.5(t+h) = PM2.5(t-1) |
| **MLP** | Feed-forward | Flatten(30×61) → 256 → 128 → horizons |
| **LSTM** | Recurrent | 2-layer LSTM(128) → FC head |
| **GRU** | Recurrent | 2-layer GRU(128) → FC head |
| **ST-UNN** | Spatio-Temporal | Input projection → 2-layer GRU → 4-head spatial attention → gated fusion → GELU FC head |

#### ST-UNN Architecture Detail

```
Input (batch, 30, 61)
  → Linear Projection (61 → 128)
  → Temporal Encoder: 2-layer GRU (hidden=128)
  → Spatial Attention: 4-head self-attention + LayerNorm (residual)
  → Gated Fusion: σ(W[temporal‖attended]) ⊙ temporal + (1-σ) ⊙ attended
  → Regression Head: 128 → 64 (GELU) → 32 (GELU) → 2 horizons
```

#### Forecasting Outputs

| Output | Description |
|--------|-------------|
| `forecast_timeseries_Xd.png` | Actual vs predicted time series + residuals |
| `forecast_error_dist_Xd.png` | Error distribution histogram per model |
| `forecast_rolling_mae_Xd.png` | Rolling MAE over time (stability check) |
| `forecast_by_aqi_band.png` | MAE by PM2.5 intensity (Good/Moderate/Unhealthy) |
| `forecast_multi_horizon_scatter.png` | Scatter pred vs actual for +1d and +3d |
| `forecast_summary.csv` | Full metrics table: MAE, RMSE, Bias, R², Extreme MAE |

---

### 4. `visualization.ipynb` — EDA & Data Quality (33 cells)

| Section | Plot | Description |
|---------|------|-------------|
| **Completeness** | `01_feature_completeness.png` | Horizontal bar chart — % available per feature, color by group |
| **Station Map** | `02_station_map.png` | 79 stations scattered by lat/lon + Bangkok center + 800km radius |
| **Temporal Coverage** | `03_temporal_coverage.png` | Gantt chart per station with train/val/test boundaries |
| **Weather Distributions** | `04_weather_distributions.png` | Histograms for 7 base weather features |
| **Split Drift** | `05_split_drift_boxplots.png` | Box plots comparing train/val/test distributions |
| **Seasonal Patterns** | `06_seasonal_patterns.png` | Monthly mean ± 1σ with Thai season backgrounds (Burning/Monsoon/Cool) |
| **Year-over-Year** | `07_yearly_comparison.png` | Multi-year monthly overlay |
| **Wind Analysis** | `08_wind_analysis.png` | U10/V10 scatter + speed histogram + wind direction rose |
| **Wind Vectors** | `09_wind_vectors_monthly.png` | Monthly mean wind arrows (shows NW → SE burning season pattern) |
| **Correlations** | `10_correlation_matrix.png` | Lower-triangle heatmap with annotation |
| **Lag Correlations** | `11_lag_rolling_correlation.png` | Heatmap for lag & rolling features |
| **Feature Scatter** | `12_feature_scatter.png` | 6 key feature-pair scatter plots with correlation |
| **Split Timeline** | `13_train_val_test_split.png` | Split bar + daily record count line |
| **PM2.5 Analysis** | `14_pm25_analysis.png` | Time series, distribution, monthly, vs temp *(when data available)* |
| **Hotspot Analysis** | `15_hotspot_analysis.png` | Daily counts, FRP, TBI *(when data available)* |

---

## Missing Data

> **Status as of February 2026** — The pipeline structure is complete but **cannot train models** until the missing data sources are resolved.

### Critical: PM2.5 / Air Quality (0% available)

| Column | Status | Impact |
|--------|--------|--------|
| `pm2_5_ugm3` | **ALL NaN** | Model target variable — nothing to predict |
| `pm10_ugm3` | ALL NaN | Co-pollutant feature |
| `co_ugm3`, `no2_ugm3`, `o3_ugm3`, `so2_ugm3` | ALL NaN | Air quality features |
| PM2.5 lag1/2/3, rolling mean/std | ALL NaN | Derived from missing PM2.5 |

**Root cause**: Open-Meteo Air Quality API returns HTTP 400 — the parameter names in `bangkok_environmental_ingestion.ipynb` are incorrect.

**Fix required**: Update `fetch_aq_station()` function with correct API parameter names, then re-run ingestion and preprocessing.

### Critical: Hotspot Data (0% available)

| Column | Status | Impact |
|--------|--------|--------|
| `hotspot_count_th` | ALL NaN | Thai fire count |
| `hotspot_count_mm` | ALL NaN | Myanmar fire count (transboundary) |
| `hotspot_count_la` | ALL NaN | Laos fire count (transboundary) |
| `hotspot_frp_sum` | ALL NaN | Fire Radiative Power |
| `transboundary_index` | ALL NaN | TBI = Σ(FRP × wind × exp(-d/decay)) |

**Root cause**: NASA FIRMS VIIRS data has not been downloaded or integrated yet.

**Fix required**: Download VIIRS archive (2014–present), filter SE Asia bbox, compute daily aggregates, implement upwind filter per `st-unn.mdc` rules.

### Partial: Weather Data (100% available, but incomplete date range)

| Issue | Detail |
|-------|--------|
| Date range | 2010-01 → 2017-09 only (7.75 years) |
| Missing period | 2017-10 → 2026-02 (~8.4 years missing) |
| ERA5 variables | No BLH (boundary layer height) or dewpoint — using Open-Meteo proxy |

**Fix required**: Resume weather backfill through present. Consider CDS API for native ERA5 variables (u10, v10, t2m, d2m, blh, tp).

### Data Availability Summary

```
Feature Group          Train      Val       Test      Status
─────────────────────────────────────────────────────────────
Weather (base)         100%       100%      100%      ✅ OK
Weather (lag/rolling)  ~99%       ~99%      ~99%      ✅ OK
Temporal encoding      100%       100%      100%      ✅ OK
Air Quality (PM2.5)      0%         0%        0%      ❌ CRITICAL
Hotspot / TBI            0%         0%        0%      ❌ CRITICAL
─────────────────────────────────────────────────────────────
Training sequences       0          0         0       ⛔ BLOCKED
```

---

## Action Plan

| Priority | Task | Notebook | Impact |
|----------|------|----------|--------|
| **P0** | Fix AQ API parameters → re-ingest PM2.5 | Step 1 | Unblocks model training |
| **P0** | Re-run preprocessing after AQ fix | Step 2 | Generates valid sequences |
| **P1** | Resume weather backfill (2017-10 → 2026) | Step 1 | More training data |
| **P1** | Integrate NASA FIRMS VIIRS hotspots | Step 1 (new) | Adds transboundary features |
| **P2** | Add ERA5 native variables (BLH, dewpoint) | Step 1 (new) | Better meteorological features |
| **P2** | Walk-forward cross-validation | Step 3 | More robust evaluation |

---

## Environment

Notebooks auto-detect the best available accelerator: **CUDA/ROCm > Apple MPS > CPU**.

### Tested Platforms

| Platform | GPU | PyTorch | Backend | Status |
|----------|-----|---------|---------|--------|
| **Linux (primary)** | AMD Radeon RX 7800 XT (16 GB) | 2.8.0+rocm7.0.2 | ROCm 7.0 | Tested |
| **macOS (secondary)** | Apple M5 Pro/Max (unified memory) | latest stable | MPS (Metal) | Supported |
| **Linux/Windows** | NVIDIA GPU | latest stable | CUDA | Supported |
| **Any** | — | latest stable | CPU | Supported (slow) |

### Setup — Linux (AMD ROCm)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# PyTorch ROCm: download wheels from https://repo.radeon.com/rocm/manylinux/
pip install torch-*.whl torchvision-*.whl torchaudio-*.whl
```

### Setup — macOS (Apple Silicon M5)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio
# MPS backend is included in official PyTorch for macOS ARM64
```

### Setup — Linux/Windows (NVIDIA CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Run

```bash
source .venv/bin/activate
jupyter lab

# Run in order:
# 1. bangkok_environmental_ingestion.ipynb  (data ingestion)
# 2. preprocessing_pipeline.ipynb           (feature engineering)
# 3. visualization.ipynb                    (EDA & plots)
# 4. model_training.ipynb                   (train + forecast)
```

## Design Principles (from `st-unn.mdc`)

- Preserve temporal order — **no random shuffle**
- Chronological train/val/test split — **no future leakage**
- Normalize using **training set statistics only**
- Compute lag features **before** train-test split
- Minimum **5-year training window**
- Reproducible: seed=42 (Python, NumPy, PyTorch, CUDA)
