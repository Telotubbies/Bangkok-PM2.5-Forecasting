# Bangkok PM2.5 Forecasting — Data Ingestion & Preprocessing

Production-grade data pipeline for ST-UNN (Spatio-Temporal Unified Neural Network) PM2.5 forecasting in Bangkok, Thailand.

## Project Structure

```
bkk-pm25-data-ingestion/
├── bangkok_environmental_ingestion.ipynb  # Step 1: Data ingestion (Bronze → Silver)
├── preprocessing_pipeline.ipynb           # Step 2: Preprocessing (Silver → Gold/Tensors)
├── requirements.txt
├── .venv/                                 # Python 3.12 + PyTorch ROCm 7.0.2
├── data/
│   ├── bronze/openmeteo_weather/          # Immutable JSON.gz (raw API responses)
│   ├── silver/openmeteo_weather/          # Schema-enforced Parquet (year=YYYY/month=MM/)
│   ├── silver/openmeteo_airquality/       # ⚠ Empty — AQ API bug pending fix
│   ├── stations/bangkok_stations.parquet  # 79 Bangkok station metadata
│   └── gold/model_ready/                  # Model-ready train/val/test outputs
├── checkpoints/                           # Backfill progress tracking
└── logs/                                  # Ingestion run logs
```

## Data Sources

| Source | API | Variables | Status |
|--------|-----|-----------|--------|
| **Weather** | Open-Meteo Archive | temp, humidity, pressure, precip, wind, radiation, cloud | 2010-01 → 2017-09 |
| **Air Quality** | Open-Meteo AQ | PM2.5, PM10, NO2, O3, SO2, CO | Not available (API parameter bug) |
| **Stations** | Air4Thai | stationID, name, lat/lon, area | 79 Bangkok stations |
| **Hotspot** | NASA FIRMS VIIRS | FRP, lat/lon, confidence | Pending integration |

---

## Step 1: `bangkok_environmental_ingestion.ipynb`

Ingests raw data from APIs into Bronze (JSON.gz) and Silver (Parquet) layers.

### Current Data Volume

- **Silver Weather**: 7,350 Parquet files, ~374 MB
- **79 stations**, hourly resolution, 2010-01-01 → 2017-09-30
- **5,255,304 total hourly records**
- Coverage: ~100% for all stations (no missing hours)

---

## Step 2: `preprocessing_pipeline.ipynb`

Transforms Silver data into model-ready tensors. Below is each cell's purpose and its execution result.

### Cell-by-Cell Reference

| Cell | Section | What It Does | Result |
|------|---------|-------------|--------|
| **0** | Title | Markdown overview of pipeline stages and rules | — |
| **1** | Deps | `_install_if_missing()` — checks & installs packages; verifies PyTorch ROCm | All packages installed, torch ROCm confirmed |
| **2** | Imports | Loads pandas, pyarrow, numpy, torch, etc. | OK |
| **3** | Config | `PipelineConfig` dataclass — paths, spatial params, split ratios, seed | Project root, Silver/Gold paths printed |
| **4** | Logger | Sets up structured logging + random seed (42) | `Random seed set to 42` |

#### Section 1 — Load Silver Data

| Cell | What It Does | Result |
|------|-------------|--------|
| **5** | Defines column lists + `load_silver_parquet()` — reads Hive-partitioned Parquet, filters `.md5` sidecars, deduplicates | Function defined |
| **6** | Loads weather Silver data | **5,255,304 rows**, 79 stations, 2010-01-01 → 2017-09-30 |
| **7** | Loads AQ Silver data | **Empty** — directory doesn't exist (AQ ingestion not yet working) |
| **8** | Loads station metadata | **79 stations** with stationID, name, areaEN, lat, lon |

#### Section 2 — Data Quality Assessment

| Cell | What It Does | Result |
|------|-------------|--------|
| **9** | `data_quality_report()` — per-column dtype, null count, null%, min/max/mean | Function defined |
| **10** | Weather DQ report | 0% missing across all columns. Temp: 12.7–40.0°C, Humidity: 19–100%, Pressure: 997–1023 hPa, Wind: 0–30.8 m/s |
| **11** | AQ DQ report | Skipped (empty DataFrame) |
| **12** | `check_date_coverage()` — per-station date range and hourly completeness | Weather: all 79 stations at ~100% coverage. AQ: no data |

#### Section 3 — Temporal Alignment (Hourly → Daily)

| Cell | What It Does | Result |
|------|-------------|--------|
| **13** | Defines aggregation rules + `compute_daily_wind_vector()` (vector-averaged direction) + `aggregate_to_daily()` (solar-noon aligned) | Functions defined |
| **14** | Aggregates weather hourly → daily | **5,255,304 hourly → 219,051 daily rows** |
| **15** | Aggregates AQ hourly → daily | Skipped (empty) |

Aggregation methods:
- Temperature, humidity, pressure, cloud cover → **daily mean**
- Precipitation, radiation → **daily sum**
- Wind → **vector mean** (u/v decomposition avoids 360°/0° wraparound)

#### Section 4 — Merge Weather + AQ

| Cell | What It Does | Result |
|------|-------------|--------|
| **16** | `merge_weather_aq()` — outer join on (stationID, date); creates NaN placeholder columns if AQ missing | **219,051 rows × 18 columns**. AQ columns (pm2_5, pm10, no2, o3, so2, co) all NaN |

#### Section 5 — Feature Engineering

| Cell | Sub-section | What It Does | Result |
|------|------------|-------------|--------|
| **17** | 5a Wind | `add_wind_components()` — speed+direction → u10 (eastward), v10 (northward) | wind_u10 mean=1.24, wind_v10 mean=3.16 (dominant southerly wind) |
| **18** | 5b Lags | `add_lag_features()` — PM2.5 lag1-3, weather lags (temp, humidity, pressure, wind u/v) lag1-2 per station | 13 lag columns created |
| **19** | 5c Rolling | `add_rolling_features()` — rolling mean & std for pm2.5, temp, humidity, wind over 3/7/14-day windows | 24 rolling features (rmean3, rstd3, rmean7, rstd7, rmean14, rstd14 × 4 vars) |
| **20** | 5d Temporal | `add_temporal_encodings()` — cyclical sin/cos for day-of-year and month | 4 columns: doy_sin, doy_cos, month_sin, month_cos |
| **21** | 5e Hotspot | `add_hotspot_placeholders()` — NaN columns for hotspot_count_th/mm/la, hotspot_frp_sum, transboundary_index | 5 placeholder columns (pending VIIRS) |

#### Section 6 — Missing Data Handling

| Cell | What It Does | Result |
|------|-------------|--------|
| **22** | `handle_missing_data()` — linear interpolation (max 3-day gap) per station; flags columns >50% missing | Interpolation applied. AQ + hotspot columns flagged as >50% missing (expected — no source data) |

#### Section 7 — Outlier Detection

| Cell | What It Does | Result |
|------|-------------|--------|
| **23** | `clip_physical_bounds()` — clips values to physical ranges (temp: -10–55°C, PM2.5: 0–1000 µg/m³, etc.) | Clipping applied to weather features |

#### Section 8 — Feature Inventory

| Cell | What It Does | Result |
|------|-------------|--------|
| **24** | Auto-detects all numeric feature columns; defines target (pm2_5_ugm3) and ID columns | Lists all feature columns used for model input |

#### Section 9 — Chronological Split

| Cell | What It Does | Result |
|------|-------------|--------|
| **25** | `chronological_split()` — 70% train / 15% val / 15% test by date (no shuffle). Validates minimum 5-year training window | Train: ~2010–2015, Val: ~2015–2016, Test: ~2016–2017 |

#### Section 10 — Normalization

| Cell | What It Does | Result |
|------|-------------|--------|
| **26** | `NormalizationStats` dataclass — z-score using **training set statistics only**; saves stats to Parquet for inference reuse | Training features: mean≈0, std≈1. Stats saved to `normalization_stats.parquet` |

#### Section 11 — Tensor Preparation

| Cell | What It Does | Result |
|------|-------------|--------|
| **27** | `PM25SequenceDataset` — sliding window over daily data: input = (30 days × N features), target = PM2.5 at +1 and +3 days. Skips sequences with any NaN | Dataset class defined |
| **28** | Builds train/val/test datasets | **0 valid sequences** (PM2.5 target is all NaN — no AQ data yet) |
| **29** | `create_dataloaders()` — wraps datasets in DataLoaders (batch_size=64, shuffle=False to preserve temporal order) | DataLoaders created (empty) |

#### Section 12 — Save Outputs

| Cell | What It Does | Result |
|------|-------------|--------|
| **30** | `save_split_parquet()` + `save_tensor_datasets()` — saves normalized train/val/test as Parquet and `.pt` tensor files | Parquet saved to `data/gold/model_ready/`. Tensor saves skipped (empty datasets) |
| **31** | `save_pipeline_manifest()` — saves config, feature list, and hyperparameters as JSON | `pipeline_manifest.json` saved |

#### Section 13 — Summary & Visualization

| Cell | What It Does | Result |
|------|-------------|--------|
| **32** | `print_pipeline_summary()` — data split sizes, tensor shapes, output file listing, data availability warnings | Summary printed with warnings: PM2.5 all NaN, hotspot all NaN |
| **33** | Matplotlib plot — PM2.5, temperature, and wind components over time for first station | Temperature and wind plotted; PM2.5 panel shows "not available" |

---

## Current Limitations

| Issue | Impact | Action Required |
|-------|--------|-----------------|
| **AQ data missing** | PM2.5 target is all NaN → 0 training sequences | Fix Open-Meteo AQ API parameter names in Step 1 |
| **Weather incomplete** | Only 2010-01 to 2017-09 (7.75 years) | Resume backfill through 2026 |
| **Hotspot not integrated** | No FRP/TBI features | Download VIIRS archive + implement upwind filter |
| **No ERA5 native variables** | Using Open-Meteo proxy (no BLH, dewpoint) | Consider direct ERA5 (CDS API) for u10, v10, t2m, d2m, blh, tp |

## Environment

- **Python**: 3.12
- **PyTorch**: 2.8.0+rocm7.0.2
- **GPU**: AMD Radeon RX 7800 XT (16 GB VRAM, gfx1101)
- **ROCm**: 7.0.1
- **Jupyter kernel**: `bkk-pm25` ("Python (bkk-pm25 ROCm)")

## Quick Start

```bash
source .venv/bin/activate
jupyter notebook preprocessing_pipeline.ipynb
# Select kernel: "Python (bkk-pm25 ROCm)"
# Run All Cells
```
