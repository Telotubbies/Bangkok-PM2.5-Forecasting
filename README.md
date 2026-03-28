# 🌍 Bangkok PM2.5 Forecasting with STC-HGAT

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced PM2.5 air quality forecasting system for Bangkok, Thailand using **STC-HGAT** (Spatio-Temporal Contrastive Heterogeneous Graph Attention Networks) with Phase 3 enhancements.

## 📊 Performance

| Forecast Horizon | MAE (µg/m³) | RMSE (µg/m³) | R² Score |
|------------------|-------------|--------------|----------|
| **+1 day** | 0.2398 | 0.3560 | **0.9146** ⭐ |
| **+3 days** | 0.3567 | 0.5412 | **0.8025** |
| **+7 days** | 0.5937 | 0.8937 | **0.4605** |

**Best Model:** `models/stc_hgat_improved_20260327_222751.pt` (91.5% accuracy for 1-day forecast)

---

## 🎯 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)

---

## 🌟 Overview

### What is STC-HGAT?

STC-HGAT is a state-of-the-art graph neural network that forecasts PM2.5 air pollution by modeling:

1. **Spatial Dependencies** - How pollution spreads between monitoring stations
2. **Temporal Patterns** - How pollution changes over time (hourly, daily, seasonal)
3. **Wind Transport** - How wind direction and speed affect pollution dispersion
4. **Multi-scale Dynamics** - Short-term and long-term pollution patterns

### Key Features

- ✅ **Graph-based modeling** - Captures spatial relationships between 79 stations in Bangkok
- ✅ **Wind-aware attention** - Prioritizes upwind stations based on wind direction
- ✅ **Multi-scale temporal** - Analyzes patterns at 1-hour, 3-hour, and 6-hour scales
- ✅ **Contrastive learning** - Improves representation quality with InfoNCE loss
- ✅ **Phase 3 enhancements** - Gated fusion, cross-attention, and multi-scale temporal blocks

### Why Graph Neural Networks?

**Traditional Time Series Models:**
```
❌ Only look at one station's history
❌ Ignore spatial relationships
❌ Don't consider wind direction
```

**Our STC-HGAT Model:**
```
✅ Analyzes all 79 stations simultaneously
✅ Models pollution transport between stations
✅ Incorporates wind direction and speed
✅ Learns which stations affect each other
```

---

## 🏗️ Model Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA (B, N, T, F)                  │
│  B=Batch, N=79 Stations, T=30 Timesteps, F=18 Features     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE EMBEDDING (Linear)                     │
│                 F=18 → H=128                                │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  SPATIAL MODULE  │    │ TEMPORAL MODULE  │
│   (HyperGAT)     │    │     (HGAT)       │
│                  │    │                  │
│ • Hypergraph     │    │ • Session-based  │
│ • Region nodes   │    │ • Attention      │
│ • 2 layers       │    │ • 2 layers       │
└────────┬─────────┘    └─────────┬────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │   GATED FUSION         │
         │   α·h_s + (1-α)·h_t    │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │  CROSS-ATTENTION       │
         │  Q=h_spatial           │
         │  K,V=h_temporal        │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │ MULTI-SCALE TEMPORAL   │
         │ Scales: [1, 3, 6]      │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │ POSITION ENCODING      │
         │ Soft Attention         │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │   OUTPUT PROJECTION    │
         │      (B, N, 1)         │
         └────────────────────────┘
```

### Component Details

#### 1. HyperGAT (Spatial Module)

**Purpose:** Model spatial relationships using hypergraph attention

**Architecture:**
```
Stations (79) + Regions (5) = 84 nodes
         ↓
  Hyperedge Construction
  (stations in same region)
         ↓
   Attention Mechanism
   α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
         ↓
    Aggregation
    h'_i = σ(Σ α_ij · W · h_j)
```

**Key Features:**
- Region-based grouping (Central, North, South, East, West)
- Multi-head attention (4 heads)
- 2 layers with residual connections

#### 2. HGAT (Temporal Module)

**Purpose:** Capture temporal dependencies with session-based attention

**Architecture:**
```
Node Embeddings (B, N, T, H)
         ↓
  Session Representation
  s = mean(h_1, h_2, ..., h_T)
         ↓
   Node-to-Session Attention
   α_is = softmax(v_is^T · tanh(W_is · [h_i || s]))
         ↓
   Session-to-Node Attention
   α_si = softmax(v_si^T · tanh(W_si · [s || h_i]))
         ↓
    Updated Embeddings
    h'_i = h_i + α_is·s + α_si·h_i
```

#### 3. Gated Fusion

**Purpose:** Adaptively combine spatial and temporal features

**Formula:**
```
Gate: α = σ(W_g · [h_spatial || h_temporal] + b_g)
Output: h_fused = α ⊙ h_spatial + (1 - α) ⊙ h_temporal
```

#### 4. Cross-Attention Fusion

**Purpose:** Allow spatial and temporal features to attend to each other

**Formula:**
```
Q = W_q · h_spatial
K = W_k · h_temporal
V = W_v · h_temporal

Attention = softmax(Q·K^T / √d_k)
Output = Attention · V
```

#### 5. Multi-Scale Temporal Block

**Purpose:** Capture patterns at different time scales

**Architecture:**
```
Input → [Conv1D(scale=1), Conv1D(scale=3), Conv1D(scale=6)] → Concat → Output
```

---

## 📐 Mathematical Formulation

### Problem Definition

Given:
- **N** = 79 monitoring stations
- **T** = 30 historical timesteps
- **F** = 18 features per timestep
- **H** = [1, 3, 7] forecast horizons (days)

Predict:
- PM2.5 values for all stations at future timesteps

### Input Representation

**Feature Matrix:**
```
X ∈ ℝ^(N×T×F)

where F = 18:
  • PM2.5 data (6): PM2.5, PM10, NO₂, O₃, SO₂, CO
  • Weather (6): temp, humidity, precip, wind_speed, wind_dir, pressure
  • Temporal (6): hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos
```

### Spatial Graph Construction

**Hypergraph:** G_s = (V, E_h)

**Nodes:**
```
V = V_stations ∪ V_regions
|V| = 79 + 5 = 84
```

**Hyperedges:**
```
E_h = {e_r | r ∈ Regions}
e_r = {v_i | station i belongs to region r}
```

**Incidence Matrix:**
```
H ∈ {0,1}^(N×|E_h|)
H_ij = 1 if station i ∈ hyperedge j, else 0
```

### Temporal Graph Construction

**Session Graph:** G_t = (V_t, E_t)

**Nodes:**
```
V_t = {v_1, v_2, ..., v_T} (T timesteps)
```

**Edges:**
```
E_t = {(v_i, v_j) | |i - j| ≤ window_size}
```

### Attention Mechanisms

#### Spatial Attention (HyperGAT)

**Attention Coefficient:**
```
e_ij = LeakyReLU(a^T · [W·h_i || W·h_j])

α_ij = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)

where:
  W ∈ ℝ^(H×H) - weight matrix
  a ∈ ℝ^(2H) - attention vector
  || - concatenation
```

**Aggregation:**
```
h'_i = σ(Σ_{j∈N(i)} α_ij · W · h_j)

where σ = ELU activation
```

#### Temporal Attention (HGAT)

**Session Representation:**
```
s = (1/T) Σ_{t=1}^T h_t
```

**Node-to-Session Attention:**
```
e_is = v_is^T · tanh(W_is · [h_i || s])
α_is = exp(e_is) / Σ_j exp(e_js)
```

**Session-to-Node Attention:**
```
e_si = v_si^T · tanh(W_si · [s || h_i])
α_si = exp(e_si) / Σ_j exp(e_sj)
```

**Update:**
```
h'_i = h_i + α_is·s + α_si·h_i
```

### Wind-Aware Weighting

**Distance Weight:**
```
w_dist(i,j) = 1 / max(d_ij, d_min)²

where:
  d_ij = haversine_distance(station_i, station_j)
  d_min = 10 km (minimum threshold)
```

**Wind Angle Weight:**
```
θ_ij = bearing(station_i → station_j)
θ_wind = wind_direction at station_i

Δθ = |θ_ij - θ_wind|

w_wind(i,j) = max(0, cos(Δθ))

Interpretation:
  • Δθ = 0° (upwind): w_wind = 1.0 ✅
  • Δθ = 90° (crosswind): w_wind = 0.0
  • Δθ = 180° (downwind): w_wind = 0.0
```

**Combined Weight:**
```
w_ij = w_dist(i,j) × w_wind(i,j) × PM2.5_j
```

### Loss Functions

#### 1. Prediction Loss (MSE)

```
L_pred = (1/N) Σ_{i=1}^N (ŷ_i - y_i)²

where:
  ŷ_i = predicted PM2.5 at station i
  y_i = ground truth PM2.5 at station i
```

#### 2. Contrastive Loss (InfoNCE)

```
L_contrast = -log(exp(sim(h_i, h_i⁺)/τ) / Σ_j exp(sim(h_i, h_j)/τ))

where:
  h_i = spatial embedding
  h_i⁺ = temporal embedding (positive pair)
  sim(·,·) = cosine similarity
  τ = temperature parameter (0.1)
```

#### 3. Total Loss

```
L_total = L_pred + λ·L_contrast

where λ = 0.1 (contrastive weight)
```

---

## 🔄 Data Pipeline

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────┐
│                  RAW DATA SOURCES                        │
├──────────────────────────────────────────────────────────┤
│ • PM2.5 Sensors (79 stations)                           │
│ • Weather Data (Open-Meteo API)                         │
│ • NASA FIRMS Fire/Hotspot Data                          │
│ • Station Metadata (coordinates, regions)               │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              BRONZE LAYER (Raw Data)                     │
│  data/bronze/                                            │
│  ├── pm25_raw/year=YYYY/month=MM/*.parquet             │
│  ├── weather_raw/year=YYYY/*.parquet                    │
│  └── firms_raw/year=YYYY/month=MM/*.parquet            │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│         SILVER LAYER (Cleaned & Validated)               │
│  data/silver/                                            │
│  ├── pm25/year=YYYY/*.parquet                          │
│  ├── weather/year=YYYY/*.parquet                        │
│  └── firms_hotspot/year=YYYY/month=MM/*.parquet        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  • Temporal encoding (sin/cos)                          │
│  • Missing value interpolation                          │
│  • Normalization (z-score)                              │
│  • Sequence creation (sliding window)                   │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              GRAPH CONSTRUCTION                          │
│  • Spatial graph (hyperedges)                           │
│  • Temporal graph (session-based)                       │
│  • Wind-aware attention weights                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  MODEL TRAINING                          │
│  Input: (B, N=79, T=30, F=18)                          │
│  Output: (B, N=79, H=3)                                 │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              TRAINED MODEL (.pt)                         │
│  models/stc_hgat_improved_YYYYMMDD_HHMMSS.pt           │
└──────────────────────────────────────────────────────────┘
```

### Data Statistics

**PM2.5 Data:**
- **Stations:** 79 (Bangkok metropolitan area)
- **Temporal Coverage:** 2017-2025 (8 years)
- **Frequency:** Hourly measurements
- **Features:** PM2.5, PM10, NO₂, O₃, SO₂, CO

**Weather Data:**
- **Source:** Open-Meteo Historical Weather API
- **Variables:** Temperature, humidity, precipitation, wind speed/direction, pressure
- **Frequency:** Hourly
- **Spatial Resolution:** Per station

**Fire Data (Experimental):**
- **Source:** NASA FIRMS (Fire Information for Resource Management System)
- **Coverage:** Southeast Asia region (500km radius from Bangkok)
- **Frequency:** Real-time detections
- **Status:** ⚠️ Under development (date normalization issues)

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 10GB+ disk space

### Setup

```bash
# Clone repository
git clone https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting.git
cd Bangkok-PM2.5-Forecasting

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Dependencies

**Core:**
- PyTorch 2.0+
- PyTorch Geometric
- pandas, numpy
- scikit-learn

**Data:**
- pyarrow (Parquet support)
- requests (API calls)

**Visualization:**
- matplotlib, seaborn
- plotly (interactive plots)

---

## 💻 Usage

### Training

**Basic Training:**
```bash
python scripts/train_improved.py \
  --epochs 100 \
  --batch-size 64 \
  --start-date 2024-01-01 \
  --end-date 2024-11-30
```

**With Custom Configuration:**
```bash
python scripts/train_improved.py \
  --config params.yaml \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --hidden-dim 128 \
  --num-heads 4
```

**Monitor Training:**
```bash
# Real-time monitoring
watch -n 5 './scripts/monitor_training.sh'

# GPU usage
watch -n 2 nvidia-smi

# Training log
tail -f training_improved.log
```

### Evaluation

**Load and Evaluate Model:**
```python
import torch
from src.models.stc_hgat_improved import ImprovedSTCHGAT

# Load checkpoint
checkpoint = torch.load('models/stc_hgat_improved_20260327_222751.pt')

# Initialize model
model = ImprovedSTCHGAT(
    num_features=18,
    hidden_dim=128,
    num_stations=79,
    num_regions=5,
    forecast_horizons=[1, 3, 7]
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get results
results = checkpoint['results']
print(f"R² (+1d): {results['horizon_1d']['r2']:.4f}")
print(f"R² (+3d): {results['horizon_3d']['r2']:.4f}")
print(f"R² (+7d): {results['horizon_7d']['r2']:.4f}")
```

### Inference

**Make Predictions:**
```python
import torch
import pandas as pd
from src.data.real_data_loader import load_pm25_data, load_weather_data, combine_features

# Load recent data
stations_df = pd.read_parquet('data/stations/bangkok_stations.parquet')
pm25_data, metadata = load_pm25_data(
    Path('data'), stations_df,
    start_date='2024-11-01', end_date='2024-11-30'
)
weather_data, _ = load_weather_data(
    Path('data'), stations_df,
    start_date='2024-11-01', end_date='2024-11-30'
)

# Combine features
data, feature_names = combine_features(
    pm25_data, weather_data=weather_data,
    add_temporal_features=True, metadata=metadata
)

# Normalize
train_mean = checkpoint['train_mean']
train_std = checkpoint['train_std']
data_norm = (data - train_mean) / train_std

# Prepare input (last 30 timesteps)
X = data_norm[-30:].unsqueeze(0)  # (1, N, T, F)

# Predict
with torch.no_grad():
    pred, _, _ = model(X, None, None)

# Denormalize
pred_denorm = pred * train_std[0, :, 0] + train_mean[0, :, 0]

print(f"Predicted PM2.5 (+1d): {pred_denorm[0].mean():.2f} µg/m³")
```

---

## 📁 Project Structure

```
bkk-pm25-data-ingestion/
├── data/
│   ├── bronze/              # Raw data
│   ├── silver/              # Cleaned data
│   │   ├── pm25/
│   │   ├── weather/
│   │   └── firms_hotspot/
│   └── stations/            # Station metadata
│       └── bangkok_stations.parquet
│
├── src/
│   ├── models/
│   │   ├── stc_hgat_model.py           # Base STC-HGAT
│   │   ├── stc_hgat_improved.py        # Phase 3 improvements
│   │   ├── stc_hgat_session.py         # Session enhancements
│   │   └── session_enhancements.py     # Session modules
│   │
│   ├── data/
│   │   ├── dataset.py                  # PyTorch Dataset
│   │   ├── real_data_loader.py         # Data loading
│   │   ├── fire_feature_loader.py      # Fire features
│   │   └── fire_feature_loader_v2.py   # Fire features v2
│   │
│   └── utils/
│       ├── graph_builder.py            # Graph construction
│       └── evaluator.py                # Metrics
│
├── scripts/
│   ├── train_improved.py               # Main training script
│   ├── train_session_simple.py         # Simplified training
│   ├── train_session_based.py          # Session-based training
│   ├── monitor_training.sh             # Training monitor
│   └── analyze_per_station.py          # Per-station analysis
│
├── models/
│   └── stc_hgat_improved_20260327_222751.pt  # Best model ⭐
│
├── docs/
│   ├── MODEL_IMPROVEMENTS.md           # Phase 3 improvements
│   ├── MODEL_EXPLANATION_TH.md         # Thai explanation
│   ├── SESSION_BASED_ANALYSIS.md       # Session analysis
│   └── FIRE_FEATURES_IMPLEMENTATION.md # Fire features
│
├── notebooks/
│   └── stc_hgat_pm25_forecasting.ipynb
│
├── params.yaml                         # Model configuration
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

---

## 📈 Results

### Model Comparison

| Model | Features | R² (+1d) | R² (+3d) | R² (+7d) | Status |
|-------|----------|----------|----------|----------|--------|
| **Baseline (Improved)** | 18 | **0.9146** | **0.8025** | **0.4605** | ✅ **Best** |
| Simplified Session | 18 | 0.8874 | 0.7802 | 0.4537 | ✅ Good |
| Complete Session | 24 | 0.6908 | 0.6273 | 0.3783 | ❌ Failed |

**Analysis:**
- Baseline model achieves excellent performance (91.5% R² for 1-day)
- Session enhancements did not improve performance
- Fire features had implementation issues (all zeros)

### Performance by Horizon

**1-Day Forecast (+1d):**
- MAE: 0.2398 µg/m³
- RMSE: 0.3560 µg/m³
- R²: 0.9146 ⭐
- **Interpretation:** 91.5% of variance explained

**3-Day Forecast (+3d):**
- MAE: 0.3567 µg/m³
- RMSE: 0.5412 µg/m³
- R²: 0.8025
- **Interpretation:** 80% of variance explained

**7-Day Forecast (+7d):**
- MAE: 0.5937 µg/m³
- RMSE: 0.8937 µg/m³
- R²: 0.4605
- **Interpretation:** 46% of variance explained

### Training Details

**Best Model:**
- **File:** `models/stc_hgat_improved_20260327_222751.pt`
- **Date:** March 27, 2026
- **Epochs:** 45 (early stopping)
- **Best Val Loss:** 0.6202
- **Parameters:** 653,697
- **Training Time:** ~11 minutes (RTX 3080 Ti)

**Hyperparameters:**
```yaml
model:
  hidden_dim: 128
  num_hypergat_layers: 2
  num_hgat_layers: 2
  num_heads: 4
  dropout: 0.2

training:
  learning_rate: 0.001
  batch_size: 64
  weight_decay: 0.0001
  gradient_clip: 1.0
  early_stopping_patience: 15

data:
  sequence_length: 30
  forecast_horizons: [1, 3, 7]
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

---

## 📚 Documentation

### Core Documentation

- **[MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md)** - Phase 3 enhancements details
- **[MODEL_EXPLANATION_TH.md](docs/MODEL_EXPLANATION_TH.md)** - Thai language explanation
- **[SESSION_BASED_ANALYSIS.md](docs/SESSION_BASED_ANALYSIS.md)** - Session-based approach analysis
- **[FIRE_FEATURES_IMPLEMENTATION.md](docs/FIRE_FEATURES_IMPLEMENTATION.md)** - Fire features implementation

### Key Concepts

**Graph Neural Networks:**
- Nodes = Monitoring stations
- Edges = Spatial/temporal relationships
- Features = PM2.5 + weather + temporal

**Attention Mechanism:**
- Learns which stations are most relevant
- Incorporates wind direction
- Multi-head for different aspects

**Contrastive Learning:**
- Aligns spatial and temporal representations
- Improves feature quality
- InfoNCE loss

---

## 🔬 Experimental Features

### Fire Features (Under Development)

**Status:** ⚠️ Implementation incomplete

**Features Designed:**
1. Fire count within 500km radius
2. Total Fire Radiative Power (FRP)
3. Upwind fire impact (wind-weighted)
4. Distance-weighted fire intensity
5. Temporal lag features (1-day, 3-day)

**Issues:**
- Date normalization bug (only 1 unique date instead of 334)
- All fire features return zeros
- Needs debugging of `pd.DatetimeIndex.normalize()`

**Expected Impact:**
- +2-3% R² improvement during fire season (March-April)
- Better 3-day and 7-day forecasts

### Session-Based Enhancements

**Status:** ✅ Implemented but not effective

**Features:**
1. Session type embeddings (weekday/weekend/holiday/fire_season)
2. Daily session boundaries (hour 0-23 markers)
3. Cross-window attention (long-range dependencies)

**Results:**
- Did not improve performance over baseline
- R² dropped from 0.91 to 0.69
- Session concepts may not suit continuous time series

---

## 🛠️ Development

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Coverage
python -m pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📊 Visualization

### Training Curves

```python
import matplotlib.pyplot as plt

history = checkpoint['history']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.show()
```

### Prediction Visualization

```python
import matplotlib.pyplot as plt

# Plot predictions vs ground truth
plt.figure(figsize=(15, 5))
plt.plot(y_true, label='Ground Truth', alpha=0.7)
plt.plot(y_pred, label='Prediction', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.title('PM2.5 Forecast vs Ground Truth')
plt.show()
```

---

## 🐛 Known Issues

1. **Fire Features All Zeros**
   - Issue: Date normalization returns only 1 unique date
   - Impact: Fire features unusable
   - Status: Under investigation

2. **Session Enhancements Ineffective**
   - Issue: Performance degradation with session features
   - Impact: R² drops from 0.91 to 0.69
   - Status: Documented, not recommended

3. **7-Day Forecast Accuracy**
   - Issue: R² only 0.46 for 7-day forecast
   - Impact: Limited long-term prediction capability
   - Status: Inherent limitation of the problem

---

## 🔮 Future Work

### Short-term (1-2 months)

- [ ] Fix fire features date normalization bug
- [ ] Implement fire features v3 with proper testing
- [ ] Add per-station performance analysis
- [ ] Create web dashboard for predictions

### Medium-term (3-6 months)

- [ ] Integrate satellite imagery (MODIS AOD)
- [ ] Add traffic data features
- [ ] Implement ensemble methods
- [ ] Deploy as REST API

### Long-term (6-12 months)

- [ ] Real-time prediction system
- [ ] Mobile app for public access
- [ ] Expand to other cities in Thailand
- [ ] Publish research paper

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Sources:**
  - Bangkok Air Quality Monitoring Network
  - Open-Meteo Historical Weather API
  - NASA FIRMS (Fire Information for Resource Management System)

- **Research:**
  - STC-HGAT paper and original implementation
  - Graph Neural Networks literature
  - Air quality forecasting research community

- **Tools:**
  - PyTorch and PyTorch Geometric
  - Windsurf IDE with Cascade AI
  - GitHub for version control

---

## 📞 Contact

**Project Team:** Telotubbies  
**Repository:** [Bangkok-PM2.5-Forecasting](https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting)  
**Issues:** [GitHub Issues](https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting/issues)

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{bangkok_pm25_forecasting_2026,
  title = {Bangkok PM2.5 Forecasting with STC-HGAT},
  author = {Telotubbies},
  year = {2026},
  url = {https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting},
  note = {Spatio-Temporal Contrastive Heterogeneous Graph Attention Networks for Air Quality Prediction}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ for cleaner air in Bangkok**

</div>
