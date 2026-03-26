# Bangkok PM2.5 Forecasting — STC-HGAT Model

PM2.5 forecasting system for Bangkok, Thailand using **STC-HGAT** (Spatio-Temporal Correlation Heterogeneous Graph Attention Network).

## Project Structure

```
bkk-pm25-data-ingestion/
├── src/
│   ├── models/
│   │   └── stc_hgat_model.py          # STC-HGAT model architecture
│   ├── data/
│   │   └── dataset.py                 # Data loading and preprocessing
│   ├── utils/
│   │   ├── graph_builder.py           # Graph construction utilities
│   │   └── evaluator.py               # Model evaluation metrics
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluation script
├── notebooks/
│   ├── stc_hgat_pm25_forecasting.ipynb    # Main forecasting notebook
│   └── 04_stc_hgat_training.ipynb         # Training notebook
├── config.yaml                        # Model configuration
├── requirements.txt                   # Dependencies
└── README.md
```

## STC-HGAT Model

The STC-HGAT (Spatio-Temporal Correlation Heterogeneous Graph Attention Network) is a graph neural network designed for PM2.5 forecasting that captures:

- **Spatial correlations** between monitoring stations using graph attention mechanisms
- **Temporal dependencies** through recurrent layers
- **Heterogeneous relationships** between different types of environmental features

### Architecture

```
Input: Multi-station time series (stations × timesteps × features)
  → Graph Construction (spatial + temporal edges)
  → Heterogeneous Graph Attention Layers
  → Temporal Aggregation (GRU/LSTM)
  → Prediction Head
Output: PM2.5 forecasts (stations × forecast_horizons)
```

## Quick Start

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
python src/train.py --config config.yaml
```

### Evaluation

```bash
python src/evaluate.py --checkpoint path/to/checkpoint.pt
```

### Using Notebooks

```bash
jupyter lab
# Open notebooks/stc_hgat_pm25_forecasting.ipynb
```

## Configuration

Model hyperparameters and training settings are defined in `config.yaml`:

- Model architecture (hidden dimensions, attention heads, layers)
- Training parameters (learning rate, batch size, epochs)
- Data settings (sequence length, forecast horizons)
- Graph construction parameters

## Data Requirements

The model expects data in the following format:
- Station metadata with coordinates
- Time series data with PM2.5 and meteorological features
- Partitioned by year/month for efficient loading

## Environment

Supports multiple backends:
- **CUDA** (NVIDIA GPUs)
- **ROCm** (AMD GPUs)
- **MPS** (Apple Silicon)
- **CPU** (fallback)

## Citation

If you use this code, please cite the STC-HGAT paper and acknowledge the Bangkok PM2.5 forecasting project.
