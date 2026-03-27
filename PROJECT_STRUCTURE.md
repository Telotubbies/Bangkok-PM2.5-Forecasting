# Bangkok PM2.5 Forecasting - Project Structure

## Clean and Organized STC-HGAT Implementation

```
bkk-pm25-data-ingestion/
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   │
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   └── stc_hgat_model.py        # STC-HGAT model implementation
│   │
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   └── dataset.py               # Dataset loader and preprocessing
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── graph_builder.py         # Graph construction utilities
│       └── evaluator.py             # Evaluation metrics
│
├── notebooks/                        # Jupyter notebooks
│   ├── stc_hgat_pm25_forecasting.ipynb   # Main forecasting notebook
│   └── 04_stc_hgat_training.ipynb        # Training notebook
│
├── data/                             # Data directory (gitignored)
│   ├── bronze/                       # Raw data
│   ├── silver/                       # Processed data
│   └── stations/                     # Station metadata
│
├── .github/                          # GitHub configurations
│
├── config.yaml                       # Model configuration
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation

```

## File Descriptions

### Core Source Files

- **`src/train.py`**: Main training script for STC-HGAT model
- **`src/evaluate.py`**: Model evaluation and metrics calculation
- **`src/models/stc_hgat_model.py`**: STC-HGAT architecture implementation
- **`src/data/dataset.py`**: Data loading and preprocessing pipeline
- **`src/utils/graph_builder.py`**: Spatial graph construction utilities
- **`src/utils/evaluator.py`**: Evaluation metrics (MAE, RMSE, R², etc.)

### Configuration

- **`config.yaml`**: Model hyperparameters, training settings, data paths

### Notebooks

- **`notebooks/stc_hgat_pm25_forecasting.ipynb`**: Interactive forecasting workflow
- **`notebooks/04_stc_hgat_training.ipynb`**: Training experiments and analysis

### Data Structure

Data files are excluded from git (see `.gitignore`):
- Bronze layer: Raw API data
- Silver layer: Cleaned and partitioned data
- Stations: Bangkok monitoring station metadata

## What Was Removed

The following files were removed during cleanup:
- ❌ Non-STC-HGAT model implementations (GB, hybrid STHGAT, etc.)
- ❌ Redundant training scripts (v2, v3, v4, v5 versions)
- ❌ Old visualization and analysis scripts
- ❌ Deprecated notebooks
- ❌ Checkpoint and log files
- ❌ Nested duplicate directories

## Usage

### Training
```bash
python src/train.py --config config.yaml
```

### Evaluation
```bash
python src/evaluate.py --checkpoint path/to/model.pt
```

### Interactive Development
```bash
jupyter lab
# Open notebooks/stc_hgat_pm25_forecasting.ipynb
```

## Git Branches

- **`data-ingestion`**: Initial code commit
- **`stc-hgat-refactor`**: Clean, organized STC-HGAT-only implementation (current)
