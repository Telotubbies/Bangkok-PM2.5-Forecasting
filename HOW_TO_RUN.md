# 🚀 How to Run - Bangkok PM2.5 Forecasting Project

## 📋 Table of Contents
1. [Environment Setup](#environment-setup)
2. [Jupyter Notebooks](#jupyter-notebooks)
3. [Python Scripts](#python-scripts)
4. [Shell Scripts](#shell-scripts)
5. [DVC Pipeline](#dvc-pipeline)
6. [Testing](#testing)

---

## 🔧 Environment Setup

### **Option 1: Use Existing Environment (Recommended for 3080ti)**
```bash
# Activate the existing environment
source ~/.venvs/stc_hgat_pm25_forecasting/bin/activate

# Verify installation
python --version  # Should be Python 3.12.3
pip list | grep torch
```

### **Option 2: Create New Environment**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### **Check GPU Availability**
```bash
# Check CUDA
nvidia-smi

# Check PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 📓 Jupyter Notebooks

### **All Available Notebooks:**

#### **1. `notebooks/train_stc_hgat_complete.ipynb`** ⭐ **RECOMMENDED**
**Purpose**: Complete end-to-end training pipeline for STC-HGAT model

**What it does:**
- ✅ Checks project structure and model readiness
- ✅ Loads station metadata (79 stations)
- ✅ Builds spatial hypergraph (proximity + regions)
- ✅ Builds temporal graph (sequential + seasonal)
- ✅ Creates mock data for demo training
- ✅ Splits data (train/val/test)
- ✅ Trains STC-HGAT model with early stopping
- ✅ Visualizes training progress
- ✅ Evaluates on test set (MAE, RMSE, R²)
- ✅ Saves trained model

**How to run:**
```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook

# Navigate to: notebooks/train_stc_hgat_complete.ipynb
# Run All Cells (Shift + Enter or Cell > Run All)
```

**Expected output:**
- Training curves visualization
- Test set metrics (MAE, RMSE, R²)
- Saved model: `models/stc_hgat_best.pt`

**Time to complete:** ~5-10 minutes (with mock data)

---

#### **2. `notebooks/stc_hgat_pm25_forecasting.ipynb`**
**Purpose**: Original STC-HGAT exploration and development notebook

**What it does:**
- Model architecture exploration
- Graph construction examples
- Feature engineering experiments

**How to run:**
```bash
jupyter lab
# Navigate to: notebooks/stc_hgat_pm25_forecasting.ipynb
```

---

#### **3. `notebooks/04_stc_hgat_training.ipynb`**
**Purpose**: Training experiments and hyperparameter tuning

**What it does:**
- Training loop experiments
- Loss function testing
- Hyperparameter exploration

**How to run:**
```bash
jupyter lab
# Navigate to: notebooks/04_stc_hgat_training.ipynb
```

---

#### **4. `bangkok_environmental_ingestion.ipynb`** (Root directory)
**Purpose**: Data ingestion from Open-Meteo API

**What it does:**
- Fetches air quality data (PM2.5, PM10, etc.)
- Fetches weather data (temperature, humidity, etc.)
- Saves to `data/silver/` in parquet format

**How to run:**
```bash
jupyter notebook bangkok_environmental_ingestion.ipynb
```

**Note:** This may take hours to run for full historical data

---

#### **5. `model_training.ipynb`** (Root directory)
**Purpose**: Model training experiments

**How to run:**
```bash
jupyter notebook model_training.ipynb
```

---

#### **6. `preprocessing_pipeline.ipynb`** (Root directory)
**Purpose**: Data preprocessing and feature engineering

**What it does:**
- Data cleaning
- Feature engineering
- Normalization

**How to run:**
```bash
jupyter notebook preprocessing_pipeline.ipynb
```

---

#### **7. `visualization.ipynb`** (Root directory)
**Purpose**: Data visualization and exploratory analysis

**What it does:**
- PM2.5 distribution plots
- Temporal patterns
- Spatial heatmaps

**How to run:**
```bash
jupyter notebook visualization.ipynb
```

---

## 🐍 Python Scripts

### **Training & Evaluation**

#### **1. `src/train.py`** - Main Training Script
```bash
# Basic training
python src/train.py

# With custom config
python src/train.py --config params.yaml

# With MLflow tracking
python src/train.py --experiment-name "stc-hgat-baseline"
```

**Expected output:**
- Training logs
- Model checkpoints in `models/`
- MLflow experiments in `mlruns/`

---

#### **2. `src/evaluate.py`** - Model Evaluation
```bash
# Evaluate saved model
python src/evaluate.py --model-path models/stc_hgat_best.pt

# Evaluate on specific dataset
python src/evaluate.py --model-path models/stc_hgat_best.pt --data-path data/processed/test.pt
```

**Expected output:**
- MAE, RMSE, R² metrics
- Prediction vs actual plots
- Error analysis

---

#### **3. `src/tune.py`** - Hyperparameter Tuning
```bash
# Run Optuna hyperparameter search
python src/tune.py --n-trials 50

# With specific search space
python src/tune.py --n-trials 100 --timeout 3600
```

**Expected output:**
- Best hyperparameters
- Optimization history
- MLflow logged trials

---

### **Data Processing**

#### **4. `src/data/dataset.py`** - Dataset Creation
```bash
# Create datasets
python src/data/dataset.py

# Or import in other scripts
from src.data.dataset import PM25SequenceDataset, split_by_date
```

---

#### **5. `src/utils/graph_builder.py`** - Graph Construction
```bash
# Build and save graphs
python src/utils/graph_builder.py --save

# Or import
from src.utils.graph_builder import build_spatial_hypergraph, build_temporal_graph
```

---

### **Analysis Scripts**

#### **6. `scripts/analyze_data.py`** - Data Quality Analysis
```bash
# Run comprehensive data analysis
python scripts/analyze_data.py

# Or with specific environment
~/.venvs/stc_hgat_pm25_forecasting/bin/python scripts/analyze_data.py
```

**Expected output:**
- Data structure summary
- Missing value analysis
- Station coverage report
- Data quality metrics

---

## 🔨 Shell Scripts

#### **1. `scripts/quick_data_check.sh`** - Quick Data Validation
```bash
# Run quick check
bash scripts/quick_data_check.sh
```

**Output:**
- Directory sizes
- File counts
- Missing years check
- Processed data status

---

#### **2. `scripts/cleanup_macos_files.sh`** - Clean Metadata Files
```bash
# Remove macOS metadata files
bash scripts/cleanup_macos_files.sh
```

**Output:**
- Removes `._*` files
- Removes `.DS_Store` files
- Verifies parquet file count

---

## 📊 DVC Pipeline

### **Run Full Pipeline**
```bash
# Initialize DVC (first time only)
dvc init

# Run entire pipeline
dvc repro

# Run specific stage
dvc repro prepare_data
dvc repro build_graphs
dvc repro train_model
dvc repro evaluate_model
```

### **Check Pipeline Status**
```bash
# Show pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Show plots
dvc plots show
```

---

## 🧪 Testing

### **Run All Tests**
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test type
pytest -m unit tests/
pytest -m integration tests/
pytest -m model tests/
```

### **Run Specific Test Files**
```bash
# Unit tests
pytest tests/unit/test_graph_builder.py
pytest tests/unit/test_dataset.py
pytest tests/unit/test_model.py

# Integration tests
pytest tests/integration/test_pipeline.py
```

---

## 📈 MLflow Tracking

### **Start MLflow UI**
```bash
# Start MLflow server
mlflow ui

# Access at: http://localhost:5000
```

### **View Experiments**
- Training runs
- Hyperparameter tuning results
- Model metrics comparison

---

## 🎯 Quick Start Guide

### **For First-Time Users:**

1. **Setup Environment**
   ```bash
   source ~/.venvs/stc_hgat_pm25_forecasting/bin/activate
   ```

2. **Check Data**
   ```bash
   bash scripts/quick_data_check.sh
   python scripts/analyze_data.py
   ```

3. **Run Training Notebook** (Recommended)
   ```bash
   jupyter lab
   # Open: notebooks/train_stc_hgat_complete.ipynb
   # Run All Cells
   ```

4. **Or Train via Script**
   ```bash
   python src/train.py
   ```

5. **Evaluate Model**
   ```bash
   python src/evaluate.py --model-path models/stc_hgat_best.pt
   ```

---

## 🔍 Troubleshooting

### **Issue: Jupyter kernel not found**
```bash
# Install ipykernel in environment
pip install ipykernel

# Register kernel
python -m ipykernel install --user --name=stc_hgat_pm25_forecasting
```

### **Issue: CUDA out of memory**
```bash
# Reduce batch size in params.yaml
training:
  batch_size: 16  # Reduce from 32
```

### **Issue: Missing data files**
```bash
# Check data structure
bash scripts/quick_data_check.sh

# Run data ingestion
jupyter notebook bangkok_environmental_ingestion.ipynb
```

### **Issue: Import errors**
```bash
# Ensure you're in project root
cd /home/supawich/Desktop/bkk-pm25-data-ingestion

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 📝 Current Data Status

Based on latest analysis:

- **Air Quality Data**: 175,272 records
  - PM2.5 coverage: 40.3% (59.7% null)
  - Years: 2019-2020, 2023-2026 (missing 2021-2022)
  - Stations: 79 unique

- **Weather Data**: Complete
  - Years: 2010-2026
  - Stations: 79 unique

- **Valid Parquet Files**: 11,341 files

---

## 🎓 Recommended Workflow

### **For Model Development:**
1. Start with `notebooks/train_stc_hgat_complete.ipynb`
2. Experiment with hyperparameters
3. Use `src/tune.py` for optimization
4. Evaluate with `src/evaluate.py`

### **For Data Processing:**
1. Check data: `scripts/analyze_data.py`
2. Clean data: `scripts/cleanup_macos_files.sh`
3. Process: `preprocessing_pipeline.ipynb`
4. Visualize: `visualization.ipynb`

### **For Production:**
1. Use DVC pipeline: `dvc repro`
2. Track with MLflow: `mlflow ui`
3. Test: `pytest tests/`
4. Deploy model from `models/`

---

## 📞 Next Steps

1. ✅ **Run** `notebooks/train_stc_hgat_complete.ipynb` for quick start
2. ⚠️ **Address** missing PM2.5 data (2021-2022)
3. 🔄 **Process** full dataset with real data
4. 🎯 **Optimize** hyperparameters with `src/tune.py`
5. 📊 **Evaluate** model performance
6. 🚀 **Deploy** trained model

---

**Last Updated**: 2026-03-27  
**Environment**: `~/.venvs/stc_hgat_pm25_forecasting`  
**Python**: 3.12.3  
**GPU**: RTX 3080 Ti (if available)
