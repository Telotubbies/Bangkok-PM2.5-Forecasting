# Development Guide - Bangkok PM2.5 Forecasting (STC-HGAT)

Complete development workflow guide for the STC-HGAT PM2.5 forecasting project using a hybrid notebook + module approach with DVC, MLflow, and Optuna.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Development Workflow](#development-workflow)
- [DVC Pipeline](#dvc-pipeline)
- [Experiment Tracking](#experiment-tracking)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Notebooks](#notebooks)

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install PyTorch with ROCm (AMD GPU)
pip install torch-*.whl torchvision-*.whl torchaudio-*.whl

# Or install PyTorch with CUDA (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or install PyTorch for CPU/MPS (Apple Silicon)
pip install torch torchvision torchaudio
```

### 3. Initialize DVC

```bash
# Initialize DVC in the project
dvc init

# Add local remote storage (or configure S3/GCS)
dvc remote add -d local /path/to/dvc-storage
dvc remote modify local type local

# Track data directories
dvc add data/bronze
dvc add data/silver
dvc add data/stations

# Commit DVC files
git add data/.gitignore data/*.dvc .dvc/config
git commit -m "Initialize DVC tracking"
```

### 4. Setup Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Test hooks
```

---

## Development Workflow

### Hybrid Approach: Notebooks + Modules

**Philosophy**: Use Jupyter notebooks for exploration and experimentation, but keep production code in `.py` modules with comprehensive tests.

#### Workflow Steps:

1. **Explore in Notebooks**
   - Use `notebooks/01_data_exploration.ipynb` for data analysis
   - Use `notebooks/02_model_experiments.ipynb` for quick experiments
   - Prototype new features interactively

2. **Refactor to Modules**
   - Move stable code to `src/` modules
   - Add type hints and docstrings
   - Write unit tests in `tests/unit/`

3. **Integrate with DVC**
   - Update `dvc.yaml` with new pipeline stages
   - Run `dvc repro` to execute pipeline

4. **Track with MLflow**
   - Log experiments automatically
   - Compare results in MLflow UI

---

## DVC Pipeline

### Pipeline Stages

The DVC pipeline consists of 4 stages:

```
prepare_data → build_graphs → train_model → evaluate_model
```

### Running the Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train_model

# Show pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Show plots
dvc plots show
```

### Modifying the Pipeline

1. Edit `params.yaml` to change hyperparameters
2. Run `dvc repro` to re-run affected stages
3. DVC automatically tracks dependencies and only re-runs what's needed

### Example: Change Learning Rate

```bash
# Edit params.yaml
# training:
#   learning_rate: 0.0005  # changed from 0.001

# Re-run training (DVC skips data prep and graph building)
dvc repro train_model
```

---

## Experiment Tracking

### MLflow Setup

MLflow is configured to track all experiments automatically.

#### Start MLflow UI

```bash
mlflow ui
# Open http://localhost:5000 in browser
```

#### Tracking Experiments in Code

```python
from src.utils.mlflow_config import setup_mlflow
import mlflow

# Setup
setup_mlflow()

# Start run
with mlflow.start_run(run_name="my-experiment"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    # Train model...
    
    # Log metrics
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

#### Tracking in Notebooks

```python
# In notebook cell
%load_ext autoreload
%autoreload 2

from src.utils.mlflow_config import setup_mlflow
import mlflow

setup_mlflow()

with mlflow.start_run(run_name="notebook-experiment"):
    # Your experiment code
    pass
```

### Comparing Experiments

```bash
# In MLflow UI:
# 1. Select multiple runs
# 2. Click "Compare"
# 3. View metrics, parameters, and plots side-by-side
```

---

## Hyperparameter Tuning

### Using Optuna

Optuna is integrated with MLflow for automated hyperparameter tuning.

#### Run Tuning

```bash
# Run 50 trials
python src/tune.py --n-trials 50

# Continue existing study
python src/tune.py --n-trials 20 --study-name stc-hgat-optimization

# Use custom params file
python src/tune.py --params params_custom.yaml
```

#### View Optimization Progress

```bash
# Start Optuna dashboard
optuna-dashboard sqlite:///optuna.db
# Open http://localhost:8080
```

#### Tuned Hyperparameters

The following hyperparameters are tuned:

- **Model**: `hidden_dim`, `num_hypergat_layers`, `num_hgat_layers`, `num_heads`, `dropout`
- **Training**: `learning_rate`, `batch_size`, `weight_decay`
- **Loss**: `lambda_contrastive`, `temperature`

#### Using Best Parameters

```bash
# After tuning, best parameters are saved to params_best.yaml
# Use them for training:
dvc repro train_model --params params_best.yaml
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual functions
├── integration/       # Integration tests for pipelines
└── models/           # Model performance tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_graph_builder.py

# Run specific test class
pytest tests/unit/test_model.py::TestSTCHGAT

# Run specific test
pytest tests/unit/test_model.py::TestSTCHGAT::test_forward_pass

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_my_module.py
import pytest
from src.my_module import my_function

class TestMyFunction:
    def test_basic_case(self):
        result = my_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

#### Using Fixtures

```python
def test_with_fixture(sample_station_data):
    # sample_station_data is provided by conftest.py
    result = process_stations(sample_station_data)
    assert len(result) == len(sample_station_data)
```

### Coverage Requirements

- Minimum coverage: **80%**
- View coverage report: `open htmlcov/index.html`

---

## Code Quality

### Formatting and Linting

```bash
# Format code with Black
black src/ tests/

# Lint with Flake8
flake8 src/ tests/

# Type check with MyPy
mypy src/

# Run all checks
pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON validation
- Black formatting
- Flake8 linting
- MyPy type checking

### Code Style Guidelines

- **Line length**: 100 characters
- **Docstrings**: Google style
- **Type hints**: Required for all functions
- **Imports**: Organized (stdlib, third-party, local)

---

## Notebooks

### Notebook Organization

```
notebooks/
├── 01_data_exploration.ipynb        # EDA and data quality
├── 02_model_experiments.ipynb       # Quick model experiments
├── stc_hgat_pm25_forecasting.ipynb  # Main forecasting workflow
└── 04_stc_hgat_training.ipynb       # Training experiments
```

### Best Practices

1. **Import from modules**
   ```python
   %load_ext autoreload
   %autoreload 2
   
   import sys
   sys.path.append('..')
   
   from src.models.stc_hgat_model import STCHGAT
   from src.data.dataset import PM25SequenceDataset
   ```

2. **Track experiments**
   ```python
   from src.utils.mlflow_config import setup_mlflow
   import mlflow
   
   setup_mlflow()
   with mlflow.start_run(run_name="notebook-exp-1"):
       # Your code
       pass
   ```

3. **Keep notebooks clean**
   - Clear outputs before committing
   - Use meaningful cell comments
   - Extract reusable code to modules

4. **Restart kernel regularly**
   - Avoid hidden state issues
   - Ensure reproducibility

---

## Common Tasks

### Add New Feature

1. **Prototype in notebook**
   ```python
   # notebooks/02_model_experiments.ipynb
   def new_feature_function(data):
       # Prototype code
       pass
   ```

2. **Write tests**
   ```python
   # tests/unit/test_new_feature.py
   def test_new_feature():
       assert new_feature_function(test_data) == expected
   ```

3. **Move to module**
   ```python
   # src/utils/new_feature.py
   def new_feature_function(data):
       """
       Description of function.
       
       Parameters
       ----------
       data : type
           Description
           
       Returns
       -------
       result : type
           Description
       """
       # Production code
       pass
   ```

4. **Update pipeline**
   ```yaml
   # dvc.yaml
   stages:
     new_stage:
       cmd: python src/utils/new_feature.py
       deps: [...]
       outs: [...]
   ```

### Debug Pipeline

```bash
# Check pipeline status
dvc status

# Show what will run
dvc repro --dry

# Run with verbose output
dvc repro -v

# Force re-run stage
dvc repro --force train_model
```

### Compare Model Versions

```bash
# In MLflow UI:
# 1. Select runs to compare
# 2. View metrics table
# 3. Plot metrics over time
# 4. Download artifacts

# Or use MLflow API:
python -c "
import mlflow
runs = mlflow.search_runs(experiment_names=['stc-hgat-pm25-forecasting'])
print(runs[['metrics.val_loss', 'params.learning_rate']].head())
"
```

---

## Troubleshooting

### DVC Issues

**Problem**: `dvc repro` fails with "output already exists"
```bash
# Solution: Remove outputs and re-run
dvc remove <stage-name>.dvc
dvc repro
```

**Problem**: DVC can't find data
```bash
# Solution: Pull data from remote
dvc pull
```

### MLflow Issues

**Problem**: MLflow UI shows no experiments
```bash
# Solution: Check tracking URI
echo $MLFLOW_TRACKING_URI
# Should be: file:./mlruns
```

**Problem**: Can't log large artifacts
```bash
# Solution: Increase artifact size limit in MLflow config
```

### Testing Issues

**Problem**: Tests fail with import errors
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Problem**: GPU tests fail on CPU machine
```bash
# Solution: Skip GPU tests
pytest -m "not gpu"
```

---

## Resources

- **DVC Documentation**: https://dvc.org/doc
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Pytest Documentation**: https://docs.pytest.org/

---

## Quick Reference

```bash
# Development cycle
dvc repro                    # Run pipeline
pytest --cov=src            # Run tests
mlflow ui                   # View experiments
python src/tune.py          # Tune hyperparameters

# Code quality
black src/ tests/           # Format
flake8 src/ tests/          # Lint
pre-commit run --all-files  # All checks

# Git workflow
git add .
git commit -m "message"     # Pre-commit hooks run
git push
```
