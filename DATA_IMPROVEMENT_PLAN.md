# Data Improvement Plan - Bangkok PM2.5 Forecasting

## 📊 Current Data Status Analysis

### Data Structure Overview

```
data/
├── bronze/           3.5 MB    (Raw weather data)
│   └── openmeteo_weather/
│       └── 2010-2017/        ✅ 8 years
│
├── silver/           15 GB     (Processed data)
│   ├── openmeteo_airquality/  6,974 parquet files
│   │   ├── year=2019         ✅ Available
│   │   ├── year=2020         ✅ Available
│   │   ├── year=2021         ❌ MISSING
│   │   ├── year=2022         ❌ MISSING
│   │   ├── year=2023         ✅ Available
│   │   ├── year=2024         ✅ Available
│   │   ├── year=2025         ✅ Available
│   │   └── year=2026         ✅ Available (partial)
│   │
│   ├── openmeteo_weather/     7,541 parquet files
│   │   └── year=2010-2026    ✅ Complete
│   │
│   ├── firms_hotspot/         Fire hotspot data
│   └── firms_hotspot_daily/   Daily aggregated
│
├── stations/         16 KB
│   └── bangkok_stations.parquet  ✅ 79 stations
│
└── processed/        ❌ NOT CREATED YET
    └── (for training data)
```

---

## 🚨 Critical Issues Identified

### 1. **Missing PM2.5 Data (2021-2022)**
- **Problem**: Air quality data missing for 2021 and 2022
- **Impact**: 2-year gap in training data
- **Evidence**: From logs - HTTP 400 errors during data ingestion
- **Root Cause**: API request failures or data not available from source

### 2. **No Processed Training Data**
- **Problem**: No `data/processed/` directory with ready-to-train datasets
- **Impact**: Cannot train model without preprocessing
- **Required**: Train/val/test splits, normalized features, graph structures

### 3. **Data Alignment Unknown**
- **Problem**: Haven't verified if stations in metadata match data files
- **Impact**: Potential mismatches between station IDs
- **Need**: Cross-validation of station IDs across all data sources

### 4. **No Data Quality Checks**
- **Problem**: Unknown data quality (missing values, outliers, errors)
- **Impact**: Poor model performance if trained on bad data
- **Need**: Comprehensive data validation pipeline

---

## 🎯 Improvement Plan (Priority Order)

### **Phase 1: Data Assessment & Validation** (Day 1-2)

#### 1.1 Create Data Validation Script
```python
# scripts/validate_data.py
- Check all parquet files are readable
- Verify column schemas match expected format
- Count missing values per column
- Identify outliers (PM2.5 > 500 µg/m³)
- Check date continuity
- Validate station ID consistency
```

**Action Items:**
- [ ] Create `scripts/validate_data.py`
- [ ] Run validation on all silver data
- [ ] Generate data quality report
- [ ] Document issues in `DATA_QUALITY_REPORT.md`

#### 1.2 Analyze Missing Data Patterns
```python
# scripts/analyze_missing_data.py
- Identify which stations have most missing data
- Find temporal patterns in missing data
- Calculate completeness percentage per station
- Visualize missing data heatmap
```

**Action Items:**
- [ ] Create missing data analysis script
- [ ] Generate visualizations
- [ ] Prioritize stations with >80% completeness

---

### **Phase 2: Fill Data Gaps** (Day 3-5)

#### 2.1 Recover Missing PM2.5 Data (2021-2022)

**Option A: Re-run Data Ingestion**
```bash
# Check if data is available from source
# Re-run ingestion with better error handling
python scripts/ingest_airquality.py --start-date 2021-01-01 --end-date 2022-12-31
```

**Option B: Alternative Data Sources**
- Thailand Pollution Control Department (PCD)
- IQAir historical data
- NASA MODIS satellite data
- Interpolation from nearby stations

**Action Items:**
- [ ] Investigate why 2021-2022 failed (check logs)
- [ ] Try re-ingesting with fixed parameters
- [ ] If unavailable, use alternative sources
- [ ] Document data provenance

#### 2.2 Handle Missing Values in Existing Data

**Strategies:**
1. **Forward Fill**: For short gaps (<3 days)
2. **Interpolation**: Linear/spline for medium gaps (3-7 days)
3. **Spatial Interpolation**: Use nearby stations
4. **Seasonal Average**: For longer gaps
5. **Remove**: If >30% missing for a station

```python
# src/data/preprocessing.py
def handle_missing_values(df, method='auto'):
    # Implement smart missing value handling
    pass
```

**Action Items:**
- [ ] Implement missing value handling
- [ ] Test on sample data
- [ ] Apply to full dataset
- [ ] Track imputation statistics

---

### **Phase 3: Data Preprocessing Pipeline** (Day 6-8)

#### 3.1 Create Unified Dataset

```python
# src/data/prepare_training_data.py

def create_unified_dataset():
    """
    Merge air quality + weather + station metadata
    
    Output schema:
    - date: datetime
    - stationID: str
    - pm2_5: float (target variable)
    - pm10: float
    - temperature_2m: float
    - relative_humidity_2m: float
    - surface_pressure: float
    - precipitation: float
    - wind_speed_10m: float
    - wind_direction_10m: float
    - cloud_cover: float
    - lat: float (from stations)
    - lon: float (from stations)
    """
    pass
```

**Action Items:**
- [ ] Create data merging script
- [ ] Handle timezone issues
- [ ] Validate merged data
- [ ] Save to `data/processed/unified.parquet`

#### 3.2 Feature Engineering

```python
# src/data/feature_engineering.py

def add_temporal_features(df):
    """Add time-based features"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map(season_mapping)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    return df

def add_spatial_features(df, stations):
    """Add location-based features"""
    # Distance to city center (Bangkok: 13.7563°N, 100.5018°E)
    df['dist_to_center'] = haversine(df['lat'], df['lon'], 13.7563, 100.5018)
    return df

def add_lagged_features(df, lags=[1, 7, 30]):
    """Add historical PM2.5 values"""
    for lag in lags:
        df[f'pm2_5_lag_{lag}'] = df.groupby('stationID')['pm2_5'].shift(lag)
    return df

def add_rolling_features(df, windows=[7, 30]):
    """Add rolling statistics"""
    for window in windows:
        df[f'pm2_5_rolling_mean_{window}'] = (
            df.groupby('stationID')['pm2_5']
            .rolling(window).mean().reset_index(0, drop=True)
        )
    return df
```

**Action Items:**
- [ ] Implement feature engineering functions
- [ ] Test on sample data
- [ ] Apply to full dataset
- [ ] Document new features in README

#### 3.3 Data Normalization

```python
# src/data/normalization.py

def normalize_features(train, val, test):
    """
    Normalize using training statistics only
    (prevent data leakage)
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    # Fit on training data only
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    
    # Save scaler
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    return train_scaled, val_scaled, test_scaled
```

**Action Items:**
- [ ] Implement normalization
- [ ] Save scaler for inference
- [ ] Verify no data leakage

#### 3.4 Create Train/Val/Test Splits

```python
# src/data/split_data.py

def create_splits(df, train_ratio=0.70, val_ratio=0.15):
    """
    Chronological split (no temporal leakage)
    
    Example for 2019-2026 data:
    - Train: 2019-01-01 to 2023-12-31 (70%)
    - Val:   2024-01-01 to 2024-12-31 (15%)
    - Test:  2025-01-01 to 2026-03-27 (15%)
    """
    df = df.sort_values('date')
    
    n = len(df['date'].unique())
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    dates = sorted(df['date'].unique())
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    train = df[df['date'].isin(train_dates)]
    val = df[df['date'].isin(val_dates)]
    test = df[df['date'].isin(test_dates)]
    
    return train, val, test
```

**Action Items:**
- [ ] Implement chronological split
- [ ] Verify no overlap
- [ ] Save splits to parquet
- [ ] Create metadata file with split info

---

### **Phase 4: Graph Construction** (Day 9-10)

#### 4.1 Build Spatial Graphs

```python
# Already implemented in src/utils/graph_builder.py
# Just need to run and save

from src.utils.graph_builder import build_spatial_hypergraph

spatial_graph = build_spatial_hypergraph(
    stations_df,
    threshold_km=50.0,
    correlation_threshold=0.3
)

torch.save(spatial_graph, 'data/processed/spatial_graph.pt')
```

**Action Items:**
- [ ] Run graph construction
- [ ] Visualize graph structure
- [ ] Save to processed directory
- [ ] Document graph properties

#### 4.2 Build Temporal Graphs

```python
from src.utils.graph_builder import build_temporal_graph

temporal_graph = build_temporal_graph(
    num_days=len(train_dates),
    seasonal_pattern=True
)

torch.save(temporal_graph, 'data/processed/temporal_graph.pt')
```

**Action Items:**
- [ ] Build temporal graph
- [ ] Save to processed directory

---

### **Phase 5: Create DVC Pipeline** (Day 11-12)

#### 5.1 Track Data with DVC

```bash
# Initialize DVC (already done)
dvc init

# Add raw data
dvc add data/bronze
dvc add data/silver
dvc add data/stations

# Commit DVC files
git add data/.gitignore data/*.dvc
git commit -m "Track data with DVC"
```

#### 5.2 Create Preprocessing Pipeline

```yaml
# dvc.yaml (update existing)

stages:
  validate_data:
    cmd: python scripts/validate_data.py
    deps:
      - data/silver
      - scripts/validate_data.py
    outs:
      - reports/data_quality_report.json

  prepare_data:
    cmd: python src/data/prepare_training_data.py
    deps:
      - data/silver/openmeteo_airquality
      - data/silver/openmeteo_weather
      - data/stations
      - src/data/prepare_training_data.py
    params:
      - data.train_ratio
      - data.val_ratio
      - data.sequence_length
    outs:
      - data/processed/train.pt
      - data/processed/val.pt
      - data/processed/test.pt
      - data/processed/scaler.pkl
    metrics:
      - data/processed/data_stats.json

  build_graphs:
    cmd: python src/utils/graph_builder.py --save
    deps:
      - data/stations/bangkok_stations.parquet
      - src/utils/graph_builder.py
    params:
      - graph.spatial_threshold_km
      - graph.correlation_threshold
    outs:
      - data/processed/spatial_graph.pt
      - data/processed/temporal_graph.pt
```

**Action Items:**
- [ ] Update dvc.yaml with preprocessing stages
- [ ] Test pipeline: `dvc repro`
- [ ] Commit pipeline configuration

---

### **Phase 6: Data Quality Monitoring** (Ongoing)

#### 6.1 Create Data Quality Dashboard

```python
# scripts/data_quality_dashboard.py

def generate_dashboard():
    """
    Create HTML dashboard with:
    - Data completeness by station
    - Missing value heatmap
    - PM2.5 distribution by year
    - Outlier detection results
    - Correlation matrix
    """
    pass
```

**Action Items:**
- [ ] Create dashboard script
- [ ] Generate initial report
- [ ] Schedule regular updates

#### 6.2 Automated Data Validation

```python
# tests/data/test_data_quality.py

def test_no_missing_dates():
    """Ensure no gaps in date range"""
    pass

def test_pm25_range():
    """PM2.5 should be 0-500 µg/m³"""
    pass

def test_station_coverage():
    """All stations should have data"""
    pass
```

**Action Items:**
- [ ] Create data quality tests
- [ ] Add to CI/CD pipeline
- [ ] Run before each training

---

## 📋 Implementation Checklist

### Week 1: Data Assessment & Recovery
- [ ] Day 1: Create validation scripts
- [ ] Day 2: Run validation, generate reports
- [ ] Day 3: Investigate 2021-2022 missing data
- [ ] Day 4: Re-ingest or find alternative sources
- [ ] Day 5: Handle missing values in existing data

### Week 2: Preprocessing & Pipeline
- [ ] Day 6: Create unified dataset
- [ ] Day 7: Feature engineering
- [ ] Day 8: Normalization & splits
- [ ] Day 9: Build graphs
- [ ] Day 10: Test preprocessing pipeline
- [ ] Day 11: DVC integration
- [ ] Day 12: Documentation & testing

---

## 🎯 Success Criteria

### Data Completeness
- ✅ PM2.5 data for 2019-2026 (or 2019-2020, 2023-2026 if 2021-2022 unavailable)
- ✅ Weather data for matching period
- ✅ All 79 stations represented
- ✅ <10% missing values after imputation

### Data Quality
- ✅ No outliers (PM2.5 < 500 µg/m³)
- ✅ Continuous date range (no gaps)
- ✅ Consistent station IDs across sources
- ✅ Validated data types and ranges

### Preprocessing Pipeline
- ✅ Reproducible with DVC
- ✅ Train/val/test splits created
- ✅ Features normalized
- ✅ Graphs constructed and saved
- ✅ Ready for model training

### Documentation
- ✅ Data quality report generated
- ✅ Preprocessing steps documented
- ✅ Feature descriptions added
- ✅ Known issues tracked

---

## 🚀 Quick Start Commands

```bash
# 1. Validate current data
python scripts/validate_data.py

# 2. Analyze missing data
python scripts/analyze_missing_data.py

# 3. Prepare training data
python src/data/prepare_training_data.py

# 4. Build graphs
python src/utils/graph_builder.py --save

# 5. Run full DVC pipeline
dvc repro

# 6. Check processed data
ls -lh data/processed/
```

---

## 📞 Next Steps

1. **Immediate**: Run data validation to understand current state
2. **Priority**: Recover 2021-2022 PM2.5 data or adjust training period
3. **Essential**: Create preprocessing pipeline
4. **Important**: Integrate with DVC for reproducibility
5. **Ongoing**: Monitor data quality

---

## 📝 Notes

- **Data Source**: Open-Meteo API (air quality & weather)
- **Station Count**: 79 Bangkok monitoring stations
- **Target Variable**: PM2.5 (µg/m³)
- **Forecast Horizons**: 1, 3, 7 days
- **Training Period**: Ideally 2019-2026 (7 years)
- **Known Issue**: HTTP 400 errors during 2021-2022 ingestion

---

**Last Updated**: 2026-03-27
**Status**: Planning Phase
**Next Review**: After Phase 1 completion
