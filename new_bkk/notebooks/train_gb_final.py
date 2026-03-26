"""
Final PM2.5 Forecasting Model using Gradient Boosting
Based on baseline analysis: GB achieves R² = 0.78 on test set
This is the best performing model for this dataset.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

print("=" * 60)
print("PM2.5 FORECASTING - GRADIENT BOOSTING MODEL")
print("=" * 60)

# ============================================================================
# Data Loading
# ============================================================================
print("\n1. Loading data...")

aq_dir = Path('../data/silver/openmeteo_airquality')
aq_files = list(aq_dir.glob('**/*.parquet'))
aq_dfs = [pd.read_parquet(f) for f in aq_files]
aq_df = pd.concat(aq_dfs, ignore_index=True)

aq_df['timestamp_utc'] = pd.to_datetime(aq_df['timestamp_utc'])
aq_df['date'] = pd.to_datetime(aq_df['timestamp_utc'].dt.date)

daily_aq = aq_df.groupby(['stationID', 'date', 'lat', 'lon']).agg({
    'pm2_5_ugm3': 'mean',
    'pm10_ugm3': 'mean',
}).reset_index().rename(columns={
    'pm2_5_ugm3': 'pm2_5',
    'pm10_ugm3': 'pm10',
})

# Load weather
weather_dir = Path('../data/silver/openmeteo_weather')
weather_files = list(weather_dir.glob('**/*.parquet'))
weather_dfs = [pd.read_parquet(f) for f in weather_files]
weather_df = pd.concat(weather_dfs, ignore_index=True)
weather_df['timestamp_utc'] = pd.to_datetime(weather_df['timestamp_utc'])
weather_df['date'] = pd.to_datetime(weather_df['timestamp_utc'].dt.date)

daily_weather = weather_df.groupby(['stationID', 'date']).agg({
    'temp_c': 'mean',
    'humidity_pct': 'mean',
    'wind_ms': 'mean',
    'pressure_hpa': 'mean',
}).reset_index()

# Merge
df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= '2023-01-01']
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET = 'pm2_5'

print(f"   Rows: {len(df):,}, Stations: {df['stationID'].nunique()}")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n2. Feature engineering...")

# Lag features - PM2.5 and PM10
for lag in [1, 2, 3, 7, 14]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')[TARGET].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

# Rolling statistics
for window in [3, 7, 14]:
    df[f'pm2_5_mean_{window}d'] = df.groupby('stationID')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f'pm2_5_std_{window}d'] = df.groupby('stationID')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )
    df[f'pm10_mean_{window}d'] = df.groupby('stationID')['pm10'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

# PM10/PM2.5 ratio
df['pm10_pm25_ratio'] = df['pm10'] / (df['pm2_5_lag1'] + 1)

# Difference features
df['pm2_5_diff1'] = df.groupby('stationID')[TARGET].diff(1)
df['pm10_diff1'] = df.groupby('stationID')['pm10'].diff(1)

# Time features
df['day_of_year'] = df['date'].dt.dayofyear
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
df['month'] = df['date'].dt.month
df['is_dry_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(float)

# Weather interactions
df['temp_humidity'] = df['temp_c'] * df['humidity_pct'] / 100
df['wind_pressure'] = df['wind_ms'] * df['pressure_hpa'] / 1000

# Drop NaN
df = df.dropna()

print(f"   After feature engineering: {len(df):,} rows")

# ============================================================================
# Feature Selection
# ============================================================================
feature_cols = [
    # PM10 - most important
    'pm10',
    'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7', 'pm10_lag14',
    'pm10_mean_3d', 'pm10_mean_7d', 'pm10_mean_14d',
    'pm10_diff1',
    
    # PM2.5 lags
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_lag14',
    'pm2_5_mean_3d', 'pm2_5_mean_7d', 'pm2_5_mean_14d',
    'pm2_5_std_3d', 'pm2_5_std_7d', 'pm2_5_std_14d',
    'pm2_5_diff1',
    
    # Ratio
    'pm10_pm25_ratio',
    
    # Weather
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'temp_humidity', 'wind_pressure',
    
    # Time
    'day_sin', 'day_cos', 'is_dry_season',
    
    # Location
    'lat', 'lon',
]

# Keep only existing columns
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# Split
# ============================================================================
print("\n3. Splitting data...")

dates = sorted(df['date'].unique())
n = len(dates)
t1, t2 = int(n * 0.7), int(n * 0.85)

train_df = df[df['date'].isin(dates[:t1])]
val_df = df[df['date'].isin(dates[t1:t2])]
test_df = df[df['date'].isin(dates[t2:])]

print(f"   Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

X_train, y_train = train_df[feature_cols].values, train_df[TARGET].values
X_val, y_val = val_df[feature_cols].values, val_df[TARGET].values
X_test, y_test = test_df[feature_cols].values, test_df[TARGET].values

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ============================================================================
# Train Gradient Boosting
# ============================================================================
print("\n4. Training Gradient Boosting...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    verbose=1,
)

model.fit(X_train_s, y_train)

# ============================================================================
# Evaluation
# ============================================================================
print("\n5. Evaluating...")

def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    mbe = np.mean(y_pred - y_true)
    
    print(f"\n{name}:")
    print(f"  R²    : {r2:.4f}")
    print(f"  RMSE  : {rmse:.2f} µg/m³")
    print(f"  MAE   : {mae:.2f} µg/m³")
    print(f"  SMAPE : {smape:.2f}%")
    print(f"  MBE   : {mbe:.2f} µg/m³")
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'MBE': mbe}

y_train_pred = model.predict(X_train_s)
y_val_pred = model.predict(X_val_s)
y_test_pred = model.predict(X_test_s)

train_metrics = evaluate(y_train, y_train_pred, "TRAIN")
val_metrics = evaluate(y_val, y_val_pred, "VALIDATION")
test_metrics = evaluate(y_test, y_test_pred, "TEST")

# ============================================================================
# Feature Importance
# ============================================================================
print("\n6. Feature Importance (Top 15):")
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importances.head(15).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# Save
# ============================================================================
print("\n7. Saving model...")

reports_dir = Path('../data/reports/gb_final')
reports_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, reports_dir / 'model.joblib')
joblib.dump(scaler, reports_dir / 'scaler.joblib')

# Save metrics
with open(reports_dir / 'metrics.json', 'w') as f:
    json.dump({
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
    }, f, indent=2)

# Save feature list
with open(reports_dir / 'features.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save feature importance
importances.to_csv(reports_dir / 'feature_importance.csv', index=False)

print(f"\nArtifacts saved to {reports_dir}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Test R²   : {test_metrics['R2']:.4f}")
print(f"Test RMSE : {test_metrics['RMSE']:.2f} µg/m³")
print(f"Test MAE  : {test_metrics['MAE']:.2f} µg/m³")

if test_metrics['R2'] > 0.9:
    print("\n🎯 TARGET R² > 0.9 ACHIEVED!")
elif test_metrics['R2'] > 0.8:
    print("\n✓ Excellent result!")
elif test_metrics['R2'] > 0.7:
    print("\n✓ Good result!")
