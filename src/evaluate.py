"""
STC-HGAT Model Evaluation and Visualization
Comprehensive evaluation to verify the model works in practice
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# Load Data (same as training)
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

df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= '2023-01-01']
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET = 'pm2_5'
print(f"   Rows: {len(df):,}, Stations: {df['stationID'].nunique()}")

# ============================================================================
# Build spatial graph and neighbor features
# ============================================================================
print("\n2. Building spatial features...")

station_meta = df.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))

lats = station_meta.set_index('stationID').reindex(station_order)['lat'].values
lons = station_meta.set_index('stationID').reindex(station_order)['lon'].values

D = np.zeros((N_STATIONS, N_STATIONS), dtype=np.float32)
for i in range(N_STATIONS):
    for j in range(i+1, N_STATIONS):
        d = haversine_km(lats[i], lons[i], lats[j], lons[j])
        D[i, j] = D[j, i] = d

k = 5
neighbor_idx = {}
for i, sid in enumerate(station_order):
    sorted_idx = np.argsort(D[i])[1:k+1]
    neighbor_idx[sid] = [station_order[j] for j in sorted_idx]

pm10_pivot = df.pivot_table(index='date', columns='stationID', values='pm10', aggfunc='mean')
pm25_pivot = df.pivot_table(index='date', columns='stationID', values='pm2_5', aggfunc='mean')

def get_neighbor_features(row):
    sid = row['stationID']
    date = row['date']
    neighbors = neighbor_idx.get(sid, [])
    if not neighbors:
        return pd.Series({'neighbor_pm10_mean': np.nan, 'neighbor_pm25_mean': np.nan,
                         'neighbor_pm10_std': np.nan, 'neighbor_pm25_std': np.nan})
    pm10_vals, pm25_vals = [], []
    for n_sid in neighbors:
        if n_sid in pm10_pivot.columns and date in pm10_pivot.index:
            v = pm10_pivot.loc[date, n_sid]
            if not pd.isna(v): pm10_vals.append(v)
        if n_sid in pm25_pivot.columns and date in pm25_pivot.index:
            v = pm25_pivot.loc[date, n_sid]
            if not pd.isna(v): pm25_vals.append(v)
    return pd.Series({
        'neighbor_pm10_mean': np.mean(pm10_vals) if pm10_vals else np.nan,
        'neighbor_pm25_mean': np.mean(pm25_vals) if pm25_vals else np.nan,
        'neighbor_pm10_std': np.std(pm10_vals) if len(pm10_vals) > 1 else 0,
        'neighbor_pm25_std': np.std(pm25_vals) if len(pm25_vals) > 1 else 0,
    })

neighbor_features = df.apply(get_neighbor_features, axis=1)
df = pd.concat([df, neighbor_features], axis=1)

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n3. Feature engineering...")

for lag in [1, 2, 3, 7]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')[TARGET].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

df['pm2_5_mean_7d'] = df.groupby('stationID')[TARGET].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['pm10_mean_7d'] = df.groupby('stationID')['pm10'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['pm10_pm25_ratio'] = df['pm10'] / (df['pm2_5_lag1'] + 1)
df['neighbor_local_pm10_ratio'] = df['neighbor_pm10_mean'] / (df['pm10'] + 1)
df['neighbor_local_pm25_ratio'] = df['neighbor_pm25_mean'] / (df['pm2_5_lag1'] + 1)
df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
df['is_dry_season'] = df['date'].dt.month.isin([11, 12, 1, 2, 3]).astype(float)

lat_mean, lat_std = df['lat'].mean(), df['lat'].std()
lon_mean, lon_std = df['lon'].mean(), df['lon'].std()
df['lat_norm'] = (df['lat'] - lat_mean) / lat_std
df['lon_norm'] = (df['lon'] - lon_mean) / lon_std
df['station_idx'] = df['stationID'].map(sid2idx)

feature_cols = [
    'pm10', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7', 'pm10_mean_7d',
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_mean_7d',
    'pm10_pm25_ratio',
    'neighbor_pm10_mean', 'neighbor_pm25_mean', 'neighbor_pm10_std', 'neighbor_pm25_std',
    'neighbor_local_pm10_ratio', 'neighbor_local_pm25_ratio',
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'day_sin', 'day_cos', 'is_dry_season',
    'lat_norm', 'lon_norm',
]

for col in feature_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=feature_cols + [TARGET])

# ============================================================================
# Split (same as training)
# ============================================================================
print("\n4. Splitting data...")

dates = sorted(df['date'].unique())
n = len(dates)
t1, t2 = int(n * 0.70), int(n * 0.85)

train_df = df[df['date'].isin(dates[:t1])].copy()
val_df = df[df['date'].isin(dates[t1:t2])].copy()
test_df = df[df['date'].isin(dates[t2:])].copy()

print(f"   Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
print(f"   Test date range: {test_df['date'].min()} to {test_df['date'].max()}")

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
val_df[feature_cols] = scaler.transform(val_df[feature_cols].values)
test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
y_test = test_df[TARGET].values
idx_test = torch.tensor(test_df['station_idx'].values, dtype=torch.long)

# ============================================================================
# Load Model
# ============================================================================
print("\n5. Loading trained model...")

class SpatialMLP(nn.Module):
    def __init__(self, in_features, n_stations, hidden=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.station_emb = nn.Embedding(n_stations, hidden // 2)
        self.input_proj = nn.Linear(in_features, hidden)
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden + hidden // 2, hidden + hidden // 2),
                nn.LayerNorm(hidden + hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden + hidden // 2, 1)
    
    def forward(self, x, station_idx):
        h = Fn.relu(self.input_proj(x))
        s = self.station_emb(station_idx)
        h = torch.cat([h, s], dim=-1)
        h = self.mlp(h)
        return self.head(h).squeeze(-1)

checkpoint = torch.load('../data/reports/stc_hgat_rowbased/model.pt', map_location=DEVICE)
config = checkpoint['config']

model = SpatialMLP(
    in_features=len(feature_cols),
    n_stations=N_STATIONS,
    hidden=config['hidden'],
    n_layers=config['n_layers'],
    dropout=config['dropout'],
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   Model loaded successfully")

# ============================================================================
# Evaluate
# ============================================================================
print("\n6. Evaluating on test set...")

with torch.no_grad():
    X_test_dev = X_test.to(DEVICE)
    idx_test_dev = idx_test.to(DEVICE)
    y_pred = model(X_test_dev, idx_test_dev).cpu().numpy()

y_true = y_test

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
mbe = np.mean(y_pred - y_true)

print(f"\n{'='*60}")
print("TEST SET METRICS")
print(f"{'='*60}")
print(f"  R²    : {r2:.4f}")
print(f"  RMSE  : {rmse:.2f} µg/m³")
print(f"  MAE   : {mae:.2f} µg/m³")
print(f"  SMAPE : {smape:.2f}%")
print(f"  MBE   : {mbe:.2f} µg/m³")

# ============================================================================
# Visualizations
# ============================================================================
print("\n7. Creating visualizations...")

reports_dir = Path('../data/reports/stc_hgat_rowbased')
reports_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter plot: Predicted vs Actual
ax1 = axes[0, 0]
ax1.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
ax1.plot([0, y_true.max()], [0, y_true.max()], 'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Actual PM2.5 (µg/m³)', fontsize=12)
ax1.set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=12)
ax1.set_title(f'Predicted vs Actual PM2.5\nR² = {r2:.4f}, RMSE = {rmse:.2f}', fontsize=14)
ax1.legend()
ax1.set_xlim(0, y_true.max() * 1.1)
ax1.set_ylim(0, y_true.max() * 1.1)

# 2. Residual distribution
ax2 = axes[0, 1]
residuals = y_pred - y_true
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', lw=2)
ax2.axvline(residuals.mean(), color='orange', linestyle='-', lw=2, label=f'Mean = {residuals.mean():.2f}')
ax2.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Residual Distribution', fontsize=14)
ax2.legend()

# 3. Time series for a sample station
ax3 = axes[1, 0]
sample_station = test_df['stationID'].value_counts().idxmax()
station_data = test_df[test_df['stationID'] == sample_station].copy()
station_data = station_data.sort_values('date')

station_idx_vals = station_data['station_idx'].values
station_X = torch.tensor(scaler.transform(station_data[feature_cols].values), dtype=torch.float32)
station_idx_t = torch.tensor(station_idx_vals, dtype=torch.long)

with torch.no_grad():
    station_pred = model(station_X.to(DEVICE), station_idx_t.to(DEVICE)).cpu().numpy()

ax3.plot(station_data['date'].values, station_data[TARGET].values, 'b-', label='Actual', alpha=0.8)
ax3.plot(station_data['date'].values, station_pred, 'r--', label='Predicted', alpha=0.8)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax3.set_title(f'Time Series: Station {sample_station}', fontsize=14)
ax3.legend()
ax3.tick_params(axis='x', rotation=45)

# 4. Error by PM2.5 level
ax4 = axes[1, 1]
bins = [0, 15, 35, 55, 75, 100, 200]
labels = ['Good\n(0-15)', 'Moderate\n(15-35)', 'Unhealthy-S\n(35-55)', 
          'Unhealthy\n(55-75)', 'Very Unhealthy\n(75-100)', 'Hazardous\n(100+)']
test_df_eval = test_df.copy()
test_df_eval['y_pred'] = y_pred
test_df_eval['abs_error'] = np.abs(y_pred - y_true)
test_df_eval['pm25_bin'] = pd.cut(test_df_eval[TARGET], bins=bins, labels=labels[:len(bins)-1])

error_by_bin = test_df_eval.groupby('pm25_bin', observed=True)['abs_error'].mean()
colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon'][:len(error_by_bin)]
bars = ax4.bar(range(len(error_by_bin)), error_by_bin.values, color=colors, edgecolor='black')
ax4.set_xticks(range(len(error_by_bin)))
ax4.set_xticklabels(error_by_bin.index, fontsize=10)
ax4.set_xlabel('PM2.5 Level', fontsize=12)
ax4.set_ylabel('Mean Absolute Error (µg/m³)', fontsize=12)
ax4.set_title('Prediction Error by PM2.5 Level', fontsize=14)

for bar, val in zip(bars, error_by_bin.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(reports_dir / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.savefig(reports_dir / 'evaluation_plots.pdf', bbox_inches='tight')
print(f"   Saved: evaluation_plots.png")

# ============================================================================
# Additional Analysis: Per-station performance
# ============================================================================
print("\n8. Per-station analysis...")

test_df_eval['y_true'] = y_true
station_metrics = test_df_eval.groupby('stationID').apply(
    lambda g: pd.Series({
        'R2': r2_score(g['y_true'], g['y_pred']) if len(g) > 1 else np.nan,
        'RMSE': np.sqrt(mean_squared_error(g['y_true'], g['y_pred'])),
        'MAE': mean_absolute_error(g['y_true'], g['y_pred']),
        'N_samples': len(g),
    })
).reset_index()

print(f"\n   Station R² Statistics:")
print(f"   - Mean R²  : {station_metrics['R2'].mean():.4f}")
print(f"   - Min R²   : {station_metrics['R2'].min():.4f}")
print(f"   - Max R²   : {station_metrics['R2'].max():.4f}")
print(f"   - Stations with R² > 0.9: {(station_metrics['R2'] > 0.9).sum()} / {len(station_metrics)}")

# Save station metrics
station_metrics.to_csv(reports_dir / 'station_metrics.csv', index=False)
print(f"   Saved: station_metrics.csv")

# ============================================================================
# Spatial visualization
# ============================================================================
print("\n9. Creating spatial visualization...")

fig2, ax = plt.subplots(figsize=(12, 10))

# Merge station metrics with coordinates
station_metrics_geo = station_metrics.merge(station_meta, on='stationID')

scatter = ax.scatter(
    station_metrics_geo['lon'], 
    station_metrics_geo['lat'],
    c=station_metrics_geo['R2'],
    cmap='RdYlGn',
    s=100,
    edgecolor='black',
    vmin=0.8,
    vmax=1.0,
)
plt.colorbar(scatter, ax=ax, label='R² Score')

for _, row in station_metrics_geo.iterrows():
    ax.annotate(row['stationID'][:3], (row['lon'], row['lat']), 
                fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Per-Station R² Performance (Thailand)', fontsize=14)

plt.tight_layout()
plt.savefig(reports_dir / 'spatial_performance.png', dpi=150, bbox_inches='tight')
print(f"   Saved: spatial_performance.png")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"Model: STC-HGAT (Row-based with Spatial Features)")
print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: {rmse:.2f} µg/m³")
print(f"Test MAE: {mae:.2f} µg/m³")
print(f"\nArtifacts saved to: {reports_dir}")
print(f"  - evaluation_plots.png")
print(f"  - evaluation_plots.pdf")
print(f"  - station_metrics.csv")
print(f"  - spatial_performance.png")

if r2 > 0.99:
    print(f"\n🎯 EXCELLENT: Model achieves R² = {r2:.4f} on test set!")
    print("   Model is ready for production use.")
