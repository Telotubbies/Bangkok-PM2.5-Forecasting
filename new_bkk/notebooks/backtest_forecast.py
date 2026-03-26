"""
Backtest Forecast Model
ทดสอบ model โดยใช้ข้อมูลในอดีตเพื่อ predict และเปรียบเทียบกับค่าจริง
เพื่อดูว่า forecast approach ของเรา FIT กับข้อมูลจริงหรือไม่
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# AQI Colors
AQI_COLORS = {'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy-S': '#FF7E00',
              'Unhealthy': '#FF0000', 'Very Unhealthy': '#8F3F97'}

def get_aqi_color(pm25):
    bins = [0, 15, 35, 55, 75, 100]
    colors = list(AQI_COLORS.values())
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if low <= pm25 < high:
            return colors[i]
    return colors[-1]

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/6] Loading data...")

aq_dir = Path('../data/silver/openmeteo_airquality')
aq_files = list(aq_dir.glob('**/*.parquet'))
aq_dfs = [pd.read_parquet(f) for f in aq_files]
aq_df = pd.concat(aq_dfs, ignore_index=True)

aq_df['timestamp_utc'] = pd.to_datetime(aq_df['timestamp_utc'])
aq_df['date'] = pd.to_datetime(aq_df['timestamp_utc'].dt.date)

daily_aq = aq_df.groupby(['stationID', 'date', 'lat', 'lon']).agg({
    'pm2_5_ugm3': 'mean', 'pm10_ugm3': 'mean',
}).reset_index().rename(columns={'pm2_5_ugm3': 'pm2_5', 'pm10_ugm3': 'pm10'})

weather_dir = Path('../data/silver/openmeteo_weather')
weather_files = list(weather_dir.glob('**/*.parquet'))
weather_dfs = [pd.read_parquet(f) for f in weather_files]
weather_df = pd.concat(weather_dfs, ignore_index=True)
weather_df['timestamp_utc'] = pd.to_datetime(weather_df['timestamp_utc'])
weather_df['date'] = pd.to_datetime(weather_df['timestamp_utc'].dt.date)

daily_weather = weather_df.groupby(['stationID', 'date']).agg({
    'temp_c': 'mean', 'humidity_pct': 'mean', 'wind_ms': 'mean', 'pressure_hpa': 'mean',
}).reset_index()

df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= '2023-01-01']
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

station_meta = df.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}

print(f"   Data: {len(df):,} rows, {N_STATIONS} stations")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n[2/6] Feature engineering...")

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
    sid, date = row['stationID'], row['date']
    neighbors = neighbor_idx.get(sid, [])
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

for lag in [1, 2, 3, 7]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')['pm2_5'].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

df['pm2_5_mean_7d'] = df.groupby('stationID')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
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

df = df.dropna(subset=feature_cols + ['pm2_5'])

# ============================================================================
# Load Model
# ============================================================================
print("\n[3/6] Loading model...")

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
print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Fit scaler on training data
dates = sorted(df['date'].unique())
n = len(dates)
t1 = int(n * 0.70)
train_df = df[df['date'].isin(dates[:t1])].copy()

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

# ============================================================================
# Backtest: Rolling Forecast on Historical Data
# ============================================================================
print("\n[4/6] Running backtest (rolling forecast on historical data)...")

# Use last 60 days of data for backtest
# For each day, predict 1-day and 2-day ahead using only past data
backtest_days = 60
latest_date = df['date'].max()
backtest_start = latest_date - timedelta(days=backtest_days + 2)

backtest_results = []

# Get unique dates for backtest
backtest_dates = sorted(df[df['date'] >= backtest_start]['date'].unique())

print(f"   Backtest period: {backtest_dates[0].strftime('%Y-%m-%d')} to {backtest_dates[-3].strftime('%Y-%m-%d')}")
print(f"   Running {len(backtest_dates) - 2} forecast iterations...")

for i, base_date in enumerate(backtest_dates[:-2]):  # Leave 2 days for validation
    if i % 10 == 0:
        print(f"   Processing {i+1}/{len(backtest_dates)-2}...")
    
    # Get data up to base_date
    base_data = df[df['date'] == base_date].copy()
    
    for day_ahead in [1, 2]:
        target_date = base_date + timedelta(days=day_ahead)
        
        # Get actual values for target_date
        actual_data = df[df['date'] == target_date]
        
        for _, row in base_data.iterrows():
            # Check if we have actual data for this station on target date
            actual_row = actual_data[actual_data['stationID'] == row['stationID']]
            if actual_row.empty:
                continue
            
            actual_pm25 = actual_row['pm2_5'].values[0]
            
            # Create feature row for forecast (same logic as forecast_2days.py)
            feat_row = {}
            
            if day_ahead == 1:
                feat_row['pm10'] = row['pm10']
                feat_row['pm10_lag1'] = row['pm10']
                feat_row['pm10_lag2'] = row['pm10_lag1']
                feat_row['pm10_lag3'] = row['pm10_lag2']
                feat_row['pm10_lag7'] = row['pm10_lag7']
                feat_row['pm10_mean_7d'] = row['pm10_mean_7d']
                feat_row['pm2_5_lag1'] = row['pm2_5']
                feat_row['pm2_5_lag2'] = row['pm2_5_lag1']
                feat_row['pm2_5_lag3'] = row['pm2_5_lag2']
                feat_row['pm2_5_lag7'] = row['pm2_5_lag7']
                feat_row['pm2_5_mean_7d'] = row['pm2_5_mean_7d']
            else:  # day_ahead == 2
                # For day 2, we need day 1 prediction first
                # Simplified: use current pm2_5 as proxy for day 1
                feat_row['pm10'] = row['pm10']
                feat_row['pm10_lag1'] = row['pm10']
                feat_row['pm10_lag2'] = row['pm10_lag1']
                feat_row['pm10_lag3'] = row['pm10_lag2']
                feat_row['pm10_lag7'] = row['pm10_lag7']
                feat_row['pm10_mean_7d'] = row['pm10_mean_7d']
                feat_row['pm2_5_lag1'] = row['pm2_5']  # Approximation
                feat_row['pm2_5_lag2'] = row['pm2_5']
                feat_row['pm2_5_lag3'] = row['pm2_5_lag1']
                feat_row['pm2_5_lag7'] = row['pm2_5_lag7']
                feat_row['pm2_5_mean_7d'] = row['pm2_5_mean_7d']
            
            feat_row['pm10_pm25_ratio'] = feat_row['pm10'] / (feat_row['pm2_5_lag1'] + 1)
            feat_row['neighbor_pm10_mean'] = row['neighbor_pm10_mean']
            feat_row['neighbor_pm25_mean'] = row['neighbor_pm25_mean']
            feat_row['neighbor_pm10_std'] = row['neighbor_pm10_std']
            feat_row['neighbor_pm25_std'] = row['neighbor_pm25_std']
            feat_row['neighbor_local_pm10_ratio'] = row['neighbor_local_pm10_ratio']
            feat_row['neighbor_local_pm25_ratio'] = row['neighbor_local_pm25_ratio']
            feat_row['temp_c'] = row['temp_c']
            feat_row['humidity_pct'] = row['humidity_pct']
            feat_row['wind_ms'] = row['wind_ms']
            feat_row['pressure_hpa'] = row['pressure_hpa']
            feat_row['day_sin'] = np.sin(2 * np.pi * target_date.dayofyear / 365)
            feat_row['day_cos'] = np.cos(2 * np.pi * target_date.dayofyear / 365)
            feat_row['is_dry_season'] = 1.0 if target_date.month in [11, 12, 1, 2, 3] else 0.0
            feat_row['lat_norm'] = row['lat_norm']
            feat_row['lon_norm'] = row['lon_norm']
            
            # Predict
            X = np.array([[feat_row[col] for col in feature_cols]])
            X_scaled = scaler.transform(X)
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            idx_t = torch.tensor([row['station_idx']], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                pred = model(X_t, idx_t).cpu().numpy()[0]
            
            backtest_results.append({
                'base_date': base_date,
                'target_date': target_date,
                'stationID': row['stationID'],
                'day_ahead': day_ahead,
                'pm2_5_actual': actual_pm25,
                'pm2_5_pred': max(0, pred),
            })

backtest_df = pd.DataFrame(backtest_results)
print(f"   Generated {len(backtest_df):,} backtest predictions")

# ============================================================================
# Evaluate Backtest Results
# ============================================================================
print("\n[5/6] Evaluating backtest results...")

reports_dir = Path('../data/reports/backtest')
reports_dir.mkdir(parents=True, exist_ok=True)

# Overall metrics
y_true = backtest_df['pm2_5_actual'].values
y_pred = backtest_df['pm2_5_pred'].values

overall_r2 = r2_score(y_true, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
overall_mae = mean_absolute_error(y_true, y_pred)

print(f"\n   OVERALL BACKTEST METRICS:")
print(f"   R²:   {overall_r2:.4f}")
print(f"   RMSE: {overall_rmse:.2f} µg/m³")
print(f"   MAE:  {overall_mae:.2f} µg/m³")

# By day ahead
print(f"\n   BY FORECAST HORIZON:")
for day in [1, 2]:
    day_df = backtest_df[backtest_df['day_ahead'] == day]
    r2 = r2_score(day_df['pm2_5_actual'], day_df['pm2_5_pred'])
    rmse = np.sqrt(mean_squared_error(day_df['pm2_5_actual'], day_df['pm2_5_pred']))
    mae = mean_absolute_error(day_df['pm2_5_actual'], day_df['pm2_5_pred'])
    print(f"   Day {day}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n[6/6] Creating backtest visualizations...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Backtest Results: Historical Forecast Validation', fontsize=20, fontweight='bold', y=0.98)

# 1. Predicted vs Actual Scatter
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(backtest_df['pm2_5_actual'], backtest_df['pm2_5_pred'], alpha=0.3, s=10, c='steelblue')
ax1.plot([0, 80], [0, 80], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual PM2.5 (µg/m³)', fontsize=12)
ax1.set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=12)
ax1.set_title(f'Predicted vs Actual\nR² = {overall_r2:.4f}', fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_xlim(0, 80)
ax1.set_ylim(0, 80)

# 2. By Day Ahead
ax2 = fig.add_subplot(2, 3, 2)
day1_df = backtest_df[backtest_df['day_ahead'] == 1]
day2_df = backtest_df[backtest_df['day_ahead'] == 2]
r2_day1 = r2_score(day1_df['pm2_5_actual'], day1_df['pm2_5_pred'])
r2_day2 = r2_score(day2_df['pm2_5_actual'], day2_df['pm2_5_pred'])

ax2.scatter(day1_df['pm2_5_actual'], day1_df['pm2_5_pred'], alpha=0.3, s=10, c='steelblue', label=f'Day 1 (R²={r2_day1:.3f})')
ax2.scatter(day2_df['pm2_5_actual'], day2_df['pm2_5_pred'], alpha=0.3, s=10, c='coral', label=f'Day 2 (R²={r2_day2:.3f})')
ax2.plot([0, 80], [0, 80], 'k--', lw=2)
ax2.set_xlabel('Actual PM2.5 (µg/m³)', fontsize=12)
ax2.set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=12)
ax2.set_title('Performance by Forecast Horizon', fontsize=14, fontweight='bold')
ax2.legend()

# 3. Time Series Comparison (Daily Mean)
ax3 = fig.add_subplot(2, 3, 3)
daily_backtest = backtest_df.groupby('target_date').agg({
    'pm2_5_actual': 'mean',
    'pm2_5_pred': 'mean',
}).reset_index()

ax3.plot(daily_backtest['target_date'], daily_backtest['pm2_5_actual'], 'b-o', lw=1.5, markersize=4, label='Actual')
ax3.plot(daily_backtest['target_date'], daily_backtest['pm2_5_pred'], 'r--s', lw=1.5, markersize=4, label='Predicted')
ax3.fill_between(daily_backtest['target_date'], daily_backtest['pm2_5_actual'], daily_backtest['pm2_5_pred'], 
                 alpha=0.3, color='gray')
ax3.axhline(35, color='orange', linestyle=':', lw=2)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax3.set_title('Daily Mean: Actual vs Predicted', fontsize=14, fontweight='bold')
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Error Distribution
ax4 = fig.add_subplot(2, 3, 4)
errors = backtest_df['pm2_5_pred'] - backtest_df['pm2_5_actual']
ax4.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', lw=2)
ax4.axvline(errors.mean(), color='orange', linestyle='-', lw=2, label=f'Mean Error = {errors.mean():.2f}')
ax4.set_xlabel('Prediction Error (µg/m³)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax4.legend()

# 5. R² by Day Ahead (Bar)
ax5 = fig.add_subplot(2, 3, 5)
days = ['Day 1', 'Day 2']
r2_values = [r2_day1, r2_day2]
rmse_day1 = np.sqrt(mean_squared_error(day1_df['pm2_5_actual'], day1_df['pm2_5_pred']))
rmse_day2 = np.sqrt(mean_squared_error(day2_df['pm2_5_actual'], day2_df['pm2_5_pred']))
rmse_values = [rmse_day1, rmse_day2]

x = np.arange(len(days))
width = 0.35
bars1 = ax5.bar(x - width/2, r2_values, width, label='R²', color='steelblue', edgecolor='black')
ax5.set_ylabel('R² Score', fontsize=12)
ax5.set_ylim(0.9, 1.0)

ax5_twin = ax5.twinx()
bars2 = ax5_twin.bar(x + width/2, rmse_values, width, label='RMSE', color='coral', edgecolor='black')
ax5_twin.set_ylabel('RMSE (µg/m³)', fontsize=12)

ax5.set_xticks(x)
ax5.set_xticklabels(days)
ax5.set_title('Performance by Forecast Horizon', fontsize=14, fontweight='bold')
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')

for bar, val in zip(bars1, r2_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}', ha='center', fontsize=11)
for bar, val in zip(bars2, rmse_values):
    ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center', fontsize=11)

# 6. Summary Statistics
ax6 = fig.add_subplot(2, 3, 6)
summary_text = f"""
BACKTEST SUMMARY
════════════════════════════════════════

Period: {backtest_df['target_date'].min().strftime('%Y-%m-%d')} to {backtest_df['target_date'].max().strftime('%Y-%m-%d')}
Total Predictions: {len(backtest_df):,}
Stations: {backtest_df['stationID'].nunique()}

OVERALL METRICS:
  R² Score:    {overall_r2:.4f}
  RMSE:        {overall_rmse:.2f} µg/m³
  MAE:         {overall_mae:.2f} µg/m³
  Mean Error:  {errors.mean():.2f} µg/m³

BY FORECAST HORIZON:
  Day 1: R²={r2_day1:.4f}, RMSE={rmse_day1:.2f}
  Day 2: R²={r2_day2:.4f}, RMSE={rmse_day2:.2f}

VERDICT: {'✅ FORECAST IS RELIABLE' if overall_r2 > 0.95 else '⚠️ NEEDS IMPROVEMENT'}
"""
ax6.text(0.05, 0.5, summary_text, fontsize=11, family='monospace', va='center', transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
ax6.axis('off')
ax6.set_title('Backtest Summary', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(reports_dir / 'backtest_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: backtest_results.png")

# ============================================================================
# Detailed Time Series Plot
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Day 1 Forecast
ax1 = axes[0]
day1_daily = day1_df.groupby('target_date').agg({'pm2_5_actual': 'mean', 'pm2_5_pred': 'mean'}).reset_index()
ax1.plot(day1_daily['target_date'], day1_daily['pm2_5_actual'], 'b-o', lw=2, markersize=6, label='Actual')
ax1.plot(day1_daily['target_date'], day1_daily['pm2_5_pred'], 'r--s', lw=2, markersize=6, label='Predicted (Day 1)')
ax1.fill_between(day1_daily['target_date'], day1_daily['pm2_5_actual'], day1_daily['pm2_5_pred'], 
                 alpha=0.3, color='gray')
ax1.axhline(35, color='orange', linestyle=':', lw=2, label='WHO Guideline')
ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax1.set_title(f'Day 1 Forecast Backtest (R² = {r2_day1:.4f})', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.grid(True, alpha=0.3)

# Day 2 Forecast
ax2 = axes[1]
day2_daily = day2_df.groupby('target_date').agg({'pm2_5_actual': 'mean', 'pm2_5_pred': 'mean'}).reset_index()
ax2.plot(day2_daily['target_date'], day2_daily['pm2_5_actual'], 'b-o', lw=2, markersize=6, label='Actual')
ax2.plot(day2_daily['target_date'], day2_daily['pm2_5_pred'], 'r--s', lw=2, markersize=6, label='Predicted (Day 2)')
ax2.fill_between(day2_daily['target_date'], day2_daily['pm2_5_actual'], day2_daily['pm2_5_pred'], 
                 alpha=0.3, color='gray')
ax2.axhline(35, color='orange', linestyle=':', lw=2, label='WHO Guideline')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax2.set_title(f'Day 2 Forecast Backtest (R² = {r2_day2:.4f})', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'backtest_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: backtest_timeseries.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("BACKTEST COMPLETE")
print("="*70)

print(f"""
BACKTEST RESULTS:
─────────────────────────────────────────────────────────────────────────
Period: {backtest_df['target_date'].min().strftime('%Y-%m-%d')} to {backtest_df['target_date'].max().strftime('%Y-%m-%d')}
Total Predictions: {len(backtest_df):,}

OVERALL:
  R² Score:    {overall_r2:.4f}
  RMSE:        {overall_rmse:.2f} µg/m³
  MAE:         {overall_mae:.2f} µg/m³

BY FORECAST HORIZON:
  Day 1: R²={r2_day1:.4f}, RMSE={rmse_day1:.2f} µg/m³
  Day 2: R²={r2_day2:.4f}, RMSE={rmse_day2:.2f} µg/m³

VERDICT: {'✅ FORECAST MODEL IS RELIABLE - predictions FIT well with actual data!' if overall_r2 > 0.95 else '⚠️ Model needs improvement'}

FILES CREATED:
  1. backtest_results.png   - Comprehensive backtest analysis
  2. backtest_timeseries.png - Day 1 & Day 2 time series comparison

LOCATION: {reports_dir}
""")

# Save results
backtest_df.to_csv(reports_dir / 'backtest_data.csv', index=False)
print(f"   Backtest data saved to: {reports_dir / 'backtest_data.csv'}")
