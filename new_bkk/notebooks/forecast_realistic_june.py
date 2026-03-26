"""
Realistic Long-term PM2.5 Forecast until June 2026
Uses existing model with realistic daily variance simulation
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

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# AQI Colors
AQI_COLORS = {'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy-S': '#FF7E00',
              'Unhealthy': '#FF0000', 'Very Unhealthy': '#8F3F97'}
AQI_BINS = [0, 15, 35, 55, 75, 100]
AQI_LABELS = ['Good', 'Moderate', 'Unhealthy-S', 'Unhealthy', 'Very Unhealthy']

def get_aqi_color(pm25):
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return list(AQI_COLORS.values())[i]
    return AQI_COLORS['Very Unhealthy']

def get_aqi_level(pm25):
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return AQI_LABELS[i]
    return 'Very Unhealthy'

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
# Analyze Historical Patterns for Mar-Jun
# ============================================================================
print("\n[2/6] Analyzing historical patterns for Mar-Jun...")

# Filter historical data for Mar-Jun
hist_mar_jun = df[df['date'].dt.month.isin([3, 4, 5, 6])].copy()
hist_mar_jun['month'] = hist_mar_jun['date'].dt.month
hist_mar_jun['day_of_month'] = hist_mar_jun['date'].dt.day

# Calculate daily statistics by month and day
daily_patterns = hist_mar_jun.groupby(['month', 'day_of_month']).agg({
    'pm2_5': ['mean', 'std', 'min', 'max'],
    'pm10': ['mean', 'std'],
    'temp_c': ['mean', 'std'],
    'humidity_pct': ['mean', 'std'],
    'wind_ms': ['mean', 'std'],
    'pressure_hpa': ['mean', 'std'],
}).reset_index()

daily_patterns.columns = ['month', 'day_of_month', 
                          'pm25_mean', 'pm25_std', 'pm25_min', 'pm25_max',
                          'pm10_mean', 'pm10_std',
                          'temp_mean', 'temp_std',
                          'humidity_mean', 'humidity_std',
                          'wind_mean', 'wind_std',
                          'pressure_mean', 'pressure_std']

# Calculate autocorrelation for PM2.5 (day-to-day correlation)
daily_mean = df.groupby('date')['pm2_5'].mean()
autocorr = daily_mean.autocorr(lag=1)
print(f"   PM2.5 autocorrelation (lag=1): {autocorr:.3f}")

# Calculate monthly statistics
monthly_stats = hist_mar_jun.groupby('month').agg({
    'pm2_5': ['mean', 'std'],
    'pm10': ['mean', 'std'],
}).reset_index()
monthly_stats.columns = ['month', 'pm25_mean', 'pm25_std', 'pm10_mean', 'pm10_std']

print("   Monthly averages (historical):")
for _, row in monthly_stats.iterrows():
    month_name = ['Mar', 'Apr', 'May', 'Jun'][int(row['month']) - 3]
    print(f"     {month_name}: PM2.5={row['pm25_mean']:.1f}±{row['pm25_std']:.1f}, PM10={row['pm10_mean']:.1f}±{row['pm10_std']:.1f}")

# ============================================================================
# Feature Engineering (same as training)
# ============================================================================
print("\n[3/6] Feature engineering...")

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
print("\n[4/6] Loading model...")

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

# Fit scaler
dates = sorted(df['date'].unique())
n = len(dates)
t1 = int(n * 0.70)
train_df = df[df['date'].isin(dates[:t1])].copy()

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

# ============================================================================
# Realistic Long-term Forecast with Daily Variance
# ============================================================================
print("\n[5/6] Generating realistic long-term forecast...")

latest_date = df['date'].max()
end_date = pd.Timestamp('2026-06-30')
n_days = (end_date - latest_date).days

print(f"   Latest data: {latest_date.strftime('%Y-%m-%d')}")
print(f"   Forecast to: {end_date.strftime('%Y-%m-%d')}")
print(f"   Days to forecast: {n_days}")

# Get latest data for each station
latest_data = df[df['date'] == latest_date].copy()

# Initialize tracking variables for each station
station_history = {}
for _, row in latest_data.iterrows():
    sid = row['stationID']
    station_history[sid] = {
        'pm25': [row['pm2_5']],  # List to track history
        'pm10': [row['pm10']],
        'pm25_7d': [row['pm2_5_mean_7d']],
    }

# Ornstein-Uhlenbeck process parameters for realistic random walk
theta = 0.3  # Mean reversion speed
sigma = 0.15  # Volatility

forecasts = []

print(f"   Forecasting {n_days} days for {N_STATIONS} stations with realistic variance...")

for day in range(1, n_days + 1):
    if day % 30 == 0:
        print(f"   Day {day}/{n_days}...")
    
    forecast_date = latest_date + timedelta(days=day)
    month = forecast_date.month
    day_of_month = forecast_date.day
    
    # Get daily pattern for this specific day
    day_pattern = daily_patterns[(daily_patterns['month'] == month) & 
                                  (daily_patterns['day_of_month'] == day_of_month)]
    
    if day_pattern.empty:
        # Fallback to monthly average
        month_stats = monthly_stats[monthly_stats['month'] == month].iloc[0]
        pm25_target = month_stats['pm25_mean']
        pm25_std = month_stats['pm25_std']
        pm10_target = month_stats['pm10_mean']
        pm10_std = month_stats['pm10_std']
        temp_target, humidity_target = 30.0, 70.0
        wind_target, pressure_target = 2.0, 1010.0
    else:
        day_pattern = day_pattern.iloc[0]
        pm25_target = day_pattern['pm25_mean']
        pm25_std = day_pattern['pm25_std']
        pm10_target = day_pattern['pm10_mean']
        pm10_std = day_pattern['pm10_std']
        temp_target = day_pattern['temp_mean']
        humidity_target = day_pattern['humidity_mean']
        wind_target = day_pattern['wind_mean']
        pressure_target = day_pattern['pressure_mean']
    
    for _, row in latest_data.iterrows():
        sid = row['stationID']
        hist = station_history[sid]
        
        # Get previous values
        prev_pm25 = hist['pm25'][-1]
        prev_pm10 = hist['pm10'][-1]
        
        # Simulate PM10 with Ornstein-Uhlenbeck process (mean-reverting random walk)
        # dX = theta * (mu - X) * dt + sigma * dW
        pm10_noise = np.random.normal(0, pm10_std * sigma)
        pm10_sim = prev_pm10 + theta * (pm10_target - prev_pm10) + pm10_noise
        pm10_sim = max(5, min(150, pm10_sim))  # Clip to realistic range
        
        # Simulate weather with small daily variation
        temp_sim = temp_target + np.random.normal(0, 2)
        humidity_sim = humidity_target + np.random.normal(0, 5)
        wind_sim = max(0.5, wind_target + np.random.normal(0, 0.5))
        pressure_sim = pressure_target + np.random.normal(0, 2)
        
        # Calculate lag features from history
        pm25_lag1 = hist['pm25'][-1] if len(hist['pm25']) >= 1 else pm25_target
        pm25_lag2 = hist['pm25'][-2] if len(hist['pm25']) >= 2 else pm25_lag1
        pm25_lag3 = hist['pm25'][-3] if len(hist['pm25']) >= 3 else pm25_lag2
        pm25_lag7 = hist['pm25'][-7] if len(hist['pm25']) >= 7 else pm25_lag1
        
        pm10_lag1 = hist['pm10'][-1] if len(hist['pm10']) >= 1 else pm10_target
        pm10_lag2 = hist['pm10'][-2] if len(hist['pm10']) >= 2 else pm10_lag1
        pm10_lag3 = hist['pm10'][-3] if len(hist['pm10']) >= 3 else pm10_lag2
        pm10_lag7 = hist['pm10'][-7] if len(hist['pm10']) >= 7 else pm10_lag1
        
        # Calculate 7-day rolling mean
        pm25_7d = np.mean(hist['pm25'][-7:]) if len(hist['pm25']) >= 7 else np.mean(hist['pm25'])
        pm10_7d = np.mean(hist['pm10'][-7:]) if len(hist['pm10']) >= 7 else np.mean(hist['pm10'])
        
        # Neighbor features (use average of all stations' current predictions)
        all_prev_pm25 = [station_history[s]['pm25'][-1] for s in station_history]
        all_prev_pm10 = [station_history[s]['pm10'][-1] for s in station_history]
        neighbor_pm25 = np.mean(all_prev_pm25)
        neighbor_pm10 = np.mean(all_prev_pm10)
        
        # Create feature row
        feat_row = {
            'pm10': pm10_sim,
            'pm10_lag1': pm10_lag1,
            'pm10_lag2': pm10_lag2,
            'pm10_lag3': pm10_lag3,
            'pm10_lag7': pm10_lag7,
            'pm10_mean_7d': pm10_7d,
            'pm2_5_lag1': pm25_lag1,
            'pm2_5_lag2': pm25_lag2,
            'pm2_5_lag3': pm25_lag3,
            'pm2_5_lag7': pm25_lag7,
            'pm2_5_mean_7d': pm25_7d,
            'pm10_pm25_ratio': pm10_sim / (pm25_lag1 + 1),
            'neighbor_pm10_mean': neighbor_pm10,
            'neighbor_pm25_mean': neighbor_pm25,
            'neighbor_pm10_std': np.std(all_prev_pm10),
            'neighbor_pm25_std': np.std(all_prev_pm25),
            'neighbor_local_pm10_ratio': neighbor_pm10 / (pm10_sim + 1),
            'neighbor_local_pm25_ratio': neighbor_pm25 / (pm25_lag1 + 1),
            'temp_c': temp_sim,
            'humidity_pct': humidity_sim,
            'wind_ms': wind_sim,
            'pressure_hpa': pressure_sim,
            'day_sin': np.sin(2 * np.pi * forecast_date.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * forecast_date.dayofyear / 365),
            'is_dry_season': 1.0 if month in [11, 12, 1, 2, 3] else 0.0,
            'lat_norm': row['lat_norm'],
            'lon_norm': row['lon_norm'],
        }
        
        # Predict using model
        X = np.array([[feat_row[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        idx_t = torch.tensor([row['station_idx']], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            pred = model(X_t, idx_t).cpu().numpy()[0]
        
        # Add realistic noise based on historical variance
        pred_noise = np.random.normal(0, pm25_std * 0.2)
        pred_final = max(0, pred + pred_noise)
        
        # Apply mean reversion to keep predictions realistic
        pred_final = pred_final * 0.7 + pm25_target * 0.3
        pred_final = max(5, min(100, pred_final))  # Clip to realistic range
        
        # Update history (keep last 14 days)
        hist['pm25'].append(pred_final)
        hist['pm10'].append(pm10_sim)
        if len(hist['pm25']) > 14:
            hist['pm25'] = hist['pm25'][-14:]
            hist['pm10'] = hist['pm10'][-14:]
        
        forecasts.append({
            'stationID': sid,
            'lat': row['lat'],
            'lon': row['lon'],
            'date': forecast_date,
            'pm2_5_pred': pred_final,
            'pm10_sim': pm10_sim,
            'month': month,
        })

forecast_df = pd.DataFrame(forecasts)
print(f"   Generated {len(forecast_df):,} forecasts")

# ============================================================================
# Visualization
# ============================================================================
print("\n[6/6] Creating visualizations...")

reports_dir = Path('../data/reports/forecast_june_realistic')
reports_dir.mkdir(parents=True, exist_ok=True)

# Historical daily mean
daily_hist = df.groupby('date')['pm2_5'].agg(['mean', 'std', 'min', 'max']).reset_index()
daily_hist.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']

# Forecast daily mean
daily_forecast = forecast_df.groupby('date')['pm2_5_pred'].agg(['mean', 'std', 'min', 'max']).reset_index()
daily_forecast.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']

# ============================================================================
# PLOT 1: Full Timeline with Realistic Forecast
# ============================================================================
fig, ax = plt.subplots(figsize=(22, 8))

# Historical
ax.fill_between(daily_hist['date'], daily_hist['pm2_5_min'], daily_hist['pm2_5_max'], 
                alpha=0.15, color='steelblue', label='Historical Range')
ax.plot(daily_hist['date'], daily_hist['pm2_5_mean'], 'b-', lw=1.5, label='Historical Mean')

# Forecast with confidence band
ax.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_mean'] - daily_forecast['pm2_5_std'], 
                daily_forecast['pm2_5_mean'] + daily_forecast['pm2_5_std'], 
                alpha=0.3, color='red', label='Forecast ±1 Std')
ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=1.5, label='Forecast Mean')

# WHO guideline
ax.axhline(35, color='orange', linestyle='--', lw=2, label='WHO Guideline')

# Today line
ax.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax.text(latest_date, ax.get_ylim()[1] * 0.95, ' Today', fontsize=12, fontweight='bold', va='top')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=14)
ax.set_title('Bangkok PM2.5: Historical + Realistic Forecast to June 2026', fontsize=18, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / '1_full_timeline_realistic.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 1_full_timeline_realistic.png")

# ============================================================================
# PLOT 2: Forecast Detail (Mar-Jun 2026)
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 8))

# Last 30 days of historical
hist_recent = daily_hist[daily_hist['date'] >= latest_date - timedelta(days=30)]

# Plot
ax.plot(hist_recent['date'], hist_recent['pm2_5_mean'], 'b-o', lw=2, markersize=3, label='Historical')
ax.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_min'], 
                daily_forecast['pm2_5_max'], 
                alpha=0.2, color='red', label='Forecast Range')
ax.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_mean'] - daily_forecast['pm2_5_std'], 
                daily_forecast['pm2_5_mean'] + daily_forecast['pm2_5_std'], 
                alpha=0.3, color='red')
ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=1.5, label='Forecast Mean')

ax.axhline(35, color='orange', linestyle='--', lw=2, label='WHO Guideline')
ax.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)

# Add monthly labels
for month in range(3, 7):
    month_start = pd.Timestamp(f'2026-{month:02d}-01')
    ax.axvline(month_start, color='gray', linestyle=':', lw=1, alpha=0.5)
    month_name = ['Mar', 'Apr', 'May', 'Jun'][month-3]
    ax.text(month_start + timedelta(days=15), ax.get_ylim()[1] * 0.95, month_name, 
            fontsize=12, ha='center', fontweight='bold')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=14)
ax.set_title('PM2.5 Realistic Forecast: March - June 2026', fontsize=18, fontweight='bold')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(reports_dir / '2_forecast_detail_realistic.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 2_forecast_detail_realistic.png")

# ============================================================================
# PLOT 3: Monthly Summary with Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3a. Monthly forecast mean
ax1 = axes[0, 0]
monthly_forecast = forecast_df.groupby('month')['pm2_5_pred'].agg(['mean', 'std']).reset_index()
month_names = {3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun'}
monthly_forecast['month_name'] = monthly_forecast['month'].map(month_names)
colors = [get_aqi_color(v) for v in monthly_forecast['mean']]

bars = ax1.bar(monthly_forecast['month_name'], monthly_forecast['mean'], color=colors, edgecolor='black')
ax1.errorbar(monthly_forecast['month_name'], monthly_forecast['mean'], 
             yerr=monthly_forecast['std'], fmt='none', color='black', capsize=5)
ax1.axhline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')
ax1.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax1.set_title('Monthly Forecast Average (2026)', fontsize=14, fontweight='bold')
ax1.legend()

for bar, val in zip(bars, monthly_forecast['mean']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
             ha='center', fontsize=12, fontweight='bold')

# 3b. Historical vs Forecast comparison
ax2 = axes[0, 1]
x = np.arange(4)
width = 0.35
hist_monthly_means = [monthly_stats[monthly_stats['month'] == m]['pm25_mean'].values[0] for m in [3, 4, 5, 6]]
forecast_monthly_means = [monthly_forecast[monthly_forecast['month'] == m]['mean'].values[0] for m in [3, 4, 5, 6]]

bars1 = ax2.bar(x - width/2, hist_monthly_means, width, label='Historical Avg', color='steelblue', edgecolor='black')
bars2 = ax2.bar(x + width/2, forecast_monthly_means, width, label='2026 Forecast', color='coral', edgecolor='black')
ax2.axhline(35, color='red', linestyle='--', lw=2)
ax2.set_xticks(x)
ax2.set_xticklabels(['Mar', 'Apr', 'May', 'Jun'])
ax2.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax2.set_title('Historical vs Forecast (Mar-Jun)', fontsize=14, fontweight='bold')
ax2.legend()

# 3c. Daily variance comparison
ax3 = axes[1, 0]
hist_mar_jun_daily = df[df['date'].dt.month.isin([3, 4, 5, 6])].groupby('date')['pm2_5'].mean()
forecast_daily = daily_forecast['pm2_5_mean']

ax3.hist(hist_mar_jun_daily, bins=30, alpha=0.6, label='Historical (Mar-Jun)', color='steelblue', edgecolor='white')
ax3.hist(forecast_daily, bins=30, alpha=0.6, label='Forecast (Mar-Jun 2026)', color='coral', edgecolor='white')
ax3.axvline(35, color='red', linestyle='--', lw=2)
ax3.set_xlabel('Daily Mean PM2.5 (µg/m³)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Daily Variance: Historical vs Forecast', fontsize=14, fontweight='bold')
ax3.legend()

# 3d. Trend
ax4 = axes[1, 1]
current_mean = daily_hist[daily_hist['date'] == latest_date]['pm2_5_mean'].values[0]
mar_mean = monthly_forecast[monthly_forecast['month'] == 3]['mean'].values[0]
apr_mean = monthly_forecast[monthly_forecast['month'] == 4]['mean'].values[0]
may_mean = monthly_forecast[monthly_forecast['month'] == 5]['mean'].values[0]
jun_mean = monthly_forecast[monthly_forecast['month'] == 6]['mean'].values[0]

dates_trend = ['Current\n(Feb 28)', 'March', 'April', 'May', 'June']
values_trend = [current_mean, mar_mean, apr_mean, may_mean, jun_mean]
colors_trend = [get_aqi_color(v) for v in values_trend]

ax4.plot(dates_trend, values_trend, 'b-o', lw=3, markersize=15, markeredgecolor='black')
for i, (d, v) in enumerate(zip(dates_trend, values_trend)):
    ax4.scatter([d], [v], c=[colors_trend[i]], s=200, edgecolor='black', zorder=5)
    ax4.annotate(f'{v:.1f}', (d, v), textcoords="offset points", xytext=(0, 15), 
                 ha='center', fontsize=12, fontweight='bold')

ax4.axhline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')
ax4.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax4.set_title('PM2.5 Trend: Current → June 2026', fontsize=14, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig(reports_dir / '3_monthly_summary_realistic.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 3_monthly_summary_realistic.png")

# ============================================================================
# PLOT 4: Sample Station Forecasts
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Realistic Forecast by Station (Sample)', fontsize=18, fontweight='bold', y=0.98)

# Select 9 sample stations
sample_stations = latest_data.nlargest(9, 'pm2_5')['stationID'].tolist()

for idx, (ax, station) in enumerate(zip(axes.flat, sample_stations)):
    # Historical
    station_hist = df[df['stationID'] == station].groupby('date')['pm2_5'].mean()
    station_recent = station_hist[station_hist.index >= latest_date - timedelta(days=30)]
    
    # Forecast
    station_forecast = forecast_df[forecast_df['stationID'] == station].sort_values('date')
    
    ax.plot(station_recent.index, station_recent.values, 'b-', lw=1.5, label='Historical')
    ax.plot(station_forecast['date'], station_forecast['pm2_5_pred'], 'r-', lw=1, alpha=0.8, label='Forecast')
    ax.axhline(35, color='orange', linestyle=':', lw=1.5)
    ax.axvline(latest_date, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_title(f'{station}', fontsize=10, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(reports_dir / '4_station_forecasts_realistic.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 4_station_forecasts_realistic.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("REALISTIC LONG-TERM FORECAST COMPLETE")
print("="*70)

print(f"""
FORECAST SUMMARY:
─────────────────────────────────────────────────────────────────────────
Forecast Period: {(latest_date + timedelta(days=1)).strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Total Days: {n_days}
Total Predictions: {len(forecast_df):,}

MONTHLY AVERAGES:
  Current (Feb 28): {current_mean:.1f} µg/m³ ({get_aqi_level(current_mean)})
  March 2026:       {mar_mean:.1f} µg/m³ ({get_aqi_level(mar_mean)})
  April 2026:       {apr_mean:.1f} µg/m³ ({get_aqi_level(apr_mean)})
  May 2026:         {may_mean:.1f} µg/m³ ({get_aqi_level(may_mean)})
  June 2026:        {jun_mean:.1f} µg/m³ ({get_aqi_level(jun_mean)})

DAILY VARIANCE:
  Historical Mar-Jun std: {hist_mar_jun_daily.std():.2f} µg/m³
  Forecast Mar-Jun std:   {forecast_daily.std():.2f} µg/m³

TREND: {'📉 PM2.5 expected to DECREASE (Wet season)' if jun_mean < mar_mean else '📈 PM2.5 expected to increase'}

FILES CREATED:
  1. 1_full_timeline_realistic.png    - Full timeline with realistic variance
  2. 2_forecast_detail_realistic.png  - Mar-Jun detail with confidence bands
  3. 3_monthly_summary_realistic.png  - Monthly analysis & comparison
  4. 4_station_forecasts_realistic.png - Station-level forecasts

LOCATION: {reports_dir}
""")

# Save forecast data
forecast_df.to_csv(reports_dir / 'forecast_realistic_june.csv', index=False)
print(f"   Forecast data saved to: {reports_dir / 'forecast_realistic_june.csv'}")
