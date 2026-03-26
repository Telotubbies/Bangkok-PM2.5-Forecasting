"""
Long-term PM2.5 Forecast until June 2026
Predict PM2.5 from current date (Feb 28, 2026) to June 30, 2026
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
print("\n[1/5] Loading data...")

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
print("\n[2/5] Feature engineering...")

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
print("\n[3/5] Loading model...")

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
# Long-term Forecast: Feb 28, 2026 -> June 30, 2026
# ============================================================================
print("\n[4/5] Generating long-term forecast to June 2026...")

latest_date = df['date'].max()
end_date = pd.Timestamp('2026-06-30')
n_days = (end_date - latest_date).days

print(f"   Latest data: {latest_date.strftime('%Y-%m-%d')}")
print(f"   Forecast to: {end_date.strftime('%Y-%m-%d')}")
print(f"   Days to forecast: {n_days}")

# Get latest data for each station
latest_data = df[df['date'] == latest_date].copy()

# Use historical seasonal patterns for PM10 estimation
# Calculate monthly averages for PM10 and weather
monthly_stats = df.groupby(df['date'].dt.month).agg({
    'pm10': ['mean', 'std'],
    'pm2_5': ['mean', 'std'],
    'temp_c': 'mean',
    'humidity_pct': 'mean',
    'wind_ms': 'mean',
    'pressure_hpa': 'mean',
}).reset_index()
monthly_stats.columns = ['month', 'pm10_mean', 'pm10_std', 'pm25_mean', 'pm25_std', 
                         'temp_mean', 'humidity_mean', 'wind_mean', 'pressure_mean']

# Generate forecasts
forecasts = []
current_pm25 = {}  # Track PM2.5 for each station
current_pm10 = {}

# Initialize with latest values
for _, row in latest_data.iterrows():
    current_pm25[row['stationID']] = row['pm2_5']
    current_pm10[row['stationID']] = row['pm10']

print(f"   Forecasting {n_days} days for {N_STATIONS} stations...")

for day in range(1, n_days + 1):
    if day % 30 == 0:
        print(f"   Day {day}/{n_days}...")
    
    forecast_date = latest_date + timedelta(days=day)
    month = forecast_date.month
    
    # Get seasonal statistics for this month
    month_stats = monthly_stats[monthly_stats['month'] == month].iloc[0]
    
    for _, row in latest_data.iterrows():
        sid = row['stationID']
        
        # Estimate PM10 based on seasonal pattern with some noise
        pm10_seasonal = month_stats['pm10_mean']
        pm10_noise = np.random.normal(0, month_stats['pm10_std'] * 0.3)
        pm10_est = max(5, pm10_seasonal + pm10_noise)
        
        # Get previous PM2.5 values
        pm25_lag1 = current_pm25.get(sid, month_stats['pm25_mean'])
        
        # Create feature row
        feat_row = {
            'pm10': pm10_est,
            'pm10_lag1': current_pm10.get(sid, pm10_est),
            'pm10_lag2': current_pm10.get(sid, pm10_est),
            'pm10_lag3': current_pm10.get(sid, pm10_est),
            'pm10_lag7': current_pm10.get(sid, pm10_est),
            'pm10_mean_7d': pm10_est,
            'pm2_5_lag1': pm25_lag1,
            'pm2_5_lag2': pm25_lag1,
            'pm2_5_lag3': pm25_lag1,
            'pm2_5_lag7': pm25_lag1,
            'pm2_5_mean_7d': pm25_lag1,
            'pm10_pm25_ratio': pm10_est / (pm25_lag1 + 1),
            'neighbor_pm10_mean': pm10_est,
            'neighbor_pm25_mean': pm25_lag1,
            'neighbor_pm10_std': month_stats['pm10_std'] * 0.5,
            'neighbor_pm25_std': month_stats['pm25_std'] * 0.5,
            'neighbor_local_pm10_ratio': 1.0,
            'neighbor_local_pm25_ratio': 1.0,
            'temp_c': month_stats['temp_mean'],
            'humidity_pct': month_stats['humidity_mean'],
            'wind_ms': month_stats['wind_mean'],
            'pressure_hpa': month_stats['pressure_mean'],
            'day_sin': np.sin(2 * np.pi * forecast_date.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * forecast_date.dayofyear / 365),
            'is_dry_season': 1.0 if month in [11, 12, 1, 2, 3] else 0.0,
            'lat_norm': row['lat_norm'],
            'lon_norm': row['lon_norm'],
        }
        
        # Predict
        X = np.array([[feat_row[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        idx_t = torch.tensor([row['station_idx']], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            pred = model(X_t, idx_t).cpu().numpy()[0]
        
        # Use prediction directly without over-adjustment
        # Add small seasonal variation based on historical patterns
        seasonal_mean = month_stats['pm25_mean']
        pred_adjusted = max(0, pred * 0.3 + seasonal_mean * 0.7)  # Blend prediction with seasonal
        
        # Update current values for next iteration
        current_pm25[sid] = pred_adjusted
        current_pm10[sid] = pm10_est
        
        forecasts.append({
            'stationID': sid,
            'lat': row['lat'],
            'lon': row['lon'],
            'date': forecast_date,
            'pm2_5_pred': pred_adjusted,
            'month': month,
        })

forecast_df = pd.DataFrame(forecasts)
print(f"   Generated {len(forecast_df):,} forecasts")

# ============================================================================
# Visualization
# ============================================================================
print("\n[5/5] Creating visualizations...")

reports_dir = Path('../data/reports/forecast_june')
reports_dir.mkdir(parents=True, exist_ok=True)

# Historical daily mean
daily_hist = df.groupby('date')['pm2_5'].agg(['mean', 'std', 'min', 'max']).reset_index()
daily_hist.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']

# Forecast daily mean
daily_forecast = forecast_df.groupby('date')['pm2_5_pred'].agg(['mean', 'std', 'min', 'max']).reset_index()
daily_forecast.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']

# ============================================================================
# PLOT 1: Full Timeline with Forecast
# ============================================================================
fig, ax = plt.subplots(figsize=(22, 8))

# Historical
ax.fill_between(daily_hist['date'], daily_hist['pm2_5_min'], daily_hist['pm2_5_max'], 
                alpha=0.15, color='steelblue', label='Historical Range')
ax.plot(daily_hist['date'], daily_hist['pm2_5_mean'], 'b-', lw=1.5, label='Historical Mean')

# Forecast
ax.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_mean'] - daily_forecast['pm2_5_std'], 
                daily_forecast['pm2_5_mean'] + daily_forecast['pm2_5_std'], 
                alpha=0.3, color='red', label='Forecast ±1 Std')
ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=2, label='Forecast Mean')

# WHO guideline
ax.axhline(35, color='orange', linestyle='--', lw=2, label='WHO Guideline')

# Today line
ax.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax.text(latest_date, ax.get_ylim()[1] * 0.95, ' Today', fontsize=12, fontweight='bold', va='top')

# Styling
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=14)
ax.set_title('Bangkok PM2.5: Historical + Forecast to June 2026', fontsize=18, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / '1_full_timeline_june.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 1_full_timeline_june.png")

# ============================================================================
# PLOT 2: Forecast Detail (Mar-Jun 2026)
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 8))

# Last 30 days of historical
hist_recent = daily_hist[daily_hist['date'] >= latest_date - timedelta(days=30)]

# Plot
ax.plot(hist_recent['date'], hist_recent['pm2_5_mean'], 'b-o', lw=2, markersize=4, label='Historical')
ax.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_mean'] - daily_forecast['pm2_5_std'], 
                daily_forecast['pm2_5_mean'] + daily_forecast['pm2_5_std'], 
                alpha=0.3, color='red')
ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=2, label='Forecast')

# Connect historical to forecast
connect_dates = [hist_recent['date'].iloc[-1], daily_forecast['date'].iloc[0]]
connect_values = [hist_recent['pm2_5_mean'].iloc[-1], daily_forecast['pm2_5_mean'].iloc[0]]
ax.plot(connect_dates, connect_values, 'g--', lw=2, alpha=0.7)

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
ax.set_title('PM2.5 Forecast: March - June 2026', fontsize=18, fontweight='bold')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(reports_dir / '2_forecast_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 2_forecast_detail.png")

# ============================================================================
# PLOT 3: Monthly Summary
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
ax1.set_title('Monthly Forecast Average', fontsize=14, fontweight='bold')
ax1.legend()

for bar, val in zip(bars, monthly_forecast['mean']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
             ha='center', fontsize=12, fontweight='bold')

# 3b. Seasonal comparison (historical vs forecast)
ax2 = axes[0, 1]
hist_monthly = df.groupby(df['date'].dt.month)['pm2_5'].mean()
forecast_monthly = forecast_df.groupby('month')['pm2_5_pred'].mean()

x = np.arange(4)
width = 0.35
bars1 = ax2.bar(x - width/2, [hist_monthly.get(m, 0) for m in [3, 4, 5, 6]], width, 
                label='Historical Avg', color='steelblue', edgecolor='black')
bars2 = ax2.bar(x + width/2, [forecast_monthly.get(m, 0) for m in [3, 4, 5, 6]], width, 
                label='2026 Forecast', color='coral', edgecolor='black')
ax2.axhline(35, color='red', linestyle='--', lw=2)
ax2.set_xticks(x)
ax2.set_xticklabels(['Mar', 'Apr', 'May', 'Jun'])
ax2.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax2.set_title('Historical vs Forecast (Mar-Jun)', fontsize=14, fontweight='bold')
ax2.legend()

# 3c. AQI Distribution
ax3 = axes[1, 0]
forecast_df['aqi'] = forecast_df['pm2_5_pred'].apply(get_aqi_level)
aqi_counts = forecast_df['aqi'].value_counts().reindex(AQI_LABELS).fillna(0)
colors_pie = [AQI_COLORS[l] for l in aqi_counts.index if l in AQI_COLORS]
ax3.pie(aqi_counts.values, labels=aqi_counts.index, colors=colors_pie, autopct='%1.1f%%',
        explode=[0.02]*len(aqi_counts))
ax3.set_title('Forecast AQI Distribution (Mar-Jun)', fontsize=14, fontweight='bold')

# 3d. Trend summary
ax4 = axes[1, 1]
current_mean = daily_hist[daily_hist['date'] == latest_date]['pm2_5_mean'].values[0]
mar_mean = forecast_df[forecast_df['month'] == 3]['pm2_5_pred'].mean()
apr_mean = forecast_df[forecast_df['month'] == 4]['pm2_5_pred'].mean()
may_mean = forecast_df[forecast_df['month'] == 5]['pm2_5_pred'].mean()
jun_mean = forecast_df[forecast_df['month'] == 6]['pm2_5_pred'].mean()

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
plt.savefig(reports_dir / '3_monthly_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 3_monthly_summary.png")

# ============================================================================
# PLOT 4: Comprehensive Dashboard
# ============================================================================
fig = plt.figure(figsize=(22, 16))
fig.suptitle('🏙️ Bangkok PM2.5 Long-term Forecast: Feb 2026 → June 2026', 
             fontsize=22, fontweight='bold', y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

# 4a. Full timeline
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(daily_hist['date'], daily_hist['pm2_5_min'], daily_hist['pm2_5_max'], 
                alpha=0.1, color='steelblue')
ax1.plot(daily_hist['date'], daily_hist['pm2_5_mean'], 'b-', lw=1, alpha=0.7, label='Historical')
ax1.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=2, label='Forecast')
ax1.axhline(35, color='orange', linestyle='--', lw=2)
ax1.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax1.set_title('Full Timeline: 2023-2026 + Forecast to June', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# 4b. Forecast zoom
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-', lw=2)
ax2.fill_between(daily_forecast['date'], 
                daily_forecast['pm2_5_mean'] - daily_forecast['pm2_5_std'], 
                daily_forecast['pm2_5_mean'] + daily_forecast['pm2_5_std'], 
                alpha=0.3, color='red')
ax2.axhline(35, color='orange', linestyle='--', lw=2)
ax2.set_ylabel('PM2.5 (µg/m³)', fontsize=11)
ax2.set_title('Forecast Detail (Mar-Jun 2026)', fontsize=12, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 4c. Current AQI
ax3 = fig.add_subplot(gs[1, 2])
ax3.text(0.5, 0.65, f'{current_mean:.1f}', fontsize=40, fontweight='bold', 
         ha='center', va='center', transform=ax3.transAxes, color=get_aqi_color(current_mean))
ax3.text(0.5, 0.4, 'µg/m³', fontsize=16, ha='center', va='center', transform=ax3.transAxes)
ax3.text(0.5, 0.2, get_aqi_level(current_mean), fontsize=18, fontweight='bold', 
         ha='center', va='center', transform=ax3.transAxes, color=get_aqi_color(current_mean))
ax3.axis('off')
ax3.set_title('Current (Feb 28)', fontsize=12, fontweight='bold')

# 4d. June forecast
ax4 = fig.add_subplot(gs[1, 3])
ax4.text(0.5, 0.65, f'{jun_mean:.1f}', fontsize=40, fontweight='bold', 
         ha='center', va='center', transform=ax4.transAxes, color=get_aqi_color(jun_mean))
ax4.text(0.5, 0.4, 'µg/m³', fontsize=16, ha='center', va='center', transform=ax4.transAxes)
ax4.text(0.5, 0.2, get_aqi_level(jun_mean), fontsize=18, fontweight='bold', 
         ha='center', va='center', transform=ax4.transAxes, color=get_aqi_color(jun_mean))
ax4.axis('off')
ax4.set_title('June 2026 Forecast', fontsize=12, fontweight='bold')

# 4e. Monthly bars
ax5 = fig.add_subplot(gs[2, 0])
bars = ax5.bar(monthly_forecast['month_name'], monthly_forecast['mean'], 
               color=[get_aqi_color(v) for v in monthly_forecast['mean']], edgecolor='black')
ax5.axhline(35, color='red', linestyle='--', lw=1.5)
ax5.set_ylabel('PM2.5', fontsize=10)
ax5.set_title('Monthly Forecast', fontsize=12, fontweight='bold')

# 4f. Trend
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(['Feb', 'Mar', 'Apr', 'May', 'Jun'], values_trend, 'g-o', lw=2, markersize=10)
ax6.axhline(35, color='red', linestyle='--', lw=1.5)
ax6.set_ylabel('PM2.5', fontsize=10)
ax6.set_title('Trend', fontsize=12, fontweight='bold')

# 4g. AQI pie
ax7 = fig.add_subplot(gs[2, 2])
ax7.pie(aqi_counts.values, labels=aqi_counts.index, colors=colors_pie, autopct='%1.0f%%',
        textprops={'fontsize': 9})
ax7.set_title('AQI Distribution', fontsize=12, fontweight='bold')

# 4h. Summary
ax8 = fig.add_subplot(gs[2, 3])
summary_text = f"""
FORECAST SUMMARY
────────────────────
Period: Mar-Jun 2026
Days: {n_days}

Monthly Averages:
  Mar: {mar_mean:.1f} µg/m³
  Apr: {apr_mean:.1f} µg/m³
  May: {may_mean:.1f} µg/m³
  Jun: {jun_mean:.1f} µg/m³

Trend: {'📉 Improving' if jun_mean < current_mean else '📈 Worsening'}
"""
ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', va='center', transform=ax8.transAxes)
ax8.axis('off')
ax8.set_title('Summary', fontsize=12, fontweight='bold')

plt.savefig(reports_dir / '4_dashboard_june.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 4_dashboard_june.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("LONG-TERM FORECAST COMPLETE")
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

TREND: {'📉 PM2.5 expected to DECREASE (Wet season approaching)' if jun_mean < current_mean else '📈 PM2.5 expected to increase'}

FILES CREATED:
  1. 1_full_timeline_june.png  - Full timeline 2023-June 2026
  2. 2_forecast_detail.png     - Forecast detail Mar-Jun
  3. 3_monthly_summary.png     - Monthly analysis
  4. 4_dashboard_june.png      - Comprehensive dashboard

LOCATION: {reports_dir}
""")

# Save forecast data
forecast_df.to_csv(reports_dir / 'forecast_to_june.csv', index=False)
print(f"   Forecast data saved to: {reports_dir / 'forecast_to_june.csv'}")
