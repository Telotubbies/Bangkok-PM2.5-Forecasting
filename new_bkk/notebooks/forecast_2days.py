"""
2-Day Ahead PM2.5 Forecast with Graph Visualization
Uses trained STC-HGAT model to predict PM2.5 for next 2 days
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
AQI_COLORS = {
    'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy-S': '#FF7E00',
    'Unhealthy': '#FF0000', 'Very Unhealthy': '#8F3F97'
}
AQI_BINS = [0, 15, 35, 55, 75, 100]

def get_aqi_color(pm25):
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return list(AQI_COLORS.values())[i]
    return AQI_COLORS['Very Unhealthy']

def get_aqi_level(pm25):
    labels = ['Good', 'Moderate', 'Unhealthy-S', 'Unhealthy', 'Very Unhealthy']
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return labels[i]
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

# Fit scaler on training data
dates = sorted(df['date'].unique())
n = len(dates)
t1 = int(n * 0.70)
train_df = df[df['date'].isin(dates[:t1])].copy()

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

# ============================================================================
# 2-Day Ahead Forecast
# ============================================================================
print("\n[4/5] Generating 2-day ahead forecast...")

latest_date = df['date'].max()
forecast_dates = [latest_date + timedelta(days=1), latest_date + timedelta(days=2)]

print(f"   Latest data: {latest_date.strftime('%Y-%m-%d')}")
print(f"   Forecasting: {forecast_dates[0].strftime('%Y-%m-%d')} and {forecast_dates[1].strftime('%Y-%m-%d')}")

# Get latest data for each station
latest_data = df[df['date'] == latest_date].copy()

# Prepare forecast data
forecasts = []

for day_ahead in [1, 2]:
    forecast_date = latest_date + timedelta(days=day_ahead)
    
    for _, row in latest_data.iterrows():
        # Create feature row for forecast
        feat_row = {}
        
        # Shift lags appropriately
        if day_ahead == 1:
            feat_row['pm10'] = row['pm10']  # Assume similar PM10
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
            # Use day 1 forecast as lag1
            day1_forecast = [f for f in forecasts if f['stationID'] == row['stationID'] and f['day_ahead'] == 1]
            if day1_forecast:
                pm25_day1 = day1_forecast[0]['pm2_5_pred']
            else:
                pm25_day1 = row['pm2_5']
            
            feat_row['pm10'] = row['pm10']
            feat_row['pm10_lag1'] = row['pm10']
            feat_row['pm10_lag2'] = row['pm10_lag1']
            feat_row['pm10_lag3'] = row['pm10_lag2']
            feat_row['pm10_lag7'] = row['pm10_lag7']
            feat_row['pm10_mean_7d'] = row['pm10_mean_7d']
            feat_row['pm2_5_lag1'] = pm25_day1
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
        feat_row['day_sin'] = np.sin(2 * np.pi * forecast_date.dayofyear / 365)
        feat_row['day_cos'] = np.cos(2 * np.pi * forecast_date.dayofyear / 365)
        feat_row['is_dry_season'] = 1.0 if forecast_date.month in [11, 12, 1, 2, 3] else 0.0
        feat_row['lat_norm'] = row['lat_norm']
        feat_row['lon_norm'] = row['lon_norm']
        
        # Create feature vector
        X = np.array([[feat_row[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        idx_t = torch.tensor([row['station_idx']], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            pred = model(X_t, idx_t).cpu().numpy()[0]
        
        forecasts.append({
            'stationID': row['stationID'],
            'lat': row['lat'],
            'lon': row['lon'],
            'date': forecast_date,
            'day_ahead': day_ahead,
            'pm2_5_pred': max(0, pred),
            'pm2_5_current': row['pm2_5'],
        })

forecast_df = pd.DataFrame(forecasts)
print(f"   Generated {len(forecast_df)} forecasts")

# ============================================================================
# Visualization
# ============================================================================
print("\n[5/5] Creating forecast visualizations...")

reports_dir = Path('../data/reports/forecast_2days')
reports_dir.mkdir(parents=True, exist_ok=True)

# Get historical data for context (last 14 days)
hist_start = latest_date - timedelta(days=13)
hist_df = df[df['date'] >= hist_start].copy()

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Bangkok PM2.5 2-Day Ahead Forecast', fontsize=20, fontweight='bold', y=0.98)

# 1. Overall forecast summary
ax1 = fig.add_subplot(2, 2, 1)
daily_hist = hist_df.groupby('date')['pm2_5'].mean()
daily_forecast = forecast_df.groupby('date')['pm2_5_pred'].mean()

# Combine for plotting
all_dates = list(daily_hist.index) + list(daily_forecast.index)
all_values = list(daily_hist.values) + [np.nan, np.nan]
forecast_values = [np.nan] * len(daily_hist) + list(daily_forecast.values)

ax1.plot(daily_hist.index, daily_hist.values, 'b-o', lw=2, markersize=6, label='Historical')
ax1.plot(daily_forecast.index, daily_forecast.values, 'r--s', lw=2, markersize=8, label='Forecast')
ax1.fill_between(daily_forecast.index, daily_forecast.values * 0.9, daily_forecast.values * 1.1, 
                 alpha=0.3, color='red', label='±10% Uncertainty')
ax1.axhline(35, color='orange', linestyle=':', lw=2, label='WHO Guideline')
ax1.axvline(latest_date, color='gray', linestyle='--', lw=1.5, alpha=0.7)
ax1.text(latest_date, ax1.get_ylim()[1], ' Today', fontsize=10, va='top')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax1.set_title('Bangkok Average PM2.5: History & Forecast', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Forecast by station (top 10)
ax2 = fig.add_subplot(2, 2, 2)
day1_forecast = forecast_df[forecast_df['day_ahead'] == 1].sort_values('pm2_5_pred', ascending=False).head(10)
colors = [get_aqi_color(v) for v in day1_forecast['pm2_5_pred']]
bars = ax2.barh(range(len(day1_forecast)), day1_forecast['pm2_5_pred'], color=colors, edgecolor='black')
ax2.set_yticks(range(len(day1_forecast)))
ax2.set_yticklabels(day1_forecast['stationID'])
ax2.axvline(35, color='red', linestyle='--', lw=2)
ax2.set_xlabel('Predicted PM2.5 (µg/m³)', fontsize=12)
ax2.set_title(f'Top 10 Stations - Day 1 ({forecast_dates[0].strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for bar, val in zip(bars, day1_forecast['pm2_5_pred']):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=10)

# 3. Day 1 vs Day 2 comparison
ax3 = fig.add_subplot(2, 2, 3)
day1_mean = forecast_df[forecast_df['day_ahead'] == 1]['pm2_5_pred'].mean()
day2_mean = forecast_df[forecast_df['day_ahead'] == 2]['pm2_5_pred'].mean()
current_mean = latest_data['pm2_5'].mean()

x = ['Current\n' + latest_date.strftime('%b %d'), 
     'Day 1\n' + forecast_dates[0].strftime('%b %d'), 
     'Day 2\n' + forecast_dates[1].strftime('%b %d')]
y = [current_mean, day1_mean, day2_mean]
colors = [get_aqi_color(v) for v in y]

bars = ax3.bar(x, y, color=colors, edgecolor='black', width=0.6)
ax3.axhline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')
ax3.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax3.set_title('PM2.5 Trend: Current → 2-Day Forecast', fontsize=14, fontweight='bold')
ax3.legend()

for bar, val in zip(bars, y):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add trend arrow
if day2_mean > current_mean:
    trend = '📈 Increasing'
    trend_color = 'red'
elif day2_mean < current_mean:
    trend = '📉 Decreasing'
    trend_color = 'green'
else:
    trend = '➡️ Stable'
    trend_color = 'gray'
ax3.text(0.5, 0.95, trend, transform=ax3.transAxes, fontsize=14, fontweight='bold',
         ha='center', va='top', color=trend_color)

# 4. Forecast distribution
ax4 = fig.add_subplot(2, 2, 4)
day1_preds = forecast_df[forecast_df['day_ahead'] == 1]['pm2_5_pred']
day2_preds = forecast_df[forecast_df['day_ahead'] == 2]['pm2_5_pred']

ax4.hist(day1_preds, bins=20, alpha=0.6, label=f'Day 1 (mean={day1_preds.mean():.1f})', color='steelblue', edgecolor='white')
ax4.hist(day2_preds, bins=20, alpha=0.6, label=f'Day 2 (mean={day2_preds.mean():.1f})', color='coral', edgecolor='white')
ax4.axvline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')
ax4.set_xlabel('Predicted PM2.5 (µg/m³)', fontsize=12)
ax4.set_ylabel('Number of Stations', fontsize=12)
ax4.set_title('Forecast Distribution by Day', fontsize=14, fontweight='bold')
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(reports_dir / 'forecast_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: forecast_summary.png")

# ============================================================================
# Detailed Station Forecast Graph
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('PM2.5 Forecast by Station (Sample)', fontsize=18, fontweight='bold', y=0.98)

# Select 9 sample stations
sample_stations = latest_data.nlargest(9, 'pm2_5')['stationID'].tolist()

for idx, (ax, station) in enumerate(zip(axes.flat, sample_stations)):
    # Historical data
    station_hist = hist_df[hist_df['stationID'] == station].sort_values('date')
    station_forecast = forecast_df[forecast_df['stationID'] == station].sort_values('date')
    
    # Plot
    ax.plot(station_hist['date'], station_hist['pm2_5'], 'b-o', lw=1.5, markersize=4, label='Historical')
    ax.plot(station_forecast['date'], station_forecast['pm2_5_pred'], 'r--s', lw=2, markersize=8, label='Forecast')
    
    # Fill uncertainty
    ax.fill_between(station_forecast['date'], 
                    station_forecast['pm2_5_pred'] * 0.85, 
                    station_forecast['pm2_5_pred'] * 1.15,
                    alpha=0.3, color='red')
    
    ax.axhline(35, color='orange', linestyle=':', lw=1.5)
    ax.axvline(latest_date, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_title(f'{station}', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(reports_dir / 'forecast_by_station.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: forecast_by_station.png")

# ============================================================================
# Forecast Table
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Create summary table
summary_data = []
for day in [1, 2]:
    day_df = forecast_df[forecast_df['day_ahead'] == day]
    summary_data.append({
        'Date': forecast_dates[day-1].strftime('%Y-%m-%d'),
        'Day': f'Day {day}',
        'Mean PM2.5': f"{day_df['pm2_5_pred'].mean():.1f} µg/m³",
        'Max PM2.5': f"{day_df['pm2_5_pred'].max():.1f} µg/m³",
        'Min PM2.5': f"{day_df['pm2_5_pred'].min():.1f} µg/m³",
        'Stations > WHO': f"{(day_df['pm2_5_pred'] > 35).sum()} / {len(day_df)}",
        'AQI Level': get_aqi_level(day_df['pm2_5_pred'].mean()),
    })

summary_table = pd.DataFrame(summary_data)

# Draw table
table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc='center',
    loc='center',
    colColours=['lightsteelblue'] * len(summary_table.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

ax.set_title('2-Day Forecast Summary', fontsize=18, fontweight='bold', pad=20)

plt.savefig(reports_dir / 'forecast_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: forecast_table.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("2-DAY FORECAST COMPLETE")
print("="*70)

print(f"""
FORECAST SUMMARY:
─────────────────────────────────────────────────────────────────────────
Current Date: {latest_date.strftime('%Y-%m-%d')}
Current Mean PM2.5: {current_mean:.1f} µg/m³

Day 1 ({forecast_dates[0].strftime('%Y-%m-%d')}):
  - Mean PM2.5: {day1_mean:.1f} µg/m³
  - Max PM2.5:  {forecast_df[forecast_df['day_ahead']==1]['pm2_5_pred'].max():.1f} µg/m³
  - AQI Level:  {get_aqi_level(day1_mean)}

Day 2 ({forecast_dates[1].strftime('%Y-%m-%d')}):
  - Mean PM2.5: {day2_mean:.1f} µg/m³
  - Max PM2.5:  {forecast_df[forecast_df['day_ahead']==2]['pm2_5_pred'].max():.1f} µg/m³
  - AQI Level:  {get_aqi_level(day2_mean)}

Trend: {trend}

FILES CREATED:
  1. forecast_summary.png     - Overall forecast visualization
  2. forecast_by_station.png  - Station-level forecasts
  3. forecast_table.png       - Summary table

LOCATION: {reports_dir}
""")

# Save forecast data
forecast_df.to_csv(reports_dir / 'forecast_data.csv', index=False)
print(f"   Forecast data saved to: {reports_dir / 'forecast_data.csv'}")
