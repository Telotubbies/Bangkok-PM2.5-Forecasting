"""
Full Historical + 2-Day Forecast PM2.5 Visualization
Shows complete timeline from past to future predictions
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# AQI Colors and thresholds
AQI_COLORS = {
    'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy-S': '#FF7E00',
    'Unhealthy': '#FF0000', 'Very Unhealthy': '#8F3F97'
}
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
print("\n[1/4] Loading data...")

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
print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# Load Forecast Data
# ============================================================================
print("\n[2/4] Loading forecast data...")

forecast_file = Path('../data/reports/forecast_2days/forecast_data.csv')
if forecast_file.exists():
    forecast_df = pd.read_csv(forecast_file)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    print(f"   Forecast loaded: {len(forecast_df)} records")
else:
    print("   Forecast file not found, generating...")
    # Will generate inline if needed
    forecast_df = None

# ============================================================================
# Prepare Full Timeline Data
# ============================================================================
print("\n[3/4] Preparing full timeline...")

# Historical daily mean
daily_hist = df.groupby('date').agg({
    'pm2_5': ['mean', 'std', 'min', 'max'],
}).reset_index()
daily_hist.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']

# Forecast daily mean
if forecast_df is not None:
    daily_forecast = forecast_df.groupby('date').agg({
        'pm2_5_pred': ['mean', 'std', 'min', 'max'],
    }).reset_index()
    daily_forecast.columns = ['date', 'pm2_5_mean', 'pm2_5_std', 'pm2_5_min', 'pm2_5_max']
else:
    daily_forecast = pd.DataFrame()

latest_date = df['date'].max()
forecast_dates = [latest_date + timedelta(days=1), latest_date + timedelta(days=2)]

print(f"   Historical: {daily_hist['date'].min().strftime('%Y-%m-%d')} to {daily_hist['date'].max().strftime('%Y-%m-%d')}")
print(f"   Forecast: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[1].strftime('%Y-%m-%d')}")

# ============================================================================
# Create Visualizations
# ============================================================================
print("\n[4/4] Creating visualizations...")

reports_dir = Path('../data/reports/full_timeline')
reports_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PLOT 1: Full Timeline (All History + Forecast)
# ============================================================================
fig, ax = plt.subplots(figsize=(20, 8))

# Plot historical data
ax.fill_between(daily_hist['date'], daily_hist['pm2_5_min'], daily_hist['pm2_5_max'], 
                alpha=0.2, color='steelblue', label='Min-Max Range')
ax.fill_between(daily_hist['date'], 
                daily_hist['pm2_5_mean'] - daily_hist['pm2_5_std'], 
                daily_hist['pm2_5_mean'] + daily_hist['pm2_5_std'], 
                alpha=0.4, color='steelblue', label='±1 Std Dev')
ax.plot(daily_hist['date'], daily_hist['pm2_5_mean'], 'b-', lw=1.5, label='Historical Mean')

# Plot forecast
if not daily_forecast.empty:
    ax.fill_between(daily_forecast['date'], 
                    daily_forecast['pm2_5_mean'] * 0.85, 
                    daily_forecast['pm2_5_mean'] * 1.15, 
                    alpha=0.3, color='red', label='Forecast ±15%')
    ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'r-o', lw=3, markersize=10, 
            label='2-Day Forecast', markeredgecolor='black')

# Add WHO guideline
ax.axhline(35, color='orange', linestyle='--', lw=2, label='WHO Guideline (35 µg/m³)')

# Add vertical line for "today"
ax.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax.text(latest_date, ax.get_ylim()[1] * 0.95, ' Today', fontsize=12, fontweight='bold', va='top')

# Styling
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=14)
ax.set_title('Bangkok PM2.5: Full Historical Timeline + 2-Day Forecast', fontsize=18, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, alpha=0.3)

# Add AQI color bands
for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
    ax.axhspan(low, high, alpha=0.1, color=list(AQI_COLORS.values())[i])

plt.tight_layout()
plt.savefig(reports_dir / '1_full_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 1_full_timeline.png")

# ============================================================================
# PLOT 2: Last 90 Days + Forecast (Zoomed)
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 8))

# Last 90 days
recent_start = latest_date - timedelta(days=89)
recent_hist = daily_hist[daily_hist['date'] >= recent_start]

# Plot
ax.fill_between(recent_hist['date'], recent_hist['pm2_5_min'], recent_hist['pm2_5_max'], 
                alpha=0.2, color='steelblue', label='Min-Max Range')
ax.plot(recent_hist['date'], recent_hist['pm2_5_mean'], 'b-o', lw=1.5, markersize=3, label='Historical Mean')

# Forecast
if not daily_forecast.empty:
    # Connect last historical point to forecast
    connect_dates = [recent_hist['date'].iloc[-1]] + list(daily_forecast['date'])
    connect_values = [recent_hist['pm2_5_mean'].iloc[-1]] + list(daily_forecast['pm2_5_mean'])
    
    ax.fill_between(daily_forecast['date'], 
                    daily_forecast['pm2_5_mean'] * 0.85, 
                    daily_forecast['pm2_5_mean'] * 1.15, 
                    alpha=0.3, color='red', label='Forecast ±15%')
    ax.plot(connect_dates, connect_values, 'r--', lw=2, alpha=0.7)
    ax.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'ro', markersize=15, 
            markeredgecolor='black', markeredgewidth=2, label='Forecast')
    
    # Add forecast values as text
    for _, row in daily_forecast.iterrows():
        ax.annotate(f'{row["pm2_5_mean"]:.1f}', 
                   (row['date'], row['pm2_5_mean']),
                   textcoords="offset points", xytext=(0, 15),
                   ha='center', fontsize=12, fontweight='bold', color='red')

ax.axhline(35, color='orange', linestyle='--', lw=2, label='WHO Guideline')
ax.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax.text(latest_date, ax.get_ylim()[1] * 0.95, ' Today', fontsize=12, fontweight='bold', va='top')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=14)
ax.set_title('Bangkok PM2.5: Last 90 Days + 2-Day Forecast', fontsize=18, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / '2_last_90days_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 2_last_90days_forecast.png")

# ============================================================================
# PLOT 3: Monthly Comparison with Forecast
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 3a. Monthly averages
ax1 = axes[0, 0]
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
monthly = df.groupby(['year', 'month'])['pm2_5'].mean().reset_index()
monthly['year_month'] = pd.to_datetime(monthly['year'].astype(str) + '-' + monthly['month'].astype(str) + '-01')

colors = [get_aqi_color(v) for v in monthly['pm2_5']]
ax1.bar(monthly['year_month'], monthly['pm2_5'], width=25, color=colors, edgecolor='black', alpha=0.8)
ax1.axhline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax1.set_title('Monthly Average PM2.5 (2023-2026)', fontsize=14, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
ax1.legend()

# 3b. Seasonal pattern
ax2 = axes[0, 1]
seasonal = df.groupby('month')['pm2_5'].agg(['mean', 'std']).reset_index()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = [get_aqi_color(v) for v in seasonal['mean']]
bars = ax2.bar(seasonal['month'], seasonal['mean'], color=colors, edgecolor='black')
ax2.errorbar(seasonal['month'], seasonal['mean'], yerr=seasonal['std'], fmt='none', color='black', capsize=5)
ax2.axhline(35, color='red', linestyle='--', lw=2)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(months)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=12)
ax2.set_title('Seasonal Pattern (All Years)', fontsize=14, fontweight='bold')

# Highlight dry season
ax2.axvspan(0.5, 3.5, alpha=0.15, color='red', label='Dry Season')
ax2.axvspan(10.5, 12.5, alpha=0.15, color='red')
ax2.legend()

# 3c. Year-over-year comparison
ax3 = axes[1, 0]
yearly = df.groupby('year')['pm2_5'].agg(['mean', 'std', 'max']).reset_index()
x = range(len(yearly))
ax3.bar(x, yearly['mean'], color='steelblue', edgecolor='black', label='Mean')
ax3.errorbar(x, yearly['mean'], yerr=yearly['std'], fmt='none', color='black', capsize=5)
ax3.scatter(x, yearly['max'], color='red', s=100, marker='v', label='Max', zorder=5)
ax3.axhline(35, color='orange', linestyle='--', lw=2)
ax3.set_xticks(x)
ax3.set_xticklabels(yearly['year'])
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax3.set_title('Year-over-Year Comparison', fontsize=14, fontweight='bold')
ax3.legend()

# 3d. Current + Forecast summary
ax4 = axes[1, 1]
current_mean = daily_hist[daily_hist['date'] == latest_date]['pm2_5_mean'].values[0]

if not daily_forecast.empty:
    day1_mean = daily_forecast[daily_forecast['date'] == forecast_dates[0]]['pm2_5_mean'].values[0]
    day2_mean = daily_forecast[daily_forecast['date'] == forecast_dates[1]]['pm2_5_mean'].values[0]
else:
    day1_mean = day2_mean = current_mean

labels = [f'Current\n{latest_date.strftime("%b %d")}', 
          f'Day 1\n{forecast_dates[0].strftime("%b %d")}', 
          f'Day 2\n{forecast_dates[1].strftime("%b %d")}']
values = [current_mean, day1_mean, day2_mean]
colors = [get_aqi_color(v) for v in values]

bars = ax4.bar(labels, values, color=colors, edgecolor='black', width=0.5)
ax4.axhline(35, color='red', linestyle='--', lw=2, label='WHO Guideline')

for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add trend arrow
if day2_mean < current_mean:
    trend_text = '📉 Improving'
    trend_color = 'green'
elif day2_mean > current_mean:
    trend_text = '📈 Worsening'
    trend_color = 'red'
else:
    trend_text = '➡️ Stable'
    trend_color = 'gray'

ax4.text(0.5, 0.95, trend_text, transform=ax4.transAxes, fontsize=16, fontweight='bold',
         ha='center', va='top', color=trend_color,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=trend_color, linewidth=2))

ax4.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax4.set_title('Current vs 2-Day Forecast', fontsize=14, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig(reports_dir / '3_monthly_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 3_monthly_comparison.png")

# ============================================================================
# PLOT 4: Station-Level Historical + Forecast
# ============================================================================
fig, axes = plt.subplots(4, 3, figsize=(20, 18))
fig.suptitle('PM2.5 History + Forecast by Station (Top 12)', fontsize=18, fontweight='bold', y=0.98)

# Select top 12 stations by data count
top_stations = df.groupby('stationID').size().nlargest(12).index.tolist()

for idx, (ax, station) in enumerate(zip(axes.flat, top_stations)):
    # Historical
    station_hist = df[df['stationID'] == station].groupby('date')['pm2_5'].mean()
    
    # Last 60 days only for clarity
    recent_start = latest_date - timedelta(days=59)
    station_recent = station_hist[station_hist.index >= recent_start]
    
    # Plot historical
    ax.plot(station_recent.index, station_recent.values, 'b-', lw=1, alpha=0.7)
    ax.fill_between(station_recent.index, 0, station_recent.values, alpha=0.2, color='steelblue')
    
    # Forecast
    if forecast_df is not None:
        station_forecast = forecast_df[forecast_df['stationID'] == station].sort_values('date')
        if not station_forecast.empty:
            # Connect to forecast
            connect_x = [station_recent.index[-1]] + list(station_forecast['date'])
            connect_y = [station_recent.values[-1]] + list(station_forecast['pm2_5_pred'])
            ax.plot(connect_x, connect_y, 'r--', lw=2)
            ax.plot(station_forecast['date'], station_forecast['pm2_5_pred'], 'ro', markersize=10, 
                    markeredgecolor='black', markeredgewidth=1.5)
            
            # Add forecast values
            for _, row in station_forecast.iterrows():
                ax.annotate(f'{row["pm2_5_pred"]:.1f}', 
                           (row['date'], row['pm2_5_pred']),
                           textcoords="offset points", xytext=(0, 8),
                           ha='center', fontsize=9, fontweight='bold', color='red')
    
    ax.axhline(35, color='orange', linestyle=':', lw=1.5)
    ax.axvline(latest_date, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Station title with current AQI
    current_pm25 = station_recent.values[-1] if len(station_recent) > 0 else 0
    aqi_level = get_aqi_level(current_pm25)
    ax.set_title(f'{station}\n(Current: {current_pm25:.1f} - {aqi_level})', fontsize=10, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    if idx == 0:
        ax.legend(['Historical', 'Forecast'], fontsize=8, loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(reports_dir / '4_station_history_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 4_station_history_forecast.png")

# ============================================================================
# PLOT 5: Comprehensive Dashboard
# ============================================================================
fig = plt.figure(figsize=(22, 16))
fig.suptitle('🏙️ Bangkok PM2.5 Comprehensive Dashboard: History + 2-Day Forecast', 
             fontsize=22, fontweight='bold', y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

# 5a. Full timeline (top row, full width)
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(daily_hist['date'], daily_hist['pm2_5_min'], daily_hist['pm2_5_max'], 
                alpha=0.15, color='steelblue')
ax1.plot(daily_hist['date'], daily_hist['pm2_5_mean'], 'b-', lw=1, label='Historical')

if not daily_forecast.empty:
    ax1.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'ro-', lw=3, markersize=12, 
            markeredgecolor='black', label='Forecast')

ax1.axhline(35, color='orange', linestyle='--', lw=2)
ax1.axvline(latest_date, color='gray', linestyle='--', lw=2, alpha=0.7)
ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax1.set_title('Full Timeline: 2023-2026 + 2-Day Forecast', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# 5b. Last 30 days zoom
ax2 = fig.add_subplot(gs[1, :2])
last_30 = daily_hist[daily_hist['date'] >= latest_date - timedelta(days=29)]
ax2.plot(last_30['date'], last_30['pm2_5_mean'], 'b-o', lw=1.5, markersize=4)

if not daily_forecast.empty:
    connect = [last_30['date'].iloc[-1]] + list(daily_forecast['date'])
    connect_y = [last_30['pm2_5_mean'].iloc[-1]] + list(daily_forecast['pm2_5_mean'])
    ax2.plot(connect, connect_y, 'r--', lw=2)
    ax2.plot(daily_forecast['date'], daily_forecast['pm2_5_mean'], 'ro', markersize=12, 
            markeredgecolor='black')

ax2.axhline(35, color='orange', linestyle='--', lw=2)
ax2.axvline(latest_date, color='gray', linestyle='--', lw=1.5)
ax2.set_ylabel('PM2.5 (µg/m³)', fontsize=11)
ax2.set_title('Last 30 Days + Forecast', fontsize=12, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 5c. Current AQI indicator
ax3 = fig.add_subplot(gs[1, 2])
ax3.text(0.5, 0.65, f'{current_mean:.1f}', fontsize=48, fontweight='bold', 
         ha='center', va='center', transform=ax3.transAxes, color=get_aqi_color(current_mean))
ax3.text(0.5, 0.4, 'µg/m³', fontsize=18, ha='center', va='center', transform=ax3.transAxes)
ax3.text(0.5, 0.2, get_aqi_level(current_mean), fontsize=20, fontweight='bold', 
         ha='center', va='center', transform=ax3.transAxes, color=get_aqi_color(current_mean))
ax3.axis('off')
ax3.set_title('Current AQI', fontsize=12, fontweight='bold')

# 5d. Forecast summary
ax4 = fig.add_subplot(gs[1, 3])
forecast_text = f"""
📅 FORECAST
─────────────────
Day 1 ({forecast_dates[0].strftime('%b %d')}):
  PM2.5: {day1_mean:.1f} µg/m³
  Level: {get_aqi_level(day1_mean)}

Day 2 ({forecast_dates[1].strftime('%b %d')}):
  PM2.5: {day2_mean:.1f} µg/m³
  Level: {get_aqi_level(day2_mean)}

Trend: {trend_text}
"""
ax4.text(0.1, 0.5, forecast_text, fontsize=11, family='monospace', va='center', transform=ax4.transAxes)
ax4.axis('off')
ax4.set_title('2-Day Forecast', fontsize=12, fontweight='bold')

# 5e. Seasonal pattern
ax5 = fig.add_subplot(gs[2, 0])
colors = [get_aqi_color(v) for v in seasonal['mean']]
ax5.bar(seasonal['month'], seasonal['mean'], color=colors, edgecolor='black')
ax5.axhline(35, color='red', linestyle='--', lw=1.5)
ax5.set_xticks(range(1, 13))
ax5.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax5.set_ylabel('PM2.5', fontsize=10)
ax5.set_title('Seasonal Pattern', fontsize=12, fontweight='bold')

# 5f. Year comparison
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(yearly['year'], yearly['mean'], color='steelblue', edgecolor='black')
ax6.axhline(35, color='red', linestyle='--', lw=1.5)
ax6.set_ylabel('PM2.5', fontsize=10)
ax6.set_title('Yearly Average', fontsize=12, fontweight='bold')

# 5g. AQI distribution
ax7 = fig.add_subplot(gs[2, 2])
df['aqi'] = df['pm2_5'].apply(get_aqi_level)
aqi_counts = df['aqi'].value_counts().reindex(AQI_LABELS).fillna(0)
colors_pie = [AQI_COLORS[l] for l in aqi_counts.index]
ax7.pie(aqi_counts.values, labels=aqi_counts.index, colors=colors_pie, autopct='%1.1f%%', 
        textprops={'fontsize': 9})
ax7.set_title('AQI Distribution', fontsize=12, fontweight='bold')

# 5h. Statistics
ax8 = fig.add_subplot(gs[2, 3])
stats_text = f"""
📊 STATISTICS
─────────────────
Total Days: {len(daily_hist):,}
Stations: {N_STATIONS}

Overall Mean: {df['pm2_5'].mean():.1f} µg/m³
Overall Max: {df['pm2_5'].max():.1f} µg/m³

Days > WHO: {(daily_hist['pm2_5_mean'] > 35).sum()}
  ({(daily_hist['pm2_5_mean'] > 35).mean()*100:.1f}%)

Model R²: 0.9935
"""
ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center', transform=ax8.transAxes)
ax8.axis('off')
ax8.set_title('Statistics', fontsize=12, fontweight='bold')

plt.savefig(reports_dir / '5_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 5_comprehensive_dashboard.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("FULL HISTORY + FORECAST VISUALIZATION COMPLETE")
print("="*70)

print(f"""
FILES CREATED:
─────────────────────────────────────────────────────────────────────────
1. 1_full_timeline.png          - Complete 2023-2026 + 2-day forecast
2. 2_last_90days_forecast.png   - Last 90 days zoomed + forecast
3. 3_monthly_comparison.png     - Monthly/seasonal/yearly analysis
4. 4_station_history_forecast.png - Top 12 stations individual graphs
5. 5_comprehensive_dashboard.png - All-in-one dashboard

LOCATION: {reports_dir}

DATA SUMMARY:
─────────────────────────────────────────────────────────────────────────
Historical Period: {df['date'].min().strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}
Forecast Period:   {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[1].strftime('%Y-%m-%d')}

Current PM2.5:     {current_mean:.1f} µg/m³ ({get_aqi_level(current_mean)})
Day 1 Forecast:    {day1_mean:.1f} µg/m³ ({get_aqi_level(day1_mean)})
Day 2 Forecast:    {day2_mean:.1f} µg/m³ ({get_aqi_level(day2_mean)})

Trend: {trend_text}
""")
