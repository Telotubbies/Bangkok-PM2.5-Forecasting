"""
Bangkok PM2.5 Map Visualization with District Boundaries
Creates maps with actual Bangkok geography and district overlays
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import requests

warnings.filterwarnings('ignore')

# AQI Colors
AQI_COLORS = {
    'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy-S': '#FF7E00',
    'Unhealthy': '#FF0000', 'Very Unhealthy': '#8F3F97', 'Hazardous': '#7E0023'
}
AQI_BINS = [0, 15, 35, 55, 75, 100, 200]

def get_aqi_color(pm25):
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return list(AQI_COLORS.values())[i]
    return AQI_COLORS['Hazardous']

def get_aqi_level(pm25):
    labels = ['Good', 'Moderate', 'Unhealthy-S', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    for i, (low, high) in enumerate(zip(AQI_BINS[:-1], AQI_BINS[1:])):
        if low <= pm25 < high:
            return labels[i]
    return 'Hazardous'

# ============================================================================
# Load Data
# ============================================================================
print("Loading data...")

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

station_meta = df.groupby('stationID')[['lat', 'lon']].first().reset_index()
N_STATIONS = len(station_meta)

print(f"Data: {len(df):,} rows, {N_STATIONS} stations")

reports_dir = Path('../data/reports/bangkok_maps')
reports_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Bangkok District GeoJSON (simplified boundaries)
# ============================================================================
print("\nCreating Bangkok district boundaries...")

# Bangkok districts with approximate center coordinates
BANGKOK_DISTRICTS = {
    'พระนคร': {'lat': 13.7563, 'lon': 100.4989, 'name_en': 'Phra Nakhon'},
    'ดุสิต': {'lat': 13.7747, 'lon': 100.5147, 'name_en': 'Dusit'},
    'หนองจอก': {'lat': 13.8556, 'lon': 100.8569, 'name_en': 'Nong Chok'},
    'บางรัก': {'lat': 13.7276, 'lon': 100.5234, 'name_en': 'Bang Rak'},
    'บางเขน': {'lat': 13.8731, 'lon': 100.5956, 'name_en': 'Bang Khen'},
    'บางกะปิ': {'lat': 13.7656, 'lon': 100.6478, 'name_en': 'Bang Kapi'},
    'ปทุมวัน': {'lat': 13.7456, 'lon': 100.5331, 'name_en': 'Pathum Wan'},
    'ป้อมปราบฯ': {'lat': 13.7544, 'lon': 100.5122, 'name_en': 'Pom Prap'},
    'พระโขนง': {'lat': 13.7022, 'lon': 100.6011, 'name_en': 'Phra Khanong'},
    'มีนบุรี': {'lat': 13.8128, 'lon': 100.7297, 'name_en': 'Min Buri'},
    'ลาดกระบัง': {'lat': 13.7289, 'lon': 100.7747, 'name_en': 'Lat Krabang'},
    'ยานนาวา': {'lat': 13.6956, 'lon': 100.5433, 'name_en': 'Yan Nawa'},
    'สัมพันธวงศ์': {'lat': 13.7378, 'lon': 100.5122, 'name_en': 'Samphanthawong'},
    'พญาไท': {'lat': 13.7778, 'lon': 100.5422, 'name_en': 'Phaya Thai'},
    'ธนบุรี': {'lat': 13.7189, 'lon': 100.4856, 'name_en': 'Thon Buri'},
    'บางกอกใหญ่': {'lat': 13.7378, 'lon': 100.4756, 'name_en': 'Bangkok Yai'},
    'ห้วยขวาง': {'lat': 13.7778, 'lon': 100.5789, 'name_en': 'Huai Khwang'},
    'คลองสาน': {'lat': 13.7289, 'lon': 100.5022, 'name_en': 'Khlong San'},
    'ตลิ่งชัน': {'lat': 13.7778, 'lon': 100.4356, 'name_en': 'Taling Chan'},
    'บางกอกน้อย': {'lat': 13.7656, 'lon': 100.4689, 'name_en': 'Bangkok Noi'},
    'บางขุนเทียน': {'lat': 13.6356, 'lon': 100.4356, 'name_en': 'Bang Khun Thian'},
    'ภาษีเจริญ': {'lat': 13.7156, 'lon': 100.4456, 'name_en': 'Phasi Charoen'},
    'หนองแขม': {'lat': 13.7056, 'lon': 100.3556, 'name_en': 'Nong Khaem'},
    'ราษฎร์บูรณะ': {'lat': 13.6756, 'lon': 100.5056, 'name_en': 'Rat Burana'},
    'บางพลัด': {'lat': 13.7889, 'lon': 100.5022, 'name_en': 'Bang Phlat'},
    'ดินแดง': {'lat': 13.7689, 'lon': 100.5556, 'name_en': 'Din Daeng'},
    'บึงกุ่ม': {'lat': 13.8056, 'lon': 100.6556, 'name_en': 'Bueng Kum'},
    'สาทร': {'lat': 13.7189, 'lon': 100.5289, 'name_en': 'Sathon'},
    'บางซื่อ': {'lat': 13.8056, 'lon': 100.5356, 'name_en': 'Bang Sue'},
    'จตุจักร': {'lat': 13.8256, 'lon': 100.5556, 'name_en': 'Chatuchak'},
    'บางคอแหลม': {'lat': 13.6889, 'lon': 100.5022, 'name_en': 'Bang Kho Laem'},
    'ประเวศ': {'lat': 13.7156, 'lon': 100.6689, 'name_en': 'Prawet'},
    'คลองเตย': {'lat': 13.7089, 'lon': 100.5689, 'name_en': 'Khlong Toei'},
    'สวนหลวง': {'lat': 13.7256, 'lon': 100.6256, 'name_en': 'Suan Luang'},
    'จอมทอง': {'lat': 13.6856, 'lon': 100.4656, 'name_en': 'Chom Thong'},
    'ดอนเมือง': {'lat': 13.9156, 'lon': 100.5956, 'name_en': 'Don Mueang'},
    'ราชเทวี': {'lat': 13.7556, 'lon': 100.5356, 'name_en': 'Ratchathewi'},
    'ลาดพร้าว': {'lat': 13.8156, 'lon': 100.6056, 'name_en': 'Lat Phrao'},
    'วัฒนา': {'lat': 13.7356, 'lon': 100.5856, 'name_en': 'Watthana'},
    'บางแค': {'lat': 13.7156, 'lon': 100.4056, 'name_en': 'Bang Khae'},
    'หลักสี่': {'lat': 13.8856, 'lon': 100.5656, 'name_en': 'Lak Si'},
    'สายไหม': {'lat': 13.9156, 'lon': 100.6556, 'name_en': 'Sai Mai'},
    'คันนายาว': {'lat': 13.8256, 'lon': 100.6856, 'name_en': 'Khan Na Yao'},
    'สะพานสูง': {'lat': 13.7756, 'lon': 100.6956, 'name_en': 'Saphan Sung'},
    'วังทองหลาง': {'lat': 13.7856, 'lon': 100.6056, 'name_en': 'Wang Thonglang'},
    'คลองสามวา': {'lat': 13.8656, 'lon': 100.7256, 'name_en': 'Khlong Sam Wa'},
    'บางนา': {'lat': 13.6689, 'lon': 100.6156, 'name_en': 'Bang Na'},
    'ทวีวัฒนา': {'lat': 13.7756, 'lon': 100.3556, 'name_en': 'Thawi Watthana'},
    'ทุ่งครุ': {'lat': 13.6356, 'lon': 100.5056, 'name_en': 'Thung Khru'},
    'บางบอน': {'lat': 13.6556, 'lon': 100.3756, 'name_en': 'Bang Bon'},
}

# ============================================================================
# 1. Interactive Map with Bangkok Base Layer
# ============================================================================
print("\n[1/4] Creating Interactive Bangkok Map...")

latest_date = df['date'].max()
latest_data = df[df['date'] == latest_date].copy()

# Bangkok center
bkk_center = [13.7563, 100.5018]

# Create map with OpenStreetMap (shows Bangkok streets)
m = folium.Map(
    location=bkk_center, 
    zoom_start=11,
    tiles=None
)

# Add multiple tile layers
folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)

# Add Bangkok boundary (approximate polygon)
bangkok_boundary = [
    [13.95, 100.35], [13.95, 100.90], [13.55, 100.90], [13.55, 100.35], [13.95, 100.35]
]
folium.Polygon(
    locations=bangkok_boundary,
    color='blue',
    weight=3,
    fill=False,
    popup='Bangkok Metropolitan Area'
).add_to(m)

# Add district markers
district_group = folium.FeatureGroup(name='Districts')
for district_th, info in BANGKOK_DISTRICTS.items():
    folium.CircleMarker(
        location=[info['lat'], info['lon']],
        radius=3,
        color='gray',
        fill=True,
        fill_color='gray',
        fill_opacity=0.5,
        tooltip=f"{info['name_en']} ({district_th})"
    ).add_to(district_group)
district_group.add_to(m)

# Add PM2.5 station markers
pm25_group = folium.FeatureGroup(name='PM2.5 Stations')
for _, row in latest_data.iterrows():
    color = get_aqi_color(row['pm2_5'])
    level = get_aqi_level(row['pm2_5'])
    
    popup_html = f"""
    <div style="font-family: Arial; width: 220px;">
        <h4 style="margin: 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;">
            📍 {row['stationID']}
        </h4>
        <table style="width: 100%; margin-top: 10px;">
            <tr><td><b>PM2.5:</b></td><td style="color: {color}; font-weight: bold;">{row['pm2_5']:.1f} µg/m³</td></tr>
            <tr><td><b>AQI Level:</b></td><td style="color: {color};">{level}</td></tr>
            <tr><td><b>PM10:</b></td><td>{row['pm10']:.1f} µg/m³</td></tr>
            <tr><td><b>Temp:</b></td><td>{row['temp_c']:.1f}°C</td></tr>
            <tr><td><b>Humidity:</b></td><td>{row['humidity_pct']:.0f}%</td></tr>
            <tr><td><b>Wind:</b></td><td>{row['wind_ms']:.1f} m/s</td></tr>
        </table>
        <p style="margin-top: 10px; font-size: 11px; color: gray;">
            📅 {row['date'].strftime('%Y-%m-%d')}
        </p>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=12,
        color='black',
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['stationID']}: {row['pm2_5']:.1f} µg/m³ ({level})"
    ).add_to(pm25_group)

pm25_group.add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; 
            padding: 15px; border-radius: 10px; border: 2px solid #333; box-shadow: 3px 3px 10px rgba(0,0,0,0.3);">
    <h4 style="margin: 0 0 10px 0; border-bottom: 1px solid #ccc; padding-bottom: 5px;">🌫️ PM2.5 AQI Level</h4>
    <div style="margin: 5px 0;"><span style="background: #00E400; padding: 3px 12px; border-radius: 3px;">  </span> Good (0-15 µg/m³)</div>
    <div style="margin: 5px 0;"><span style="background: #FFFF00; padding: 3px 12px; border-radius: 3px;">  </span> Moderate (15-35)</div>
    <div style="margin: 5px 0;"><span style="background: #FF7E00; padding: 3px 12px; border-radius: 3px;">  </span> Unhealthy-S (35-55)</div>
    <div style="margin: 5px 0;"><span style="background: #FF0000; padding: 3px 12px; border-radius: 3px;">  </span> Unhealthy (55-75)</div>
    <div style="margin: 5px 0;"><span style="background: #8F3F97; padding: 3px 12px; border-radius: 3px;">  </span> Very Unhealthy (75+)</div>
    <hr style="margin: 10px 0;">
    <p style="margin: 0; font-size: 11px; color: gray;">📅 Data: {date}</p>
    <p style="margin: 0; font-size: 11px; color: gray;">📍 Stations: {n_stations}</p>
</div>
'''.format(date=latest_date.strftime('%Y-%m-%d'), n_stations=len(latest_data))

m.get_root().html.add_child(folium.Element(legend_html))

# Add title
title_html = '''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; 
            background: white; padding: 10px 20px; border-radius: 10px; border: 2px solid #333;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);">
    <h2 style="margin: 0; text-align: center;">🏙️ Bangkok PM2.5 Air Quality Map</h2>
    <p style="margin: 5px 0 0 0; text-align: center; color: gray;">Real-time monitoring from {n} stations</p>
</div>
'''.format(n=len(latest_data))
m.get_root().html.add_child(folium.Element(title_html))

# Add layer control
folium.LayerControl().add_to(m)

m.save(reports_dir / 'bangkok_pm25_map.html')
print("   Saved: bangkok_pm25_map.html")

# ============================================================================
# 2. Heatmap with Bangkok Streets
# ============================================================================
print("\n[2/4] Creating Bangkok Heatmap...")

m2 = folium.Map(location=bkk_center, zoom_start=11, tiles='cartodbdark_matter')

# Add heatmap
heat_data = [[row['lat'], row['lon'], row['pm2_5']] for _, row in latest_data.iterrows()]
HeatMap(heat_data, radius=30, blur=20, max_zoom=13, gradient={
    0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'
}).add_to(m2)

# Add station markers on top
for _, row in latest_data.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,
        color='white',
        weight=1,
        fill=True,
        fill_color='white',
        fill_opacity=0.7,
        tooltip=f"{row['stationID']}: {row['pm2_5']:.1f}"
    ).add_to(m2)

# Add title
title_html2 = '''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; 
            background: rgba(0,0,0,0.8); padding: 10px 20px; border-radius: 10px;">
    <h2 style="margin: 0; text-align: center; color: white;">🔥 Bangkok PM2.5 Heatmap</h2>
</div>
'''
m2.get_root().html.add_child(folium.Element(title_html2))

m2.save(reports_dir / 'bangkok_heatmap.html')
print("   Saved: bangkok_heatmap.html")

# ============================================================================
# 3. Static Map with Bangkok Layout
# ============================================================================
print("\n[3/4] Creating Static Bangkok Map...")

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Bangkok boundary for all plots
bkk_lon_min, bkk_lon_max = 100.35, 100.90
bkk_lat_min, bkk_lat_max = 13.55, 13.95

# 3a. Current PM2.5 Map
ax1 = axes[0, 0]
scatter = ax1.scatter(latest_data['lon'], latest_data['lat'], 
                      c=latest_data['pm2_5'], cmap='RdYlGn_r',
                      s=200, edgecolor='black', linewidth=1.5, vmin=10, vmax=60)
plt.colorbar(scatter, ax=ax1, label='PM2.5 (µg/m³)', shrink=0.8)

# Add district labels
for district_th, info in list(BANGKOK_DISTRICTS.items())[:20]:  # Top 20 districts
    ax1.annotate(info['name_en'], (info['lon'], info['lat']), 
                fontsize=7, alpha=0.6, ha='center')

ax1.set_xlim(bkk_lon_min, bkk_lon_max)
ax1.set_ylim(bkk_lat_min, bkk_lat_max)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_title(f'Bangkok PM2.5 Map ({latest_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')

# Add Bangkok label
ax1.text(0.02, 0.98, '🏙️ BANGKOK', transform=ax1.transAxes, fontsize=14, 
         fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3b. Mean PM2.5 by Station
ax2 = axes[0, 1]
station_mean = df.groupby('stationID').agg({'pm2_5': 'mean', 'lat': 'first', 'lon': 'first'}).reset_index()
scatter2 = ax2.scatter(station_mean['lon'], station_mean['lat'], 
                       c=station_mean['pm2_5'], cmap='RdYlGn_r',
                       s=200, edgecolor='black', linewidth=1.5, vmin=15, vmax=45)
plt.colorbar(scatter2, ax=ax2, label='Mean PM2.5 (µg/m³)', shrink=0.8)
ax2.set_xlim(bkk_lon_min, bkk_lon_max)
ax2.set_ylim(bkk_lat_min, bkk_lat_max)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.set_title('Average PM2.5 by Station (2023-2026)', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')

# 3c. Station Density
ax3 = axes[1, 0]
# Create grid
lon_bins = np.linspace(bkk_lon_min, bkk_lon_max, 20)
lat_bins = np.linspace(bkk_lat_min, bkk_lat_max, 20)
H, xedges, yedges = np.histogram2d(station_mean['lon'], station_mean['lat'], bins=[lon_bins, lat_bins])
im = ax3.imshow(H.T, origin='lower', extent=[bkk_lon_min, bkk_lon_max, bkk_lat_min, bkk_lat_max],
                cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax3, label='Number of Stations', shrink=0.8)
ax3.scatter(station_mean['lon'], station_mean['lat'], c='red', s=30, alpha=0.5, marker='x')
ax3.set_xlabel('Longitude', fontsize=12)
ax3.set_ylabel('Latitude', fontsize=12)
ax3.set_title('Station Density Map', fontsize=14, fontweight='bold')

# 3d. AQI Level Distribution Map
ax4 = axes[1, 1]
latest_data['aqi_level'] = latest_data['pm2_5'].apply(get_aqi_level)
for level, color in AQI_COLORS.items():
    mask = latest_data['aqi_level'] == level
    if mask.sum() > 0:
        ax4.scatter(latest_data.loc[mask, 'lon'], latest_data.loc[mask, 'lat'],
                   c=color, s=200, edgecolor='black', linewidth=1.5, label=level)
ax4.set_xlim(bkk_lon_min, bkk_lon_max)
ax4.set_ylim(bkk_lat_min, bkk_lat_max)
ax4.set_xlabel('Longitude', fontsize=12)
ax4.set_ylabel('Latitude', fontsize=12)
ax4.set_title('AQI Level Distribution', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig(reports_dir / 'bangkok_static_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: bangkok_static_maps.png")

# ============================================================================
# 4. Time Animation Data (for future use)
# ============================================================================
print("\n[4/4] Creating Time Series Map Data...")

# Prepare data for last 7 days
last_7_days = df[df['date'] >= latest_date - pd.Timedelta(days=6)].copy()
daily_data = []

for date in sorted(last_7_days['date'].unique()):
    day_data = last_7_days[last_7_days['date'] == date]
    day_list = [[row['lat'], row['lon'], row['pm2_5']] for _, row in day_data.iterrows()]
    daily_data.append(day_list)

# Create time-based heatmap
m3 = folium.Map(location=bkk_center, zoom_start=11, tiles='cartodbpositron')

# Add time index
time_index = [d.strftime('%Y-%m-%d') for d in sorted(last_7_days['date'].unique())]

HeatMapWithTime(
    daily_data,
    index=time_index,
    auto_play=True,
    max_opacity=0.8,
    radius=25,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
).add_to(m3)

title_html3 = '''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; 
            background: white; padding: 10px 20px; border-radius: 10px; border: 2px solid #333;">
    <h2 style="margin: 0; text-align: center;">📅 Bangkok PM2.5 Time Animation (Last 7 Days)</h2>
    <p style="margin: 5px 0 0 0; text-align: center; color: gray;">Press play to see daily changes</p>
</div>
'''
m3.get_root().html.add_child(folium.Element(title_html3))

m3.save(reports_dir / 'bangkok_time_animation.html')
print("   Saved: bangkok_time_animation.html")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("BANGKOK MAP VISUALIZATION COMPLETE")
print("="*70)

summary = f"""
FILES CREATED:
─────────────────────────────────────────────────────────────────────────
1. bangkok_pm25_map.html      - Interactive map with Bangkok streets & districts
2. bangkok_heatmap.html       - PM2.5 heatmap overlay
3. bangkok_static_maps.png    - 4-panel static map analysis
4. bangkok_time_animation.html - 7-day time animation

LOCATION: {reports_dir}

DATA SUMMARY:
- Stations: {N_STATIONS}
- Latest Date: {latest_date.strftime('%Y-%m-%d')}
- Mean PM2.5: {latest_data['pm2_5'].mean():.1f} µg/m³
- Max PM2.5: {latest_data['pm2_5'].max():.1f} µg/m³
"""

print(summary)

with open(reports_dir / 'README.txt', 'w') as f:
    f.write(summary)

print(f"\nAll Bangkok maps saved to: {reports_dir}")
