"""
fire_feature_loader_v2.py
--------------------------
Simplified and working fire feature loader
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import pytz


def load_fire_data(
    data_dir: Path,
    start_date: str,
    end_date: str,
    radius_km: float = 500.0
) -> pd.DataFrame:
    """Load NASA FIRMS fire data"""
    
    print(f"\n🔥 Loading Fire Data from {start_date} to {end_date}...")
    
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    
    years = list(range(start.year, end.year + 1))
    print(f"  Loading {len(years)} files from year(s) {years}...")
    
    dfs = []
    for year in years:
        fire_dir = data_dir / 'silver' / 'firms_hotspot' / f'year={year}'
        if not fire_dir.exists():
            continue
        
        # Read from month subfolders
        for month_dir in fire_dir.glob('month=*'):
            for file in month_dir.glob('*.parquet'):
                df = pd.read_parquet(file)
                dfs.append(df)
    
    if not dfs:
        print("  ⚠️  No fire data found!")
        return pd.DataFrame()
    
    fire_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total records: {len(fire_df):,}")
    
    # Filter by confidence
    fire_df = fire_df[fire_df['confidence'].isin(['n', 'h'])]
    print(f"  After confidence filter: {len(fire_df):,}")
    
    # Filter by FRP
    fire_df = fire_df[fire_df['frp'] >= 5.0]
    print(f"  After FRP filter (>=5.0 MW): {len(fire_df):,}")
    
    # Filter by radius (Bangkok center: 13.7563, 100.5018)
    bangkok_lat, bangkok_lon = 13.7563, 100.5018
    
    def calc_distance(row):
        from math import radians, cos, sin, asin, sqrt
        lat1, lon1 = radians(bangkok_lat), radians(bangkok_lon)
        lat2, lon2 = radians(row['latitude']), radians(row['longitude'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c
    
    fire_df['distance_km'] = fire_df.apply(calc_distance, axis=1)
    fire_df = fire_df[fire_df['distance_km'] <= radius_km]
    print(f"  Within {radius_km}km of Bangkok: {len(fire_df):,}")
    
    # Filter by date range
    fire_df['timestamp_utc'] = pd.to_datetime(fire_df['timestamp_utc'])
    fire_df = fire_df[(fire_df['timestamp_utc'] >= start) & (fire_df['timestamp_utc'] <= end)]
    
    if len(fire_df) > 0:
        print(f"✅ Fire data loaded: {len(fire_df):,} fires")
        print(f"   Date range: {fire_df['timestamp_utc'].min()} to {fire_df['timestamp_utc'].max()}")
    else:
        print("  ⚠️  No fires in date range!")
    
    return fire_df


def compute_fire_features_simple(
    fire_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    radius_km: float = 500.0
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute simplified fire features that actually work
    
    Returns
    -------
    fire_features : (T, N, 3) tensor
    feature_names : List of feature names
    """
    
    T = len(timestamps)
    N = len(stations_df)
    
    print(f"\n🔧 Computing Fire Features (Simplified)...")
    print(f"  Timestamps: {T}")
    print(f"  Stations: {N}")
    
    # Initialize features
    fire_count = np.zeros((T, N))
    fire_frp_sum = np.zeros((T, N))
    fire_distance_avg = np.zeros((T, N))
    
    if len(fire_df) == 0:
        print("  ⚠️  No fire data - returning zeros")
        fire_features = np.stack([fire_count, fire_frp_sum, fire_distance_avg], axis=2)
        return torch.from_numpy(fire_features).float(), ['fire_count', 'fire_frp_sum', 'fire_distance_avg']
    
    # Convert timestamps to dates (remove time component)
    # Convert DatetimeIndex to array of dates
    timestamps_dates = pd.DatetimeIndex([pd.Timestamp(ts).normalize() for ts in timestamps])
    
    # Add date column to fire_df
    fire_df = fire_df.copy()
    fire_df['date'] = fire_df['timestamp_utc'].dt.normalize()
    
    print(f"  Unique fire dates: {fire_df['date'].nunique()}")
    print(f"  Unique PM2.5 dates: {timestamps_dates.nunique()}")
    
    # Group fires by date
    fires_by_date = fire_df.groupby('date')
    
    # For each station
    for station_idx, station in stations_df.iterrows():
        if station_idx % 20 == 0:
            print(f"    Processing station {station_idx+1}/{N}...")
        
        station_lat = station['lat']
        station_lon = station['lon']
        
        # For each unique date
        for date in timestamps_dates.unique():
            # Get fires for this date
            if date not in fires_by_date.groups:
                continue
            
            fires_today = fires_by_date.get_group(date)
            
            if len(fires_today) == 0:
                continue
            
            # Calculate distances for all fires to this station
            from math import radians, cos, sin, asin, sqrt
            
            distances = []
            frps = []
            
            for _, fire in fires_today.iterrows():
                lat1, lon1 = radians(station_lat), radians(station_lon)
                lat2, lon2 = radians(fire['latitude']), radians(fire['longitude'])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                dist = 6371 * c
                
                if dist <= radius_km:
                    distances.append(dist)
                    frps.append(fire['frp'])
            
            if len(distances) == 0:
                continue
            
            # Find all timesteps for this date
            date_mask = timestamps_dates == date
            date_indices = np.where(date_mask)[0]
            
            # Aggregate features
            count = len(distances)
            frp_sum = sum(frps)
            dist_avg = sum(distances) / len(distances)
            
            # Apply to all timesteps for this date
            for t_idx in date_indices:
                fire_count[t_idx, station_idx] = count
                fire_frp_sum[t_idx, station_idx] = frp_sum
                fire_distance_avg[t_idx, station_idx] = dist_avg
    
    # Stack features
    fire_features = np.stack([fire_count, fire_frp_sum, fire_distance_avg], axis=2)
    
    # Check non-zero values
    non_zero = (fire_features != 0).sum()
    total = fire_features.size
    
    print(f"\n✅ Fire features computed:")
    print(f"   Shape: {fire_features.shape}")
    print(f"   Non-zero values: {non_zero:,} / {total:,} ({non_zero/total*100:.2f}%)")
    print(f"   Fire count range: [{fire_count.min():.0f}, {fire_count.max():.0f}]")
    print(f"   FRP sum range: [{fire_frp_sum.min():.1f}, {fire_frp_sum.max():.1f}]")
    print(f"   Distance avg range: [{fire_distance_avg[fire_distance_avg>0].min():.1f}, {fire_distance_avg.max():.1f}] km")
    
    feature_names = ['fire_count', 'fire_frp_sum', 'fire_distance_avg']
    
    return torch.from_numpy(fire_features).float(), feature_names
