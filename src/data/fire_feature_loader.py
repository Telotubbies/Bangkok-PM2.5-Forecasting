"""
fire_feature_loader.py
----------------------
Load and engineer NASA FIRMS fire/hotspot features for PM2.5 forecasting

Features:
- Fire count within radius
- Fire Radiative Power (FRP) aggregation
- Wind-direction weighted fire impact (upwind fires)
- Distance-weighted fire intensity
- Temporal lag features (1-day, 3-day)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import pytz


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers
    
    Parameters
    ----------
    lon1, lat1 : First point coordinates
    lon2, lat2 : Second point coordinates
    
    Returns
    -------
    distance_km : Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360)
    
    Returns
    -------
    bearing : Bearing in degrees (0=North, 90=East, 180=South, 270=West)
    """
    from math import radians, degrees, atan2, cos, sin
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    bearing = degrees(atan2(x, y))
    bearing = (bearing + 360) % 360
    
    return bearing


def distance_weight(distance_km: float, min_distance: float = 10.0) -> float:
    """
    Calculate distance weight using inverse square law
    
    Parameters
    ----------
    distance_km : Distance in kilometers
    min_distance : Minimum distance threshold to avoid division by zero
    
    Returns
    -------
    weight : Distance weight (higher for closer fires)
    """
    return 1.0 / max(distance_km, min_distance)**2


def wind_angle_weight(wind_direction: float, fire_bearing: float) -> float:
    """
    Calculate wind angle weight for upwind fires
    
    Parameters
    ----------
    wind_direction : Wind direction in degrees (direction wind is blowing FROM)
    fire_bearing : Bearing from station to fire in degrees
    
    Returns
    -------
    weight : Wind angle weight (0-1, only for upwind fires)
    """
    from math import radians, cos
    
    # Calculate angle difference
    # Wind blows FROM wind_direction, so fire is upwind if bearing ≈ wind_direction
    angle_diff = abs(wind_direction - fire_bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # Only upwind fires (angle < 90°)
    if angle_diff <= 90:
        return cos(radians(angle_diff))
    
    return 0.0


def get_adaptive_lag_days(distance_km: float, max_lag: int = 3) -> int:
    """
    Calculate adaptive lag based on distance
    Assumes smoke travels ~200km/day
    
    Parameters
    ----------
    distance_km : Distance in kilometers
    max_lag : Maximum lag in days
    
    Returns
    -------
    lag_days : Lag in days
    """
    lag = int(distance_km / 200)
    return min(lag, max_lag)


def load_fire_data(
    data_dir: Path,
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    radius_km: float = 500.0,
    min_frp: float = 5.0,
    confidence_levels: list = ['n', 'h']
) -> Tuple[pd.DataFrame, dict]:
    """
    Load NASA FIRMS fire/hotspot data for given date range
    
    Parameters
    ----------
    data_dir : Path to data directory
    stations_df : DataFrame with station information (lat, lon, stationID)
    start_date : Start date (YYYY-MM-DD)
    end_date : End date (YYYY-MM-DD)
    radius_km : Radius in km to search for fires around Bangkok
    min_frp : Minimum Fire Radiative Power threshold (MW)
    confidence_levels : List of confidence levels to include ('n', 'h', 'l')
    
    Returns
    -------
    fire_df : DataFrame with fire data
    metadata : Dictionary with metadata
    """
    print(f"\n🔥 Loading Fire Data from {start_date} to {end_date}...")
    
    # Bangkok center coordinates
    bkk_lat, bkk_lon = 13.7563, 100.5018
    
    # Parse dates
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Determine years to load
    years = list(range(start.year, end.year + 1))
    
    fire_dfs = []
    total_files = 0
    
    for year in years:
        year_dir = data_dir / 'silver' / 'firms_hotspot' / f'year={year}'
        
        if not year_dir.exists():
            print(f"  ⚠️  Year {year} directory not found, skipping...")
            continue
        
        # Get all month directories
        month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir()])
        
        for month_dir in month_dirs:
            # Get parquet files
            parquet_files = list(month_dir.glob('*.parquet'))
            
            for file in parquet_files:
                try:
                    df = pd.read_parquet(file)
                    
                    # Convert timestamp to UTC
                    if 'timestamp_utc' in df.columns:
                        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
                    
                    # Filter by date range
                    df = df[(df['timestamp_utc'] >= start) & (df['timestamp_utc'] <= end)]
                    
                    if len(df) > 0:
                        fire_dfs.append(df)
                        total_files += 1
                
                except Exception as e:
                    print(f"  ⚠️  Error loading {file.name}: {e}")
                    continue
    
    if not fire_dfs:
        print("  ❌ No fire data found!")
        return pd.DataFrame(), {}
    
    print(f"  Loading {total_files} files from year(s) {years}...")
    
    # Combine all data
    fire_df = pd.concat(fire_dfs, ignore_index=True)
    print(f"  Total records: {len(fire_df):,}")
    
    # Filter by confidence
    fire_df = fire_df[fire_df['confidence'].isin(confidence_levels)]
    print(f"  After confidence filter: {len(fire_df):,}")
    
    # Filter by FRP
    fire_df = fire_df[fire_df['frp'] >= min_frp]
    print(f"  After FRP filter (>={min_frp} MW): {len(fire_df):,}")
    
    # Filter by distance from Bangkok
    fire_df['distance_km'] = fire_df.apply(
        lambda row: haversine_distance(bkk_lon, bkk_lat, row['longitude'], row['latitude']),
        axis=1
    )
    
    fire_df = fire_df[fire_df['distance_km'] <= radius_km]
    print(f"  Within {radius_km}km of Bangkok: {len(fire_df):,}")
    
    # Sort by timestamp
    fire_df = fire_df.sort_values('timestamp_utc').reset_index(drop=True)
    
    metadata = {
        'start_date': start,
        'end_date': end,
        'radius_km': radius_km,
        'min_frp': min_frp,
        'confidence_levels': confidence_levels,
        'total_fires': len(fire_df)
    }
    
    print(f"✅ Fire data loaded: {len(fire_df):,} fires")
    print(f"   Date range: {fire_df['timestamp_utc'].min()} to {fire_df['timestamp_utc'].max()}")
    
    return fire_df, metadata


def compute_fire_features(
    fire_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    weather_data: Optional[torch.Tensor],
    timestamps: pd.DatetimeIndex,
) -> Tuple[torch.Tensor, list]:
    """
    Compute fire features for each station and timestamp
    
    Parameters
    ----------
    fire_df : DataFrame with fire data (must have: timestamp_utc, latitude, longitude, frp, distance_km)
    stations_df : DataFrame with station info (must have: lat, lon, stationID)
    weather_data : Weather data tensor (T, N, F_weather) - used for wind direction/speed
    timestamps : DatetimeIndex with timestamps for each time step
    
    Returns
    -------
    fire_features : Tensor of shape (T, N, 6) with fire features
    feature_names : List of feature names
    """
    print("\n🔧 Computing Fire Features...")
    
    T = len(timestamps)
    N = len(stations_df)
    
    # Initialize feature arrays
    fire_count = np.zeros((T, N))
    fire_frp_total = np.zeros((T, N))
    upwind_fire_impact = np.zeros((T, N))
    fire_distance_weighted = np.zeros((T, N))
    fire_lag1d = np.zeros((T, N))
    fire_lag3d = np.zeros((T, N))
    
    # Convert timestamps to daily resolution for fire data
    timestamps_daily = pd.to_datetime(timestamps.date)
    unique_dates = timestamps_daily.unique()
    
    print(f"  Processing {len(unique_dates)} unique dates...")
    
    # Group fires by date
    fire_df['date'] = pd.to_datetime(fire_df['timestamp_utc'].dt.date)
    fires_by_date = fire_df.groupby('date')
    
    # Process each station
    for station_idx, station in stations_df.iterrows():
        station_lat = station['lat']
        station_lon = station['lon']
        
        # Process each date
        for date_idx, date in enumerate(unique_dates):
            # Get fires for this date
            if date not in fires_by_date.groups:
                continue
            
            fires_today = fires_by_date.get_group(date)
            
            if len(fires_today) == 0:
                continue
            
            # Calculate distance and bearing for each fire
            fires_today = fires_today.copy()
            fires_today['dist_to_station'] = fires_today.apply(
                lambda row: haversine_distance(station_lon, station_lat, row['longitude'], row['latitude']),
                axis=1
            )
            fires_today['bearing_to_fire'] = fires_today.apply(
                lambda row: calculate_bearing(station_lon, station_lat, row['longitude'], row['latitude']),
                axis=1
            )
            
            # Get wind data for this station and date (if available)
            wind_direction = 0.0  # Default
            wind_speed = 0.0
            
            if weather_data is not None:
                # Find timesteps for this date
                date_mask = timestamps_daily == date
                date_indices = np.where(date_mask)[0]
                
                if len(date_indices) > 0:
                    # Use average wind for the day
                    # Assuming weather_data has wind_direction at index 4 and wind_speed at index 3
                    if weather_data.shape[2] >= 5:
                        wind_direction = weather_data[date_indices, station_idx, 4].mean().item()
                        wind_speed = weather_data[date_indices, station_idx, 3].mean().item()
            
            # Compute features
            for fire_idx, fire in fires_today.iterrows():
                dist = fire['dist_to_station']
                frp = fire['frp']
                bearing = fire['bearing_to_fire']
                
                # Distance weight
                dist_w = distance_weight(dist)
                
                # Wind angle weight
                wind_w = wind_angle_weight(wind_direction, bearing)
                
                # Aggregate features for this date
                date_mask = timestamps_daily == date
                date_indices = np.where(date_mask)[0]
                
                for t_idx in date_indices:
                    # Feature 1: Fire count
                    fire_count[t_idx, station_idx] += 1
                    
                    # Feature 2: Total FRP
                    fire_frp_total[t_idx, station_idx] += frp
                    
                    # Feature 3: Upwind fire impact (wind-weighted)
                    upwind_fire_impact[t_idx, station_idx] += frp * wind_w * dist_w
                    
                    # Feature 4: Distance-weighted fire intensity
                    fire_distance_weighted[t_idx, station_idx] += frp * dist_w
                
                # Temporal lag features
                lag1_days = get_adaptive_lag_days(dist, max_lag=1)
                lag3_days = get_adaptive_lag_days(dist, max_lag=3)
                
                # Apply lag
                lag1_date = date + pd.Timedelta(days=lag1_days)
                lag3_date = date + pd.Timedelta(days=lag3_days)
                
                # Feature 5: 1-day lag
                if lag1_date in unique_dates:
                    lag1_mask = timestamps_daily == lag1_date
                    lag1_indices = np.where(lag1_mask)[0]
                    for t_idx in lag1_indices:
                        fire_lag1d[t_idx, station_idx] += frp * wind_w * dist_w
                
                # Feature 6: 3-day lag
                if lag3_date in unique_dates:
                    lag3_mask = timestamps_daily == lag3_date
                    lag3_indices = np.where(lag3_mask)[0]
                    for t_idx in lag3_indices:
                        fire_lag3d[t_idx, station_idx] += frp * wind_w * dist_w
    
    # Stack features
    fire_features = np.stack([
        fire_count,
        fire_frp_total,
        upwind_fire_impact,
        fire_distance_weighted,
        fire_lag1d,
        fire_lag3d
    ], axis=2)  # (T, N, 6)
    
    # Convert to tensor
    fire_features = torch.from_numpy(fire_features).float()
    
    feature_names = [
        'fire_count_500km',
        'fire_frp_total',
        'upwind_fire_impact',
        'fire_distance_weighted',
        'fire_lag1d',
        'fire_lag3d'
    ]
    
    print(f"✅ Fire features computed: {fire_features.shape}")
    print(f"   Features: {feature_names}")
    print(f"   Non-zero values: {(fire_features > 0).sum().item():,}")
    
    return fire_features, feature_names
