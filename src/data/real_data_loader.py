"""
real_data_loader.py
-------------------
Load real PM2.5 and meteorological data from silver layer
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from datetime import datetime, timedelta


def load_pm25_data(
    data_dir: Path,
    stations: pd.DataFrame,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    features: List[str] = None
) -> Tuple[Tensor, pd.DataFrame]:
    """
    Load real PM2.5 and air quality data from silver layer
    
    Parameters
    ----------
    data_dir : Path to data directory
    stations : DataFrame with station metadata
    start_date : Start date (YYYY-MM-DD)
    end_date : End date (YYYY-MM-DD)
    features : List of feature columns to include
    
    Returns
    -------
    data : Tensor of shape (T, N, F) where T=timesteps, N=stations, F=features
    metadata : DataFrame with timestamp information
    """
    
    if features is None:
        features = ['pm2_5_ugm3', 'pm10_ugm3', 'no2_ugm3', 'o3_ugm3', 
                   'so2_ugm3', 'co_ugm3']
    
    print(f"Loading PM2.5 data from {start_date} to {end_date}...")
    
    # Parse dates with UTC timezone
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Find all parquet files in date range
    airquality_dir = data_dir / 'silver' / 'openmeteo_airquality'
    
    all_data = []
    years = range(start.year, end.year + 1)
    
    for year in years:
        year_dir = airquality_dir / f'year={year}'
        if not year_dir.exists():
            print(f"  Warning: {year_dir} not found, skipping...")
            continue
        
        # Read all parquet files for this year
        parquet_files = list(year_dir.rglob('*.parquet'))
        
        if not parquet_files:
            print(f"  Warning: No parquet files found in {year_dir}")
            continue
        
        print(f"  Loading {len(parquet_files)} files from year {year}...")
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                all_data.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No data loaded! Check data directory and date range.")
    
    # Combine all data
    print("  Combining data...")
    df = pd.concat(all_data, ignore_index=True)
    
    # Filter by date range
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df = df[(df['timestamp_utc'] >= start) & (df['timestamp_utc'] <= end)]
    
    print(f"  Total records after filtering: {len(df):,}")
    
    # Get unique timestamps and stations
    timestamps = sorted(df['timestamp_utc'].unique())
    station_ids = sorted(df['stationID'].unique())
    
    # Filter to only stations in our station list
    valid_stations = set(stations['stationID'].values)
    station_ids = [s for s in station_ids if s in valid_stations]
    
    print(f"  Unique timestamps: {len(timestamps)}")
    print(f"  Unique stations: {len(station_ids)}")
    
    # Create station ID to index mapping
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
    
    # Initialize tensor
    T = len(timestamps)
    N = len(station_ids)
    F = len(features)
    
    data = torch.full((T, N, F), float('nan'), dtype=torch.float32)
    
    # Fill tensor
    print("  Filling tensor...")
    for _, row in df.iterrows():
        if row['stationID'] not in station_to_idx:
            continue
        
        try:
            t_idx = timestamps.index(row['timestamp_utc'])
            n_idx = station_to_idx[row['stationID']]
            
            for f_idx, feat in enumerate(features):
                if feat in row and pd.notna(row[feat]):
                    data[t_idx, n_idx, f_idx] = float(row[feat])
        except (ValueError, KeyError):
            continue
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'timestamp': timestamps,
        'hour': [t.hour for t in timestamps],
        'day': [t.day for t in timestamps],
        'month': [t.month for t in timestamps],
        'year': [t.year for t in timestamps],
        'dayofweek': [t.dayofweek for t in timestamps],
        'dayofyear': [t.dayofyear for t in timestamps],
    })
    
    # Handle missing values
    nan_count = torch.isnan(data).sum().item()
    total_values = data.numel()
    nan_pct = (nan_count / total_values) * 100
    
    print(f"  Missing values: {nan_count:,} / {total_values:,} ({nan_pct:.2f}%)")
    
    # Interpolate missing values (linear interpolation along time axis)
    print("  Interpolating missing values...")
    data_np = data.numpy()
    
    for n in range(N):
        for f in range(F):
            series = data_np[:, n, f]
            mask = ~np.isnan(series)
            
            if mask.sum() > 1:  # Need at least 2 points to interpolate
                indices = np.arange(len(series))
                series[~mask] = np.interp(indices[~mask], indices[mask], series[mask])
            elif mask.sum() == 1:  # Only one value, fill with that value
                series[~mask] = series[mask][0]
            else:  # No values, fill with 0
                series[:] = 0.0
            
            data_np[:, n, f] = series
    
    data = torch.from_numpy(data_np)
    
    # Final check
    remaining_nans = torch.isnan(data).sum().item()
    print(f"  Remaining NaNs after interpolation: {remaining_nans}")
    
    print(f"✅ Data loaded: shape={data.shape}")
    print(f"   Features: {features}")
    print(f"   Date range: {timestamps[0]} to {timestamps[-1]}")
    
    return data, metadata


def load_weather_data(
    data_dir: Path,
    stations: pd.DataFrame,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> Tuple[Tensor, List[str]]:
    """
    Load meteorological data from silver layer
    
    Returns
    -------
    weather_data : Tensor of shape (T, N, F_weather)
    weather_features : List of weather feature names
    """
    
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'surface_pressure'
    ]
    
    print(f"Loading weather data from {start_date} to {end_date}...")
    
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    
    weather_dir = data_dir / 'silver' / 'openmeteo_weather'
    
    all_data = []
    years = range(start.year, end.year + 1)
    
    for year in years:
        year_dir = weather_dir / f'year={year}'
        if not year_dir.exists():
            continue
        
        parquet_files = list(year_dir.rglob('*.parquet'))
        print(f"  Loading {len(parquet_files)} files from year {year}...")
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                all_data.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}")
                continue
    
    if not all_data:
        print("  Warning: No weather data found, returning None")
        return None, weather_features
    
    df = pd.concat(all_data, ignore_index=True)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df = df[(df['timestamp_utc'] >= start) & (df['timestamp_utc'] <= end)]
    
    print(f"  Total weather records: {len(df):,}")
    
    timestamps = sorted(df['timestamp_utc'].unique())
    station_ids = sorted(df['stationID'].unique())
    
    valid_stations = set(stations['stationID'].values)
    station_ids = [s for s in station_ids if s in valid_stations]
    
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
    
    T = len(timestamps)
    N = len(station_ids)
    F = len(weather_features)
    
    data = torch.full((T, N, F), float('nan'), dtype=torch.float32)
    
    for _, row in df.iterrows():
        if row['stationID'] not in station_to_idx:
            continue
        
        try:
            t_idx = timestamps.index(row['timestamp_utc'])
            n_idx = station_to_idx[row['stationID']]
            
            for f_idx, feat in enumerate(weather_features):
                if feat in row and pd.notna(row[feat]):
                    data[t_idx, n_idx, f_idx] = float(row[feat])
        except (ValueError, KeyError):
            continue
    
    # Interpolate
    data_np = data.numpy()
    for n in range(N):
        for f in range(F):
            series = data_np[:, n, f]
            mask = ~np.isnan(series)
            if mask.sum() > 1:
                indices = np.arange(len(series))
                series[~mask] = np.interp(indices[~mask], indices[mask], series[mask])
            elif mask.sum() == 1:
                series[~mask] = series[mask][0]
            else:
                series[:] = 0.0
            data_np[:, n, f] = series
    
    data = torch.from_numpy(data_np)
    
    print(f"✅ Weather data loaded: shape={data.shape}")
    
    return data, weather_features


def combine_features(
    pm25_data: Tensor,
    weather_data: Optional[Tensor] = None,
    fire_data: Optional[Tensor] = None,
    add_temporal_features: bool = True,
    metadata: Optional[pd.DataFrame] = None
) -> Tuple[Tensor, List[str]]:
    """
    Combine PM2.5, weather, fire, and temporal features
    
    Parameters
    ----------
    pm25_data : (T, N, F_pm25)
    weather_data : (T, N, F_weather) or None
    fire_data : (T, N, F_fire) or None
    add_temporal_features : Add cyclical time features
    metadata : DataFrame with timestamp info
    
    Returns
    -------
    combined_data : (T, N, F_total)
    feature_names : List of all feature names
    """
    
    T, N, F_pm25 = pm25_data.shape
    
    features = [pm25_data]
    feature_names = ['pm2_5_ugm3', 'pm10_ugm3', 'no2_ugm3', 'o3_ugm3', 'so2_ugm3', 'co_ugm3']
    
    # Add weather features
    if weather_data is not None:
        features.append(weather_data)
        feature_names.extend(['temperature_2m', 'relative_humidity_2m', 'precipitation',
                            'wind_speed_10m', 'wind_direction_10m', 'surface_pressure'])
    
    # Add fire features
    if fire_data is not None:
        features.append(fire_data)
        feature_names.extend(['fire_count_500km', 'fire_frp_total', 'upwind_fire_impact',
                            'fire_distance_weighted', 'fire_lag1d', 'fire_lag3d'])
    
    # Add temporal features
    if add_temporal_features and metadata is not None:
        temporal_feats = []
        
        # Hour (cyclical encoding)
        hour_sin = np.sin(2 * np.pi * metadata['hour'] / 24)
        hour_cos = np.cos(2 * np.pi * metadata['hour'] / 24)
        
        # Day of week (cyclical encoding)
        dow_sin = np.sin(2 * np.pi * metadata['dayofweek'] / 7)
        dow_cos = np.cos(2 * np.pi * metadata['dayofweek'] / 7)
        
        # Day of year (cyclical encoding)
        doy_sin = np.sin(2 * np.pi * metadata['dayofyear'] / 365)
        doy_cos = np.cos(2 * np.pi * metadata['dayofyear'] / 365)
        
        # Stack temporal features (T, 6)
        temporal = np.stack([hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos], axis=1)
        
        # Expand to (T, N, 6)
        temporal = torch.from_numpy(temporal).float().unsqueeze(1).expand(T, N, 6)
        
        features.append(temporal)
        feature_names.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos'])
    
    # Combine all features
    combined = torch.cat(features, dim=2)
    
    print(f"✅ Combined features: {combined.shape}")
    print(f"   Feature names ({len(feature_names)}): {feature_names}")
    
    return combined, feature_names
