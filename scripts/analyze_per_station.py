#!/usr/bin/env python3
"""
analyze_per_station.py
----------------------
Analyze model performance per station
Show which stations have best/worst predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.stc_hgat_improved import ImprovedSTCHGAT
from src.data.dataset import PM25SequenceDataset
from src.data.real_data_loader import load_pm25_data, load_weather_data, combine_features
from src.utils.evaluator import calculate_mae, calculate_rmse, calculate_r2


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def main():
    print("=" * 60)
    print("📊 Per-Station Performance Analysis")
    print("=" * 60)
    
    device = setup_device()
    
    # Load model
    model_path = project_root / 'models/stc_hgat_improved_20260327_222751.pt'
    print(f"\n📦 Loading model: {model_path.name}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load stations
    stations_df = pd.read_parquet(project_root / 'data/stations/bangkok_stations.parquet')
    num_stations = len(stations_df)
    
    print(f"✅ Loaded {num_stations} stations")
    
    # Load data
    print("\n📊 Loading data...")
    pm25_data, metadata = load_pm25_data(
        project_root / 'data', stations_df,
        start_date='2024-01-01', end_date='2024-11-30'
    )
    
    weather_data, _ = load_weather_data(
        project_root / 'data', stations_df,
        start_date='2024-01-01', end_date='2024-11-30'
    )
    
    data, feature_names = combine_features(
        pm25_data, weather_data=weather_data,
        add_temporal_features=True, metadata=metadata
    )
    
    # Prepare test data
    train_ratio = 0.7
    val_ratio = 0.15
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    test_data = data[val_end:]
    
    train_mean = checkpoint['train_mean'].cpu()
    train_std = checkpoint['train_std'].cpu()
    
    test_data_norm = (test_data - train_mean) / train_std
    
    # Create test loader
    seq_len = 30
    horizons = [1, 3, 7]
    batch_size = 32
    
    test_dataset = PM25SequenceDataset(test_data_norm, seq_len, horizons)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load model
    num_features = data.shape[2]
    
    model = ImprovedSTCHGAT(
        num_features=num_features,
        hidden_dim=128,
        num_stations=num_stations,
        num_regions=5,
        num_hypergat_layers=2,
        num_hgat_layers=1,
        num_heads=4,
        dropout=0.2,
        forecast_horizons=horizons,
        use_gated_fusion=True,
        use_cross_attention=True,
        use_multiscale_temporal=True,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate per station
    print("\n🔍 Evaluating per station...")
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            if X.shape[1] != num_stations:
                X = X.permute(0, 2, 1, 3)
            
            pred, _, _ = model(X, None, None)
            pred_expanded = pred.unsqueeze(-1).expand(-1, -1, len(horizons))
            
            test_predictions.append(pred_expanded.cpu())
            test_targets.append(y.cpu())
    
    test_predictions = torch.cat(test_predictions, dim=0)  # (samples, stations, horizons)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Calculate metrics per station for horizon +1d
    horizon_idx = 0  # +1 day
    
    station_metrics = []
    
    for station_idx in range(num_stations):
        pred_station = test_predictions[:, station_idx, horizon_idx]
        target_station = test_targets[:, station_idx, horizon_idx]
        
        mae = calculate_mae(target_station.unsqueeze(0), pred_station.unsqueeze(0))
        rmse = calculate_rmse(target_station.unsqueeze(0), pred_station.unsqueeze(0))
        r2 = calculate_r2(target_station.unsqueeze(0), pred_station.unsqueeze(0))
        
        station_info = stations_df.iloc[station_idx]
        
        station_metrics.append({
            'station_id': station_info['stationID'],
            'station_name': station_info.get('name', station_info['stationID']),
            'lat': station_info['lat'],
            'lon': station_info['lon'],
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    # Create DataFrame
    df_metrics = pd.DataFrame(station_metrics)
    df_metrics = df_metrics.sort_values('r2', ascending=False)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 Per-Station Performance Summary (Horizon +1d)")
    print("=" * 60)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Mean R²:   {df_metrics['r2'].mean():.4f}")
    print(f"  Median R²: {df_metrics['r2'].median():.4f}")
    print(f"  Std R²:    {df_metrics['r2'].std():.4f}")
    print(f"  Min R²:    {df_metrics['r2'].min():.4f}")
    print(f"  Max R²:    {df_metrics['r2'].max():.4f}")
    
    # Top 10 stations
    print(f"\n🏆 Top 10 Best Performing Stations:")
    print("-" * 60)
    for i, row in df_metrics.head(10).iterrows():
        print(f"{row['station_id']:12s} | R²={row['r2']:.4f} | MAE={row['mae']:.4f} | RMSE={row['rmse']:.4f}")
    
    # Bottom 10 stations
    print(f"\n⚠️  Top 10 Worst Performing Stations:")
    print("-" * 60)
    for i, row in df_metrics.tail(10).iterrows():
        print(f"{row['station_id']:12s} | R²={row['r2']:.4f} | MAE={row['mae']:.4f} | RMSE={row['rmse']:.4f}")
    
    # Save results
    output_file = project_root / 'results' / 'per_station_performance.csv'
    output_file.parent.mkdir(exist_ok=True)
    df_metrics.to_csv(output_file, index=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Performance distribution
    print(f"\n📊 R² Distribution:")
    print(f"  R² >= 0.95: {(df_metrics['r2'] >= 0.95).sum()} stations ({(df_metrics['r2'] >= 0.95).sum()/len(df_metrics)*100:.1f}%)")
    print(f"  R² >= 0.90: {(df_metrics['r2'] >= 0.90).sum()} stations ({(df_metrics['r2'] >= 0.90).sum()/len(df_metrics)*100:.1f}%)")
    print(f"  R² >= 0.80: {(df_metrics['r2'] >= 0.80).sum()} stations ({(df_metrics['r2'] >= 0.80).sum()/len(df_metrics)*100:.1f}%)")
    print(f"  R² >= 0.70: {(df_metrics['r2'] >= 0.70).sum()} stations ({(df_metrics['r2'] >= 0.70).sum()/len(df_metrics)*100:.1f}%)")
    print(f"  R² <  0.70: {(df_metrics['r2'] < 0.70).sum()} stations ({(df_metrics['r2'] < 0.70).sum()/len(df_metrics)*100:.1f}%)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
