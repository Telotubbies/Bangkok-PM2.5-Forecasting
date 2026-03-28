#!/usr/bin/env python3
"""
train_improved.py
-----------------
Training script for Improved STC-HGAT (Phase 3)
With gated fusion, cross-attention, and multi-scale temporal modeling
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.stc_hgat_improved import ImprovedSTCHGAT
from src.data.dataset import PM25SequenceDataset
from src.data.real_data_loader import load_pm25_data, load_weather_data, combine_features
from src.utils.graph_builder import build_spatial_hypergraph, build_temporal_graph
from src.utils.evaluator import calculate_mae, calculate_rmse, calculate_r2


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple MPS")
    else:
        device = torch.device('cpu')
        print("⚠️  CPU")
    return device


def prepare_data(data, params):
    print("\n✂️  Data Splits:")
    print("=" * 50)
    
    train_ratio = params['data']['train_ratio']
    val_ratio = params['data']['val_ratio']
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    train_mean = train_data.mean(dim=(0, 1), keepdim=True)
    train_std = train_data.std(dim=(0, 1), keepdim=True) + 1e-8
    
    train_data_norm = (train_data - train_mean) / train_std
    val_data_norm = (val_data - train_mean) / train_std
    test_data_norm = (test_data - train_mean) / train_std
    
    print("=" * 50)
    
    return train_data_norm, val_data_norm, test_data_norm, train_mean, train_std


def create_dataloaders(train_data, val_data, test_data, params):
    seq_len = params['data']['sequence_length']
    horizons = params['data']['forecast_horizons']
    batch_size = params['training']['batch_size']
    
    train_dataset = PM25SequenceDataset(train_data, seq_len, horizons)
    val_dataset = PM25SequenceDataset(val_data, seq_len, horizons)
    test_dataset = PM25SequenceDataset(test_data, seq_len, horizons)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, num_stations, gradient_clip, epoch):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if X.shape[1] != num_stations:
            X = X.permute(0, 2, 1, 3)
        
        pred, h_s, h_t = model(X, None, None)
        y_target = y[:, :, 0] if len(y.shape) == 3 else y
        
        # Use model's compute_loss for combined loss
        loss, loss_dict = model.compute_loss(pred, y_target, h_s, h_t)
        
        loss.backward()
        
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def validate(model, val_loader, device, num_stations):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            
            if X.shape[1] != num_stations:
                X = X.permute(0, 2, 1, 3)
            
            pred, h_s, h_t = model(X, None, None)
            y_target = y[:, :, 0] if len(y.shape) == 3 else y
            
            loss, _ = model.compute_loss(pred, y_target, h_s, h_t)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def evaluate(model, test_loader, device, num_stations, horizons):
    print("\n📈 Test Evaluation:")
    print("=" * 50)
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            if X.shape[1] != num_stations:
                X = X.permute(0, 2, 1, 3)
            
            pred, h_s, h_t = model(X, None, None)
            pred_expanded = pred.unsqueeze(-1).expand(-1, -1, len(horizons))
            
            test_predictions.append(pred_expanded.cpu())
            test_targets.append(y.cpu())
    
    test_predictions = torch.cat(test_predictions, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    results = {}
    for i, horizon in enumerate(horizons):
        pred_h = test_predictions[:, :, i]
        target_h = test_targets[:, :, i]
        
        mae = calculate_mae(target_h, pred_h)
        rmse = calculate_rmse(target_h, pred_h)
        r2 = calculate_r2(target_h, pred_h)
        
        results[f'horizon_{horizon}d'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"\nHorizon +{horizon}d:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    print("=" * 50)
    
    return results, test_predictions, test_targets


def main():
    parser = argparse.ArgumentParser(description='Train Improved STC-HGAT')
    parser.add_argument('--config', type=str, default='params.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default='2024-11-30')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--use-weather', action='store_true', default=True)
    parser.add_argument('--no-gated-fusion', action='store_true', help='Disable gated fusion')
    parser.add_argument('--no-cross-attention', action='store_true', help='Disable cross-attention')
    parser.add_argument('--no-multiscale', action='store_true', help='Disable multi-scale temporal')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Improved STC-HGAT Training (Phase 3)")
    print("=" * 60)
    
    device = setup_device()
    
    with open(project_root / args.config) as f:
        params = yaml.safe_load(f)
    
    if args.epochs:
        params['training']['epochs'] = args.epochs
    if args.batch_size:
        params['training']['batch_size'] = args.batch_size
    
    # Load data
    print("\n🗺️  Loading Station Data:")
    stations_df = pd.read_parquet(project_root / 'data/stations/bangkok_stations.parquet')
    num_stations = len(stations_df)
    print(f"Stations: {num_stations}")
    
    print("\n📊 Loading PM2.5 Data:")
    pm25_data, metadata = load_pm25_data(
        project_root / 'data', stations_df,
        start_date=args.start_date, end_date=args.end_date
    )
    
    weather_data = None
    if args.use_weather:
        print("\n🌤️  Loading Weather Data:")
        weather_data, _ = load_weather_data(
            project_root / 'data', stations_df,
            start_date=args.start_date, end_date=args.end_date
        )
    
    print("\n🔧 Combining Features:")
    data, feature_names = combine_features(
        pm25_data, weather_data=weather_data,
        add_temporal_features=True, metadata=metadata
    )
    
    num_features = data.shape[2]
    print(f"Total features: {num_features}")
    
    # Prepare data
    train_data, val_data, test_data, train_mean, train_std = prepare_data(data, params)
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, params)
    
    # Initialize improved model
    print("\n🤖 Initializing Improved Model:")
    print("=" * 50)
    print(f"Enhancements:")
    print(f"  ✅ Gated Fusion: {not args.no_gated_fusion}")
    print(f"  ✅ Cross-Attention: {not args.no_cross_attention}")
    print(f"  ✅ Multi-Scale Temporal: {not args.no_multiscale}")
    
    model = ImprovedSTCHGAT(
        num_features=num_features,
        hidden_dim=params['model']['hidden_dim'],
        num_stations=num_stations,
        num_regions=params['graph']['num_regions'],
        num_hypergat_layers=params['model']['num_hypergat_layers'],
        num_hgat_layers=params['model']['num_hgat_layers'],
        num_heads=params['model']['num_heads'],
        dropout=params['model']['dropout'],
        forecast_horizons=params['data']['forecast_horizons'],
        use_gated_fusion=not args.no_gated_fusion,
        use_cross_attention=not args.no_cross_attention,
        use_multiscale_temporal=not args.no_multiscale,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print("=" * 50)
    
    # Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['training']['learning_rate'],
        weight_decay=params['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['training']['scheduler_factor'],
        patience=params['training']['scheduler_patience'],
        verbose=True
    )
    
    # Training loop
    print("\n🚀 Training:")
    print("=" * 50)
    
    num_epochs = params['training']['epochs']
    early_stopping_patience = params['training']['early_stopping_patience']
    gradient_clip = params['training']['gradient_clip_value']
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, 
                                num_stations, gradient_clip, epoch)
        val_loss = validate(model, val_loader, device, num_stations)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f} ⭐")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Best model loaded (val_loss: {best_val_loss:.4f})")
    
    # Evaluate
    results, test_preds, test_targets = evaluate(
        model, test_loader, device, num_stations, params['data']['forecast_horizons']
    )
    
    # Save model
    save_dir = project_root / args.save_dir
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_dir / f'stc_hgat_improved_{timestamp}.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'params': params,
        'train_mean': train_mean,
        'train_std': train_std,
        'feature_names': feature_names,
        'results': results,
        'model_config': {
            'use_gated_fusion': not args.no_gated_fusion,
            'use_cross_attention': not args.no_cross_attention,
            'use_multiscale_temporal': not args.no_multiscale,
        }
    }, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE - IMPROVED MODEL")
    print("=" * 60)
    print(f"\n📊 Best val loss: {best_val_loss:.4f}")
    print(f"Epochs: {len(history['train_loss'])}")
    print(f"\n🎯 Test Performance:")
    for horizon in params['data']['forecast_horizons']:
        r = results[f'horizon_{horizon}d']
        print(f"  +{horizon}d: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, R²={r['r2']:.4f}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
