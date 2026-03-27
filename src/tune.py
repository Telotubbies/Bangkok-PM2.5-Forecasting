"""
Hyperparameter tuning with Optuna and MLflow integration.
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.stc_hgat_model import STCHGAT
from src.data.dataset import PM25SequenceDataset
from src.utils.mlflow_config import setup_mlflow


def load_data(params: Dict[str, Any]):
    """Load processed data for training"""
    processed_dir = Path(params['paths']['processed_data'])
    
    train_data = torch.load(processed_dir / 'train.pt')
    val_data = torch.load(processed_dir / 'val.pt')
    spatial_graph = torch.load(processed_dir / 'spatial_graph.pt')
    temporal_graph = torch.load(processed_dir / 'temporal_graph.pt')
    
    return train_data, val_data, spatial_graph, temporal_graph


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    spatial_graph: Dict,
    temporal_graph: Dict,
    params: Dict[str, Any],
    device: torch.device
) -> float:
    """
    Train model and return validation loss.
    
    Returns
    -------
    val_loss : float
        Best validation loss achieved
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['training']['learning_rate'],
        weight_decay=params['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['training']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(X, spatial_graph, temporal_graph)
            loss = criterion(out, y)
            loss.backward()
            
            # Gradient clipping
            if params['training'].get('gradient_clip_value'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    params['training']['gradient_clip_value']
                )
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X, spatial_graph, temporal_graph)
                loss = criterion(out, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= params['training']['early_stopping_patience']:
            break
    
    return best_val_loss


def objective(trial: optuna.Trial, base_params: Dict[str, Any], device: torch.device) -> float:
    """
    Optuna objective function.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    base_params : dict
        Base configuration parameters
    device : torch.device
        Device to run on
        
    Returns
    -------
    val_loss : float
        Validation loss to minimize
    """
    # Suggest hyperparameters
    params = base_params.copy()
    params['model']['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 256, step=64)
    params['model']['num_hypergat_layers'] = trial.suggest_int('num_hypergat_layers', 1, 3)
    params['model']['num_hgat_layers'] = trial.suggest_int('num_hgat_layers', 1, 3)
    params['model']['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
    params['model']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
    
    params['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    params['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    params['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    params['loss']['lambda_contrastive'] = trial.suggest_float('lambda_contrastive', 0.01, 0.5)
    params['loss']['temperature'] = trial.suggest_float('temperature', 0.05, 0.2)
    
    # Load data
    train_data, val_data, spatial_graph, temporal_graph = load_data(params)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=params['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params['training']['batch_size'],
        shuffle=False
    )
    
    # Move graphs to device
    spatial_graph = {k: v.to(device) for k, v in spatial_graph.items()}
    temporal_graph = {k: v.to(device) for k, v in temporal_graph.items()}
    
    # Create model
    num_features = train_data[0][0].shape[-1]
    num_stations = train_data[0][0].shape[-2]
    
    model = STCHGAT(
        num_features=num_features,
        hidden_dim=params['model']['hidden_dim'],
        num_stations=num_stations,
        num_regions=params['graph']['num_regions'],
        num_hypergat_layers=params['model']['num_hypergat_layers'],
        num_hgat_layers=params['model']['num_hgat_layers'],
        num_heads=params['model']['num_heads'],
        dropout=params['model']['dropout'],
        forecast_horizons=params['data']['forecast_horizons']
    ).to(device)
    
    # Train and get validation loss
    val_loss = train_model(
        model, train_loader, val_loader,
        spatial_graph, temporal_graph,
        params, device
    )
    
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--params', default='params.yaml', help='Path to params file')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--study-name', default='stc-hgat-optimization', help='Study name')
    args = parser.parse_args()
    
    # Load base parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup MLflow
    setup_mlflow(
        tracking_uri=params['mlflow']['tracking_uri'],
        experiment_name=params['mlflow']['experiment_name']
    )
    
    # Create MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=params['optuna']['metric']
    )
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        direction=params['optuna']['direction'],
        storage=params['optuna']['storage'],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, params, device),
        n_trials=args.n_trials,
        callbacks=[mlflc],
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params = params.copy()
    for key, value in study.best_params.items():
        if key in ['hidden_dim', 'num_hypergat_layers', 'num_hgat_layers', 'num_heads', 'dropout']:
            best_params['model'][key] = value
        elif key in ['learning_rate', 'batch_size', 'weight_decay']:
            best_params['training'][key] = value
        elif key in ['lambda_contrastive', 'temperature']:
            best_params['loss'][key] = value
    
    output_path = Path('params_best.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    print(f"\nBest parameters saved to: {output_path}")


if __name__ == '__main__':
    main()
