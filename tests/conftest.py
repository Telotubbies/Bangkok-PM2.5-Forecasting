"""
Pytest configuration and shared fixtures for STC-HGAT testing.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_station_data():
    """Mock station data for testing"""
    return pd.DataFrame({
        'stationID': ['BKK01', 'BKK02', 'BKK03', 'BKK04', 'BKK05'],
        'name': ['Station 1', 'Station 2', 'Station 3', 'Station 4', 'Station 5'],
        'lat': [13.7563, 13.7467, 13.7308, 13.7200, 13.7100],
        'lon': [100.5018, 100.5349, 100.5418, 100.5500, 100.5600],
        'region': ['Central', 'Central', 'Central', 'East', 'East'],
        'area': ['Bangkok', 'Bangkok', 'Bangkok', 'Bangkok', 'Bangkok']
    })


@pytest.fixture
def sample_timeseries():
    """Mock time series data (seq_len, num_stations, features)"""
    return torch.randn(100, 5, 32)


@pytest.fixture
def sample_pm25_data():
    """Mock PM2.5 time series data"""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    stations = ['BKK01', 'BKK02', 'BKK03']
    
    data = []
    for station in stations:
        for date in dates:
            data.append({
                'date': date,
                'stationID': station,
                'pm2_5': np.random.uniform(10, 100),
                'pm10': np.random.uniform(20, 150),
                'temperature_2m': np.random.uniform(25, 35),
                'relative_humidity_2m': np.random.uniform(50, 90),
                'surface_pressure': np.random.uniform(1010, 1020),
                'precipitation': np.random.uniform(0, 10),
                'wind_speed_10m': np.random.uniform(0, 15),
                'wind_direction_10m': np.random.uniform(0, 360),
                'cloud_cover': np.random.uniform(0, 100)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_graph():
    """Mock graph structure (edge_index, edge_attr)"""
    num_nodes = 5
    num_edges = 10
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 16)
    return {'edge_index': edge_index, 'edge_attr': edge_attr}


@pytest.fixture
def sample_hypergraph():
    """Mock hypergraph structure"""
    num_nodes = 5
    num_hyperedges = 3
    # Each hyperedge connects multiple nodes
    hyperedge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3, 2, 3, 4],  # node indices
        [0, 0, 0, 1, 1, 1, 2, 2, 2]   # hyperedge indices
    ])
    return {'hyperedge_index': hyperedge_index}


@pytest.fixture
def device():
    """Get available device (CUDA/ROCm/MPS/CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure"""
    data_dir = tmp_path / "data"
    (data_dir / "bronze").mkdir(parents=True)
    (data_dir / "silver").mkdir(parents=True)
    (data_dir / "stations").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_config():
    """Sample configuration parameters"""
    return {
        'data': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'sequence_length': 30,
            'forecast_horizons': [1, 3, 7]
        },
        'graph': {
            'spatial_threshold_km': 50.0,
            'correlation_threshold': 0.3,
            'num_regions': 5
        },
        'model': {
            'hidden_dim': 64,
            'num_hypergat_layers': 2,
            'num_hgat_layers': 2,
            'num_heads': 4,
            'dropout': 0.2
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 10
        }
    }


@pytest.fixture
def mock_model_output():
    """Mock model predictions"""
    batch_size = 4
    num_stations = 5
    num_horizons = 3
    return torch.randn(batch_size, num_stations, num_horizons)
