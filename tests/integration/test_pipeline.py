"""
Integration tests for end-to-end pipeline
"""
import pytest
import torch
import pandas as pd
from pathlib import Path
from src.data.dataset import prepare_datasets, PM25SequenceDataset
from src.utils.graph_builder import build_spatial_hypergraph, build_temporal_graph
from src.models.stc_hgat_model import STCHGAT


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete pipeline from data to model"""
    
    def test_data_to_model_flow(self, sample_station_data, sample_pm25_data, device):
        """Test data flows correctly through entire pipeline"""
        # 1. Build graphs
        spatial_graph = build_spatial_hypergraph(
            sample_station_data,
            threshold_km=50.0
        )
        temporal_graph = build_temporal_graph(num_days=365)
        
        # 2. Create sequences from data
        # Simulate processed data
        num_stations = len(sample_station_data)
        seq_len = 30
        num_features = 9
        
        data = torch.randn(100, num_stations, num_features)
        
        dataset = PM25SequenceDataset(
            data,
            sequence_length=seq_len,
            forecast_horizon=1
        )
        
        # 3. Create model
        model = STCHGAT(
            num_features=num_features,
            hidden_dim=64,
            num_stations=num_stations,
            num_regions=5,
            forecast_horizons=[1]
        ).to(device)
        
        # 4. Forward pass
        X, y = dataset[0]
        X = X.unsqueeze(0).to(device)  # Add batch dimension
        
        spatial_graph_device = {k: v.to(device) for k, v in spatial_graph.items()}
        temporal_graph_device = {k: v.to(device) for k, v in temporal_graph.items()}
        
        out = model(X, spatial_graph_device, temporal_graph_device)
        
        assert out.shape == (1, num_stations, 1)
        assert not torch.isnan(out).any()
    
    def test_training_step(self, sample_station_data, device):
        """Test a complete training step"""
        # Setup
        num_stations = len(sample_station_data)
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=num_stations,
            num_regions=5
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Create dummy data
        batch_size = 4
        X = torch.randn(batch_size, 30, num_stations, 32).to(device)
        y = torch.randn(batch_size, num_stations, 3).to(device)
        
        # Create graphs
        edge_index = torch.randint(0, num_stations, (2, 20)).to(device)
        spatial_graph = {'edge_index': edge_index}
        temporal_graph = {'edge_index': torch.randint(0, 30, (2, 50)).to(device)}
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        out = model(X, spatial_graph, temporal_graph)
        loss = criterion(out, y)
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_evaluation_step(self, sample_station_data, device):
        """Test evaluation mode"""
        num_stations = len(sample_station_data)
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=num_stations,
            num_regions=5
        ).to(device)
        
        model.eval()
        
        X = torch.randn(4, 30, num_stations, 32).to(device)
        edge_index = torch.randint(0, num_stations, (2, 20)).to(device)
        spatial_graph = {'edge_index': edge_index}
        temporal_graph = {'edge_index': torch.randint(0, 30, (2, 50)).to(device)}
        
        with torch.no_grad():
            out = model(X, spatial_graph, temporal_graph)
        
        assert out.shape == (4, num_stations, 3)
        assert not torch.isnan(out).any()
    
    def test_save_and_load_model(self, sample_station_data, tmp_path, device):
        """Test model can be saved and loaded"""
        num_stations = len(sample_station_data)
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=num_stations,
            num_regions=5
        ).to(device)
        
        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=num_stations,
            num_regions=5
        ).to(device)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Test both models produce same output
        X = torch.randn(2, 30, num_stations, 32).to(device)
        edge_index = torch.randint(0, num_stations, (2, 20)).to(device)
        spatial_graph = {'edge_index': edge_index}
        temporal_graph = {'edge_index': torch.randint(0, 30, (2, 50)).to(device)}
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            out1 = model(X, spatial_graph, temporal_graph)
            out2 = loaded_model(X, spatial_graph, temporal_graph)
        
        assert torch.allclose(out1, out2, atol=1e-6)
