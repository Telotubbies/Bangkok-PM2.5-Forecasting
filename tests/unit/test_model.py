"""
Unit tests for stc_hgat_model.py
"""
import pytest
import torch
import torch.nn as nn
from src.models.stc_hgat_model import (
    HyperGATLayer,
    HGATLayer,
    STCHGAT
)


class TestHyperGATLayer:
    """Test HyperGAT layer"""
    
    def test_forward_pass(self, device):
        """Test basic forward pass"""
        layer = HyperGATLayer(
            in_dim=32,
            out_dim=64,
            num_heads=4
        ).to(device)
        
        x = torch.randn(10, 32).to(device)  # 10 nodes, 32 features
        hyperedge_index = torch.randint(0, 10, (2, 20)).to(device)
        
        out = layer(x, hyperedge_index)
        
        assert out.shape == (10, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_attention_weights(self, device):
        """Test attention weight computation"""
        layer = HyperGATLayer(
            in_dim=32,
            out_dim=64,
            num_heads=4,
            return_attention=True
        ).to(device)
        
        x = torch.randn(10, 32).to(device)
        hyperedge_index = torch.randint(0, 10, (2, 20)).to(device)
        
        out, attn = layer(x, hyperedge_index)
        
        # Attention weights should be positive
        assert (attn >= 0).all()
        assert not torch.isnan(attn).any()
    
    def test_different_dimensions(self, device):
        """Test with different input/output dimensions"""
        for in_dim, out_dim in [(16, 32), (64, 128), (128, 64)]:
            layer = HyperGATLayer(in_dim, out_dim, num_heads=2).to(device)
            x = torch.randn(5, in_dim).to(device)
            hyperedge_index = torch.randint(0, 5, (2, 10)).to(device)
            
            out = layer(x, hyperedge_index)
            assert out.shape == (5, out_dim)


class TestHGATLayer:
    """Test HGAT layer"""
    
    def test_forward_pass(self, device):
        """Test basic forward pass"""
        layer = HGATLayer(
            in_dim=32,
            out_dim=64,
            num_heads=4
        ).to(device)
        
        x = torch.randn(10, 32).to(device)
        edge_index = torch.randint(0, 10, (2, 20)).to(device)
        
        out = layer(x, edge_index)
        
        assert out.shape == (10, 64)
        assert not torch.isnan(out).any()
    
    def test_with_edge_attributes(self, device):
        """Test with edge attributes"""
        layer = HGATLayer(
            in_dim=32,
            out_dim=64,
            num_heads=4,
            edge_dim=16
        ).to(device)
        
        x = torch.randn(10, 32).to(device)
        edge_index = torch.randint(0, 10, (2, 20)).to(device)
        edge_attr = torch.randn(20, 16).to(device)
        
        out = layer(x, edge_index, edge_attr)
        
        assert out.shape == (10, 64)


class TestSTCHGAT:
    """Test full STC-HGAT model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=10,
            num_regions=5,
            forecast_horizons=[1, 3, 7]
        )
        
        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 64
        assert model.num_stations == 10
    
    def test_forward_pass(self, device, sample_graph):
        """Test forward pass through full model"""
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=10,
            num_regions=5,
            forecast_horizons=[1, 3, 7]
        ).to(device)
        
        batch_size = 8
        seq_len = 30
        num_stations = 10
        num_features = 32
        
        x = torch.randn(batch_size, seq_len, num_stations, num_features).to(device)
        spatial_graph = {k: v.to(device) for k, v in sample_graph.items()}
        temporal_graph = {k: v.to(device) for k, v in sample_graph.items()}
        
        out = model(x, spatial_graph, temporal_graph)
        
        assert out.shape == (batch_size, num_stations, 3)  # 3 horizons
        assert not torch.isnan(out).any()
    
    def test_gradient_flow(self, device):
        """Test that gradients flow through the model"""
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=10,
            num_regions=5
        ).to(device)
        
        x = torch.randn(4, 30, 10, 32, requires_grad=True).to(device)
        
        # Create simple graphs
        edge_index = torch.randint(0, 10, (2, 20)).to(device)
        spatial_graph = {'edge_index': edge_index}
        temporal_graph = {'edge_index': torch.randint(0, 30, (2, 50)).to(device)}
        
        out = model(x, spatial_graph, temporal_graph)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_different_batch_sizes(self, device):
        """Test model with different batch sizes"""
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=10,
            num_regions=5
        ).to(device)
        
        edge_index = torch.randint(0, 10, (2, 20)).to(device)
        spatial_graph = {'edge_index': edge_index}
        temporal_graph = {'edge_index': torch.randint(0, 30, (2, 50)).to(device)}
        
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 30, 10, 32).to(device)
            out = model(x, spatial_graph, temporal_graph)
            assert out.shape[0] == batch_size
    
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        model = STCHGAT(
            num_features=32,
            hidden_dim=64,
            num_stations=10,
            num_regions=5
        )
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check parameters require gradients
        assert all(p.requires_grad for p in params)
        
        # Check parameters are not all zeros
        assert not all(torch.allclose(p, torch.zeros_like(p)) for p in params)
