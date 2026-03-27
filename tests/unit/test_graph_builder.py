"""
Unit tests for graph_builder.py
"""
import pytest
import torch
import pandas as pd
import numpy as np
from src.utils.graph_builder import (
    haversine_km,
    build_spatial_hypergraph,
    build_temporal_graph,
    compute_region_embeddings
)


class TestHaversineDistance:
    """Test haversine distance calculation"""
    
    def test_same_point(self):
        """Distance between same point should be 0"""
        dist = haversine_km(13.7563, 100.5018, 13.7563, 100.5018)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)
    
    def test_known_distance(self):
        """Test against known distance (Bangkok to Chiang Mai ~600km)"""
        dist = haversine_km(13.7563, 100.5018, 18.7883, 98.9853)
        assert 580 < dist < 620
    
    def test_vectorized(self):
        """Test vectorized computation"""
        lat1 = torch.tensor([13.7563, 13.7467])
        lon1 = torch.tensor([100.5018, 100.5349])
        lat2 = torch.tensor([13.7308, 13.7308])
        lon2 = torch.tensor([100.5418, 100.5418])
        
        dists = haversine_km(lat1, lon1, lat2, lon2)
        assert dists.shape == (2,)
        assert all(dists > 0)
    
    def test_symmetry(self):
        """Distance should be symmetric"""
        dist1 = haversine_km(13.7563, 100.5018, 13.7467, 100.5349)
        dist2 = haversine_km(13.7467, 100.5349, 13.7563, 100.5018)
        assert torch.isclose(dist1, dist2)


class TestSpatialGraph:
    """Test spatial graph construction"""
    
    def test_graph_structure(self, sample_station_data):
        """Test basic graph structure"""
        graph = build_spatial_hypergraph(
            sample_station_data,
            threshold_km=50.0
        )
        
        assert 'edge_index' in graph or 'hyperedge_index' in graph
        if 'edge_index' in graph:
            assert graph['edge_index'].shape[0] == 2
            assert graph['edge_index'].max() < len(sample_station_data)
    
    def test_threshold_filtering(self, sample_station_data):
        """Larger threshold should produce more edges"""
        graph_small = build_spatial_hypergraph(
            sample_station_data, threshold_km=1.0
        )
        graph_large = build_spatial_hypergraph(
            sample_station_data, threshold_km=100.0
        )
        
        # Count edges
        if 'edge_index' in graph_small and 'edge_index' in graph_large:
            edges_small = graph_small['edge_index'].shape[1]
            edges_large = graph_large['edge_index'].shape[1]
            assert edges_large >= edges_small
    
    def test_no_self_loops(self, sample_station_data):
        """Graph should not contain self-loops"""
        graph = build_spatial_hypergraph(sample_station_data)
        
        if 'edge_index' in graph:
            edge_index = graph['edge_index']
            # Check no self-loops
            assert not torch.any(edge_index[0] == edge_index[1])


class TestTemporalGraph:
    """Test temporal graph construction"""
    
    def test_sequential_edges(self):
        """Test sequential day connections"""
        graph = build_temporal_graph(num_days=365, seasonal_pattern=False)
        
        assert 'edge_index' in graph
        # Should have at least 364 sequential edges (day i -> day i+1)
        assert graph['edge_index'].shape[1] >= 364
    
    def test_seasonal_edges(self):
        """Seasonal pattern should add more edges"""
        graph_with_seasonal = build_temporal_graph(
            num_days=365, seasonal_pattern=True
        )
        graph_without_seasonal = build_temporal_graph(
            num_days=365, seasonal_pattern=False
        )
        
        edges_with = graph_with_seasonal['edge_index'].shape[1]
        edges_without = graph_without_seasonal['edge_index'].shape[1]
        
        assert edges_with > edges_without
    
    def test_edge_index_bounds(self):
        """Edge indices should be within valid range"""
        num_days = 100
        graph = build_temporal_graph(num_days=num_days)
        
        edge_index = graph['edge_index']
        assert edge_index.min() >= 0
        assert edge_index.max() < num_days


class TestRegionEmbeddings:
    """Test region embedding computation"""
    
    def test_embedding_shape(self, sample_station_data):
        """Test output shape of region embeddings"""
        num_regions = 5
        embeddings = compute_region_embeddings(
            sample_station_data,
            num_regions=num_regions
        )
        
        assert embeddings.shape[0] == num_regions
        assert embeddings.dim() == 2  # (num_regions, embedding_dim)
    
    def test_embedding_values(self, sample_station_data):
        """Embeddings should be finite"""
        embeddings = compute_region_embeddings(sample_station_data)
        
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
