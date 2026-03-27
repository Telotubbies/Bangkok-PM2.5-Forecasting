"""
Unit tests for dataset.py
"""
import pytest
import pandas as pd
import torch
from src.data.dataset import (
    split_by_date,
    PM25SequenceDataset,
    create_sequences
)


class TestDateSplit:
    """Test date-based train/val/test splitting"""
    
    def test_split_ratios(self):
        """Test that split ratios are correct"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'stationID': ['BKK01'] * 100,
            'value': range(100)
        })
        
        train, val, test = split_by_date(df, train_ratio=0.7, val_ratio=0.15)
        
        total_dates = len(df['date'].unique())
        train_dates = len(train['date'].unique())
        val_dates = len(val['date'].unique())
        test_dates = len(test['date'].unique())
        
        assert train_dates == int(total_dates * 0.7)
        assert val_dates == int(total_dates * 0.15)
        assert train_dates + val_dates + test_dates == total_dates
    
    def test_no_overlap(self):
        """Test that splits don't overlap"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'stationID': ['BKK01'] * 100,
            'value': range(100)
        })
        
        train, val, test = split_by_date(df)
        
        train_dates = set(train['date'].unique())
        val_dates = set(val['date'].unique())
        test_dates = set(test['date'].unique())
        
        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0
    
    def test_chronological_order(self):
        """Test that splits maintain chronological order"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'stationID': ['BKK01'] * 100,
            'value': range(100)
        })
        
        train, val, test = split_by_date(df)
        
        assert train['date'].max() < val['date'].min()
        assert val['date'].max() < test['date'].min()
    
    def test_multiple_stations(self):
        """Test splitting with multiple stations"""
        dates = pd.date_range('2020-01-01', periods=50)
        df = pd.DataFrame({
            'date': list(dates) * 3,
            'stationID': ['BKK01'] * 50 + ['BKK02'] * 50 + ['BKK03'] * 50,
            'value': range(150)
        })
        
        train, val, test = split_by_date(df)
        
        # All stations should be present in all splits
        assert set(train['stationID'].unique()) == {'BKK01', 'BKK02', 'BKK03'}
        assert set(val['stationID'].unique()) == {'BKK01', 'BKK02', 'BKK03'}
        assert set(test['stationID'].unique()) == {'BKK01', 'BKK02', 'BKK03'}


class TestSequenceDataset:
    """Test PM25SequenceDataset"""
    
    def test_dataset_length(self, sample_timeseries):
        """Test dataset length calculation"""
        seq_len = 10
        horizon = 1
        
        dataset = PM25SequenceDataset(
            sample_timeseries,
            sequence_length=seq_len,
            forecast_horizon=horizon
        )
        
        expected_len = len(sample_timeseries) - seq_len - horizon + 1
        assert len(dataset) == expected_len
    
    def test_sequence_shape(self, sample_timeseries):
        """Test output sequence shapes"""
        seq_len = 10
        horizon = 1
        
        dataset = PM25SequenceDataset(
            sample_timeseries,
            sequence_length=seq_len,
            forecast_horizon=horizon
        )
        
        X, y = dataset[0]
        
        assert X.shape[0] == seq_len
        assert X.shape[1:] == sample_timeseries.shape[1:]
        assert y.shape == (sample_timeseries.shape[1],)  # (num_stations,)
    
    def test_multiple_horizons(self, sample_timeseries):
        """Test with multiple forecast horizons"""
        seq_len = 10
        horizons = [1, 3, 7]
        
        dataset = PM25SequenceDataset(
            sample_timeseries,
            sequence_length=seq_len,
            forecast_horizons=horizons
        )
        
        X, y = dataset[0]
        
        assert X.shape[0] == seq_len
        assert y.shape == (sample_timeseries.shape[1], len(horizons))
    
    def test_indexing(self, sample_timeseries):
        """Test that indexing works correctly"""
        seq_len = 10
        dataset = PM25SequenceDataset(
            sample_timeseries,
            sequence_length=seq_len,
            forecast_horizon=1
        )
        
        # Test first and last indices
        X_first, y_first = dataset[0]
        X_last, y_last = dataset[len(dataset) - 1]
        
        assert X_first.shape == X_last.shape
        assert y_first.shape == y_last.shape
    
    def test_no_data_leakage(self, sample_timeseries):
        """Test that future data doesn't leak into input"""
        seq_len = 10
        horizon = 3
        
        dataset = PM25SequenceDataset(
            sample_timeseries,
            sequence_length=seq_len,
            forecast_horizon=horizon
        )
        
        X, y = dataset[0]
        
        # Input should be from indices 0:seq_len
        # Target should be from index seq_len + horizon - 1
        assert torch.allclose(X, sample_timeseries[:seq_len])


class TestCreateSequences:
    """Test sequence creation function"""
    
    def test_basic_sequence_creation(self):
        """Test basic sequence creation"""
        data = torch.randn(100, 5, 10)  # (time, stations, features)
        seq_len = 20
        horizon = 1
        
        sequences = create_sequences(data, seq_len, horizon)
        
        assert len(sequences) == 100 - seq_len - horizon + 1
        assert sequences[0][0].shape == (seq_len, 5, 10)
    
    def test_edge_cases(self):
        """Test edge cases"""
        data = torch.randn(30, 5, 10)
        
        # Sequence length equal to data length
        with pytest.raises((ValueError, AssertionError)):
            create_sequences(data, sequence_length=30, forecast_horizon=1)
        
        # Sequence length + horizon > data length
        with pytest.raises((ValueError, AssertionError)):
            create_sequences(data, sequence_length=20, forecast_horizon=15)
