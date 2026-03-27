"""
dataset.py
----------
Data loading, sequence creation, and PyTorch Dataset classes for STC-HGAT.

Key fixes vs original code:
  ❌ row-based split  → ✅ date-based split (no temporal leakage)
  ❌ variable node count not tracked → ✅ fixed station-to-index mapping
  ❌ fillna(0)        → ✅ forward-fill per station then median fill
  ❌ no normalization → ✅ StandardScaler fit on train only
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Date-based split  (fix: was row-based → data leakage)
# ---------------------------------------------------------------------------

def split_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique dates to avoid temporal leakage.

    Parameters
    ----------
    df         : must contain `date_col` and be sorted by it
    train_ratio: fraction of unique dates for training
    val_ratio  : fraction of unique dates for validation
    """
    dates = sorted(df[date_col].unique())
    n     = len(dates)
    t1    = int(n * train_ratio)
    t2    = int(n * (train_ratio + val_ratio))

    train_dates = set(dates[:t1])
    val_dates   = set(dates[t1:t2])
    test_dates  = set(dates[t2:])

    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df   = df[df[date_col].isin(val_dates)].copy()
    test_df  = df[df[date_col].isin(test_dates)].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. Missing-value handler  (fix: fillna(0) → forward-fill per station)
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    feature_cols: List[str],
    station_col: str = "stationID",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    1. Forward-fill within each station (temporal continuity).
    2. Backward-fill for leading NaNs.
    3. Fill remaining NaNs with feature median (global fallback).
    """
    df = df.sort_values([station_col, date_col]).copy()
    df[feature_cols] = (
        df.groupby(station_col)[feature_cols]
        .transform(lambda g: g.ffill().bfill())
    )
    # Global median fallback
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ---------------------------------------------------------------------------
# 3. Normalization  (fix: fit on train only, transform all)
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Wrapper around StandardScaler that tracks fit/transform state."""

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._scaler.fit_transform(X)
        self._fitted = True
        return result.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit_transform on train set first"
        return self._scaler.transform(X).astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(X)


class TargetScaler:
    """Per-station StandardScaler for the PM2.5 target."""

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y2d = y.reshape(-1, 1)
        result = self._scaler.fit_transform(y2d).ravel()
        self._fitted = True
        return result.astype(np.float32)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(y.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# 4. Sequence creation  (fix: fixed station index, no cross-station bleed)
# ---------------------------------------------------------------------------

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    station_order: List[str],
    lookback: int = 7,
    target_col: str = "pm2_5_mean",
    date_col: str = "date",
    station_col: str = "stationID",
    min_stations: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Create (X, y, masks) sequences for STC-HGAT.

    Station order is FIXED (station_order) so node index i always refers
    to the same physical station — fixes the original mask bug.

    Returns
    -------
    X_list   : list of ndarray (N_fixed, lookback, F)
    y_list   : list of ndarray (N_fixed,)
    mask_list: list of bool ndarray (N_fixed,)  True = station available
    """
    N_fixed  = len(station_order)
    F        = len(feature_cols)
    sid2idx  = {sid: i for i, sid in enumerate(station_order)}

    df = df.sort_values(date_col).copy()
    dates = sorted(df[date_col].unique())

    X_list, y_list, mask_list = [], [], []

    for t in range(len(dates) - lookback):
        window_dates = dates[t: t + lookback]
        target_date  = dates[t + lookback]

        # Pre-allocate padded tensors (zeros for missing stations)
        X    = np.zeros((N_fixed, lookback, F), dtype=np.float32)
        y    = np.zeros(N_fixed,               dtype=np.float32)
        mask = np.zeros(N_fixed,               dtype=bool)

        target_rows = df[df[date_col] == target_date]
        n_available = 0

        for _, row in target_rows.iterrows():
            sid = row[station_col]
            if sid not in sid2idx:
                continue
            idx = sid2idx[sid]

            if pd.isna(row[target_col]):
                continue

            # Gather lookback window for this station
            window_data = df[
                (df[station_col] == sid) & (df[date_col].isin(window_dates))
            ].sort_values(date_col)

            if len(window_data) < lookback:
                continue                          # skip incomplete windows

            X[idx]    = window_data[feature_cols].values.astype(np.float32)
            y[idx]    = float(row[target_col])
            mask[idx] = True
            n_available += 1

        if n_available >= min_stations:
            X_list.append(X)
            y_list.append(y)
            mask_list.append(mask)

    return X_list, y_list, mask_list


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset
# ---------------------------------------------------------------------------

class PM25GraphDataset(Dataset):
    """
    PyTorch Dataset for STC-HGAT.

    Each sample:
        x    : FloatTensor (N, T, F)   node features
        y    : FloatTensor (N,)        PM2.5 targets (0 for masked nodes)
        mask : BoolTensor  (N,)        True = node has valid target
    """

    def __init__(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        mask_list: List[np.ndarray],
    ):
        assert len(X_list) == len(y_list) == len(mask_list)
        self.X    = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y    = [torch.tensor(y, dtype=torch.float32) for y in y_list]
        self.mask = [torch.tensor(m, dtype=torch.bool)    for m in mask_list]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.mask[idx]

    @property
    def n_nodes(self) -> int:
        return self.X[0].shape[0]

    @property
    def seq_len(self) -> int:
        return self.X[0].shape[1]

    @property
    def n_features(self) -> int:
        return self.X[0].shape[2]


class PM25SequenceDataset(Dataset):
    """
    Simple sequence dataset for multi-horizon PM2.5 forecasting.
    
    Used in notebooks for demo training with mock data.
    Each sample: (X, y) where X is input sequence, y is multi-horizon targets.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        sequence_length: int = 7,
        forecast_horizons: List[int] = [1, 3, 7]
    ):
        """
        Parameters
        ----------
        data : Tensor of shape (T, N, F)
            T = timesteps, N = stations, F = features
        sequence_length : int
            Number of past timesteps to use as input
        forecast_horizons : List[int]
            Future timesteps to predict (e.g., [1, 3, 7] for 1-day, 3-day, 7-day)
        """
        self.data = data
        self.seq_len = sequence_length
        self.horizons = forecast_horizons
        self.max_horizon = max(forecast_horizons)
        
        # Valid indices where we have enough history and future
        self.valid_indices = list(range(
            sequence_length,
            len(data) - self.max_horizon
        ))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        X : Tensor (seq_len, N, F)
            Input sequence
        y : Tensor (N, len(horizons))
            Target values at each forecast horizon
        """
        t = self.valid_indices[idx]
        
        # Input: past seq_len timesteps
        X = self.data[t - self.seq_len : t]  # (seq_len, N, F)
        
        # Targets: future values at each horizon
        # Assume first feature is PM2.5
        y_list = []
        for h in self.horizons:
            y_h = self.data[t + h - 1, :, 0]  # (N,) - PM2.5 at horizon h
            y_list.append(y_h)
        
        y = torch.stack(y_list, dim=1)  # (N, len(horizons))
        
        return X, y


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack variable-node batches (all have same N_fixed here)."""
    X    = torch.stack([b[0] for b in batch])   # (B, N, T, F)
    y    = torch.stack([b[1] for b in batch])   # (B, N)
    mask = torch.stack([b[2] for b in batch])   # (B, N)
    return X, y, mask


# ---------------------------------------------------------------------------
# 6. Full data loading pipeline
# ---------------------------------------------------------------------------

def load_and_prepare(
    data_dir: str | Path,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "pm2_5_mean",
    lookback: int = 7,
    min_stations: int = 20,
    start_date: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end data loading pipeline.

    Returns
    -------
    dict with keys:
      train_ds, val_ds, test_ds  : PM25GraphDataset
      feature_scaler             : FeatureScaler (fitted on train)
      target_scaler              : TargetScaler  (fitted on train)
      station_order              : list[str]
      feature_cols               : list[str]
      station_meta               : DataFrame (stationID, lat, lon)
    """
    data_dir = Path(data_dir)

    # Load
    train_raw = pd.read_parquet(data_dir / "train.parquet")
    val_raw   = pd.read_parquet(data_dir / "val.parquet")
    test_raw  = pd.read_parquet(data_dir / "test.parquet")
    df_all    = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])

    if start_date:
        df_all = df_all[df_all["date"] >= start_date]

    df_all = df_all[df_all[target_col].notna()].copy()
    df_all = df_all.sort_values("date").reset_index(drop=True)

    # Feature columns
    exclude = {"date", "stationID", target_col, "split", "load_id", "lat", "lon"}
    if feature_cols is None:
        feature_cols = [c for c in df_all.columns if c not in exclude]

    # Station metadata
    station_meta = (
        df_all.groupby("stationID")[["lat", "lon"]]
        .first()
        .reset_index()
        .sort_values("stationID")
    )
    station_order = station_meta["stationID"].tolist()

    # Date-based split  ← FIX: was row-based
    train_df, val_df, test_df = split_by_date(df_all, train_ratio, val_ratio)

    # Fill missing values  ← FIX: was fillna(0)
    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = fill_missing(
            split_df, feature_cols
        )[feature_cols]

    # Normalization — fit on TRAIN ONLY  ← FIX
    feat_scaler   = FeatureScaler()
    target_scaler = TargetScaler()

    train_df[feature_cols] = feat_scaler.fit_transform(train_df[feature_cols].values)
    val_df[feature_cols]   = feat_scaler.transform(val_df[feature_cols].values)
    test_df[feature_cols]  = feat_scaler.transform(test_df[feature_cols].values)

    train_df[target_col] = target_scaler.fit_transform(train_df[target_col].values)
    val_df[target_col]   = target_scaler.transform(val_df[target_col].values)
    test_df[target_col]  = target_scaler.transform(test_df[target_col].values)

    # Create sequences with fixed station order  ← FIX
    def make_seqs(split_df):
        return create_sequences(
            split_df, feature_cols, station_order,
            lookback=lookback, target_col=target_col,
            min_stations=min_stations,
        )

    X_tr, y_tr, m_tr = make_seqs(train_df)
    X_va, y_va, m_va = make_seqs(val_df)
    X_te, y_te, m_te = make_seqs(test_df)

    if verbose:
        print(f"Stations    : {len(station_order)}")
        print(f"Features    : {len(feature_cols)}")
        print(f"Train seqs  : {len(X_tr)}")
        print(f"Val   seqs  : {len(X_va)}")
        print(f"Test  seqs  : {len(X_te)}")

    return {
        "train_ds":      PM25GraphDataset(X_tr, y_tr, m_tr),
        "val_ds":        PM25GraphDataset(X_va, y_va, m_va),
        "test_ds":       PM25GraphDataset(X_te, y_te, m_te),
        "feature_scaler": feat_scaler,
        "target_scaler":  target_scaler,
        "station_order":  station_order,
        "feature_cols":   feature_cols,
        "station_meta":   station_meta,
    }
