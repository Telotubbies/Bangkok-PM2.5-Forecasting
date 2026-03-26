# ============================================================================
# CELL 1: Setup & Imports
# ============================================================================
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

# ============================================================================
# CELL 2: Configuration
# ============================================================================
CONFIG = {
    # Data - USE SILVER LAYER (has actual PM2.5 data)
    'aq_data_dir': '../data/silver/openmeteo_airquality',
    'weather_data_dir': '../data/silver/openmeteo_weather',
    'start_date': '2022-01-01',
    'lookback': 7,
    'min_stations': 20,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    
    # Model architecture
    'hidden': 128,
    'n_regions': 5,
    'hypergat_layers': 2,
    'dropout': 0.1,
    
    # Loss
    'contrastive_lambda': 0.1,
    'aw_gamma': 2.0,
    'extreme_threshold': 50.0,  # PM2.5 threshold for extreme events
    
    # Graph construction
    'spatial_thresholds_km': (50.0, 100.0, 200.0),
    'spatial_edge_km': 150.0,
    'corr_threshold': 0.70,
    
    # Training
    'epochs': 100,
    'batch_size': 8,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
    'grad_clip': 1.0,
    'device': str(DEVICE),
}

THAILAND_REGIONS: Dict[str, List[str]] = {
    "North":     ["CM", "LM", "LP", "PY", "NAN", "PR", "PL", "MSH", "TAK", "SUK"],
    "Northeast": ["KKN", "UDT", "NKP", "SK", "BRM", "ROI", "MKM", "YST", "SR"],
    "Central":   ["BKK", "NPT", "AYA", "SBR", "NBR", "CNB", "KRI", "PKN", "SMT"],
    "East":      ["RY", "CH", "TRT", "PKO", "SA"],
    "South":     ["HYI", "PKT", "SNG", "NRT", "PT", "KBI", "TG", "PAN"],
}
REGION_NAMES = list(THAILAND_REGIONS.keys())

print("Configuration loaded.")
print(f"Regions: {REGION_NAMES}")

# ============================================================================
# CELL 3: Data Utilities - Split, Fill, Scale
# ============================================================================

def split_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique dates to avoid temporal leakage."""
    dates = sorted(df[date_col].unique())
    n = len(dates)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    train_dates = set(dates[:t1])
    val_dates = set(dates[t1:t2])
    test_dates = set(dates[t2:])

    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[df[date_col].isin(val_dates)].copy()
    test_df = df[df[date_col].isin(test_dates)].copy()

    return train_df, val_df, test_df


def fill_missing(
    df: pd.DataFrame,
    feature_cols: List[str],
    station_col: str = "stationID",
    date_col: str = "date",
) -> pd.DataFrame:
    """Forward-fill within each station, then median fill."""
    df = df.sort_values([station_col, date_col]).copy()
    df[feature_cols] = (
        df.groupby(station_col)[feature_cols]
        .transform(lambda g: g.ffill().bfill())
    )
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


class FeatureScaler:
    """StandardScaler wrapper for features."""
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
    """StandardScaler for PM2.5 target."""
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


print("Data utilities defined.")

# ============================================================================
# CELL 4: Sequence Creation & PyTorch Dataset
# ============================================================================

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    station_order: List[str],
    lookback: int = 7,
    target_col: str = "pm2_5",
    date_col: str = "date",
    station_col: str = "stationID",
    min_stations: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Create (X, y, masks) sequences with fixed station-to-index mapping."""
    N_fixed = len(station_order)
    F = len(feature_cols)
    sid2idx = {sid: i for i, sid in enumerate(station_order)}

    df = df.sort_values(date_col).copy()
    dates = sorted(df[date_col].unique())

    X_list, y_list, mask_list = [], [], []

    for t in range(len(dates) - lookback):
        window_dates = dates[t: t + lookback]
        target_date = dates[t + lookback]

        X = np.zeros((N_fixed, lookback, F), dtype=np.float32)
        y = np.zeros(N_fixed, dtype=np.float32)
        mask = np.zeros(N_fixed, dtype=bool)

        target_rows = df[df[date_col] == target_date]
        n_available = 0

        for _, row in target_rows.iterrows():
            sid = row[station_col]
            if sid not in sid2idx:
                continue
            idx = sid2idx[sid]

            if pd.isna(row[target_col]):
                continue

            window_data = df[
                (df[station_col] == sid) & (df[date_col].isin(window_dates))
            ].sort_values(date_col)

            if len(window_data) < lookback:
                continue

            X[idx] = window_data[feature_cols].values.astype(np.float32)
            y[idx] = float(row[target_col])
            mask[idx] = True
            n_available += 1

        if n_available >= min_stations:
            X_list.append(X)
            y_list.append(y)
            mask_list.append(mask)

    return X_list, y_list, mask_list


class PM25GraphDataset(Dataset):
    """PyTorch Dataset for STC-HGAT."""

    def __init__(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        mask_list: List[np.ndarray],
    ):
        assert len(X_list) == len(y_list) == len(mask_list)
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y = [torch.tensor(y, dtype=torch.float32) for y in y_list]
        self.mask = [torch.tensor(m, dtype=torch.bool) for m in mask_list]

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


def collate_fn(batch):
    """Stack batches."""
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    return X, y, mask


print("Sequence creation & dataset defined.")

# ============================================================================
# CELL 5: Graph Construction - Haversine, Edges, Hyperedges
# ============================================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in kilometres."""
    R = 6371.0
    
    def to_rad(x):
        return x * math.pi / 180.0
    
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(to_rad(lat1))
         * math.cos(to_rad(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def pairwise_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return (N, N) symmetric haversine distance matrix in km."""
    N = len(lats)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = D[j, i] = d
    return D


def build_spatial_edges(
    lats: np.ndarray,
    lons: np.ndarray,
    threshold_km: float = 150.0,
) -> Tensor:
    """Connect stations within threshold_km."""
    D = pairwise_distance_matrix(lats, lons)
    src, dst = np.where((D < threshold_km) & (D > 0))

    if len(src) == 0:
        src_list, dst_list = [], []
        for i in range(len(lats)):
            row = D[i].copy()
            row[i] = np.inf
            for j in np.argsort(row)[:3]:
                src_list += [i, j]
                dst_list += [j, i]
        src, dst = np.array(src_list), np.array(dst_list)

    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def build_semantic_edges(
    pm25_history: np.ndarray,
    corr_threshold: float = 0.70,
) -> Tensor:
    """Connect stations with PM2.5 correlation >= threshold."""
    corr = np.corrcoef(pm25_history)
    eye = ~np.eye(len(corr), dtype=bool)
    src, dst = np.where((corr >= corr_threshold) & eye)

    if len(src) == 0:
        N = len(corr)
        src_list, dst_list = [], []
        for i in range(N):
            row = corr[i].copy()
            row[i] = -np.inf
            for j in np.argsort(row)[-3:]:
                src_list += [i, j]
                dst_list += [j, i]
        src, dst = np.array(src_list), np.array(dst_list)

    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def build_hyperedges(
    dist_matrix: np.ndarray,
    thresholds_km: Tuple[float, ...] = (50.0, 100.0, 200.0),
) -> List[List[int]]:
    """Multi-scale spatial hyperedges."""
    N = dist_matrix.shape[0]
    hyperedges = []
    for th in thresholds_km:
        for i in range(N):
            members = [i] + [j for j in range(N) if j != i and dist_matrix[i, j] < th]
            if len(members) >= 2:
                hyperedges.append(members)
    return hyperedges


def hyperedges_to_incidence(hyperedges: List[List[int]], n_nodes: int) -> Tensor:
    """Convert hyperedge list to incidence matrix H (N, E)."""
    E = len(hyperedges)
    H = torch.zeros(n_nodes, E, dtype=torch.float32)
    for e_idx, members in enumerate(hyperedges):
        for node in members:
            H[node, e_idx] = 1.0
    return H


def build_region_membership(
    station_ids: List[str],
    region_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Assign each station to a region based on prefix matching."""
    if region_map is None:
        region_map = THAILAND_REGIONS

    region_names = list(region_map.keys())
    membership = np.zeros(len(station_ids), dtype=np.int64)

    for idx, sid in enumerate(station_ids):
        assigned = False
        for r_idx, (region, prefixes) in enumerate(region_map.items()):
            for prefix in prefixes:
                if sid.upper().startswith(prefix.upper()):
                    membership[idx] = r_idx
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            membership[idx] = len(region_names) - 1

    return membership, region_names


print("Graph construction functions defined.")

# ============================================================================
# CELL 6: HyperGAT Layer & Module (Spatial)
# ============================================================================

class HyperGATLayer(nn.Module):
    """One HyperGAT layer: nodes -> hyperedges -> nodes."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden

        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W1_hat = nn.Linear(hidden, hidden, bias=False)
        self.ctx1 = nn.Parameter(torch.empty(hidden))

        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.W2_hat = nn.Linear(hidden, hidden, bias=False)
        self.W3 = nn.Linear(hidden, hidden, bias=False)

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W1_hat.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W2_hat.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        nn.init.normal_(self.ctx1, std=0.02)

    def _scaled_dot(self, a: Tensor, b: Tensor) -> Tensor:
        return (a * b).sum(-1) / math.sqrt(self.hidden)

    def forward(self, node_emb: Tensor, H_inc: Tensor) -> Tensor:
        total_nodes, H = node_emb.shape
        E_h = H_inc.shape[1]

        # Stage 1: nodes -> hyperedges
        proj_hat = self.W1_hat(node_emb)
        ctx_exp = self.ctx1.unsqueeze(0).expand_as(proj_hat)
        raw_score = self._scaled_dot(proj_hat, ctx_exp)

        score_mat = raw_score.unsqueeze(1).expand(-1, E_h)
        mask_inf = (H_inc == 0) * (-1e9)
        score_mat = score_mat + mask_inf
        alpha = F.softmax(score_mat, dim=0)
        alpha = alpha * H_inc

        proj1 = self.W1(node_emb)
        hyper_emb = torch.mm(alpha.T, proj1)

        # Stage 2: hyperedges -> nodes
        proj2_hat = self.W2_hat(hyper_emb)
        proj2 = self.W2(hyper_emb)

        score2 = torch.mm(self.W3(node_emb), proj2_hat.T)
        score2 = score2 + (H_inc == 0) * (-1e9)
        beta = F.softmax(score2, dim=1)
        beta = beta * H_inc

        out = torch.mm(beta, proj2)
        return self.norm(self.drop(out) + node_emb)


class HyperGATModule(nn.Module):
    """Stack of HyperGAT layers + region embedding."""

    def __init__(
        self,
        hidden: int,
        n_regions: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, hidden)
        self.layers = nn.ModuleList([
            HyperGATLayer(hidden, dropout) for _ in range(n_layers)
        ])

    def forward(
        self,
        node_emb: Tensor,
        H_inc: Tensor,
        membership: np.ndarray,
    ) -> Tensor:
        B, N, H = node_emb.shape
        n_regions = self.region_emb.num_embeddings

        mem_t = torch.tensor(membership, device=node_emb.device, dtype=torch.long)
        r_emb = self.region_emb(torch.arange(n_regions, device=node_emb.device))

        outputs = []
        for b in range(B):
            x_b = node_emb[b]
            x_aug = torch.cat([x_b, r_emb], dim=0)

            for layer in self.layers:
                x_aug = layer(x_aug, H_inc)

            outputs.append(x_aug[:N])

        return torch.stack(outputs, dim=0)


print("HyperGAT module defined.")

# ============================================================================
# CELL 7: HGAT Module (Temporal)
# ============================================================================

class HGATModule(nn.Module):
    """Temporal HGAT: captures sequential and seasonal patterns."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.v_is = nn.Parameter(torch.empty(hidden))
        self.v_si = nn.Parameter(torch.empty(hidden))

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        nn.init.normal_(self.v_is, std=0.02)
        nn.init.normal_(self.v_si, std=0.02)

    def forward(self, node_time_emb: Tensor) -> Tensor:
        B, N, T, H = node_time_emb.shape

        h0 = node_time_emb.mean(dim=2)

        # Stage 1: nodes -> session
        h0_exp = h0.unsqueeze(2).expand(-1, -1, T, -1)
        e_is = F.leaky_relu(
            (node_time_emb * h0_exp * self.v_is).sum(-1)
        )
        beta_is = F.softmax(e_is, dim=2)
        h_s1 = (beta_is.unsqueeze(-1) * node_time_emb).sum(2)

        # Stage 2: session -> nodes
        h_s1_exp = h_s1.unsqueeze(2).expand(-1, -1, T, -1)
        e_si = F.leaky_relu(
            (node_time_emb * h_s1_exp * self.v_si).sum(-1)
        )
        beta_si = F.softmax(e_si, dim=2)
        h_t1 = (beta_si.unsqueeze(-1) * node_time_emb).sum(2)

        return self.norm(self.drop(h_t1) + h0)


print("HGAT module defined.")

# ============================================================================
# CELL 8: Position Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Reversed position encoding with soft attention."""

    def __init__(self, hidden: int, max_len: int = 14):
        super().__init__()
        self.W4 = nn.Linear(hidden * 2, hidden)
        self.W5 = nn.Linear(hidden, hidden, bias=False)
        self.W6 = nn.Linear(hidden, hidden, bias=False)
        self.p = nn.Parameter(torch.zeros(1, hidden))

        pe = torch.zeros(max_len, hidden)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:hidden // 2])
        self.register_buffer("pe", pe)

    def forward(self, h: Tensor) -> Tensor:
        B, N, T, H = h.shape

        pos = self.pe[:T].flip(0)
        pos_exp = pos.unsqueeze(0).unsqueeze(0).expand(B, N, T, H)

        h_star = torch.tanh(self.W4(torch.cat([h, pos_exp], dim=-1)))

        p_exp = self.p.unsqueeze(0).unsqueeze(0).expand(B, N, 1, H)
        rho = (p_exp * (self.W5(h_star) + self.W6(h))).sum(-1)
        rho = F.softmax(rho, dim=2)
        s = (rho.unsqueeze(-1) * h).sum(2)
        return s


print("Position encoding defined.")

# ============================================================================
# CELL 9: Loss Functions
# ============================================================================

def infonce_loss(
    h_spatial: Tensor,
    h_temporal: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE contrastive loss between spatial and temporal embeddings."""
    B, N, H = h_spatial.shape

    hs = F.normalize(h_spatial.reshape(B * N, H), dim=-1)
    ht = F.normalize(h_temporal.reshape(B * N, H), dim=-1)

    sim = torch.mm(hs, ht.T) / temperature

    labels = torch.arange(B * N, device=h_spatial.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def adaptive_weight_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mask: Tensor,
    gamma: float = 2.0,
    pm25_extreme_threshold: float = 50.0,
) -> Tensor:
    """AW Loss: upweight samples where prediction deviates most."""
    yp = y_pred[mask]
    yt = y_true[mask]

    if yp.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=y_pred.device)

    rel_err = torch.abs(yp - yt) / (torch.abs(yt) + 1e-6)
    p_i = torch.exp(-rel_err).clamp(0.01, 0.99)

    weight = (2 - 2 * p_i) ** gamma

    extreme_mask = yt > pm25_extreme_threshold
    weight = torch.where(extreme_mask, weight * 2.0, weight)

    loss = (weight * (yp - yt) ** 2).mean()
    return loss


print("Loss functions defined.")

# ============================================================================
# CELL 10: Full STC-HGAT Model
# ============================================================================

class STCHGAT(nn.Module):
    """STC-HGAT: Spatio-Temporal Contrastive HGAT for PM2.5 Forecasting."""

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        n_regions: int = 5,
        n_hyperedges: int = 100,
        hypergat_layers: int = 2,
        seq_len: int = 7,
        dropout: float = 0.1,
        contrastive_lambda: float = 0.1,
        aw_gamma: float = 2.0,
        extreme_threshold: float = 50.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.contrastive_lambda = contrastive_lambda
        self.aw_gamma = aw_gamma
        self.extreme_threshold = extreme_threshold

        self.feat_embed = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.hypergat = HyperGATModule(
            hidden, n_regions, n_layers=hypergat_layers, dropout=dropout
        )

        self.hgat = HGATModule(hidden, dropout=dropout)

        self.pos_enc = PositionalEncoding(hidden, max_len=seq_len)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        H_inc: Tensor,
        membership: np.ndarray,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, T, F = x.shape

        x_flat = x.reshape(B * N * T, F)
        h_flat = self.feat_embed(x_flat)
        h = h_flat.reshape(B, N, T, self.hidden)

        h_mean = h.mean(dim=2)
        h_spatial = self.hypergat(h_mean, H_inc, membership)

        h_temporal = self.hgat(h)

        h_fused = h_spatial + h_temporal

        h_enrich = h + h_fused.unsqueeze(2)

        session_repr = self.pos_enc(h_enrich)

        pred = self.head(session_repr)

        return pred, h_spatial, h_temporal

    def compute_loss(
        self,
        pred: Tensor,
        y: Tensor,
        mask: Tensor,
        h_spatial: Tensor,
        h_temporal: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        yp = pred.squeeze(-1)

        l_r = adaptive_weight_loss(
            yp, y, mask, gamma=self.aw_gamma,
            pm25_extreme_threshold=self.extreme_threshold,
        )

        l_c = infonce_loss(h_spatial, h_temporal)

        total = l_r + self.contrastive_lambda * l_c

        return total, {
            "total": float(total),
            "aw_loss": float(l_r),
            "contrastive": float(l_c),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print("STC-HGAT model defined.")

# ============================================================================
# CELL 11: Model Wrapper with Training Loop
# ============================================================================

class STCHGATModel:
    """Sklearn-style wrapper for STC-HGAT with full training loop."""

    def __init__(
        self,
        config: Dict,
        H_inc: Tensor,
        membership: np.ndarray,
    ):
        self.config = config
        self.membership = membership
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.net = STCHGAT(
            in_channels=config.get("in_channels", 50),
            hidden=config.get("hidden", 128),
            n_regions=config.get("n_regions", 5),
            n_hyperedges=H_inc.shape[1],
            hypergat_layers=config.get("hypergat_layers", 2),
            seq_len=config.get("seq_len", 7),
            dropout=config.get("dropout", 0.1),
            contrastive_lambda=config.get("contrastive_lambda", 0.1),
            aw_gamma=config.get("aw_gamma", 2.0),
            extreme_threshold=config.get("extreme_threshold", 50.0),
        ).to(self.device)

        self._H_inc = H_inc.to(self.device)
        self._history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_r2": [], "val_r2": [],
        }

    def _forward_batch(self, xb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.net(xb, self._H_inc, self.membership)

    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            return 0.0
        yt, yp = y_true[mask], y_pred[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean()) ** 2)
        return float(1 - ss_r / (ss_t + 1e-8))

    def fit(self, train_ds, val_ds):
        cfg = self.config
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 8)
        lr = cfg.get("lr", 1e-3)
        patience = cfg.get("patience", 15)
        grad_clip = cfg.get("grad_clip", 1.0)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn
        )

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr,
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False,
        )
        scaler = torch.amp.GradScaler(
            enabled=(self.device.type == "cuda")
        )

        best_val_loss = float("inf")
        best_state = None
        patience_cnt = 0

        for epoch in range(1, epochs + 1):
            self.net.train()
            t_losses, t_preds, t_trues = [], [], []

            for xb, yb, mb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    pred, h_s, h_t = self._forward_batch(xb)
                    loss, ld = self.net.compute_loss(pred, yb, mb, h_s, h_t)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                t_losses.append(ld["total"])
                t_preds.append(pred.squeeze(-1)[mb].detach().cpu().numpy())
                t_trues.append(yb[mb].detach().cpu().numpy())

            avg_train = float(np.mean(t_losses))

            self.net.eval()
            v_losses, v_preds, v_trues = [], [], []
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    mb = mb.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                        pred, h_s, h_t = self._forward_batch(xb)
                        _, ld = self.net.compute_loss(pred, yb, mb, h_s, h_t)
                    v_losses.append(ld["total"])
                    v_preds.append(pred.squeeze(-1)[mb].cpu().numpy())
                    v_trues.append(yb[mb].cpu().numpy())

            avg_val = float(np.mean(v_losses))
            scheduler.step(avg_val)

            tp = np.concatenate(t_preds)
            tt = np.concatenate(t_trues)
            vp = np.concatenate(v_preds)
            vt = np.concatenate(v_trues)
            tr2 = self._r2(tt, tp)
            vr2 = self._r2(vt, vp)

            self._history["train_loss"].append(avg_train)
            self._history["val_loss"].append(avg_val)
            self._history["train_r2"].append(tr2)
            self._history["val_r2"].append(vr2)

            if epoch % 5 == 0 or epoch == 1:
                lr_cur = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"Train L={avg_train:.4f} R²={tr2:.4f} | "
                    f"Val   L={avg_val:.4f} R²={vr2:.4f} | "
                    f"LR={lr_cur:.2e}"
                )

            if vr2 > 0.9:
                print(f"🎯 TARGET ACHIEVED! Val R² = {vr2:.4f}")

            if avg_val < best_val_loss - 1e-5:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone()
                              for k, v in self.net.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.net.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

    def predict(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size", 8),
            shuffle=False, collate_fn=collate_fn
        )
        self.net.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb, mb in loader:
                xb = xb.to(self.device)
                mb_dev = mb.to(self.device)
                pred, _, _ = self._forward_batch(xb)
                preds.append(pred.squeeze(-1)[mb_dev].cpu().numpy())
                trues.append(yb[mb].numpy())

        return np.concatenate(preds), np.concatenate(trues)

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "config": self.config,
            "history": self._history,
        }, path)
        print(f"Model saved to {path}")


print("STCHGATModel wrapper defined.")

# ============================================================================
# CELL 12: Evaluation Metrics
# ============================================================================

def compute_rmse(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(yt - yp)))


def compute_r2(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_r / (ss_t + 1e-8))


def compute_smape(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    denom = np.abs(yt) + np.abs(yp) + 1e-8
    return float(np.mean(2 * np.abs(yt - yp) / denom) * 100)


def compute_mbe(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(np.mean(yp - yt))


def evaluate_all(y_true, y_pred) -> Dict[str, float]:
    mask = np.isfinite(np.asarray(y_true)) & np.isfinite(np.asarray(y_pred))
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    return {
        "RMSE": compute_rmse(yt, yp),
        "MAE": compute_mae(yt, yp),
        "R2": compute_r2(yt, yp),
        "SMAPE": compute_smape(yt, yp),
        "MBE": compute_mbe(yt, yp),
    }


print("Evaluation metrics defined.")

# ============================================================================
# CELL 13: Load Air Quality Data from Silver Layer
# ============================================================================

print("Loading air quality data from silver layer...")

aq_dir = Path(CONFIG['aq_data_dir'])
aq_files = list(aq_dir.glob('**/*.parquet'))
print(f"Found {len(aq_files)} air quality parquet files")

# Load and combine
aq_dfs = []
for f in aq_files:
    df = pd.read_parquet(f)
    aq_dfs.append(df)

aq_df = pd.concat(aq_dfs, ignore_index=True)
print(f"Total air quality rows: {len(aq_df):,}")

# Convert timestamp to date
aq_df['timestamp_utc'] = pd.to_datetime(aq_df['timestamp_utc'])
aq_df['date'] = aq_df['timestamp_utc'].dt.date
aq_df['date'] = pd.to_datetime(aq_df['date'])

# Aggregate to daily
daily_aq = aq_df.groupby(['stationID', 'date', 'lat', 'lon']).agg({
    'pm2_5_ugm3': 'mean',
    'pm10_ugm3': 'mean',
    'nitrogen_dioxide_ugm3': 'mean',
    'ozone_ugm3': 'mean',
    'sulphur_dioxide_ugm3': 'mean',
    'carbon_monoxide_ugm3': 'mean',
}).reset_index()

# Rename columns
daily_aq = daily_aq.rename(columns={
    'pm2_5_ugm3': 'pm2_5',
    'pm10_ugm3': 'pm10',
    'nitrogen_dioxide_ugm3': 'no2',
    'ozone_ugm3': 'o3',
    'sulphur_dioxide_ugm3': 'so2',
    'carbon_monoxide_ugm3': 'co',
})

print(f"Daily aggregated rows: {len(daily_aq):,}")
print(f"Stations: {daily_aq['stationID'].nunique()}")
print(f"Date range: {daily_aq['date'].min()} to {daily_aq['date'].max()}")
print(f"PM2.5 valid: {daily_aq['pm2_5'].notna().sum():,}")

# ============================================================================
# CELL 14: Load Weather Data and Merge
# ============================================================================

print("\nLoading weather data from silver layer...")

weather_dir = Path(CONFIG['weather_data_dir'])
weather_files = list(weather_dir.glob('**/*.parquet'))
print(f"Found {len(weather_files)} weather parquet files")

# Load and combine weather
weather_dfs = []
for f in weather_files:
    df = pd.read_parquet(f)
    weather_dfs.append(df)

weather_df = pd.concat(weather_dfs, ignore_index=True)
print(f"Total weather rows: {len(weather_df):,}")

# Check columns
print(f"Weather columns: {weather_df.columns.tolist()[:10]}...")

# Convert timestamp
if 'timestamp_utc' in weather_df.columns:
    weather_df['timestamp_utc'] = pd.to_datetime(weather_df['timestamp_utc'])
    weather_df['date'] = weather_df['timestamp_utc'].dt.date
    weather_df['date'] = pd.to_datetime(weather_df['date'])

# Aggregate to daily - select numeric columns
weather_numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
weather_numeric_cols = [c for c in weather_numeric_cols if c not in ['lat', 'lon', 'timestamp_unix_ms']]

daily_weather = weather_df.groupby(['stationID', 'date']).agg({
    col: 'mean' for col in weather_numeric_cols if col in weather_df.columns
}).reset_index()

print(f"Daily weather rows: {len(daily_weather):,}")

# ============================================================================
# CELL 15: Merge AQ and Weather, Create Features
# ============================================================================

print("\nMerging air quality and weather data...")

# Merge on stationID and date
df_all = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
print(f"Merged rows: {len(df_all):,}")

# Filter by start date
if CONFIG['start_date']:
    df_all = df_all[df_all['date'] >= CONFIG['start_date']]
    print(f"After date filter: {len(df_all):,}")

# Remove rows without PM2.5 target
TARGET_COL = 'pm2_5'
df_all = df_all[df_all[TARGET_COL].notna()].copy()
df_all = df_all.sort_values(['stationID', 'date']).reset_index(drop=True)

print(f"After removing NaN target: {len(df_all):,}")
print(f"Stations: {df_all['stationID'].nunique()}")
print(f"Date range: {df_all['date'].min()} to {df_all['date'].max()}")

# Create lag features
for lag in [1, 2, 3]:
    df_all[f'pm2_5_lag{lag}'] = df_all.groupby('stationID')[TARGET_COL].shift(lag)

# Create rolling features
for window in [3, 7]:
    df_all[f'pm2_5_rolling_mean_{window}d'] = (
        df_all.groupby('stationID')[TARGET_COL]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

# Add day of year cyclical encoding
df_all['day_of_year'] = df_all['date'].dt.dayofyear
df_all['day_of_year_sin'] = np.sin(2 * np.pi * df_all['day_of_year'] / 365)
df_all['day_of_year_cos'] = np.cos(2 * np.pi * df_all['day_of_year'] / 365)

# Feature columns (exclude target, identifiers, and coordinates)
exclude = {'date', 'stationID', TARGET_COL, 'lat', 'lon', 'day_of_year'}
feature_cols = [c for c in df_all.columns if c not in exclude and df_all[c].dtype in [np.float64, np.float32, np.int64]]

print(f"Features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Station metadata
station_meta = (
    df_all.groupby('stationID')[['lat', 'lon']]
    .first()
    .reset_index()
    .sort_values('stationID')
)
station_order = station_meta['stationID'].tolist()

print(f"\nTotal stations: {len(station_order)}")

# ============================================================================
# CELL 16: Split, Fill, Scale
# ============================================================================

print("\nSplitting data by date...")

# Date-based split
train_df, val_df, test_df = split_by_date(
    df_all, CONFIG['train_ratio'], CONFIG['val_ratio']
)

print(f"Train: {len(train_df):,} rows, {train_df['date'].nunique()} days")
print(f"Val:   {len(val_df):,} rows, {val_df['date'].nunique()} days")
print(f"Test:  {len(test_df):,} rows, {test_df['date'].nunique()} days")

# Fill missing values
print("\nFilling missing values...")
train_df = fill_missing(train_df, feature_cols)
val_df = fill_missing(val_df, feature_cols)
test_df = fill_missing(test_df, feature_cols)

# Check for remaining NaNs
train_nan = train_df[feature_cols].isna().sum().sum()
val_nan = val_df[feature_cols].isna().sum().sum()
test_nan = test_df[feature_cols].isna().sum().sum()
print(f"Remaining NaNs - Train: {train_nan}, Val: {val_nan}, Test: {test_nan}")

# Normalization - fit on TRAIN ONLY
print("\nNormalizing features...")
feat_scaler = FeatureScaler()
target_scaler = TargetScaler()

train_df[feature_cols] = feat_scaler.fit_transform(train_df[feature_cols].values)
val_df[feature_cols] = feat_scaler.transform(val_df[feature_cols].values)
test_df[feature_cols] = feat_scaler.transform(test_df[feature_cols].values)

train_df[TARGET_COL] = target_scaler.fit_transform(train_df[TARGET_COL].values)
val_df[TARGET_COL] = target_scaler.transform(val_df[TARGET_COL].values)
test_df[TARGET_COL] = target_scaler.transform(test_df[TARGET_COL].values)

print("Normalization complete.")

# ============================================================================
# CELL 17: Create Sequences & Datasets
# ============================================================================

print("\nCreating sequences...")

def make_seqs(split_df):
    return create_sequences(
        split_df, feature_cols, station_order,
        lookback=CONFIG['lookback'], target_col=TARGET_COL,
        min_stations=CONFIG['min_stations'],
    )

X_tr, y_tr, m_tr = make_seqs(train_df)
X_va, y_va, m_va = make_seqs(val_df)
X_te, y_te, m_te = make_seqs(test_df)

train_ds = PM25GraphDataset(X_tr, y_tr, m_tr)
val_ds = PM25GraphDataset(X_va, y_va, m_va)
test_ds = PM25GraphDataset(X_te, y_te, m_te)

print(f"Train sequences: {len(train_ds)}")
print(f"Val sequences: {len(val_ds)}")
print(f"Test sequences: {len(test_ds)}")

if len(train_ds) > 0:
    print(f"Nodes: {train_ds.n_nodes}, Seq len: {train_ds.seq_len}, Features: {train_ds.n_features}")
else:
    print("WARNING: No training sequences created!")

# ============================================================================
# CELL 18: Build Station Graph
# ============================================================================

print("\nBuilding station graph...")

# Get coordinates
lats = station_meta.set_index('stationID').reindex(station_order)['lat'].values
lons = station_meta.set_index('stationID').reindex(station_order)['lon'].values

# Build PM2.5 history matrix for semantic edges
pm25_pivot = (
    df_all.groupby(['stationID', 'date'])[TARGET_COL]
    .mean().unstack(fill_value=0)
    .reindex(station_order).fillna(0)
)
pm25_history = pm25_pivot.values

# Build distance matrix
dist_matrix = pairwise_distance_matrix(lats, lons)

# Build edges
spatial_edges = build_spatial_edges(lats, lons, CONFIG['spatial_edge_km'])
semantic_edges = build_semantic_edges(pm25_history, CONFIG['corr_threshold'])

# Build hyperedges
hyperedges = build_hyperedges(dist_matrix, CONFIG['spatial_thresholds_km'])
H_base = hyperedges_to_incidence(hyperedges, len(station_order))

# Region membership
membership, region_names = build_region_membership(station_order)
n_regions = len(region_names)

# Pad incidence matrix for region nodes
H_pad = torch.zeros(n_regions, H_base.shape[1])
H_inc = torch.cat([H_base, H_pad], dim=0)

print(f"Spatial edges: {spatial_edges.shape[1]}")
print(f"Semantic edges: {semantic_edges.shape[1]}")
print(f"Hyperedges: {len(hyperedges)}")
print(f"Incidence matrix H: {H_inc.shape}")
print(f"Regions: {n_regions} ({', '.join(region_names)})")

# ============================================================================
# CELL 19: Instantiate & Train Model
# ============================================================================

# Update config with actual values
CONFIG['in_channels'] = train_ds.n_features
CONFIG['seq_len'] = train_ds.seq_len
CONFIG['n_regions'] = n_regions

# Create model
model = STCHGATModel(config=CONFIG, H_inc=H_inc, membership=membership)
print(f"Parameters: {model.net.count_parameters():,}")

# ============================================================================
# CELL 20: Training with MLflow
# ============================================================================

# Create reports directory
reports_dir = Path('../data/reports/stc_hgat')
reports_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment('pm25_stc_hgat')

with mlflow.start_run(run_name='stc_hgat_v1'):
    # Log parameters
    mlflow.log_params({k: v for k, v in CONFIG.items() 
                       if not isinstance(v, (torch.Tensor, tuple))})
    mlflow.log_param('spatial_thresholds_km', str(CONFIG['spatial_thresholds_km']))
    
    # Train
    model.fit(train_ds, val_ds)
    
    # Evaluate on test set
    y_pred_norm, y_true_norm = model.predict(test_ds)
    
    # Inverse transform to original PM2.5 scale
    y_pred = target_scaler.inverse_transform(y_pred_norm)
    y_true = target_scaler.inverse_transform(y_true_norm)
    
    metrics = evaluate_all(y_true, y_pred)
    mlflow.log_metrics(metrics)
    
    # Log training history
    for i, (tl, vl, tr, vr) in enumerate(zip(
        model._history['train_loss'],
        model._history['val_loss'],
        model._history['train_r2'],
        model._history['val_r2']
    )):
        mlflow.log_metrics({
            'train_loss': tl,
            'val_loss': vl,
            'train_r2': tr,
            'val_r2': vr
        }, step=i+1)

print('\n-- Test Metrics (original PM2.5 scale) --')
for k, v in metrics.items():
    print(f'  {k:6s}: {v:.4f}')

if metrics['R2'] > 0.9:
    print('\n🎯 TARGET R2 > 0.9 ACHIEVED!')
elif metrics['R2'] > 0.8:
    print(f'\nGood result: R2 = {metrics["R2"]:.4f}')

# ============================================================================
# CELL 21: Training Curves
# ============================================================================

hist = model._history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(hist['train_loss'], label='Train', lw=2)
axes[0].plot(hist['val_loss'], label='Val', lw=2, ls='--')
axes[0].set(title='Combined Loss (L_r + lambda*L_c)', xlabel='Epoch', ylabel='Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(hist['train_r2'], label='Train R2', lw=2)
axes[1].plot(hist['val_r2'], label='Val R2', lw=2, ls='--')
axes[1].axhline(0.9, color='red', ls=':', label='Target R2=0.9')
axes[1].set(title='R2 Score', xlabel='Epoch', ylabel='R2', ylim=(0, 1))
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 22: Prediction vs Actual Scatter
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_true, y_pred, alpha=0.3, s=10)
ax.plot([0, y_true.max()], [0, y_true.max()], 'r--', lw=2, label='Perfect prediction')
ax.set(xlabel='Actual PM2.5 (ug/m3)', ylabel='Predicted PM2.5 (ug/m3)',
       title=f'STC-HGAT Predictions (R2 = {metrics["R2"]:.4f})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'prediction_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 23: Error Distribution
# ============================================================================

errors = y_pred - y_true

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', ls='--', lw=2)
axes[0].set(title='Prediction Error Distribution', xlabel='Error (ug/m3)', ylabel='Count')
axes[0].grid(alpha=0.3)

axes[1].scatter(y_true, errors, alpha=0.3, s=10)
axes[1].axhline(0, color='red', ls='--', lw=2)
axes[1].set(title='Residual Plot', xlabel='Actual PM2.5 (ug/m3)', ylabel='Error (ug/m3)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 24: Save Model & Metrics
# ============================================================================

# Save model checkpoint
model_path = reports_dir / 'stc_hgat_model.pt'
model.save(str(model_path))

# Save metrics
metrics_path = reports_dir / 'test_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

# Save config
config_path = reports_dir / 'config.json'
config_save = {k: v for k, v in CONFIG.items() if not isinstance(v, torch.Tensor)}
config_save['spatial_thresholds_km'] = list(CONFIG['spatial_thresholds_km'])
with open(config_path, 'w') as f:
    json.dump(config_save, f, indent=2)
print(f"Config saved to {config_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Final Test R2: {metrics['R2']:.4f}")
print(f"Final Test RMSE: {metrics['RMSE']:.4f} ug/m3")
print(f"Artifacts saved to: {reports_dir}")
