"""
STC-HGAT PM2.5 Forecasting - Spatial Learning Focus
Key changes:
1. NO PM10 feature - force model to learn spatial patterns
2. Use weather, spatial coordinates, and PM2.5 lags only
3. Proper graph construction for spatial relationships
4. Stronger regularization to prevent overfitting
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'aq_data_dir': '../data/silver/openmeteo_airquality',
    'weather_data_dir': '../data/silver/openmeteo_weather',
    'start_date': '2023-01-01',
    'lookback': 14,  # Longer lookback for temporal patterns
    'min_stations': 30,  # Require more stations per sample
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    
    # Model - larger for better capacity
    'hidden': 128,
    'n_heads': 8,
    'hypergat_layers': 2,
    'dropout': 0.2,
    
    # Spatial graph
    'spatial_thresholds_km': (30.0, 60.0, 100.0),  # Tighter spatial connections
    'k_neighbors': 5,  # K-nearest neighbors for graph
    
    # Training - maximize GPU utilization
    'epochs': 200,
    'batch_size': 64,  # Larger batch for GPU
    'lr': 1e-3,  # Higher LR for larger batch
    'weight_decay': 1e-3,  # Strong L2 regularization
    'patience': 30,
    'grad_clip': 0.5,
    
    # Contrastive learning
    'use_contrastive': True,
    'contrastive_lambda': 0.05,
}

THAILAND_REGIONS = {
    "North": ["CM", "LM", "LP", "PY", "NAN", "PR", "PL", "MSH", "TAK", "SUK"],
    "Northeast": ["KKN", "UDT", "NKP", "SK", "BRM", "ROI", "MKM", "YST", "SR"],
    "Central": ["BKK", "NPT", "AYA", "SBR", "NBR", "CNB", "KRI", "PKN", "SMT"],
    "East": ["RY", "CH", "TRT", "PKO", "SA"],
    "South": ["HYI", "PKT", "SNG", "NRT", "PT", "KBI", "TG", "PAN"],
}

# ============================================================================
# Data Loading - NO PM10!
# ============================================================================
print("\n1. Loading data (NO PM10 - spatial learning focus)...")

aq_dir = Path(CONFIG['aq_data_dir'])
aq_files = list(aq_dir.glob('**/*.parquet'))
aq_dfs = [pd.read_parquet(f) for f in aq_files]
aq_df = pd.concat(aq_dfs, ignore_index=True)

aq_df['timestamp_utc'] = pd.to_datetime(aq_df['timestamp_utc'])
aq_df['date'] = pd.to_datetime(aq_df['timestamp_utc'].dt.date)

# Only PM2.5 - NO PM10!
daily_aq = aq_df.groupby(['stationID', 'date', 'lat', 'lon']).agg({
    'pm2_5_ugm3': 'mean',
}).reset_index().rename(columns={'pm2_5_ugm3': 'pm2_5'})

# Load weather - important for spatial patterns
weather_dir = Path(CONFIG['weather_data_dir'])
weather_files = list(weather_dir.glob('**/*.parquet'))
weather_dfs = [pd.read_parquet(f) for f in weather_files]
weather_df = pd.concat(weather_dfs, ignore_index=True)
weather_df['timestamp_utc'] = pd.to_datetime(weather_df['timestamp_utc'])
weather_df['date'] = pd.to_datetime(weather_df['timestamp_utc'].dt.date)

daily_weather = weather_df.groupby(['stationID', 'date']).agg({
    'temp_c': 'mean',
    'humidity_pct': 'mean',
    'wind_ms': 'mean',
    'pressure_hpa': 'mean',
}).reset_index()

# Merge
df_all = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df_all = df_all[df_all['date'] >= CONFIG['start_date']]
df_all = df_all[df_all['pm2_5'].notna()].copy()
df_all = df_all.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET = 'pm2_5'

print(f"   Rows: {len(df_all):,}, Stations: {df_all['stationID'].nunique()}")

# ============================================================================
# Feature Engineering - Focus on spatial-temporal patterns
# ============================================================================
print("\n2. Feature engineering (spatial-temporal focus)...")

# PM2.5 lag features - critical for temporal patterns
for lag in [1, 2, 3, 7, 14]:
    df_all[f'pm2_5_lag{lag}'] = df_all.groupby('stationID')[TARGET].shift(lag)

# Rolling statistics
for window in [3, 7, 14]:
    df_all[f'pm2_5_mean_{window}d'] = df_all.groupby('stationID')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df_all[f'pm2_5_std_{window}d'] = df_all.groupby('stationID')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

# Difference features (trend)
df_all['pm2_5_diff1'] = df_all.groupby('stationID')[TARGET].diff(1)
df_all['pm2_5_diff7'] = df_all.groupby('stationID')[TARGET].diff(7)

# Time features - important for seasonality
df_all['day_of_year'] = df_all['date'].dt.dayofyear
df_all['day_sin'] = np.sin(2 * np.pi * df_all['day_of_year'] / 365)
df_all['day_cos'] = np.cos(2 * np.pi * df_all['day_of_year'] / 365)
df_all['month'] = df_all['date'].dt.month
df_all['month_sin'] = np.sin(2 * np.pi * df_all['month'] / 12)
df_all['month_cos'] = np.cos(2 * np.pi * df_all['month'] / 12)
df_all['is_dry_season'] = df_all['month'].isin([11, 12, 1, 2, 3]).astype(float)

# Normalized spatial coordinates (for spatial encoding)
lat_mean, lat_std = df_all['lat'].mean(), df_all['lat'].std()
lon_mean, lon_std = df_all['lon'].mean(), df_all['lon'].std()
df_all['lat_norm'] = (df_all['lat'] - lat_mean) / lat_std
df_all['lon_norm'] = (df_all['lon'] - lon_mean) / lon_std

# Weather interactions
df_all['temp_humidity'] = df_all['temp_c'] * df_all['humidity_pct'] / 100
df_all['wind_temp'] = df_all['wind_ms'] * df_all['temp_c']

# Feature columns - NO PM10!
feature_cols = [
    # PM2.5 temporal features
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_lag14',
    'pm2_5_mean_3d', 'pm2_5_mean_7d', 'pm2_5_mean_14d',
    'pm2_5_std_3d', 'pm2_5_std_7d', 'pm2_5_std_14d',
    'pm2_5_diff1', 'pm2_5_diff7',
    
    # Weather features
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'temp_humidity', 'wind_temp',
    
    # Time features
    'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_dry_season',
    
    # Spatial features
    'lat_norm', 'lon_norm',
]

# Fill missing
df_all[feature_cols] = df_all.groupby('stationID')[feature_cols].transform(
    lambda g: g.ffill().bfill()
)
for col in feature_cols:
    if df_all[col].isna().any():
        df_all[col] = df_all[col].fillna(df_all[col].median())

df_all = df_all.dropna(subset=feature_cols + [TARGET])

print(f"   Features: {len(feature_cols)} (NO PM10)")
print(f"   After cleaning: {len(df_all):,} rows")

# ============================================================================
# Station metadata
# ============================================================================
station_meta = df_all.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}
N_FEATURES = len(feature_cols)

print(f"   Stations: {N_STATIONS}")

# ============================================================================
# Split
# ============================================================================
print("\n3. Splitting data...")

dates = sorted(df_all['date'].unique())
n = len(dates)
t1 = int(n * CONFIG['train_ratio'])
t2 = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

train_df = df_all[df_all['date'].isin(dates[:t1])].copy()
val_df = df_all[df_all['date'].isin(dates[t1:t2])].copy()
test_df = df_all[df_all['date'].isin(dates[t2:])].copy()

print(f"   Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Normalize
feat_scaler = StandardScaler()
train_df[feature_cols] = feat_scaler.fit_transform(train_df[feature_cols].values)
val_df[feature_cols] = feat_scaler.transform(val_df[feature_cols].values)
test_df[feature_cols] = feat_scaler.transform(test_df[feature_cols].values)

target_mean = train_df[TARGET].mean()
target_std = train_df[TARGET].std()
train_df[TARGET] = (train_df[TARGET] - target_mean) / target_std
val_df[TARGET] = (val_df[TARGET] - target_mean) / target_std
test_df[TARGET] = (test_df[TARGET] - target_mean) / target_std

print(f"   Target: mean={target_mean:.2f}, std={target_std:.2f}")

# ============================================================================
# Create Sequences
# ============================================================================
print("\n4. Creating sequences...")

def create_sequences(df, lookback=14, min_stations=30):
    dates = sorted(df['date'].unique())
    X_list, y_list, mask_list = [], [], []
    
    for t in range(len(dates) - lookback):
        window_dates = dates[t:t+lookback]
        target_date = dates[t+lookback]
        
        X = np.zeros((N_STATIONS, lookback, N_FEATURES), dtype=np.float32)
        y = np.zeros(N_STATIONS, dtype=np.float32)
        mask = np.zeros(N_STATIONS, dtype=bool)
        
        target_rows = df[df['date'] == target_date]
        
        for _, row in target_rows.iterrows():
            sid = row['stationID']
            if sid not in sid2idx or pd.isna(row[TARGET]):
                continue
            idx = sid2idx[sid]
            
            window = df[(df['stationID'] == sid) & (df['date'].isin(window_dates))].sort_values('date')
            if len(window) < lookback:
                continue
            
            X[idx] = window[feature_cols].values
            y[idx] = row[TARGET]
            mask[idx] = True
        
        if mask.sum() >= min_stations:
            X_list.append(X)
            y_list.append(y)
            mask_list.append(mask)
    
    return X_list, y_list, mask_list

X_tr, y_tr, m_tr = create_sequences(train_df, CONFIG['lookback'], CONFIG['min_stations'])
X_va, y_va, m_va = create_sequences(val_df, CONFIG['lookback'], CONFIG['min_stations'])
X_te, y_te, m_te = create_sequences(test_df, CONFIG['lookback'], CONFIG['min_stations'])

print(f"   Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

if len(X_tr) == 0:
    print("ERROR: No training sequences! Reducing min_stations...")
    X_tr, y_tr, m_tr = create_sequences(train_df, CONFIG['lookback'], 10)
    X_va, y_va, m_va = create_sequences(val_df, CONFIG['lookback'], 10)
    X_te, y_te, m_te = create_sequences(test_df, CONFIG['lookback'], 10)
    print(f"   Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

# ============================================================================
# Dataset
# ============================================================================
class PM25Dataset(Dataset):
    def __init__(self, X, y, m):
        self.X = [torch.tensor(x) for x in X]
        self.y = [torch.tensor(yi) for yi in y]
        self.m = [torch.tensor(mi) for mi in m]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.m[i]

def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))

train_ds = PM25Dataset(X_tr, y_tr, m_tr)
val_ds = PM25Dataset(X_va, y_va, m_va)
test_ds = PM25Dataset(X_te, y_te, m_te)

# ============================================================================
# Build Spatial Graph
# ============================================================================
print("\n5. Building spatial graph...")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))

lats = station_meta.set_index('stationID').reindex(station_order)['lat'].values
lons = station_meta.set_index('stationID').reindex(station_order)['lon'].values

# Distance matrix
D = np.zeros((N_STATIONS, N_STATIONS), dtype=np.float32)
for i in range(N_STATIONS):
    for j in range(i+1, N_STATIONS):
        d = haversine_km(lats[i], lons[i], lats[j], lons[j])
        D[i, j] = D[j, i] = d

# K-nearest neighbors adjacency
k = CONFIG['k_neighbors']
adj = np.zeros((N_STATIONS, N_STATIONS), dtype=np.float32)
for i in range(N_STATIONS):
    neighbors = np.argsort(D[i])[1:k+1]  # Exclude self
    for j in neighbors:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Symmetric

# Normalize adjacency (GCN-style)
D_sum = adj.sum(axis=1, keepdims=True) + 1e-8
adj_norm = adj / D_sum
A_norm = torch.tensor(adj_norm, dtype=torch.float32)

# Build hyperedges for multi-scale spatial
hyperedges = []
for th in CONFIG['spatial_thresholds_km']:
    for i in range(N_STATIONS):
        members = [i] + [j for j in range(N_STATIONS) if j != i and D[i, j] < th]
        if len(members) >= 2:
            hyperedges.append(members)

# Incidence matrix
H_base = torch.zeros(N_STATIONS, len(hyperedges), dtype=torch.float32)
for e_idx, members in enumerate(hyperedges):
    for node in members:
        H_base[node, e_idx] = 1.0

# Region membership
region_names = list(THAILAND_REGIONS.keys())
n_regions = len(region_names)
membership = np.zeros(N_STATIONS, dtype=np.int64)
for idx, sid in enumerate(station_order):
    for r_idx, (region, prefixes) in enumerate(THAILAND_REGIONS.items()):
        if any(sid.upper().startswith(p.upper()) for p in prefixes):
            membership[idx] = r_idx
            break

# Pad for region nodes
H_pad = torch.zeros(n_regions, H_base.shape[1])
H_inc = torch.cat([H_base, H_pad], dim=0)

print(f"   K-NN edges: {int(adj.sum())}")
print(f"   Hyperedges: {len(hyperedges)}")
print(f"   Regions: {n_regions}")

# ============================================================================
# STC-HGAT Model
# ============================================================================
print("\n6. Building STC-HGAT model...")

class HyperGATLayer(nn.Module):
    """Simplified HyperGAT layer."""
    
    def __init__(self, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.W_node = nn.Linear(hidden, hidden)
        self.W_edge = nn.Linear(hidden, hidden)
        self.attn = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, node_emb: Tensor, H_inc: Tensor) -> Tensor:
        N, H = node_emb.shape
        E = H_inc.shape[1]
        
        # Node to hyperedge aggregation
        node_proj = self.W_node(node_emb)  # (N, H)
        
        # Compute hyperedge embeddings
        H_t = H_inc.T  # (E, N)
        edge_emb = torch.mm(H_t, node_proj) / (H_t.sum(dim=1, keepdim=True) + 1e-8)  # (E, H)
        edge_emb = self.W_edge(edge_emb)
        
        # Hyperedge to node aggregation with attention
        scores = self.attn(edge_emb).squeeze(-1)  # (E,)
        
        # Masked softmax per node
        out = torch.zeros_like(node_proj)
        for i in range(N):
            mask = H_inc[i] > 0
            if mask.sum() > 0:
                edge_scores = scores[mask]
                edge_weights = Fn.softmax(edge_scores, dim=0)
                out[i] = (edge_weights.unsqueeze(-1) * edge_emb[mask]).sum(dim=0)
        
        return self.norm(self.drop(out) + node_emb)


class TemporalAttention(nn.Module):
    """Temporal attention for sequence modeling."""
    
    def __init__(self, hidden: int, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (B*N, T, H)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(self.drop(attn_out) + x)


class STCHGAT(nn.Module):
    """STC-HGAT: Spatio-Temporal Contrastive HGAT."""
    
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_regions: int = 5,
        n_heads: int = 4,
        n_hypergat_layers: int = 1,
        dropout: float = 0.3,
        use_contrastive: bool = True,
        contrastive_lambda: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        self.use_contrastive = use_contrastive
        self.contrastive_lambda = contrastive_lambda
        
        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Temporal encoder
        self.temporal_attn = TemporalAttention(hidden, n_heads, dropout)
        self.temporal_lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Linear(hidden * 2, hidden)
        
        # Spatial encoder (HyperGAT)
        self.hypergat_layers = nn.ModuleList([
            HyperGATLayer(hidden, dropout) for _ in range(n_hypergat_layers)
        ])
        
        # Region embedding
        self.region_emb = nn.Embedding(n_regions, hidden)
        
        # Spatial GCN
        self.gcn = nn.Linear(hidden, hidden)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
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
        A_norm: Tensor,
        H_inc: Tensor,
        membership: np.ndarray,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, N, T, F)
        Returns: pred (B, N), h_spatial (B, N, H), h_temporal (B, N, H)
        """
        B, N, T, F = x.shape
        
        # Project features
        x_flat = x.reshape(B * N, T, F)
        h = self.input_proj(x_flat)  # (B*N, T, H)
        
        # Temporal encoding
        h = self.temporal_attn(h)  # (B*N, T, H)
        lstm_out, _ = self.temporal_lstm(h)  # (B*N, T, 2H)
        h_temporal = self.temporal_proj(lstm_out[:, -1, :])  # (B*N, H)
        h_temporal = h_temporal.reshape(B, N, self.hidden)
        
        # Spatial encoding
        h_spatial_list = []
        for b in range(B):
            h_b = h_temporal[b]  # (N, H)
            
            # HyperGAT
            for layer in self.hypergat_layers:
                h_b = layer(h_b, H_inc[:N])  # Only node part of H_inc
            
            # GCN
            h_gcn = Fn.relu(self.gcn(torch.mm(A_norm, h_b)))
            h_b = h_b + h_gcn  # Residual
            
            # Add region embedding
            mem_t = torch.tensor(membership, device=x.device, dtype=torch.long)
            r_emb = self.region_emb(mem_t)
            h_b = h_b + r_emb
            
            h_spatial_list.append(h_b)
        
        h_spatial = torch.stack(h_spatial_list, dim=0)  # (B, N, H)
        
        # Fusion
        h_fused = self.fusion(torch.cat([h_temporal, h_spatial], dim=-1))
        
        # Predict
        pred = self.head(h_fused).squeeze(-1)  # (B, N)
        
        return pred, h_spatial, h_temporal
    
    def compute_loss(
        self,
        pred: Tensor,
        y: Tensor,
        mask: Tensor,
        h_spatial: Tensor,
        h_temporal: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        # MSE loss
        mse_loss = Fn.mse_loss(pred[mask], y[mask])
        
        # Contrastive loss (InfoNCE)
        if self.use_contrastive and self.training:
            B, N, H = h_spatial.shape
            hs = Fn.normalize(h_spatial.reshape(B * N, H), dim=-1)
            ht = Fn.normalize(h_temporal.reshape(B * N, H), dim=-1)
            
            sim = torch.mm(hs, ht.T) / 0.1
            labels = torch.arange(B * N, device=pred.device)
            contrastive = Fn.cross_entropy(sim, labels)
            
            total = mse_loss + self.contrastive_lambda * contrastive
        else:
            contrastive = torch.tensor(0.0)
            total = mse_loss
        
        return total, {
            'total': float(total),
            'mse': float(mse_loss),
            'contrastive': float(contrastive),
        }


model = STCHGAT(
    in_features=N_FEATURES,
    hidden=CONFIG['hidden'],
    n_regions=n_regions,
    n_heads=CONFIG['n_heads'],
    n_hypergat_layers=CONFIG['hypergat_layers'],
    dropout=CONFIG['dropout'],
    use_contrastive=CONFIG['use_contrastive'],
    contrastive_lambda=CONFIG['contrastive_lambda'],
).to(DEVICE)

A_norm = A_norm.to(DEVICE)
H_inc = H_inc.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parameters: {n_params:,}")

# ============================================================================
# Training
# ============================================================================
print("\n7. Training STC-HGAT...")

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_val_r2 = -float('inf')
best_state = None
patience_cnt = 0
history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

for epoch in range(1, CONFIG['epochs'] + 1):
    # Train
    model.train()
    train_losses, train_preds, train_trues = [], [], []
    
    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        
        optimizer.zero_grad()
        pred, h_s, h_t = model(xb, A_norm, H_inc, membership)
        loss, ld = model.compute_loss(pred, yb, mb, h_s, h_t)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        train_losses.append(ld['total'])
        train_preds.append(pred[mb].detach().cpu().numpy())
        train_trues.append(yb[mb].cpu().numpy())
    
    scheduler.step()
    
    avg_train_loss = np.mean(train_losses)
    train_r2 = compute_r2(np.concatenate(train_trues), np.concatenate(train_preds))
    
    # Validate
    model.eval()
    val_losses, val_preds, val_trues = [], [], []
    
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            pred, h_s, h_t = model(xb, A_norm, H_inc, membership)
            loss, ld = model.compute_loss(pred, yb, mb, h_s, h_t)
            val_losses.append(ld['mse'])  # Use MSE for validation
            val_preds.append(pred[mb].cpu().numpy())
            val_trues.append(yb[mb].cpu().numpy())
    
    avg_val_loss = np.mean(val_losses)
    val_r2 = compute_r2(np.concatenate(val_trues), np.concatenate(val_preds))
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_r2'].append(train_r2)
    history['val_r2'].append(val_r2)
    
    if epoch % 10 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train L={avg_train_loss:.4f} R²={train_r2:.4f} | "
              f"Val L={avg_val_loss:.4f} R²={val_r2:.4f} | LR={lr:.2e}")
    
    if val_r2 > 0.9:
        print(f"🎯 TARGET ACHIEVED! Val R² = {val_r2:.4f}")
    
    if val_r2 > best_val_r2 + 1e-4:
        best_val_r2 = val_r2
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
if best_state:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"Loaded best model with Val R² = {best_val_r2:.4f}")

# ============================================================================
# Test Evaluation
# ============================================================================
print("\n8. Evaluating on test set...")

model.eval()
test_preds, test_trues = [], []

with torch.no_grad():
    for xb, yb, mb in test_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        pred, _, _ = model(xb, A_norm, H_inc, membership)
        test_preds.append(pred[mb].cpu().numpy())
        test_trues.append(yb[mb].cpu().numpy())

y_pred_norm = np.concatenate(test_preds)
y_true_norm = np.concatenate(test_trues)

# Inverse transform
y_pred = y_pred_norm * target_std + target_mean
y_true = y_true_norm * target_std + target_mean

# Metrics
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
r2 = compute_r2(y_true, y_pred)
smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
mbe = np.mean(y_pred - y_true)

print(f"\n{'='*60}")
print("STC-HGAT TEST METRICS (original PM2.5 scale)")
print(f"{'='*60}")
print(f"  RMSE  : {rmse:.4f} µg/m³")
print(f"  MAE   : {mae:.4f} µg/m³")
print(f"  R²    : {r2:.4f}")
print(f"  SMAPE : {smape:.2f}%")
print(f"  MBE   : {mbe:.4f} µg/m³")

if r2 > 0.9:
    print(f"\n🎯 TARGET R² > 0.9 ACHIEVED!")
elif r2 > 0.8:
    print(f"\n✓ Excellent result: R² = {r2:.4f}")
elif r2 > 0.7:
    print(f"\n✓ Good result: R² = {r2:.4f}")

# ============================================================================
# Save
# ============================================================================
reports_dir = Path('../data/reports/stc_hgat_spatial')
reports_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'history': history,
    'target_mean': target_mean,
    'target_std': target_std,
    'feature_cols': feature_cols,
    'station_order': station_order,
}, reports_dir / 'model.pt')

metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2), 
           'SMAPE': float(smape), 'MBE': float(mbe)}
with open(reports_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nArtifacts saved to {reports_dir}")
