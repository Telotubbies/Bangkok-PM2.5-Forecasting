"""
True Spatial-Temporal Hypergraph Attention Network (STHGAT) for PM2.5 Forecasting

Architecture:
1. Spatial Graph Attention (GAT) - learns attention weights between stations
2. Temporal Attention - captures temporal dependencies in sequences
3. Hypergraph Convolution - models higher-order relationships
4. Multi-head attention for robust feature learning

Reference: Based on STHGAT paper concepts for air quality prediction
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    
    # Spatial
    'k_neighbors': 5,
    
    # Sequence
    'seq_len': 7,  # 7 days lookback
    'pred_len': 1,  # 1 day ahead
    
    # Model Architecture
    'node_features': 8,  # Features per node per timestep
    'hidden_dim': 128,
    'num_heads': 4,
    'num_gat_layers': 2,
    'num_temporal_layers': 2,
    'dropout': 0.3,
    
    # Hypergraph
    'hyperedge_dim': 32,
    
    # Training
    'epochs': 200,
    'batch_size': 64,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'patience': 15,
}

# ============================================================================
# Data Loading
# ============================================================================
print("\n[1/8] Loading data...")

aq_dir = Path(CONFIG['aq_data_dir'])
aq_files = list(aq_dir.glob('**/*.parquet'))
aq_dfs = [pd.read_parquet(f) for f in aq_files]
aq_df = pd.concat(aq_dfs, ignore_index=True)

aq_df['timestamp_utc'] = pd.to_datetime(aq_df['timestamp_utc'])
aq_df['date'] = pd.to_datetime(aq_df['timestamp_utc'].dt.date)

daily_aq = aq_df.groupby(['stationID', 'date', 'lat', 'lon']).agg({
    'pm2_5_ugm3': 'mean',
    'pm10_ugm3': 'mean',
}).reset_index().rename(columns={
    'pm2_5_ugm3': 'pm2_5',
    'pm10_ugm3': 'pm10',
})

# Load weather
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
df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= CONFIG['start_date']]
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

print(f"   Rows: {len(df):,}, Stations: {df['stationID'].nunique()}")

# ============================================================================
# Build Spatial Graph
# ============================================================================
print("\n[2/8] Building spatial graph...")

station_meta = df.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}
idx2sid = {i: s for s, i in sid2idx.items()}

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

# Build edge index (K-nearest neighbors graph)
k = CONFIG['k_neighbors']
edge_index = []
edge_weight = []

for i in range(N_STATIONS):
    sorted_idx = np.argsort(D[i])[1:k+1]  # Exclude self
    for j in sorted_idx:
        edge_index.append([i, j])
        # Weight inversely proportional to distance
        edge_weight.append(1.0 / (D[i, j] + 1e-6))

edge_index = torch.tensor(edge_index, dtype=torch.long).T  # (2, E)
edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
edge_weight = edge_weight / edge_weight.max()  # Normalize

print(f"   Nodes: {N_STATIONS}, Edges: {edge_index.shape[1]}")

# ============================================================================
# Build Hypergraph (Regional clusters)
# ============================================================================
print("\n[3/8] Building hypergraph...")

# Create hyperedges based on spatial proximity (regional clusters)
from sklearn.cluster import KMeans

coords = np.column_stack([lats, lons])
n_clusters = min(10, N_STATIONS // 3)  # Regional clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)

# Hyperedge incidence matrix H: (N_STATIONS, N_HYPEREDGES)
H = np.zeros((N_STATIONS, n_clusters), dtype=np.float32)
for i, label in enumerate(cluster_labels):
    H[i, label] = 1.0

H = torch.tensor(H, dtype=torch.float32)
print(f"   Hyperedges (regional clusters): {n_clusters}")

# ============================================================================
# Prepare Sequences
# ============================================================================
print("\n[4/8] Preparing sequences...")

# Features per node (include PM10 and lags, exclude current pm2_5)
node_features = ['pm10', 'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa']

# Add lag features to avoid data leakage
for lag in [1, 2, 3, 7]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')['pm2_5'].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

# Rolling features
df['pm10_mean_7d'] = df.groupby('stationID')['pm10'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['pm2_5_mean_7d'] = df.groupby('stationID')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Fill missing values
for col in node_features:
    df[col] = df[col].fillna(df[col].median())

# Add temporal features
df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)

# Update feature list
node_features = [
    'pm10', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7', 'pm10_mean_7d',
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_mean_7d',
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'day_sin', 'day_cos'
]

# Fill lag features
for col in node_features:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

CONFIG['node_features'] = len(node_features)

# Pivot to get (date, station, features) structure
dates = sorted(df['date'].unique())
n_dates = len(dates)
date2idx = {d: i for i, d in enumerate(dates)}

# Create 3D tensor: (T, N, F)
data_tensor = np.zeros((n_dates, N_STATIONS, len(node_features)), dtype=np.float32)
target_tensor = np.zeros((n_dates, N_STATIONS), dtype=np.float32)

for _, row in df.iterrows():
    t = date2idx[row['date']]
    n = sid2idx[row['stationID']]
    for f_idx, feat in enumerate(node_features):
        data_tensor[t, n, f_idx] = row[feat]
    target_tensor[t, n] = row['pm2_5']

# Fill any remaining zeros with column means
for f_idx in range(len(node_features)):
    col_mean = data_tensor[:, :, f_idx][data_tensor[:, :, f_idx] != 0].mean()
    data_tensor[:, :, f_idx][data_tensor[:, :, f_idx] == 0] = col_mean

print(f"   Data tensor shape: {data_tensor.shape}")  # (T, N, F)

# ============================================================================
# Create Sequences Dataset
# ============================================================================
class STHGATDataset(Dataset):
    def __init__(self, data, targets, seq_len, pred_len, start_idx, end_idx):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Valid indices
        self.valid_indices = list(range(
            max(start_idx, seq_len),
            min(end_idx, len(data) - pred_len)
        ))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input sequence: (seq_len, N, F)
        x = self.data[t - self.seq_len:t]
        
        # Target: (N,) - PM2.5 at t+pred_len-1
        y = self.targets[t + self.pred_len - 1]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Split
t1 = int(n_dates * CONFIG['train_ratio'])
t2 = int(n_dates * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

train_dataset = STHGATDataset(data_tensor, target_tensor, CONFIG['seq_len'], CONFIG['pred_len'], 0, t1)
val_dataset = STHGATDataset(data_tensor, target_tensor, CONFIG['seq_len'], CONFIG['pred_len'], t1, t2)
test_dataset = STHGATDataset(data_tensor, target_tensor, CONFIG['seq_len'], CONFIG['pred_len'], t2, n_dates)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============================================================================
# Model Components
# ============================================================================
print("\n[5/8] Building STHGAT model...")

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer (GAT) with sparse softmax"""
    
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, h, edge_index, edge_weight=None):
        """
        h: (B, N, F) or (N, F)
        edge_index: (2, E)
        """
        # Handle batched input
        if h.dim() == 3:
            B, N, F = h.shape
            # Process each batch
            outputs = []
            for b in range(B):
                out = self._forward_single(h[b], edge_index, edge_weight)
                outputs.append(out)
            return torch.stack(outputs, dim=0)
        else:
            return self._forward_single(h, edge_index, edge_weight)
    
    def _forward_single(self, h, edge_index, edge_weight=None):
        """Single graph forward with sparse softmax"""
        N = h.size(0)
        
        # Linear transformation
        Wh = self.W(h)  # (N, out_features)
        
        # Attention mechanism
        src, dst = edge_index[0], edge_index[1]
        
        # Concatenate source and destination features
        edge_h = torch.cat([Wh[src], Wh[dst]], dim=1)  # (E, 2*out_features)
        
        # Attention coefficients
        e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze())  # (E,)
        
        # Apply edge weights if provided
        if edge_weight is not None:
            e = e * edge_weight
        
        # Sparse softmax: only over actual neighbors
        attention = torch.zeros(N, N, device=h.device)
        attention[src, dst] = e
        
        # Mask zeros before softmax
        mask = (attention == 0)
        attention = attention.masked_fill(mask, float('-inf'))
        attention = F.softmax(attention, dim=1)
        attention = attention.masked_fill(mask, 0)  # Set back to 0 for non-neighbors
        attention = self.dropout_layer(attention)
        
        # Aggregate
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention"""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, concat=True)
            for _ in range(num_heads)
        ])
        
        if concat:
            self.out_proj = nn.Linear(out_features * num_heads, out_features)
        else:
            self.out_proj = nn.Linear(out_features, out_features)
    
    def forward(self, h, edge_index, edge_weight=None):
        # Apply each attention head
        head_outputs = [head(h, edge_index, edge_weight) for head in self.attention_heads]
        
        if self.concat:
            # Concatenate heads
            h = torch.cat(head_outputs, dim=-1)
        else:
            # Average heads
            h = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        return self.out_proj(h)


class HypergraphConv(nn.Module):
    """Hypergraph Convolution Layer"""
    
    def __init__(self, in_features, out_features, H):
        super().__init__()
        self.H = H  # Incidence matrix (N, M)
        self.linear = nn.Linear(in_features, out_features)
        
        # Compute degree matrices
        D_v = torch.diag(H.sum(dim=1).pow(-0.5))  # Node degree
        D_e = torch.diag(H.sum(dim=0).pow(-1))    # Hyperedge degree
        
        # Normalized Laplacian: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        self.register_buffer('laplacian', D_v @ H @ D_e @ H.T @ D_v)
    
    def forward(self, x):
        """
        x: (B, N, F) or (N, F)
        """
        if x.dim() == 3:
            B, N, F = x.shape
            x_flat = x.view(B * N, F)
            x_transformed = self.linear(x_flat).view(B, N, -1)
            # Apply hypergraph convolution
            out = torch.einsum('nm,bmf->bnf', self.laplacian, x_transformed)
            return out
        else:
            x_transformed = self.linear(x)
            return self.laplacian @ x_transformed


class TemporalAttention(nn.Module):
    """Temporal Self-Attention Layer with Causal Mask"""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, T, N, F) -> process temporal dimension with causal mask
        """
        B, T, N, F = x.shape
        
        # Reshape to (B*N, T, F) for temporal attention
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        # Create causal mask: prevent attending to future timesteps
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # Self-attention over time with causal mask
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm(x + self.dropout(attn_out))
        
        # Reshape back to (B, T, N, F)
        x = x.reshape(B, N, T, F).permute(0, 2, 1, 3)
        
        return x


class STHGAT(nn.Module):
    """
    Spatial-Temporal Hypergraph Attention Network
    
    Components:
    1. Input projection
    2. Spatial GAT layers (learn station relationships)
    3. Hypergraph convolution (capture regional patterns)
    4. Temporal attention (capture time dependencies)
    5. Output projection
    """
    
    def __init__(self, config, edge_index, edge_weight, H, n_stations):
        super().__init__()
        
        self.config = config
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        self.n_stations = n_stations
        
        node_features = config['node_features']
        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        dropout = config['dropout']
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Spatial GAT layers
        self.gat_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(config['num_gat_layers'])
        ])
        
        # Hypergraph convolution
        self.hypergraph_conv = HypergraphConv(hidden_dim, hidden_dim, H)
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(hidden_dim, num_heads, dropout)
            for _ in range(config['num_temporal_layers'])
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Temporal pooling (instead of hard-coded seq_len)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        x: (B, T, N, F) - batch of sequences
        Returns: (B, N) - predictions for each station
        """
        B, T, N, F = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, T, N, hidden_dim)
        
        # Spatial processing (per timestep)
        spatial_out = []
        for t in range(T):
            h = x[:, t]  # (B, N, hidden_dim)
            
            # GAT layers
            for gat in self.gat_layers:
                h = h + gat(h, self.edge_index, self.edge_weight)  # Residual
            
            # Hypergraph convolution
            h_hyper = self.hypergraph_conv(h)
            
            # Combine GAT and Hypergraph
            h = self.fusion(torch.cat([h, h_hyper], dim=-1))
            
            spatial_out.append(h)
        
        # Stack temporal: (B, T, N, hidden_dim)
        x = torch.stack(spatial_out, dim=1)
        
        # Temporal attention
        for temporal in self.temporal_layers:
            x = temporal(x)
        
        # Temporal pooling: (B, T, N, hidden_dim) -> (B, N, hidden_dim)
        # Average over time dimension
        x = x.mean(dim=1)  # (B, N, hidden_dim)
        
        # Output projection: (B, N, hidden_dim) -> (B, N)
        out = self.output_proj(x).squeeze(-1)
        
        return out


# Initialize model
model = STHGAT(
    config=CONFIG,
    edge_index=edge_index.to(DEVICE),
    edge_weight=edge_weight.to(DEVICE),
    H=H.to(DEVICE),
    n_stations=N_STATIONS
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   STHGAT Parameters: {n_params:,}")

# ============================================================================
# Training
# ============================================================================
print("\n[6/8] Training STHGAT...")

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.SmoothL1Loss()

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(CONFIG['epochs']):
    # Train
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item() * x.size(0)
    
    train_loss /= len(train_dataset)
    
    # Validate
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)
            
            val_preds.append(pred.cpu())
            val_targets.append(y.cpu())
    
    val_loss /= len(val_dataset)
    
    val_preds = torch.cat(val_preds).numpy().flatten()
    val_targets = torch.cat(val_targets).numpy().flatten()
    val_r2 = r2_score(val_targets, val_preds)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"   Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)

# ============================================================================
# Evaluation
# ============================================================================
print("\n[7/8] Evaluating on test set...")

model.eval()
test_preds, test_targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        test_preds.append(pred.cpu())
        test_targets.append(y.cpu())

test_preds = torch.cat(test_preds).numpy().flatten()
test_targets = torch.cat(test_targets).numpy().flatten()

test_r2 = r2_score(test_targets, test_preds)
test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
test_mae = mean_absolute_error(test_targets, test_preds)

print(f"\n   TEST RESULTS (True STHGAT):")
print(f"   R²:   {test_r2:.4f}")
print(f"   RMSE: {test_rmse:.2f} µg/m³")
print(f"   MAE:  {test_mae:.2f} µg/m³")

# ============================================================================
# Save Model
# ============================================================================
print("\n[8/8] Saving model...")

save_dir = Path('../data/reports/true_sthgat')
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'metrics': {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
    },
    'station_order': station_order,
    'edge_index': edge_index,
    'edge_weight': edge_weight,
    'H': H,
}, save_dir / 'model.pt')

print(f"   Model saved to: {save_dir / 'model.pt'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("TRUE STHGAT TRAINING COMPLETE")
print("="*70)

print(f"""
MODEL ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────
✅ Graph Attention Network (GAT): {CONFIG['num_gat_layers']} layers, {CONFIG['num_heads']} heads
✅ Hypergraph Convolution: {n_clusters} regional clusters
✅ Temporal Attention: {CONFIG['num_temporal_layers']} layers
✅ Sequence Length: {CONFIG['seq_len']} days

PARAMETERS: {n_params:,}

TEST RESULTS:
  R² Score:    {test_r2:.4f}
  RMSE:        {test_rmse:.2f} µg/m³
  MAE:         {test_mae:.2f} µg/m³

COMPARISON:
  Previous (SpatialMLP): R² = 0.9935
  True STHGAT:           R² = {test_r2:.4f}

SAVED TO: {save_dir}
""")
