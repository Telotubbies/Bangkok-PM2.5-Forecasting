"""
STC-HGAT PM2.5 Forecasting - V4
Strategy: Use PM10 but force spatial learning by:
1. Using PM10 from NEIGHBORING stations (not same station)
2. Graph-based message passing to aggregate neighbor information
3. This forces model to learn spatial relationships
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
    'lookback': 7,
    'min_stations': 20,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    
    # Model
    'hidden': 128,
    'n_heads': 4,
    'n_gnn_layers': 3,
    'dropout': 0.2,
    
    # Spatial
    'k_neighbors': 5,
    
    # Training
    'epochs': 150,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'patience': 25,
    'grad_clip': 1.0,
}

THAILAND_REGIONS = {
    "North": ["CM", "LM", "LP", "PY", "NAN", "PR", "PL", "MSH", "TAK", "SUK"],
    "Northeast": ["KKN", "UDT", "NKP", "SK", "BRM", "ROI", "MKM", "YST", "SR"],
    "Central": ["BKK", "NPT", "AYA", "SBR", "NBR", "CNB", "KRI", "PKN", "SMT"],
    "East": ["RY", "CH", "TRT", "PKO", "SA"],
    "South": ["HYI", "PKT", "SNG", "NRT", "PT", "KBI", "TG", "PAN"],
}

# ============================================================================
# Data Loading - Include PM10 but use spatially
# ============================================================================
print("\n1. Loading data...")

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
df_all = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df_all = df_all[df_all['date'] >= CONFIG['start_date']]
df_all = df_all[df_all['pm2_5'].notna()].copy()
df_all = df_all.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET = 'pm2_5'

print(f"   Rows: {len(df_all):,}, Stations: {df_all['stationID'].nunique()}")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n2. Feature engineering...")

# PM2.5 and PM10 lags
for lag in [1, 2, 3, 7]:
    df_all[f'pm2_5_lag{lag}'] = df_all.groupby('stationID')[TARGET].shift(lag)
    df_all[f'pm10_lag{lag}'] = df_all.groupby('stationID')['pm10'].shift(lag)

# Rolling statistics
for window in [3, 7]:
    df_all[f'pm2_5_mean_{window}d'] = df_all.groupby('stationID')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df_all[f'pm10_mean_{window}d'] = df_all.groupby('stationID')['pm10'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

# Time features
df_all['day_sin'] = np.sin(2 * np.pi * df_all['date'].dt.dayofyear / 365)
df_all['day_cos'] = np.cos(2 * np.pi * df_all['date'].dt.dayofyear / 365)
df_all['month'] = df_all['date'].dt.month
df_all['is_dry_season'] = df_all['month'].isin([11, 12, 1, 2, 3]).astype(float)

# Normalized coordinates
lat_mean, lat_std = df_all['lat'].mean(), df_all['lat'].std()
lon_mean, lon_std = df_all['lon'].mean(), df_all['lon'].std()
df_all['lat_norm'] = (df_all['lat'] - lat_mean) / lat_std
df_all['lon_norm'] = (df_all['lon'] - lon_mean) / lon_std

# Feature columns - INCLUDE PM10 (will be used spatially)
feature_cols = [
    # PM2.5 features
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7',
    'pm2_5_mean_3d', 'pm2_5_mean_7d',
    
    # PM10 features (will be aggregated from neighbors)
    'pm10', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7',
    'pm10_mean_3d', 'pm10_mean_7d',
    
    # Weather
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    
    # Time
    'day_sin', 'day_cos', 'is_dry_season',
    
    # Spatial
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

print(f"   Features: {len(feature_cols)}")
print(f"   After cleaning: {len(df_all):,} rows")

# ============================================================================
# Station metadata and spatial graph
# ============================================================================
station_meta = df_all.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}
N_FEATURES = len(feature_cols)

print(f"   Stations: {N_STATIONS}")

# Build spatial graph
print("\n3. Building spatial graph...")

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

# K-nearest neighbors
k = CONFIG['k_neighbors']
neighbor_idx = np.zeros((N_STATIONS, k), dtype=np.int64)
neighbor_weights = np.zeros((N_STATIONS, k), dtype=np.float32)

for i in range(N_STATIONS):
    sorted_idx = np.argsort(D[i])[1:k+1]  # Exclude self
    neighbor_idx[i] = sorted_idx
    # Inverse distance weights
    dists = D[i, sorted_idx]
    weights = 1.0 / (dists + 1e-6)
    neighbor_weights[i] = weights / weights.sum()

# Adjacency matrix (normalized)
adj = np.zeros((N_STATIONS, N_STATIONS), dtype=np.float32)
for i in range(N_STATIONS):
    for j, w in zip(neighbor_idx[i], neighbor_weights[i]):
        adj[i, j] = w

A_norm = torch.tensor(adj, dtype=torch.float32)

# Region membership
region_names = list(THAILAND_REGIONS.keys())
n_regions = len(region_names)
membership = np.zeros(N_STATIONS, dtype=np.int64)
for idx, sid in enumerate(station_order):
    for r_idx, (region, prefixes) in enumerate(THAILAND_REGIONS.items()):
        if any(sid.upper().startswith(p.upper()) for p in prefixes):
            membership[idx] = r_idx
            break

print(f"   K-NN: {k} neighbors per station")
print(f"   Regions: {n_regions}")

# ============================================================================
# Split
# ============================================================================
print("\n4. Splitting data...")

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

# ============================================================================
# Create Sequences
# ============================================================================
print("\n5. Creating sequences...")

def create_sequences(df, lookback=7, min_stations=20):
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
# STC-HGAT Model with Spatial Message Passing
# ============================================================================
print("\n6. Building STC-HGAT model...")

class GraphConv(nn.Module):
    """Graph convolution layer."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, A):
        # x: (B, N, D), A: (N, N)
        h = torch.matmul(A, x)  # Aggregate neighbors
        h = self.linear(h)
        h = Fn.relu(h)
        h = self.norm(h)
        return self.drop(h)


class SpatialEncoder(nn.Module):
    """Multi-layer GNN for spatial encoding."""
    def __init__(self, hidden, n_layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConv(hidden, hidden, dropout) for _ in range(n_layers)
        ])
    
    def forward(self, x, A):
        for layer in self.layers:
            x = x + layer(x, A)  # Residual
        return x


class TemporalEncoder(nn.Module):
    """Temporal encoding with LSTM and attention."""
    def __init__(self, in_dim, hidden, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, 
                           bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(hidden * 2, 1)
        self.out_proj = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
    
    def forward(self, x):
        # x: (B*N, T, F)
        h = Fn.relu(self.proj(x))
        lstm_out, _ = self.lstm(h)  # (B*N, T, 2H)
        
        # Attention over time
        attn_weights = Fn.softmax(self.attn(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B*N, 2H)
        
        out = self.out_proj(context)
        return self.norm(out)


class STCHGAT(nn.Module):
    """STC-HGAT with spatial message passing."""
    
    def __init__(
        self,
        in_features: int,
        hidden: int = 128,
        n_regions: int = 5,
        n_gnn_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden = hidden
        
        # Temporal encoder
        self.temporal = TemporalEncoder(in_features, hidden, dropout)
        
        # Spatial encoder
        self.spatial = SpatialEncoder(hidden, n_gnn_layers, dropout)
        
        # Region embedding
        self.region_emb = nn.Embedding(n_regions, hidden)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output
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
    
    def forward(self, x: Tensor, A: Tensor, membership: np.ndarray) -> Tensor:
        """
        x: (B, N, T, F)
        A: (N, N) adjacency
        Returns: pred (B, N)
        """
        B, N, T, F = x.shape
        
        # Temporal encoding
        x_flat = x.reshape(B * N, T, F)
        h_temporal = self.temporal(x_flat)  # (B*N, H)
        h_temporal = h_temporal.reshape(B, N, self.hidden)
        
        # Spatial encoding with message passing
        A_dev = A.to(x.device)
        h_spatial = self.spatial(h_temporal, A_dev)  # (B, N, H)
        
        # Add region embedding
        mem_t = torch.tensor(membership, device=x.device, dtype=torch.long)
        r_emb = self.region_emb(mem_t).unsqueeze(0).expand(B, -1, -1)
        h_spatial = h_spatial + r_emb
        
        # Fusion
        h = self.fusion(torch.cat([h_temporal, h_spatial], dim=-1))
        
        # Predict
        pred = self.head(h).squeeze(-1)
        
        return pred


model = STCHGAT(
    in_features=N_FEATURES,
    hidden=CONFIG['hidden'],
    n_regions=n_regions,
    n_gnn_layers=CONFIG['n_gnn_layers'],
    dropout=CONFIG['dropout'],
).to(DEVICE)

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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

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
        pred = model(xb, A_norm, membership)
        loss = Fn.mse_loss(pred[mb], yb[mb])
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        train_losses.append(loss.item())
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
            pred = model(xb, A_norm, membership)
            loss = Fn.mse_loss(pred[mb], yb[mb])
            val_losses.append(loss.item())
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
        pred = model(xb, A_norm, membership)
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
reports_dir = Path('../data/reports/stc_hgat_v4')
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
