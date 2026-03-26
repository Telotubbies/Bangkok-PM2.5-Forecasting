"""
STC-HGAT PM2.5 Forecasting - V3
Key improvements:
1. Focus on lag features as primary predictors
2. Use simpler architecture with residual connection
3. Better feature engineering
4. Larger batch size for stability
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
    'lookback': 14,  # Increased lookback
    'min_stations': 10,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    
    # Model
    'hidden': 128,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.2,
    
    # Training
    'epochs': 200,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 30,
    'grad_clip': 1.0,
}

# ============================================================================
# Data Loading
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
}).reset_index()

# Merge
df_all = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df_all = df_all[df_all['date'] >= CONFIG['start_date']]
df_all = df_all[df_all['pm2_5'].notna()].copy()
df_all = df_all.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET_COL = 'pm2_5'

# Create lag features - CRITICAL for time series
for lag in [1, 2, 3, 7, 14]:
    df_all[f'pm2_5_lag{lag}'] = df_all.groupby('stationID')[TARGET_COL].shift(lag)

# Rolling statistics
for window in [3, 7, 14]:
    df_all[f'pm2_5_mean_{window}d'] = df_all.groupby('stationID')[TARGET_COL].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df_all[f'pm2_5_std_{window}d'] = df_all.groupby('stationID')[TARGET_COL].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

# PM10/PM2.5 ratio
df_all['pm10_pm25_ratio'] = df_all['pm10'] / (df_all['pm2_5'] + 1)

# Time features
df_all['day_sin'] = np.sin(2 * np.pi * df_all['date'].dt.dayofyear / 365)
df_all['day_cos'] = np.cos(2 * np.pi * df_all['date'].dt.dayofyear / 365)
df_all['month'] = df_all['date'].dt.month
df_all['is_dry_season'] = df_all['month'].isin([11, 12, 1, 2, 3]).astype(float)

print(f"   Rows: {len(df_all):,}, Stations: {df_all['stationID'].nunique()}")

# ============================================================================
# Split & Normalize
# ============================================================================
print("\n2. Splitting and normalizing...")

# Feature columns - prioritize lag features
exclude = {'date', 'stationID', TARGET_COL, 'lat', 'lon', 'month'}
feature_cols = [c for c in df_all.columns if c not in exclude 
                and df_all[c].dtype in [np.float64, np.float32, np.int64]]

# Fill missing
df_all[feature_cols] = df_all.groupby('stationID')[feature_cols].transform(
    lambda g: g.ffill().bfill()
)
for col in feature_cols:
    if df_all[col].isna().any():
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Date-based split
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

target_mean = train_df[TARGET_COL].mean()
target_std = train_df[TARGET_COL].std()
train_df[TARGET_COL] = (train_df[TARGET_COL] - target_mean) / target_std
val_df[TARGET_COL] = (val_df[TARGET_COL] - target_mean) / target_std
test_df[TARGET_COL] = (test_df[TARGET_COL] - target_mean) / target_std

print(f"   Features: {len(feature_cols)}")
print(f"   Key features: {[c for c in feature_cols if 'lag' in c]}")

# Station order
station_meta = df_all.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}
N_FEATURES = len(feature_cols)

# ============================================================================
# Create Sequences
# ============================================================================
print("\n3. Creating sequences...")

def create_sequences(df, lookback=14, min_stations=10):
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
            if sid not in sid2idx or pd.isna(row[TARGET_COL]):
                continue
            idx = sid2idx[sid]
            
            window = df[(df['stationID'] == sid) & (df['date'].isin(window_dates))].sort_values('date')
            if len(window) < lookback:
                continue
            
            X[idx] = window[feature_cols].values
            y[idx] = row[TARGET_COL]
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
# Build Spatial Graph
# ============================================================================
print("\n4. Building spatial graph...")

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

# Adjacency: connect stations within 150km
adj = (D < 150) & (D > 0)
edge_index = torch.tensor(np.stack(np.where(adj)), dtype=torch.long)

# Normalize adjacency for GCN
A = adj.astype(np.float32)
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(1) + 1e-8))
A_norm = torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)

print(f"   Edges: {edge_index.shape[1]}")

# ============================================================================
# Model: Temporal + Spatial with Residual
# ============================================================================
print("\n5. Building model...")

class SpatioTemporalModel(nn.Module):
    """Combined spatio-temporal model with residual connections."""
    
    def __init__(self, in_features, hidden=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        
        # Feature projection
        self.input_proj = nn.Linear(in_features, hidden)
        
        # Temporal: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            dropout=dropout, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Spatial: GCN layers
        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
    
    def forward(self, x, A_norm):
        """
        x: (B, N, T, F)
        A_norm: (N, N) normalized adjacency
        """
        B, N, T, F = x.shape
        
        # Project features
        x = x.reshape(B * N, T, F)
        h = self.input_proj(x)  # (B*N, T, H)
        
        # Temporal encoding
        h = self.temporal_encoder(h)  # (B*N, T, H)
        h = h[:, -1, :]  # Take last timestep: (B*N, H)
        h = h.reshape(B, N, self.hidden)
        
        # Spatial GCN
        A = A_norm.to(h.device)
        h_spatial = Fn.relu(self.gcn1(torch.matmul(A, h)))
        h_spatial = self.dropout(h_spatial)
        h_spatial = self.gcn2(torch.matmul(A, h_spatial))
        
        # Residual connection
        h = self.norm(h + h_spatial)
        
        # Output
        pred = self.output(h).squeeze(-1)  # (B, N)
        
        return pred

model = SpatioTemporalModel(
    in_features=N_FEATURES,
    hidden=CONFIG['hidden'],
    n_heads=CONFIG['n_heads'],
    n_layers=CONFIG['n_layers'],
    dropout=CONFIG['dropout'],
).to(DEVICE)

A_norm = A_norm.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parameters: {n_params:,}")

# ============================================================================
# Training
# ============================================================================
print("\n6. Training...")

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
        pred = model(xb, A_norm)
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
            pred = model(xb, A_norm)
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
print("\n7. Evaluating on test set...")

model.eval()
test_preds, test_trues = [], []

with torch.no_grad():
    for xb, yb, mb in test_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        pred = model(xb, A_norm)
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
print("TEST METRICS (original PM2.5 scale)")
print(f"{'='*60}")
print(f"  RMSE  : {rmse:.4f} µg/m³")
print(f"  MAE   : {mae:.4f} µg/m³")
print(f"  R²    : {r2:.4f}")
print(f"  SMAPE : {smape:.2f}%")
print(f"  MBE   : {mbe:.4f} µg/m³")

if r2 > 0.9:
    print(f"\n🎯 TARGET R² > 0.9 ACHIEVED!")
elif r2 > 0.8:
    print(f"\n✓ Good result: R² = {r2:.4f}")
elif r2 > 0.6:
    print(f"\n⚠ Moderate result: R² = {r2:.4f}")

# ============================================================================
# Save
# ============================================================================
reports_dir = Path('../data/reports/stc_hgat_v3')
reports_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'history': history,
    'target_mean': target_mean,
    'target_std': target_std,
    'feature_cols': feature_cols,
}, reports_dir / 'model.pt')

with open(reports_dir / 'metrics.json', 'w') as f:
    json.dump({'RMSE': rmse, 'MAE': mae, 'R2': r2, 'SMAPE': smape, 'MBE': mbe}, f, indent=2)

print(f"\nArtifacts saved to {reports_dir}")
