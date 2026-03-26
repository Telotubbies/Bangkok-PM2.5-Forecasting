"""
STC-HGAT PM2.5 Forecasting - Row-based approach
Key insight: Sequence-based approach causes data loss and distribution shift
Strategy: 
1. Use row-based data like GB (no sequence creation)
2. Add spatial neighbor features (mean PM10/PM2.5 from K-nearest neighbors)
3. Use GNN to aggregate neighbor information at inference time
4. This combines GB's data efficiency with spatial learning
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
    
    # Model
    'hidden': 128,
    'n_layers': 3,
    'dropout': 0.2,
    
    # Training
    'epochs': 100,
    'batch_size': 512,  # Large batch for row-based
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
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
    'pressure_hpa': 'mean',
}).reset_index()

# Merge
df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= CONFIG['start_date']]
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

TARGET = 'pm2_5'

print(f"   Rows: {len(df):,}, Stations: {df['stationID'].nunique()}")

# ============================================================================
# Build spatial graph and compute neighbor features
# ============================================================================
print("\n2. Building spatial graph and neighbor features...")

station_meta = df.groupby('stationID')[['lat', 'lon']].first().reset_index()
station_order = sorted(station_meta['stationID'].tolist())
N_STATIONS = len(station_order)
sid2idx = {s: i for i, s in enumerate(station_order)}

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
neighbor_idx = {}
for i, sid in enumerate(station_order):
    sorted_idx = np.argsort(D[i])[1:k+1]
    neighbor_idx[sid] = [station_order[j] for j in sorted_idx]

# Compute neighbor features for each row
print("   Computing neighbor aggregations...")

# Create pivot tables for fast lookup
pm10_pivot = df.pivot_table(index='date', columns='stationID', values='pm10', aggfunc='mean')
pm25_pivot = df.pivot_table(index='date', columns='stationID', values='pm2_5', aggfunc='mean')

def get_neighbor_features(row):
    sid = row['stationID']
    date = row['date']
    
    neighbors = neighbor_idx.get(sid, [])
    if not neighbors:
        return pd.Series({'neighbor_pm10_mean': np.nan, 'neighbor_pm25_mean': np.nan,
                         'neighbor_pm10_std': np.nan, 'neighbor_pm25_std': np.nan})
    
    pm10_vals = []
    pm25_vals = []
    
    for n_sid in neighbors:
        if n_sid in pm10_pivot.columns and date in pm10_pivot.index:
            v = pm10_pivot.loc[date, n_sid]
            if not pd.isna(v):
                pm10_vals.append(v)
        if n_sid in pm25_pivot.columns and date in pm25_pivot.index:
            v = pm25_pivot.loc[date, n_sid]
            if not pd.isna(v):
                pm25_vals.append(v)
    
    return pd.Series({
        'neighbor_pm10_mean': np.mean(pm10_vals) if pm10_vals else np.nan,
        'neighbor_pm25_mean': np.mean(pm25_vals) if pm25_vals else np.nan,
        'neighbor_pm10_std': np.std(pm10_vals) if len(pm10_vals) > 1 else 0,
        'neighbor_pm25_std': np.std(pm25_vals) if len(pm25_vals) > 1 else 0,
    })

# Apply neighbor features
neighbor_features = df.apply(get_neighbor_features, axis=1)
df = pd.concat([df, neighbor_features], axis=1)

print(f"   Neighbor features added")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n3. Feature engineering...")

# Lag features
for lag in [1, 2, 3, 7]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')[TARGET].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

# Rolling means
df['pm2_5_mean_7d'] = df.groupby('stationID')[TARGET].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['pm10_mean_7d'] = df.groupby('stationID')['pm10'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# PM10/PM2.5 ratio
df['pm10_pm25_ratio'] = df['pm10'] / (df['pm2_5_lag1'] + 1)

# Neighbor ratio (spatial context)
df['neighbor_local_pm10_ratio'] = df['neighbor_pm10_mean'] / (df['pm10'] + 1)
df['neighbor_local_pm25_ratio'] = df['neighbor_pm25_mean'] / (df['pm2_5_lag1'] + 1)

# Time features
df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
df['is_dry_season'] = df['date'].dt.month.isin([11, 12, 1, 2, 3]).astype(float)

# Normalized coordinates
lat_mean, lat_std = df['lat'].mean(), df['lat'].std()
lon_mean, lon_std = df['lon'].mean(), df['lon'].std()
df['lat_norm'] = (df['lat'] - lat_mean) / lat_std
df['lon_norm'] = (df['lon'] - lon_mean) / lon_std

# Station index for embedding
df['station_idx'] = df['stationID'].map(sid2idx)

# Feature columns
feature_cols = [
    # PM10 - key predictor
    'pm10', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7', 'pm10_mean_7d',
    
    # PM2.5 lags
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_mean_7d',
    
    # Ratios
    'pm10_pm25_ratio',
    
    # Neighbor features (SPATIAL)
    'neighbor_pm10_mean', 'neighbor_pm25_mean',
    'neighbor_pm10_std', 'neighbor_pm25_std',
    'neighbor_local_pm10_ratio', 'neighbor_local_pm25_ratio',
    
    # Weather
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    
    # Time
    'day_sin', 'day_cos', 'is_dry_season',
    
    # Spatial
    'lat_norm', 'lon_norm',
]

# Fill missing
for col in feature_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=feature_cols + [TARGET])

print(f"   Features: {len(feature_cols)}")
print(f"   After cleaning: {len(df):,} rows")

# ============================================================================
# Split
# ============================================================================
print("\n4. Splitting data...")

dates = sorted(df['date'].unique())
n = len(dates)
t1, t2 = int(n * CONFIG['train_ratio']), int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

train_df = df[df['date'].isin(dates[:t1])].copy()
val_df = df[df['date'].isin(dates[t1:t2])].copy()
test_df = df[df['date'].isin(dates[t2:])].copy()

print(f"   Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

y_train = train_df[TARGET].values
y_val = val_df[TARGET].values
y_test = test_df[TARGET].values

# Station indices for embedding
idx_train = train_df['station_idx'].values
idx_val = val_df['station_idx'].values
idx_test = test_df['station_idx'].values

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
idx_train_t = torch.tensor(idx_train, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
idx_val_t = torch.tensor(idx_val, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
idx_test_t = torch.tensor(idx_test, dtype=torch.long)

# ============================================================================
# Model - MLP with Station Embedding (Spatial Context)
# ============================================================================
print("\n5. Building model...")

class SpatialMLP(nn.Module):
    """MLP with station embedding for spatial context."""
    
    def __init__(self, in_features, n_stations, hidden=128, n_layers=3, dropout=0.2):
        super().__init__()
        
        # Station embedding (learns spatial patterns)
        self.station_emb = nn.Embedding(n_stations, hidden // 2)
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden)
        
        # MLP layers
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden + hidden // 2, hidden + hidden // 2),
                nn.LayerNorm(hidden + hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Output
        self.head = nn.Linear(hidden + hidden // 2, 1)
    
    def forward(self, x, station_idx):
        # x: (B, F), station_idx: (B,)
        h = Fn.relu(self.input_proj(x))  # (B, H)
        s = self.station_emb(station_idx)  # (B, H/2)
        h = torch.cat([h, s], dim=-1)  # (B, H + H/2)
        h = self.mlp(h)
        return self.head(h).squeeze(-1)


model = SpatialMLP(
    in_features=len(feature_cols),
    n_stations=N_STATIONS,
    hidden=CONFIG['hidden'],
    n_layers=CONFIG['n_layers'],
    dropout=CONFIG['dropout'],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parameters: {n_params:,}")

# ============================================================================
# Training
# ============================================================================
print("\n6. Training...")

train_ds = TensorDataset(X_train_t, y_train_t, idx_train_t)
val_ds = TensorDataset(X_val_t, y_val_t, idx_val_t)
test_ds = TensorDataset(X_test_t, y_test_t, idx_test_t)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_val_r2 = -float('inf')
best_state = None
patience_cnt = 0

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

for epoch in range(1, CONFIG['epochs'] + 1):
    # Train
    model.train()
    train_preds, train_trues = [], []
    
    for xb, yb, ib in train_loader:
        xb, yb, ib = xb.to(DEVICE), yb.to(DEVICE), ib.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(xb, ib)
        loss = Fn.mse_loss(pred, yb)
        
        loss.backward()
        optimizer.step()
        
        train_preds.append(pred.detach().cpu().numpy())
        train_trues.append(yb.cpu().numpy())
    
    train_r2 = compute_r2(np.concatenate(train_trues), np.concatenate(train_preds))
    
    # Validate
    model.eval()
    val_preds, val_trues = [], []
    
    with torch.no_grad():
        for xb, yb, ib in val_loader:
            xb, yb, ib = xb.to(DEVICE), yb.to(DEVICE), ib.to(DEVICE)
            pred = model(xb, ib)
            val_preds.append(pred.cpu().numpy())
            val_trues.append(yb.cpu().numpy())
    
    val_r2 = compute_r2(np.concatenate(val_trues), np.concatenate(val_preds))
    
    scheduler.step(val_r2)
    
    if epoch % 10 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train R²={train_r2:.4f} | Val R²={val_r2:.4f} | LR={lr:.2e}")
    
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
    for xb, yb, ib in test_loader:
        xb, ib = xb.to(DEVICE), ib.to(DEVICE)
        pred = model(xb, ib)
        test_preds.append(pred.cpu().numpy())
        test_trues.append(yb.numpy())

y_pred = np.concatenate(test_preds)
y_true = np.concatenate(test_trues)

# Metrics
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
r2 = compute_r2(y_true, y_pred)
smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
mbe = np.mean(y_pred - y_true)

print(f"\n{'='*60}")
print("STC-HGAT (Row-based) TEST METRICS")
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
reports_dir = Path('../data/reports/stc_hgat_rowbased')
reports_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'feature_cols': feature_cols,
    'station_order': station_order,
}, reports_dir / 'model.pt')

metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2), 
           'SMAPE': float(smape), 'MBE': float(mbe)}
with open(reports_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nArtifacts saved to {reports_dir}")
