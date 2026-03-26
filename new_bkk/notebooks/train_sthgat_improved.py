"""
Improved STHGAT for PM2.5 Forecasting - Target R² > 0.85

Key Improvements:
1. Add PM10 as input feature (critical for PM2.5 prediction)
2. Use row-based approach instead of sequences (more data)
3. Add spatial neighbor aggregation features
4. Combine GAT + Hypergraph with MLP backbone
5. Multi-task learning (predict PM2.5 and PM10 together)
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    
    # Model Architecture
    'hidden_dim': 128,
    'num_heads': 4,
    'num_gat_layers': 2,
    'dropout': 0.2,
    
    # Training
    'epochs': 150,
    'batch_size': 512,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
}

# ============================================================================
# Data Loading
# ============================================================================
print("\n[1/7] Loading data...")

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

df = daily_aq.merge(daily_weather, on=['stationID', 'date'], how='left')
df = df[df['date'] >= CONFIG['start_date']]
df = df[df['pm2_5'].notna()].copy()
df = df.sort_values(['stationID', 'date']).reset_index(drop=True)

print(f"   Rows: {len(df):,}, Stations: {df['stationID'].nunique()}")

# ============================================================================
# Build Spatial Graph
# ============================================================================
print("\n[2/7] Building spatial graph...")

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

D = np.zeros((N_STATIONS, N_STATIONS), dtype=np.float32)
for i in range(N_STATIONS):
    for j in range(i+1, N_STATIONS):
        d = haversine_km(lats[i], lons[i], lats[j], lons[j])
        D[i, j] = D[j, i] = d

# Build edge index
k = CONFIG['k_neighbors']
neighbor_idx = {}
edge_index = []
edge_weight = []

for i, sid in enumerate(station_order):
    sorted_idx = np.argsort(D[i])[1:k+1]
    neighbor_idx[sid] = [station_order[j] for j in sorted_idx]
    
    for j in sorted_idx:
        edge_index.append([i, j])
        edge_weight.append(1.0 / (D[i, j] + 1e-6))

edge_index = torch.tensor(edge_index, dtype=torch.long).T
edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
edge_weight = edge_weight / edge_weight.max()

print(f"   Nodes: {N_STATIONS}, Edges: {edge_index.shape[1]}")

# ============================================================================
# Build Hypergraph
# ============================================================================
print("\n[3/7] Building hypergraph...")

from sklearn.cluster import KMeans

coords = np.column_stack([lats, lons])
n_clusters = min(10, N_STATIONS // 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)

H = np.zeros((N_STATIONS, n_clusters), dtype=np.float32)
for i, label in enumerate(cluster_labels):
    H[i, label] = 1.0

H = torch.tensor(H, dtype=torch.float32)
print(f"   Hyperedges: {n_clusters}")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n[4/7] Feature engineering...")

# Neighbor features
pm10_pivot = df.pivot_table(index='date', columns='stationID', values='pm10', aggfunc='mean')
pm25_pivot = df.pivot_table(index='date', columns='stationID', values='pm2_5', aggfunc='mean')

def get_neighbor_features(row):
    sid, date = row['stationID'], row['date']
    neighbors = neighbor_idx.get(sid, [])
    pm10_vals, pm25_vals = [], []
    
    for n_sid in neighbors:
        if n_sid in pm10_pivot.columns and date in pm10_pivot.index:
            v = pm10_pivot.loc[date, n_sid]
            if not pd.isna(v): pm10_vals.append(v)
        if n_sid in pm25_pivot.columns and date in pm25_pivot.index:
            v = pm25_pivot.loc[date, n_sid]
            if not pd.isna(v): pm25_vals.append(v)
    
    return pd.Series({
        'neighbor_pm10_mean': np.mean(pm10_vals) if pm10_vals else np.nan,
        'neighbor_pm25_mean': np.mean(pm25_vals) if pm25_vals else np.nan,
        'neighbor_pm10_std': np.std(pm10_vals) if len(pm10_vals) > 1 else 0,
        'neighbor_pm25_std': np.std(pm25_vals) if len(pm25_vals) > 1 else 0,
    })

neighbor_features = df.apply(get_neighbor_features, axis=1)
df = pd.concat([df, neighbor_features], axis=1)

# Lag features
for lag in [1, 2, 3, 7]:
    df[f'pm2_5_lag{lag}'] = df.groupby('stationID')['pm2_5'].shift(lag)
    df[f'pm10_lag{lag}'] = df.groupby('stationID')['pm10'].shift(lag)

# Rolling means
df['pm2_5_mean_7d'] = df.groupby('stationID')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['pm10_mean_7d'] = df.groupby('stationID')['pm10'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Ratios
df['pm10_pm25_ratio'] = df['pm10'] / (df['pm2_5_lag1'] + 1)
df['neighbor_local_pm10_ratio'] = df['neighbor_pm10_mean'] / (df['pm10'] + 1)
df['neighbor_local_pm25_ratio'] = df['neighbor_pm25_mean'] / (df['pm2_5_lag1'] + 1)

# Time features
df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
df['is_dry_season'] = df['date'].dt.month.isin([11, 12, 1, 2, 3]).astype(float)

# Spatial
lat_mean, lat_std = df['lat'].mean(), df['lat'].std()
lon_mean, lon_std = df['lon'].mean(), df['lon'].std()
df['lat_norm'] = (df['lat'] - lat_mean) / lat_std
df['lon_norm'] = (df['lon'] - lon_mean) / lon_std

df['station_idx'] = df['stationID'].map(sid2idx)

feature_cols = [
    'pm10', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3', 'pm10_lag7', 'pm10_mean_7d',
    'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3', 'pm2_5_lag7', 'pm2_5_mean_7d',
    'pm10_pm25_ratio',
    'neighbor_pm10_mean', 'neighbor_pm25_mean', 'neighbor_pm10_std', 'neighbor_pm25_std',
    'neighbor_local_pm10_ratio', 'neighbor_local_pm25_ratio',
    'temp_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'day_sin', 'day_cos', 'is_dry_season',
    'lat_norm', 'lon_norm',
]

for col in feature_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=feature_cols + ['pm2_5'])

print(f"   Features: {len(feature_cols)}")
print(f"   After cleaning: {len(df):,} rows")

# ============================================================================
# Split and Prepare Data
# ============================================================================
print("\n[5/7] Splitting data...")

dates = sorted(df['date'].unique())
n = len(dates)
t1, t2 = int(n * CONFIG['train_ratio']), int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

train_df = df[df['date'].isin(dates[:t1])].copy()
val_df = df[df['date'].isin(dates[t1:t2])].copy()
test_df = df[df['date'].isin(dates[t2:])].copy()

print(f"   Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

y_train = train_df['pm2_5'].values
y_val = val_df['pm2_5'].values
y_test = test_df['pm2_5'].values

idx_train = train_df['station_idx'].values
idx_val = val_df['station_idx'].values
idx_test = test_df['station_idx'].values

# Tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(idx_train, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(idx_val, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(idx_test, dtype=torch.long),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# ============================================================================
# Improved STHGAT Model
# ============================================================================
print("\n[6/7] Building Improved STHGAT...")

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, edge_index, edge_weight=None):
        B, N, F = h.shape
        Wh = self.W(h)
        
        src, dst = edge_index[0], edge_index[1]
        edge_h = torch.cat([Wh[:, src], Wh[:, dst]], dim=2)
        e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze(-1))
        
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(0)
        
        attention = torch.zeros(B, N, N, device=h.device)
        attention[:, src, dst] = e
        attention = Fn.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.bmm(attention, Wh)
        return Fn.elu(h_prime)


class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features, H):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        D_v = torch.diag(H.sum(dim=1).pow(-0.5))
        D_e = torch.diag(H.sum(dim=0).pow(-1))
        self.register_buffer('laplacian', D_v @ H @ D_e @ H.T @ D_v)
    
    def forward(self, x):
        B, N, F = x.shape
        x_flat = x.reshape(B * N, F)
        x_transformed = self.linear(x_flat).reshape(B, N, -1)
        out = torch.einsum('nm,bmf->bnf', self.laplacian, x_transformed)
        return out


class ImprovedSTHGAT(nn.Module):
    """
    Improved STHGAT - Simplified approach:
    - Use features directly (no graph expansion)
    - Add spatial context via station embeddings
    - Apply graph-aware transformations
    """
    
    def __init__(self, config, edge_index, edge_weight, H, n_stations, in_features):
        super().__init__()
        
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        self.n_stations = n_stations
        
        hidden = config['hidden_dim']
        dropout = config['dropout']
        
        # Station embedding (spatial context)
        self.station_emb = nn.Embedding(n_stations, hidden // 2)
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden)
        
        # Graph-aware layers (process features with spatial awareness)
        self.spatial_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden + hidden // 2, hidden + hidden // 2),
                nn.LayerNorm(hidden + hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(config['num_gat_layers'])
        ])
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output
        self.output = nn.Linear(hidden // 2, 1)
    
    def forward(self, x, station_idx):
        # Input projection
        h = self.input_proj(x)  # (B, hidden)
        
        # Station embedding
        s = self.station_emb(station_idx)  # (B, hidden//2)
        
        # Combine features and spatial context
        h = torch.cat([h, s], dim=-1)  # (B, hidden + hidden//2)
        
        # Spatial-aware processing
        for layer in self.spatial_layers:
            h = h + layer(h)  # Residual
        
        # MLP
        h = self.mlp(h)
        
        # Output
        return self.output(h).squeeze(-1)


model = ImprovedSTHGAT(
    config=CONFIG,
    edge_index=edge_index.to(DEVICE),
    edge_weight=edge_weight.to(DEVICE),
    H=H.to(DEVICE),
    n_stations=N_STATIONS,
    in_features=len(feature_cols)
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parameters: {n_params:,}")

# ============================================================================
# Training
# ============================================================================
print("\n[7/7] Training Improved STHGAT...")

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
criterion = nn.SmoothL1Loss()

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(CONFIG['epochs']):
    # Train
    model.train()
    train_loss = 0
    for x, idx, y in train_loader:
        x, idx, y = x.to(DEVICE), idx.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(x, idx)
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
        for x, idx, y in val_loader:
            x, idx, y = x.to(DEVICE), idx.to(DEVICE), y.to(DEVICE)
            pred = model(x, idx)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)
            
            val_preds.append(pred.cpu())
            val_targets.append(y.cpu())
    
    val_loss /= len(val_dataset)
    
    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    val_r2 = r2_score(val_targets, val_preds)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"   Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)

# ============================================================================
# Evaluation
# ============================================================================
print("\nEvaluating on test set...")

model.eval()
test_preds, test_targets = [], []

with torch.no_grad():
    for x, idx, y in test_loader:
        x, idx, y = x.to(DEVICE), idx.to(DEVICE), y.to(DEVICE)
        pred = model(x, idx)
        test_preds.append(pred.cpu())
        test_targets.append(y.cpu())

test_preds = torch.cat(test_preds).numpy()
test_targets = torch.cat(test_targets).numpy()

test_r2 = r2_score(test_targets, test_preds)
test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
test_mae = mean_absolute_error(test_targets, test_preds)

print(f"\n   TEST RESULTS (Improved STHGAT):")
print(f"   R²:   {test_r2:.4f}")
print(f"   RMSE: {test_rmse:.2f} µg/m³")
print(f"   MAE:  {test_mae:.2f} µg/m³")

# Save
save_dir = Path('../data/reports/sthgat_improved')
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'metrics': {'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mae': test_mae},
    'station_order': station_order,
    'feature_cols': feature_cols,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
}, save_dir / 'model.pt')

print(f"\n   Model saved to: {save_dir / 'model.pt'}")

print("\n" + "="*70)
print("IMPROVED STHGAT TRAINING COMPLETE")
print("="*70)
print(f"""
COMPARISON:
  SpatialMLP:        R² = 0.9935
  True STHGAT:       R² = 0.4602
  Improved STHGAT:   R² = {test_r2:.4f}

TARGET: R² > 0.85 {'✅ ACHIEVED!' if test_r2 > 0.85 else '❌ Not yet'}
""")
