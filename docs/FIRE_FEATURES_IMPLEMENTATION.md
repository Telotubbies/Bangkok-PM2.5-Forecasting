# Fire Features Implementation Summary

**Date:** March 28, 2026  
**Status:** ✅ Implementation Complete, Training In Progress

---

## 🎯 Objective

Integrate NASA FIRMS fire/hotspot data into STC-HGAT model to improve PM2.5 forecasting accuracy, especially for 3-day and 7-day horizons during peak fire season (March-April).

---

## 📊 Implementation Overview

### What Was Added:

**6 New Fire Features:**
1. `fire_count_500km` - Number of fires within 500km radius
2. `fire_frp_total` - Total Fire Radiative Power (MW)
3. `upwind_fire_impact` - Wind-direction weighted fire impact (hybrid: angle + distance)
4. `fire_distance_weighted` - Distance-weighted fire intensity (inverse square law)
5. `fire_lag1d` - 1-day lagged upwind fire impact (short-range transport)
6. `fire_lag3d` - 3-day lagged upwind fire impact (long-range transport)

**Total Features:** 18 → **24 features**

---

## 🔥 Fire Data Analysis (March 2024 - Peak Season)

### Data Statistics:
- **Total fires worldwide:** 2.87M
- **Southeast Asia:** 622K fires
- **Within 500km of Bangkok:** 12,512 fires
- **Daily average:** 403 fires/day
- **Mean FRP:** 6.36 MW
- **Max FRP:** 668.22 MW

### Distance Distribution:
- Within 100km: 1,944 fires
- Within 200km: 9,202 fires
- Within 500km: 77,996 fires (full year)

### Confidence Levels:
- Nominal (n): 85.2%
- Low (l): 11.9%
- High (h): 2.9%

---

## 🏗️ Implementation Details

### 1. Fire Feature Loader (`src/data/fire_feature_loader.py`)

**Key Functions:**

```python
# Distance calculation
haversine_distance(lon1, lat1, lon2, lat2) -> float

# Bearing calculation (for wind direction)
calculate_bearing(lon1, lat1, lon2, lat2) -> float

# Distance weighting (inverse square law)
distance_weight(distance_km, min_distance=10.0) -> float
# Formula: 1.0 / max(distance_km, 10.0)²

# Wind angle weighting (upwind fires only)
wind_angle_weight(wind_direction, fire_bearing) -> float
# Formula: cos(angle_diff) if angle_diff ≤ 90°, else 0

# Adaptive lag based on distance
get_adaptive_lag_days(distance_km, max_lag=3) -> int
# Assumes smoke travels ~200km/day

# Main loader
load_fire_data(data_dir, stations_df, start_date, end_date, 
               radius_km=500.0, min_frp=5.0) -> (DataFrame, dict)

# Feature computation
compute_fire_features(fire_df, stations_df, weather_data, 
                     timestamps) -> (Tensor, list)
```

**Filters Applied:**
- Confidence: 'n' (nominal) and 'h' (high) only
- FRP threshold: ≥ 5.0 MW
- Distance: ≤ 500km from Bangkok center

---

### 2. Data Loader Integration (`src/data/real_data_loader.py`)

**Modified Function:**
```python
def combine_features(
    pm25_data: Tensor,
    weather_data: Optional[Tensor] = None,
    fire_data: Optional[Tensor] = None,  # NEW
    add_temporal_features: bool = True,
    metadata: Optional[pd.DataFrame] = None
) -> Tuple[Tensor, List[str]]:
```

**Feature Stacking:**
1. PM2.5 features (6)
2. Weather features (6)
3. **Fire features (6)** ← NEW
4. Temporal features (6)

**Total:** 24 features

---

### 3. Training Script (`scripts/train_with_fire.py`)

**New Arguments:**
- `--use-fire` - Enable fire features (default: True)
- `--fire-radius` - Fire search radius in km (default: 500.0)

**Workflow:**
1. Load PM2.5 data
2. Load weather data
3. **Load fire data** ← NEW
4. **Compute fire features** ← NEW
5. Combine all features
6. Train model
7. Evaluate and save

**Model Naming:**
- With fire: `stc_hgat_improved_fire_YYYYMMDD_HHMMSS.pt`
- Without fire: `stc_hgat_improved_nofire_YYYYMMDD_HHMMSS.pt`

---

## 🧮 Feature Engineering Formulas

### 1. Distance Weight
```
weight = 1.0 / max(distance_km, 10.0)²
```
- Inverse square law for smoke dispersion
- Minimum distance threshold prevents division by zero

### 2. Wind Angle Weight
```
angle_diff = |wind_direction - fire_bearing|
if angle_diff > 180: angle_diff = 360 - angle_diff

if angle_diff ≤ 90°:
    weight = cos(angle_diff)
else:
    weight = 0  # Downwind fires ignored
```
- Only upwind fires contribute (angle ≤ 90°)
- Maximum weight when fire is directly upwind (angle = 0°)

### 3. Upwind Fire Impact
```
impact = Σ(FRP × wind_weight × distance_weight)
```
- Combines fire intensity, wind direction, and distance
- Hybrid approach: angle weighting + distance decay

### 4. Adaptive Lag
```
lag_days = min(int(distance_km / 200), max_lag)
```
- Assumes smoke travels ~200km/day
- Closer fires = shorter lag
- Distant fires = longer lag (up to max_lag)

---

## 📈 Expected Improvements

### Baseline (Without Fire Features):
| Horizon | R² | MAE | RMSE |
|---------|-----|-----|------|
| +1 day | 0.9146 | 0.2398 | 0.3560 |
| +3 days | 0.8025 | 0.3567 | 0.5412 |
| +7 days | 0.4605 | 0.5937 | 0.8937 |

### Target (With Fire Features):
| Horizon | Target R² | Expected Gain |
|---------|-----------|---------------|
| +1 day | 0.92+ | Marginal (+1%) |
| +3 days | **0.85+** | **+6%** |
| +7 days | **0.60+** | **+30%** |

**Rationale:**
- Fire features most impactful for longer horizons
- Peak fire season (March-April) should show largest gains
- Upwind fire detection captures regional transport patterns

---

## 🔍 Alignment with STC-HGAT Paper

### Paper Architecture (Preserved):
1. ✅ HyperGAT - Spatial hypergraph attention
2. ✅ HGAT - Temporal heterogeneous graph
3. ✅ Contrastive Learning - InfoNCE
4. ✅ Position Encoding - Reversed + Soft Attention
5. ✅ Region nodes - 5 geographic regions

### Enhancements (Additive):
1. ⚡ Gated Fusion (Phase 3)
2. ⚡ Cross-Attention (Phase 3)
3. ⚡ Multi-Scale Temporal (Phase 3)
4. 🔥 **Fire Features (Phase 4)** ← NEW

**Key:** Fire features are input enhancements that don't break core architecture

---

## 📁 Files Created/Modified

### New Files:
1. ✅ `src/data/fire_feature_loader.py` - Fire feature engineering (450 lines)
2. ✅ `scripts/train_with_fire.py` - Training script with fire features (400 lines)
3. ✅ `docs/FIRE_FEATURES_IMPLEMENTATION.md` - This documentation

### Modified Files:
1. ✅ `src/data/real_data_loader.py` - Added fire_data parameter to combine_features()

---

## 🚀 Training Status

### Current Run:
- **Script:** `train_with_fire.py`
- **Status:** 🔄 In Progress
- **Features:** 24 (18 baseline + 6 fire)
- **Data:** 2024-01-01 to 2024-11-30
- **Fire radius:** 500km
- **Epochs:** 50 (with early stopping)
- **Batch size:** 32

### Process Info:
- **PID:** 153512
- **CPU:** 170% (multi-threaded)
- **Memory:** 7.5% (~2.4GB)
- **Log:** `training_with_fire.log`

---

## 🧪 Testing & Validation

### Fire Data Loading Test:
```bash
✅ Test successful!
Loaded 12,512 fires (March 2024)
Date range: 2024-03-01 to 2024-03-31
```

### Feature Computation:
- Distance calculations: Haversine formula
- Wind weighting: Cosine angle weighting
- Temporal lag: Adaptive based on distance
- All functions tested and working

---

## 📊 Next Steps

1. ✅ **Implementation Complete**
   - Fire feature loader created
   - Data loader integration done
   - Training script ready

2. 🔄 **Training In Progress**
   - Model training with fire features
   - Expected completion: ~30-40 minutes

3. ⏳ **Pending Evaluation**
   - Compare R² scores (with vs without fire)
   - Analyze feature importance
   - Test on peak fire season (March-April)
   - Per-station analysis with fire features

4. ⏳ **Future Enhancements**
   - Fire-weighted hypergraph construction
   - Satellite-specific fire filtering
   - Real-time fire data integration

---

## 💡 Key Insights

### Why Fire Features Matter:
1. **Seasonal Impact:** March-April peak fire season significantly affects PM2.5
2. **Regional Transport:** Upwind fires from Myanmar, Laos impact Bangkok
3. **Temporal Lag:** Smoke takes 1-3 days to reach Bangkok from distant fires
4. **Distance Decay:** Closer fires have exponentially higher impact

### Technical Innovations:
1. **Hybrid Weighting:** Combines angle and distance weighting
2. **Adaptive Lag:** Distance-based temporal lag (not fixed)
3. **Upwind Detection:** Only considers fires upwind based on wind direction
4. **Confidence Filtering:** Uses only nominal and high confidence detections

---

## 🎯 Success Criteria

- ✅ Fire features successfully integrated (24 total features)
- ✅ Fire data loader working correctly
- ✅ Model accepts 24-feature input
- 🔄 Model training without errors (in progress)
- ⏳ R² improvement on 3-day and 7-day forecasts
- ⏳ Peak fire season shows largest gains
- ✅ Core STC-HGAT architecture preserved

---

## 📚 References

1. **NASA FIRMS:** https://firms.modaps.eosdis.nasa.gov/
2. **STC-HGAT Paper:** Yang & Peng, 2024, Mathematics 12(8), 1193
3. **Fire Data:** MODIS and VIIRS satellites
4. **Smoke Transport:** ~200km/day assumption based on atmospheric studies

---

**Implementation completed by:** Cascade AI  
**Date:** March 28, 2026, 09:48 AM UTC+07:00  
**Status:** ✅ Ready for evaluation
