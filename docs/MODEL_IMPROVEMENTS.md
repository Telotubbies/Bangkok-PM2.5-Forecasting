# Model Improvement Plan - STC-HGAT for Bangkok PM2.5 Forecasting

## Current Model: STC-HGAT (Yang & Peng, 2024)
Based on: Mathematics 12(8), 1193

### Current Architecture
- **Spatial**: HyperGAT with geographic region nodes
- **Temporal**: HGAT with sequential + seasonal edges
- **Fusion**: Sum pooling (h = h_spatial + h_temporal)
- **Loss**: Adaptive Weight Loss + Contrastive Learning (InfoNCE)

---

## 🎯 Target: 90% Accuracy (R² > 0.9)

### Current Issues Identified
1. ❌ **Shape mismatch error** - Model output (B, 30) instead of (B, 79)
2. ❌ **Poor predictions** - Model outputs constant ~0.0
3. ⚠️ **Using mock data** - Not real PM2.5 measurements

---

## 📈 Improvement Strategy (Phased Approach)

### Phase 1: Fix Critical Bugs (URGENT)
**Priority: CRITICAL**

1. **Fix Shape Mismatch**
   - Debug output shows N and T might be swapped
   - Model expects (B, N, T, F) but receives (B, T, N, F)
   - Fix transpose in training loop or model forward
   
2. **Verify Model Architecture**
   - Ensure PositionalEncoding outputs correct shape
   - Check all intermediate tensor shapes
   - Add comprehensive debug logging

**Expected Improvement**: Model will train without errors

---

### Phase 2: Data Quality Enhancement
**Priority: HIGH**

1. **Replace Mock Data with Real Data**
   ```python
   # Current: Synthetic data
   # Target: Load from data/silver/pm25_hourly.parquet
   ```
   
2. **Feature Engineering**
   - Add meteorological features:
     - Temperature
     - Humidity
     - Wind speed/direction
     - Atmospheric pressure
   - Add temporal features:
     - Hour of day (cyclical encoding)
     - Day of week (cyclical encoding)
     - Month (cyclical encoding)
     - Holidays/weekends
   - Add spatial features:
     - Distance to traffic sources
     - Land use type
     - Population density

3. **Data Augmentation**
   - Time series augmentation (jittering, scaling)
   - Mixup for temporal sequences
   - Add noise for robustness

**Expected Improvement**: +10-15% accuracy

---

### Phase 3: Model Architecture Enhancements
**Priority: HIGH**

1. **Multi-Scale Temporal Modeling**
   ```python
   # Add multiple temporal resolutions
   - Hourly patterns (24h cycle)
   - Daily patterns (7-day cycle)
   - Weekly patterns (4-week cycle)
   - Seasonal patterns (yearly cycle)
   ```

2. **Attention Mechanism Improvements**
   - Replace simple attention with **Multi-Head Self-Attention**
   - Add **Cross-Attention** between spatial and temporal
   - Implement **Temporal Attention** with learnable position bias

3. **Graph Structure Enhancement**
   - **Dynamic Graph Learning**: Learn edge weights from data
   - **Multi-Relational Graphs**: Different edge types (proximity, similarity, causality)
   - **Hierarchical Graphs**: City → District → Station levels

4. **Advanced Fusion**
   ```python
   # Current: Simple sum pooling
   h_fused = h_spatial + h_temporal
   
   # Improved: Gated fusion
   gate = sigmoid(W_gate([h_spatial, h_temporal]))
   h_fused = gate * h_spatial + (1 - gate) * h_temporal
   
   # Or: Cross-modal attention fusion
   h_fused = CrossAttention(h_spatial, h_temporal)
   ```

**Expected Improvement**: +15-20% accuracy

---

### Phase 4: Training Strategy Optimization
**Priority: MEDIUM**

1. **Hyperparameter Tuning with Optuna**
   ```python
   # Tune:
   - Learning rate: [1e-5, 1e-2]
   - Hidden dimension: [64, 128, 256, 512]
   - Number of layers: [1, 2, 3, 4]
   - Dropout: [0.1, 0.2, 0.3, 0.5]
   - Batch size: [16, 32, 64, 128]
   - Attention heads: [2, 4, 8]
   ```

2. **Advanced Training Techniques**
   - **Curriculum Learning**: Start with easy predictions (1-day) → harder (7-day)
   - **Teacher Forcing** with scheduled sampling
   - **Gradient Accumulation** for larger effective batch size
   - **Mixed Precision Training** (FP16) for faster training
   - **Learning Rate Warmup** + Cosine Annealing

3. **Loss Function Improvements**
   ```python
   # Multi-task learning
   L_total = α*L_regression + β*L_contrastive + γ*L_classification
   
   # Where:
   - L_regression: Adaptive Weight MSE (current)
   - L_contrastive: InfoNCE (current)
   - L_classification: Classify PM2.5 levels (Good/Moderate/Unhealthy)
   ```

4. **Ensemble Methods**
   - Train multiple models with different seeds
   - Use different architectures (LSTM, Transformer, GCN)
   - Weighted averaging or stacking

**Expected Improvement**: +5-10% accuracy

---

### Phase 5: Advanced Techniques
**Priority: MEDIUM-LOW**

1. **Transformer-Based Temporal Modeling**
   - Replace HGAT with **Temporal Transformer**
   - Add **Relative Position Encoding**
   - Use **Causal Masking** for autoregressive prediction

2. **External Knowledge Integration**
   - **Weather Forecasts**: Use predicted weather as features
   - **Traffic Data**: Real-time traffic congestion
   - **Event Calendar**: Festivals, holidays, construction

3. **Uncertainty Quantification**
   - **Monte Carlo Dropout** for prediction intervals
   - **Bayesian Neural Networks**
   - **Quantile Regression** for confidence bounds

4. **Multi-Horizon Joint Training**
   ```python
   # Current: Train for horizon +1d only
   # Improved: Joint training for all horizons
   pred_1d, pred_3d, pred_7d = model(x)
   loss = w1*MSE(pred_1d, y_1d) + w3*MSE(pred_3d, y_3d) + w7*MSE(pred_7d, y_7d)
   ```

**Expected Improvement**: +5-10% accuracy

---

## 🔬 Experimental Setup

### Baseline Metrics (Current)
```
Using mock data:
- MAE: TBD (after fixing bugs)
- RMSE: TBD
- R²: TBD
```

### Target Metrics
```
Horizon +1d:
- MAE: < 5.0 μg/m³
- RMSE: < 8.0 μg/m³
- R²: > 0.90

Horizon +3d:
- MAE: < 8.0 μg/m³
- RMSE: < 12.0 μg/m³
- R²: > 0.85

Horizon +7d:
- MAE: < 12.0 μg/m³
- RMSE: < 18.0 μg/m³
- R²: > 0.75
```

---

## 📝 Implementation Roadmap

### Week 1: Critical Fixes
- [ ] Fix shape mismatch error
- [ ] Verify model trains successfully
- [ ] Establish baseline with mock data
- [ ] Load real PM2.5 data

### Week 2: Data & Features
- [ ] Feature engineering (meteorological, temporal, spatial)
- [ ] Data quality checks and cleaning
- [ ] Train with real data
- [ ] Evaluate baseline performance

### Week 3: Model Improvements
- [ ] Implement gated fusion
- [ ] Add multi-head attention
- [ ] Dynamic graph learning
- [ ] Multi-scale temporal modeling

### Week 4: Hyperparameter Tuning
- [ ] Setup Optuna optimization
- [ ] Run 100+ trials
- [ ] Analyze best configurations
- [ ] Train final model with best params

### Week 5: Advanced Techniques
- [ ] Implement ensemble methods
- [ ] Add uncertainty quantification
- [ ] Multi-horizon joint training
- [ ] External knowledge integration

### Week 6: Evaluation & Deployment
- [ ] Comprehensive evaluation on test set
- [ ] Error analysis and visualization
- [ ] Model interpretation (attention weights)
- [ ] Deploy to production

---

## 🎓 Key Papers to Reference

1. **Original Paper**: Yang & Peng (2024) - STC-HGAT
2. **Graph Neural Networks**: 
   - Kipf & Welling (2017) - GCN
   - Veličković et al. (2018) - GAT
3. **Temporal Modeling**:
   - Vaswani et al. (2017) - Transformer
   - Hochreiter & Schmidhuber (1997) - LSTM
4. **Contrastive Learning**:
   - Chen et al. (2020) - SimCLR
   - He et al. (2020) - MoCo
5. **PM2.5 Forecasting**:
   - Recent MDPI papers on deep learning for air quality

---

## 💡 Innovation Points

### What Makes Our Model Better:
1. **Bangkok-Specific**: Optimized for Bangkok's geography and climate
2. **Real-Time**: Can use real-time traffic and weather data
3. **Multi-Scale**: Captures hourly, daily, and seasonal patterns
4. **Uncertainty**: Provides confidence intervals
5. **Interpretable**: Attention weights show important stations/times

### Potential Publications:
- "Enhanced STC-HGAT for Bangkok PM2.5 Forecasting"
- "Multi-Scale Spatio-Temporal Graph Networks for Air Quality Prediction"
- "Uncertainty-Aware PM2.5 Forecasting with Contrastive Learning"

---

## 📊 Expected Timeline

- **Phase 1 (Week 1)**: Fix bugs → Model trains successfully
- **Phase 2 (Week 2)**: Real data → Baseline R² ~ 0.60-0.70
- **Phase 3 (Week 3)**: Architecture improvements → R² ~ 0.75-0.85
- **Phase 4 (Week 4)**: Hyperparameter tuning → R² ~ 0.85-0.90
- **Phase 5 (Week 5-6)**: Advanced techniques → R² > 0.90 ✅

**Total Time**: 6 weeks to reach 90% accuracy target
