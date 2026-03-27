# 📊 MLflow Tracking Setup for STC-HGAT Training

## ✅ MLflow UI Status

**MLflow UI is already running!**
- **URL**: http://127.0.0.1:5000
- **Status**: ✅ Active (Process 109264)
- **Backend**: file:./mlruns

---

## 🚀 How to Add MLflow Tracking to Notebook

### **Step 1: Add MLflow Setup Cell (After Cell 4 - Imports)**

```python
# ============================================================================
# MLflow Tracking Setup
# ============================================================================

import mlflow
import mlflow.pytorch

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
experiment_name = "STC-HGAT-PM25-Forecasting"
mlflow.set_experiment(experiment_name)

print("✅ MLflow tracking configured")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experiment: {experiment_name}")
print(f"   View UI at: http://127.0.0.1:5000")
```

---

### **Step 2: Start MLflow Run (Before Cell 24 - Training Loop)**

Add this new cell **before** the training loop:

```python
# ============================================================================
# Start MLflow Run
# ============================================================================

# Start MLflow run
run_name = f"stchgat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=run_name)

try:
    # Log parameters
    mlflow.log_params({
        # Model parameters
        "hidden_dim": params['model']['hidden_dim'],
        "num_hypergat_layers": params['model']['num_hypergat_layers'],
        "num_hgat_layers": params['model']['num_hgat_layers'],
        "num_heads": params['model']['num_heads'],
        "dropout": params['model']['dropout'],
        
        # Training parameters
        "batch_size": params['training']['batch_size'],
        "learning_rate": params['training']['learning_rate'],
        "epochs": params['training']['epochs'],
        "weight_decay": params['training']['weight_decay'],
        "gradient_clip": params['training']['gradient_clip_value'],
        
        # Data parameters
        "sequence_length": params['data']['sequence_length'],
        "forecast_horizons": str(params['data']['forecast_horizons']),
        "num_stations": num_stations,
        "num_features": num_features,
        
        # Graph parameters
        "spatial_threshold_km": params['graph']['spatial_threshold_km'],
        "num_regions": params['graph']['num_regions'],
    })
    
    # Log tags
    mlflow.set_tags({
        "model_type": "STC-HGAT",
        "task": "PM2.5 Forecasting",
        "device": str(device),
        "data_type": "mock"
    })
    
    print(f"✅ MLflow run started: {run_name}")
    print(f"   View at: http://127.0.0.1:5000")
    
except Exception as e:
    print(f"⚠️  MLflow logging error: {e}")
    print("   Training will continue without MLflow tracking")
```

---

### **Step 3: Modify Training Loop (Cell 24)**

**Replace the training loop with this version that includes MLflow logging:**

```python
print("🚀 Starting Training:")
print("="*50)

num_epochs = params['training']['epochs']
early_stopping_patience = params['training']['early_stopping_patience']
gradient_clip = params['training']['gradient_clip_value']

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X, spatial_graph_device, temporal_graph_device)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X, spatial_graph_device, temporal_graph_device)
            loss = criterion(output, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(current_lr)
    
    # ========== MLflow Logging ==========
    try:
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }, step=epoch)
    except:
        pass  # Continue training even if MLflow fails
    # ====================================
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ⭐ (Best)")
        
        # Log best model checkpoint
        try:
            mlflow.log_metrics({
                "best_val_loss": best_val_loss,
                "best_epoch": epoch + 1
            }, step=epoch)
        except:
            pass
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if patience_counter >= early_stopping_patience:
        print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
        break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"\n✅ Loaded best model (val_loss: {best_val_loss:.4f})")

print("="*50)
print("Training completed!")
```

---

### **Step 4: Log Final Results (After Cell 28 - Test Evaluation)**

Add this new cell **after** test evaluation:

```python
# ============================================================================
# Log Final Results to MLflow
# ============================================================================

try:
    # Log test metrics for each horizon
    print("\n📊 Logging test metrics to MLflow...")
    
    for i, horizon in enumerate(horizons):
        pred_h = test_predictions[:, :, i]
        target_h = test_targets[:, :, i]
        
        mae = calculate_mae(target_h, pred_h)
        rmse = calculate_rmse(target_h, pred_h)
        r2 = calculate_r2(target_h, pred_h)
        
        mlflow.log_metrics({
            f"test_mae_horizon_{horizon}d": mae,
            f"test_rmse_horizon_{horizon}d": rmse,
            f"test_r2_horizon_{horizon}d": r2,
        })
        
        print(f"   Horizon +{horizon}d: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # Log training curves as artifact
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['learning_rate'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close()
    
    # Log model
    print("\n💾 Logging model to MLflow...")
    mlflow.pytorch.log_model(
        model, 
        "model",
        registered_model_name="STC-HGAT-PM25"
    )
    
    print("\n✅ MLflow logging completed!")
    print(f"   View results at: http://127.0.0.1:5000")
    
except Exception as e:
    print(f"\n⚠️  MLflow logging error: {e}")
    print("   Results saved locally but not logged to MLflow")

finally:
    # End MLflow run
    mlflow.end_run()
    print("   MLflow run ended")
```

---

## 📋 Quick Summary

### **Cells to Add:**

1. **After Cell 4 (Imports)**: MLflow setup
2. **Before Cell 24 (Training)**: Start MLflow run
3. **Modify Cell 24**: Add MLflow logging in training loop
4. **After Cell 28 (Evaluation)**: Log final results

### **What Gets Tracked:**

✅ **Parameters**: Model config, training config, data config  
✅ **Metrics**: Train/val loss, learning rate, test metrics  
✅ **Artifacts**: Training curves, model checkpoint  
✅ **Tags**: Model type, task, device, data type  

---

## 🎯 How to Use

1. **Add the cells** to your notebook as described above
2. **Restart kernel** to ensure fresh imports
3. **Run all cells** including the new MLflow cells
4. **During training**, metrics will be logged automatically
5. **After training**, open http://127.0.0.1:5000 to view results

---

## 🔍 Verify MLflow Tracking

After training completes:

1. Open http://127.0.0.1:5000 in browser
2. Click on experiment "STC-HGAT-PM25-Forecasting"
3. You should see your run with:
   - All parameters logged
   - Training metrics over time
   - Test metrics for each horizon
   - Training curves visualization
   - Saved model

---

## 💡 Tips

- MLflow logging is wrapped in try-except to not interrupt training
- Metrics are logged every epoch
- Model is registered as "STC-HGAT-PM25" for easy retrieval
- Training curves are saved as PNG artifacts
- All runs are timestamped for easy identification

---

**MLflow UI**: http://127.0.0.1:5000 ✅ **Already Running!**
