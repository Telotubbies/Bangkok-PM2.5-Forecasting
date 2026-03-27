"""
MLflow Tracking Code for STC-HGAT Training Notebook

Add these cells to your notebook to enable MLflow experiment tracking.
"""

# ============================================================================
# Cell: Setup MLflow Tracking (Add after imports)
# ============================================================================

import mlflow
import mlflow.pytorch

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
experiment_name = "STC-HGAT-PM25-Forecasting"
mlflow.set_experiment(experiment_name)

print(f"✅ MLflow tracking configured")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experiment: {experiment_name}")
print(f"   View UI at: http://127.0.0.1:5000")


# ============================================================================
# Cell: Start MLflow Run (Add before training loop)
# ============================================================================

# Start MLflow run
run_name = f"stchgat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=run_name)

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
    "data_type": "mock" if "mock" in str(data) else "real"
})

print(f"✅ MLflow run started: {run_name}")


# ============================================================================
# Cell: Log Metrics During Training (Modify training loop)
# ============================================================================

# Inside training loop, after calculating losses:
for epoch in range(num_epochs):
    # ... training code ...
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr,
    }, step=epoch)
    
    # If you have additional metrics
    if epoch % 5 == 0:  # Log less frequently to reduce overhead
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, step=epoch)


# ============================================================================
# Cell: Log Final Results and Model (Add after training)
# ============================================================================

# Log final metrics
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

# Log model
mlflow.pytorch.log_model(
    model, 
    "model",
    registered_model_name="STC-HGAT-PM25"
)

# Log training curves as artifacts
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[1].plot(history['learning_rate'])
axes[1].set_title('Learning Rate')
plt.tight_layout()
mlflow.log_figure(fig, "training_curves.png")
plt.close()

# End MLflow run
mlflow.end_run()

print("✅ MLflow run completed!")
print(f"   View results at: http://127.0.0.1:5000")
