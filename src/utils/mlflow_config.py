"""
MLflow configuration and utilities for experiment tracking.
"""
import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional


def setup_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "stc-hgat-pm25-forecasting",
    artifact_location: Optional[str] = None
) -> None:
    """
    Setup MLflow tracking configuration.
    
    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI (default: local file system)
    experiment_name : str
        Name of the experiment
    artifact_location : str, optional
        Location to store artifacts
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            if artifact_location:
                mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Could not create experiment: {e}")
    
    mlflow.set_experiment(experiment_name)


def log_params_from_dict(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log nested dictionary of parameters to MLflow.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters (can be nested)
    prefix : str
        Prefix for parameter names
    """
    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            log_params_from_dict(value, prefix=param_name)
        else:
            try:
                mlflow.log_param(param_name, value)
            except Exception as e:
                print(f"Warning: Could not log param {param_name}: {e}")


def log_metrics_from_dict(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log dictionary of metrics to MLflow.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value
    step : int, optional
        Step number for the metrics
    """
    for key, value in metrics.items():
        try:
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Could not log metric {key}: {e}")


def save_model_with_mlflow(
    model,
    model_name: str = "stc_hgat_model",
    save_path: Optional[Path] = None
) -> None:
    """
    Save PyTorch model with MLflow.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    model_name : str
        Name for the model
    save_path : Path, optional
        Local path to also save the model
    """
    try:
        mlflow.pytorch.log_model(model, model_name)
        
        if save_path:
            import torch
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(str(save_path))
            
    except Exception as e:
        print(f"Warning: Could not save model: {e}")


def load_model_from_mlflow(run_id: str, model_name: str = "stc_hgat_model"):
    """
    Load PyTorch model from MLflow.
    
    Parameters
    ----------
    run_id : str
        MLflow run ID
    model_name : str
        Name of the model artifact
        
    Returns
    -------
    model : torch.nn.Module
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{model_name}"
    return mlflow.pytorch.load_model(model_uri)
