"""
evaluator.py
------------
Regression metrics for PM2.5 forecasting evaluation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict


def _validate(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Empty arrays")
    return y_true.ravel(), y_pred.ravel()


def compute_rmse(y_true, y_pred) -> float:
    yt, yp = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    yt, yp = _validate(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def compute_r2(y_true, y_pred) -> float:
    yt, yp = _validate(y_true, y_pred)
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_r / (ss_t + 1e-8))


def compute_smape(y_true, y_pred) -> float:
    yt, yp = _validate(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp) + 1e-8
    return float(np.mean(2 * np.abs(yt - yp) / denom) * 100)


def compute_mbe(y_true, y_pred) -> float:
    yt, yp = _validate(y_true, y_pred)
    return float(np.mean(yp - yt))


def evaluate_all(y_true, y_pred) -> Dict[str, float]:
    mask = np.isfinite(np.asarray(y_true)) & np.isfinite(np.asarray(y_pred))
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    return {
        "RMSE":  compute_rmse(yt, yp),
        "MAE":   compute_mae(yt, yp),
        "R2":    compute_r2(yt, yp),
        "SMAPE": compute_smape(yt, yp),
        "MBE":   compute_mbe(yt, yp),
    }


# Aliases for notebook compatibility
calculate_mae = compute_mae
calculate_rmse = compute_rmse
calculate_r2 = compute_r2
calculate_smape = compute_smape
calculate_mbe = compute_mbe
