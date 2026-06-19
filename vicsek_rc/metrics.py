"""Reservoir performance metrics (NRMSE, MC_k)."""
import numpy as np


def nrmse(y_true, y_pred) -> float:
    """Normalized RMSE: sqrt(MSE / mean(y_true^2))."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse / np.mean(y_true ** 2)))


def nrmse2(y_true, y_pred) -> float:
    """Normalized MSE by variance: MSE / Var(y_true)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def corrcoef(a, b) -> float:
    """Pearson correlation with size guard (returns nan for empty input)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size != b.size:
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
    return float(np.corrcoef(a, b)[0, 1])


def mck_score(y_true, y_pred) -> float:
    """Memory Capacity at delay k: squared Pearson correlation."""
    return float(corrcoef(y_true, y_pred) ** 2)
