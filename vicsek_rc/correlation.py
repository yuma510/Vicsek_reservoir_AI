"""Adjacency-network correlation decay analysis."""
import numpy as np


def correlation_at_lag(matrices: list[np.ndarray], dt: int) -> float:
    """
    Mean normalised correlation at time lag dt.

      C(dt) = mean_t [ sum(A(t) * A(t+dt)) / sum(A(t)) ]

    Frames where A(t) is all-zero are skipped.
    """
    if dt == 0:
        return 1.0
    values = []
    for t in range(len(matrices) - dt):
        A_t    = matrices[t].astype(np.float32)
        A_tdt  = matrices[t + dt].astype(np.float32)
        n_edges = A_t.sum()
        if n_edges > 0:
            values.append((A_t * A_tdt).sum() / n_edges)
    return float(np.mean(values)) if values else 0.0


def compute_correlation_decay(matrices: list[np.ndarray],
                              dt_max: int) -> tuple[list[int], list[float]]:
    """Return (dt_list, correlation_list) for dt = 0 .. dt_max."""
    dts   = list(range(dt_max + 1))
    corrs = [correlation_at_lag(matrices, dt) for dt in dts]
    return dts, corrs
