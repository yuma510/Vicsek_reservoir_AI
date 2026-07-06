"""Adjacency-network correlation decay analysis."""
import numpy as np
import pandas as pd


def correlation_at_lag(matrices: list[np.ndarray], dt: int,
                       n_base: int | None = None) -> float:
    """
    Mean normalised correlation at time lag dt.

      C(dt) = mean_t [ sum(A(t) * A(t+dt)) / sum(A(t)) ]

    n_base: number of base frames to use (for equalising sample count across lags).
    If None, uses len(matrices) - dt. Frames where A(t) is all-zero are skipped.
    """
    if dt == 0:
        return 1.0
    n = n_base if n_base is not None else len(matrices) - dt
    values = []
    for t in range(n):
        A_t   = matrices[t].astype(np.float32)
        A_tdt = matrices[t + dt].astype(np.float32)
        n_edges = A_t.sum()
        if n_edges > 0:
            values.append((A_t * A_tdt).sum() / n_edges)
    return float(np.mean(values)) if values else 0.0


def compute_correlation_decay(matrices: list[np.ndarray],
                              dt_max: int) -> tuple[list[int], list[float]]:
    """Return (dt_list, correlation_list) for dt = 0 .. dt_max.

    Uses the same n_base = T - dt_max frames for all lags to equalise sample count.
    """
    n_base = len(matrices) - dt_max
    dts   = list(range(dt_max + 1))
    corrs = [correlation_at_lag(matrices, dt, n_base) for dt in dts]
    return dts, corrs


def load_position_dat(path, N: int,
                      frame_start: int = 0,
                      frame_end: int | None = None) -> np.ndarray:
    """Load a frame range from position.dat.

    Returns ndarray shape (T, N, 3) where T = frame_end - frame_start.
    Uses pandas C engine (same as load_position_fast) for speed.
    """
    skip = frame_start * N
    nrows = None if frame_end is None else (frame_end - frame_start) * N
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     skiprows=skip if skip > 0 else None,
                     nrows=nrows, dtype=np.float64, engine="c")
    data = df.values
    T = data.shape[0] // N
    return data.reshape(T, N, 3)


def compute_adjacency_from_positions(pos_frame: np.ndarray,
                                     rcut: float,
                                     boxsize: float) -> np.ndarray:
    """Compute N×N bool adjacency matrix from one frame of positions.

    pos_frame: shape (N, 3) — columns are x, y, theta.
    Returns bool array shape (N, N) with diagonal False.
    Applies periodic boundary conditions.
    """
    xy = pos_frame[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]   # (N, N, 2)
    diff -= np.round(diff / boxsize) * boxsize
    r2 = (diff ** 2).sum(axis=-1)
    A = r2 < rcut ** 2
    np.fill_diagonal(A, False)
    return A
