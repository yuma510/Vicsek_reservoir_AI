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

    FFT-based implementation of the same estimator as correlation_at_lag:
      C(dt) = mean_t [ sum(A(t) * A(t+dt)) / sum(A(t)) ]
            = (1/n_valid) * Σ_t Σ_ij [A_ij(t)/sum(A(t))] * A_ij(t+dt)
    i.e. a cross-correlation in t between the edge-normalised base frames and
    the raw frames, summed over matrix entries. Zero-edge base frames get
    weight 0 and are excluded from n_valid (same as the loop version).
    """
    T = len(matrices)
    n_base = T - dt_max
    if n_base < 1:
        raise ValueError(f"dt_max={dt_max} requires > {dt_max} frames (got {T})")
    dts = list(range(dt_max + 1))

    X = np.stack([np.asarray(m, dtype=bool).ravel() for m in matrices])  # (T, M)
    edges = X[:n_base].sum(axis=1).astype(np.float64)
    valid = edges > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return dts, [1.0] + [0.0] * dt_max
    w_scale = np.zeros(n_base, dtype=np.float32)
    w_scale[valid] = 1.0 / edges[valid]

    L = 1 << (T - 1).bit_length()          # power of 2 ≥ T → no circular wrap
    num = np.zeros(dt_max + 1, dtype=np.float64)
    chunk = 4096                            # columns per FFT batch (memory cap)
    for s in range(0, X.shape[1], chunk):
        xb = X[:, s:s + chunk].astype(np.float32)
        wb = xb[:n_base] * w_scale[:, None]
        Fx = np.fft.rfft(xb, n=L, axis=0)
        Fw = np.fft.rfft(wb, n=L, axis=0)
        c = np.fft.irfft(np.conj(Fw) * Fx, n=L, axis=0)[:dt_max + 1]
        num += c.sum(axis=1, dtype=np.float64)

    corrs = num / n_valid
    corrs[0] = 1.0                          # exact, matches correlation_at_lag
    return dts, [float(v) for v in corrs]


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
