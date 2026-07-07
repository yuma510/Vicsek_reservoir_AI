"""Data loading and reservoir-state construction utilities."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_position_fast(filepath) -> np.ndarray:
    """Read position.dat faster than np.loadtxt using pandas C engine."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None,
                     names=["x", "y", "theta"], engine="c")
    return df.values


def build_states(data: np.ndarray, N: int, utime: int,
                 readout: int = 1, add_bias: bool = True,
                 pre_subsampled: bool = False) -> np.ndarray:
    """Build (n_frames, N+1) reservoir state matrix (bias column first when add_bias).

    pre_subsampled=True: position.dat が既に utime ごとのフレームのみ含む新フォーマット。
    pre_subsampled=False (default): 旧フォーマット（全ステップ保存）。
    """
    if data.shape[0] % N != 0:
        raise ValueError(f"data rows ({data.shape[0]}) not divisible by N ({N}).")
    frame_count = data.shape[0] // N
    theta = data[:, 2].reshape(frame_count, N)
    idx = (np.arange(0, frame_count) if pre_subsampled
           else np.arange(utime - 1, frame_count, utime))
    feat  = np.sin(theta[idx]) if readout == 1 else theta[idx]
    if add_bias:
        feat = np.hstack([np.ones((feat.shape[0], 1)), feat])
    return feat


def load_theta(pos_path, N: int) -> np.ndarray:
    """Return the (n_frames, N) theta array from position.dat."""
    data = load_position_fast(pos_path)
    frame_count = data.shape[0] // N
    return data[:, 2].reshape(frame_count, N)


def load_adjacency_dir(adj_dir: Path) -> tuple[list[int], list[np.ndarray]]:
    """Return (sorted timesteps, list of NxN int8 matrices)."""
    adj_dir = Path(adj_dir)
    files = sorted(adj_dir.glob("adjacency_*.dat"))
    if not files:
        raise FileNotFoundError(f"No adjacency files in {adj_dir}")
    timesteps = [int(f.stem.split("_")[-1]) for f in files]
    matrices = [np.loadtxt(f, dtype=np.int8) for f in files]
    return timesteps, matrices


def delayed_input(raw_input: np.ndarray, delay: int, length: int) -> np.ndarray:
    """Prepend `delay` zeros to raw_input and return first `length` samples."""
    padded = np.concatenate([np.zeros(delay, dtype=raw_input.dtype), raw_input])
    if padded.size < length:
        padded = np.concatenate([padded, np.zeros(length - padded.size, dtype=padded.dtype)])
    return padded[:length]


def find_exp_dir(data_dir, *, rcut=None, sgm=None, v0=None, seed=None,
                 seed_index=None, seed_key=None) -> Path | None:
    """
    Return the newest experiment directory under `data_dir` whose
    params_model.json matches the given filters.

    - rcut / sgm / v0: matched within 1e-6 tolerance when not None.
    - seed_key: new format — matched against params[seed_key] directly.
    - seed_index: old format — matched against params["seed"][seed_index].
    """
    data_dir = Path(data_dir)
    candidates = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        params_file = d / "params_model.json"
        if not params_file.exists():
            continue
        with open(params_file) as f:
            p = json.load(f)
        if rcut is not None and abs(p.get("rcut", -1) - rcut) >= 1e-6:
            continue
        if sgm is not None and abs(p.get("sgm", -1) - sgm) >= 1e-6:
            continue
        if v0 is not None and abs(p.get("v0", -1) - v0) >= 1e-6:
            continue
        if seed is not None:
            if seed_key is not None:
                if p.get(seed_key) != seed:
                    continue
            elif seed_index is not None:
                seeds = p.get("seed", [])
                if seed_index >= len(seeds) or seeds[seed_index] != seed:
                    continue
        candidates.append(d)
    return candidates[-1] if candidates else None
