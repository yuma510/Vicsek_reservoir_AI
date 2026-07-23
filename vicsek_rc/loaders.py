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


def _find_exp_dir_from_index(df, data_dir, *, rcut, sgm, v0, ntime, utime,
                             narma_seed, seed, seed_index, seed_key):
    """index.csv（DataFrame）から find_exp_dir と同じ選択規則で最新 dir を返す。

    全 dir 走査の代わりに index を引く高速経路。走査版と同一の結果を返すよう、
    フィルタ条件・「dir 名昇順の最後（=最新）」の選択規則を厳密に一致させる。
    """
    m = df["has_params"].fillna(False).astype(bool)
    if rcut is not None:
        m &= (df["rcut"] - rcut).abs() < 1e-6
    if sgm is not None:
        m &= (df["sgm"] - sgm).abs() < 1e-6
    if v0 is not None:
        m &= (df["v0"] - v0).abs() < 1e-6
    if ntime is not None:
        m &= df["ntime"] == ntime
    if utime is not None:
        m &= df["utime"] == utime
    if narma_seed is not None:
        # 別 NARMA で駆動された（と記録された）実験を除外。input_file 未記録(NaN)は
        # 検証不能なので残す（旧スイープシムを取りこぼさないためのフォールバック）
        inf = df["input_file"] if "input_file" in df.columns else None
        if inf is not None:
            m &= inf.isna() | inf.astype(str).str.contains(f"seed{narma_seed}.dat", regex=False)
    if seed is not None:
        if seed_key is not None:
            # 新フォーマット: params[seed_key] を直接照合（列が無ければ一致なし）
            m &= (df[seed_key] == seed) if seed_key in df.columns else False
        elif seed_index is not None:
            # 旧フォーマット: 連結文字列 "s0,s1,.." の seed_index 番目を照合
            def _seed_at(s):
                if not isinstance(s, str) or not s:
                    return None
                parts = s.split(",")
                return int(parts[seed_index]) if seed_index < len(parts) else None
            m &= df["seed"].map(_seed_at) == seed
    matched = df[m]
    if matched.empty:
        return None
    newest = matched.sort_values("dir").iloc[-1]["dir"]
    return Path(data_dir) / str(newest)


def find_exp_dir(data_dir, *, rcut=None, sgm=None, v0=None, ntime=None, utime=None,
                 narma_seed=None, seed=None, seed_index=None, seed_key=None) -> Path | None:
    """
    Return the newest experiment directory under `data_dir` whose
    params_model.json matches the given filters.

    - rcut / sgm / v0: matched within 1e-6 tolerance when not None.
    - ntime / utime: matched exactly when not None（フラットな data/ に混在する
      別実験（例: rcut_utime_heatmap の ntime≠140000）の誤マッチを防ぐ）。
    - narma_seed: input_file が別 NARMA seed で駆動された実験を除外する。
      input_file 未記録（NaN）のシムは検証不能なため残す（旧スイープ互換）。
    - seed_key: new format — matched against params[seed_key] directly.
    - seed_index: old format — matched against params["seed"][seed_index].

    `<data_dir>/index.csv` が存在すればそれを引いて高速化する。無ければ全 dir を走査する
    （後方互換）。両経路は同一の dir を返す。
    """
    data_dir = Path(data_dir)

    from .catalog import load_catalog  # 遅延 import（循環回避）
    df = load_catalog(data_dir)
    if df is not None:
        return _find_exp_dir_from_index(
            df, data_dir, rcut=rcut, sgm=sgm, v0=v0, ntime=ntime, utime=utime,
            narma_seed=narma_seed, seed=seed, seed_index=seed_index, seed_key=seed_key)

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
        if ntime is not None and p.get("ntime") != ntime:
            continue
        if utime is not None and p.get("utime") != utime:
            continue
        if narma_seed is not None:
            inf = p.get("input_file")
            if inf and f"seed{narma_seed}.dat" not in inf:
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
