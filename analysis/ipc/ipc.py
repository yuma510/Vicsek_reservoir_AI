"""IPC (Information Processing Capacity) measurement for Vicsek reservoir.

Dambre et al. 2012: C[X, z_l] = corr²(z_l_test, X_test @ w*)
where w* = (X_tr^T X_tr + λI)^{-1} X_tr^T z_l_tr.

The Vicsek reservoir is DRIVEN by external input u(t) via a torque term
F·sin(c·u(t) − θ_i) in the dynamics. The reservoir states x(t) = sin(θ_i(t))
therefore carry memory of past inputs. IPC quantifies this memory.

Basis functions (orthonormal for u ~ Uniform[0, 0.5]):
  degree-1:  P̃₁(u(t-k)) = √3·(4u-1)         [MC_k ≡ C_d1(k)]
  degree-2a: P̃₂(u(t-k)) = √5·(3(4u-1)²-1)/2  [pure quadratic at single lag]
  degree-2b: P̃₁(u(t-k₁))·P̃₁(u(t-k₂))         [cross-lag product, k₁ < k₂]

C_total ≤ N (N=500 for Vicsek default).

IMPORTANT: u(t) must be the SAME sequence that drove the reservoir simulation
(via the narma10 input file). Each experiment was driven by narma_seed=seed_pos-3
with length 24000. Provide --narma-seeds to match --seeds.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import (
    apply_style,
    build_states,
    load_position_fast,
    load_reservoir_defaults,
    mck_score,
    write_params_used,
)

_RC = load_reservoir_defaults()
_IP = json.loads((Path(__file__).parent / "default_params.json").read_text())


# ---------------------------------------------------------------------------
# Experiment lookup (with optional ntime filter)
# ---------------------------------------------------------------------------

def _find_exp(data_dir, rcut, sgm, seed_pos, ntime=None):
    """Return newest experiment dir matching rcut, sgm, seed_pos, and ntime."""
    candidates = []
    for d in sorted(Path(data_dir).iterdir()):
        pmf = d / "params_model.json"
        if not pmf.exists():
            continue
        p = json.loads(pmf.read_text())
        if abs(p.get("rcut", -1) - rcut) >= 1e-6:
            continue
        if abs(p.get("sgm", -1) - sgm) >= 1e-6:
            continue
        if p.get("seed_pos") != seed_pos:
            continue
        if ntime is not None and p.get("ntime") != ntime:
            continue
        candidates.append(d)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# NARMA input loading
# ---------------------------------------------------------------------------

def _load_narma_input(narma_seed: int, n_frames: int,
                      narma_dir: str = "narma_data",
                      tmp_dir: str = "tmp") -> np.ndarray:
    """Load narma input sequence (first n_frames samples) for given seed.

    Searches narma_data/ subdirs, narma_data/ root, and tmp/ for a file
    matching narma10_input_*_seed{narma_seed}*.dat with >= n_frames lines.
    Falls back to in-process generation using generate_narma10.
    """
    search_roots = [Path(narma_dir), Path(tmp_dir)]
    search_dirs: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            subdirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
            search_dirs.extend(subdirs)
            search_dirs.append(root)

    for d in search_dirs:
        for f in sorted(d.glob(f"narma10_input_*_seed{narma_seed}*.dat")):
            n_lines = sum(1 for _ in open(f))
            if n_lines >= n_frames:
                u = np.loadtxt(f)
                print(f"    [narma] loaded {f.name} ({n_lines} lines) from {d.name}")
                return u[:n_frames]

    # Fallback: regenerate deterministically
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from generate_narma10 import generate_narma10
    print(f"    [narma] regenerating seed={narma_seed} length={n_frames + 100}")
    u, _ = generate_narma10(n_frames + 100, 0.0, 0.5, narma_seed)
    return u[:n_frames]


# ---------------------------------------------------------------------------
# Legendre basis functions for u ~ Uniform[0, 0.5]
# ---------------------------------------------------------------------------

def _p1(u: np.ndarray) -> np.ndarray:
    """Normalized degree-1 Legendre: E=0, Var=1 for u ~ U[0,0.5]."""
    return np.sqrt(3.0) * (4.0 * u - 1.0)


def _p2(u: np.ndarray) -> np.ndarray:
    """Normalized degree-2 Legendre: E=0, Var=1 for u ~ U[0,0.5]."""
    x = 4.0 * u - 1.0
    return np.sqrt(5.0) * (3.0 * x * x - 1.0) / 2.0


def _delay(arr: np.ndarray, k: int, T: int) -> np.ndarray:
    """Return arr delayed by k steps, zero-padded head, truncated to T."""
    if k == 0:
        return arr[:T]
    return np.concatenate([np.zeros(k, dtype=arr.dtype), arr])[:T]


# ---------------------------------------------------------------------------
# Core IPC computation for one reservoir realization
# ---------------------------------------------------------------------------

def _compute_ipc(states: np.ndarray, u: np.ndarray,
                 washout: int, train_num: int,
                 k_max_d1: int, k_max_d2a: int, k_max_d2b: int,
                 lam: float) -> pd.DataFrame:
    """Compute IPC capacity for all basis functions up to degree 2.

    Returns DataFrame with columns: type, k1, k2, capacity.
      type='d1'  — degree-1 at lag k1 (k2=-1)
      type='d2a' — degree-2a at lag k1=k2 (pure quadratic)
      type='d2b' — degree-2b at lags k1<k2 (cross-lag product)
    """
    T = min(states.shape[0], u.shape[0])
    states = states[:T]
    u = u[:T]

    train_end = washout + train_num
    if train_end >= T:
        raise ValueError(f"train_end={train_end} must be < T={T}")

    X_tr = states[washout:train_end]
    X_te = states[train_end:T]

    M = X_tr.T @ X_tr
    M_inv = np.linalg.solve(M + lam * np.eye(M.shape[0]), np.eye(M.shape[0]))

    p1_all = _p1(u)
    p2_all = _p2(u)

    def cap(z: np.ndarray) -> float:
        z_tr = z[washout:train_end]
        z_te = z[train_end:T]
        w = M_inv @ (X_tr.T @ z_tr)
        return mck_score(z_te, X_te @ w)

    rows = []

    # Degree 1: P̃₁(u(t-k))
    for k in range(k_max_d1 + 1):
        z = _delay(p1_all, k, T)
        rows.append({"type": "d1", "k1": k, "k2": -1, "capacity": cap(z)})

    # Degree 2a: P̃₂(u(t-k))
    for k in range(k_max_d2a + 1):
        z = _delay(p2_all, k, T)
        rows.append({"type": "d2a", "k1": k, "k2": k, "capacity": cap(z)})

    # Degree 2b: P̃₁(u(t-k₁)) · P̃₁(u(t-k₂)) for k₁ < k₂
    delayed_p1 = [_delay(p1_all, k, T) for k in range(k_max_d2b + 1)]
    for k1 in range(k_max_d2b + 1):
        for k2 in range(k1 + 1, k_max_d2b + 1):
            z = delayed_p1[k1] * delayed_p1[k2]
            rows.append({"type": "d2b", "k1": k1, "k2": k2, "capacity": cap(z)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(df_avg: pd.DataFrame, rcut: float, sgm: float, out_dir: Path) -> None:
    apply_style()

    d1  = df_avg[df_avg["type"] == "d1"].sort_values("k1")
    d2a = df_avg[df_avg["type"] == "d2a"].sort_values("k1")

    # Plot 1: IPC vs delay (d1 and d2a)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].bar(d1["k1"], d1["capacity"].clip(0), width=0.8, color="tab:blue")
    axes[0].set_ylabel("C(k)")
    axes[0].set_xlabel("Delay k")
    axes[0].set_title(f"Degree-1 linear memory  (rcut={rcut}, sgm={sgm})")
    axes[0].set_xlim(-0.5, d1["k1"].max() + 0.5)

    axes[1].bar(d2a["k1"], d2a["capacity"].clip(0), width=0.8, color="tab:orange")
    axes[1].set_ylabel("C(k)")
    axes[1].set_xlabel("Delay k")
    axes[1].set_title("Degree-2a quadratic memory")
    axes[1].set_xlim(-0.5, d2a["k1"].max() + 0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "ipc_vs_delay.png", dpi=150)
    plt.close(fig)

    # Plot 2: totals by degree
    totals = df_avg.groupby("type")["capacity"].sum().clip(0)
    c_d1  = float(totals.get("d1",  0.0))
    c_d2a = float(totals.get("d2a", 0.0))
    c_d2b = float(totals.get("d2b", 0.0))
    c_d2  = c_d2a + c_d2b
    c_tot = c_d1 + c_d2

    labels = ["C_d1\n(linear)", "C_d2a\n(quadratic)", "C_d2b\n(cross-lag)",
              "C_d2\n(2a+2b)", "C_total"]
    values = [c_d1, c_d2a, c_d2b, c_d2, c_tot]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Total IPC")
    ax.set_title(f"IPC by degree  (rcut={rcut}, sgm={sgm})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(c_tot * 0.01, 0.02),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ipc_by_degree.png", dpi=150)
    plt.close(fig)

    _make_heatmap_d2(df_avg, rcut, sgm, out_dir)


def _make_heatmap_d2(df_avg: pd.DataFrame, rcut: float, sgm: float, out_dir: Path) -> None:
    """2D heatmap of degree-2 IPC.

    Diagonal (k1==k2): d2a capacity.
    Off-diagonal (k1!=k2): d2b capacity, displayed symmetrically.
    """
    apply_style()

    d2a = df_avg[df_avg["type"] == "d2a"].set_index("k1")["capacity"]
    d2b = df_avg[df_avg["type"] == "d2b"]

    k_max = int(df_avg[df_avg["type"] == "d2b"][["k1", "k2"]].max().max())
    mat = np.full((k_max + 1, k_max + 1), np.nan)

    for k in range(k_max + 1):
        if k in d2a.index:
            mat[k, k] = float(d2a[k])

    for _, row in d2b.iterrows():
        k1, k2, c = int(row["k1"]), int(row["k2"]), float(row["capacity"])
        mat[k1, k2] = c
        mat[k2, k1] = c  # symmetric display

    vmax = float(np.nanmax(mat))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat.clip(0), origin="lower", aspect="equal",
                   cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Capacity C(k₁, k₂)")
    ax.set_xlabel("Delay k₂")
    ax.set_ylabel("Delay k₁")
    ax.set_title(f"Degree-2 IPC  (rcut={rcut}, sgm={sgm})\n"
                 "diagonal = d2a,  off-diagonal = d2b (symmetric)")
    fig.tight_layout()
    fig.savefig(out_dir / "ipc_heatmap_d2.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="IPC measurement for Vicsek reservoir")
    ap.add_argument("--data-dir",     default="data")
    ap.add_argument("--rcut",         type=float, default=13.0)
    ap.add_argument("--sgm",          type=float, default=0.0)
    ap.add_argument("--seeds",        type=int, nargs="+", required=True,
                    help="seed_pos values; must match --narma-seeds one-to-one")
    ap.add_argument("--narma-seeds",  type=int, nargs="+", required=True,
                    help="NARMA input seed for each seed_pos (same order as --seeds)")
    ap.add_argument("--narma-dir",    default="narma_data",
                    help="directory containing narma10 data subdirs (default: narma_data)")
    ap.add_argument("--tmp-dir",      default="tmp",
                    help="fallback directory for narma input files (default: tmp)")
    ap.add_argument("--ntime",        type=int, default=None,
                    help="filter experiments by ntime")
    ap.add_argument("--output-dir",   default="analysis/ipc")
    ap.add_argument("--washout",      type=int,   default=_RC["washout"])
    ap.add_argument("--train-num",    type=int,   default=_RC["train_num"])
    ap.add_argument("--ridge-lambda", type=float, default=_RC["ridge_lambda"])
    ap.add_argument("--k-max-d1",     type=int,   default=_IP["k_max_d1"])
    ap.add_argument("--k-max-d2a",    type=int,   default=_IP["k_max_d2a"])
    ap.add_argument("--k-max-d2b",    type=int,   default=_IP["k_max_d2b"])
    args = ap.parse_args()

    if len(args.seeds) != len(args.narma_seeds):
        ap.error("--seeds and --narma-seeds must have the same number of values")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    found_params: list[dict] = []

    for seed_pos, narma_seed in zip(args.seeds, args.narma_seeds):
        exp_dir = _find_exp(args.data_dir, args.rcut, args.sgm, seed_pos, args.ntime)
        if exp_dir is None:
            print(f"  [skip] seed_pos={seed_pos}: no matching experiment")
            continue

        params = json.loads((exp_dir / "params_model.json").read_text())
        n_frames = params["ntime"] // params["utime"]
        t_test = n_frames - args.washout - args.train_num
        print(f"  seed_pos={seed_pos} (narma_seed={narma_seed}): "
              f"{exp_dir.name}  n_frames={n_frames}  T_test={t_test}")

        u = _load_narma_input(narma_seed, n_frames, args.narma_dir, args.tmp_dir)

        data = load_position_fast(exp_dir / "position.dat")
        pre_sub = (data.shape[0] == params["N"] * n_frames)
        states = build_states(data, params["N"], params["utime"], pre_subsampled=pre_sub)

        df = _compute_ipc(
            states, u,
            args.washout, args.train_num,
            args.k_max_d1, args.k_max_d2a, args.k_max_d2b,
            args.ridge_lambda,
        )
        df["seed"] = seed_pos
        all_dfs.append(df)
        found_params.append(params)

        totals = df.groupby("type")["capacity"].sum()
        print(f"    d1={totals.get('d1', 0):.3f}  "
              f"d2a={totals.get('d2a', 0):.3f}  "
              f"d2b={totals.get('d2b', 0):.3f}")

    if not all_dfs:
        print("No experiments found. Exiting.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(out_dir / "ipc_data.csv", index=False)

    df_avg = (df_all
              .groupby(["type", "k1", "k2"])["capacity"]
              .mean()
              .reset_index())
    df_avg.to_csv(out_dir / "ipc_avg.csv", index=False)

    totals = df_avg.groupby("type")["capacity"].sum().clip(0)
    c_d1  = float(totals.get("d1",  0.0))
    c_d2a = float(totals.get("d2a", 0.0))
    c_d2b = float(totals.get("d2b", 0.0))
    summary_rows = [
        {"metric": "C_d1",    "value": c_d1},
        {"metric": "C_d2a",   "value": c_d2a},
        {"metric": "C_d2b",   "value": c_d2b},
        {"metric": "C_d2",    "value": c_d2a + c_d2b},
        {"metric": "C_total", "value": c_d1 + c_d2a + c_d2b},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "ipc_summary.csv", index=False)

    print("\n=== IPC Summary ===")
    for row in summary_rows:
        print(f"  {row['metric']:12s}: {row['value']:.4f}")
    print(f"  N (theory max): {found_params[0]['N']}")

    _make_plots(df_avg, args.rcut, args.sgm, out_dir)

    write_params_used(out_dir, found_params, {
        "washout":      args.washout,
        "train_num":    args.train_num,
        "ridge_lambda": args.ridge_lambda,
        "k_max_d1":     args.k_max_d1,
        "k_max_d2a":    args.k_max_d2a,
        "k_max_d2b":    args.k_max_d2b,
        "narma_seeds":  list(args.narma_seeds),
    })

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
