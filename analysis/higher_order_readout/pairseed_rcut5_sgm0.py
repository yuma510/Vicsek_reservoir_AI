"""Task 08 派生: rcut=5 / sgm=0 に絞り、高次リードアウト(P=500 ランダムペア積項)を
複数の pair_seed で試し、baseline(P=0) と比較プロットする。

「追加したリードアウトの取り方」= ランダムペア選択 pair_seed に依存するため、
pair_seed を複数振って MC/NRMSE のばらつきを baseline に対して可視化する。
既存 higher_order_readout.py の評価関数を再利用（シム・入力の解決やガードを共有）。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import load_reservoir_defaults, write_params_used
from vicsek_rc.loaders import load_position_fast, build_states
from vicsek_rc.plotting import apply_style

import higher_order_readout as hor

_RC = load_reservoir_defaults()


def parse_args():
    p = argparse.ArgumentParser(
        description="rcut=5 / sgm=0 の高次リードアウトを複数 pair_seed で試し baseline と比較")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--v0", type=float, default=0.5)
    p.add_argument("--rcut", type=float, default=5.0)
    p.add_argument("--sgm", type=float, default=0.0)
    p.add_argument("--n-pairs", type=int, default=500,
                   help="追加ペア数 P（固定, default 500）")
    p.add_argument("--n-pair-seeds", type=int, default=20,
                   help="振る pair_seed の数（0..n-1, default 20）")
    p.add_argument("--narma-root", default="narma_data")
    p.add_argument("--output-dir",
                   default="analysis/higher_order_readout/pairseed_rcut5_sgm0")
    p.add_argument("--washout",      type=int,   default=_RC["washout"])
    p.add_argument("--train-num",    type=int,   default=_RC["train_num"])
    p.add_argument("--ridge-lambda", type=float, default=_RC["ridge_lambda"])
    p.add_argument("--k-max",        type=int,   default=_RC["k_max"])
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir   = Path(args.data_dir)
    narma_root = Path(args.narma_root)

    sim_list = hor._scan_sims(data_dir, args.v0, args.rcut, args.sgm)
    if not sim_list:
        print(f"[ERROR] v0={args.v0} rcut={args.rcut} sgm={args.sgm}: sim が見つかりません。",
              flush=True)
        return
    sim_list = sorted(sim_list)

    rows = []
    model_params_list = []
    for seed_pos, sim_dir in sim_list:
        print(f"[EVAL] seed_pos={seed_pos}  {sim_dir.name} …", flush=True)
        with open(sim_dir / "params_model.json") as f:
            params = json.load(f)

        try:
            u, y = hor._load_narma(sim_dir, params, narma_root)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)
            continue

        data = load_position_fast(sim_dir / "position.dat")
        N = int(params["N"])
        utime = int(params["utime"])
        n_frames = int(params["ntime"]) // utime
        pre = (data.shape[0] == N * n_frames)
        X_base = build_states(data, N, utime, pre_subsampled=pre)

        # baseline P=0
        nr0, mc0 = hor._evaluate_ext(
            X_base, u, y, args.washout, args.train_num,
            args.ridge_lambda, args.k_max, N)
        # ガード: 退化 sim（駆動入力の不一致等）は除外
        if mc0 < hor.MC_DEGENERATE_THRESHOLD or not np.isfinite(nr0):
            print(f"  [SKIP] seed_pos={seed_pos}: 退化 (base MC={mc0:.2e}, NRMSE={nr0}) — 除外",
                  flush=True)
            continue
        model_params_list.append(params)
        rows.append({"seed_pos": seed_pos, "n_pairs": 0, "pair_seed": -1,
                     "nrmse_test": nr0, "mc_test": mc0})
        print(f"  P=0    NRMSE={nr0:.4f}  MC={mc0:.3f}", flush=True)

        # 高次 P=500 を pair_seed 0..n-1 で
        for ps in range(args.n_pair_seeds):
            X_ext = hor._build_ext(X_base, args.n_pairs, ps, N)
            nr, mc = hor._evaluate_ext(
                X_ext, u, y, args.washout, args.train_num,
                args.ridge_lambda, args.k_max, N)
            rows.append({"seed_pos": seed_pos, "n_pairs": args.n_pairs,
                         "pair_seed": ps, "nrmse_test": nr, "mc_test": mc})
        hi = [r for r in rows[-args.n_pair_seeds:]]
        print(f"  P={args.n_pairs}  NRMSE={np.mean([r['nrmse_test'] for r in hi]):.4f}"
              f"  MC={np.mean([r['mc_test'] for r in hi]):.3f} "
              f"(avg over {args.n_pair_seeds} pair seeds)", flush=True)

    if not rows:
        print("[ERROR] 結果が 0 件。", flush=True)
        return

    df = pd.DataFrame(rows)
    csv_name = f"pairseed_rcut{args.rcut:g}_sgm{args.sgm:g}_data.csv"
    df.to_csv(out_dir / csv_name, index=False)
    print(f"  {csv_name} ({len(df)} rows)", flush=True)

    _plot(df, args, out_dir)
    _plot_pooled(df, args, out_dir)

    write_params_used(out_dir, model_params_list, {
        "readout": _RC["readout"], "washout": args.washout,
        "train_num": args.train_num, "ridge_lambda": args.ridge_lambda,
        "k_max": args.k_max, "n_pairs": args.n_pairs,
        "n_pair_seeds": args.n_pair_seeds,
    })
    print(f"  params_used.json\nOutput: {out_dir}", flush=True)


def _plot(df: pd.DataFrame, args, out_dir: Path):
    """seed_pos ごとに baseline(P=0) と高次(P=500, pair_seed 散布) を 2 パネルで比較。"""
    apply_style()
    base = df[df["n_pairs"] == 0]
    hi   = df[df["n_pairs"] == args.n_pairs]
    seeds = sorted(df["seed_pos"].unique())
    xpos = {s: i for i, s in enumerate(seeds)}

    metrics = [("mc_test", "MC_test"), ("nrmse_test", "NRMSE_test")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)

    for col, (metric, label) in enumerate(metrics):
        ax = axes[0][col]
        # 高次: pair_seed ごとの散布点
        for s in seeds:
            hs = hi[hi["seed_pos"] == s]
            xs = np.full(len(hs), xpos[s]) + np.random.uniform(-0.12, 0.12, len(hs))
            ax.scatter(xs, hs[metric], color="tab:blue", alpha=0.5, s=22,
                       label=f"higher-order (P={args.n_pairs})" if s == seeds[0] else "")
        # 高次: seed_pos 平均
        hmean = hi.groupby("seed_pos")[metric].mean()
        ax.scatter([xpos[s] for s in seeds], [hmean[s] for s in seeds],
                   color="tab:blue", marker="_", s=420, linewidths=2,
                   label="higher-order mean")
        # baseline P=0
        bmap = base.set_index("seed_pos")[metric]
        ax.scatter([xpos[s] for s in seeds], [bmap[s] for s in seeds],
                   color="black", marker="*", s=90, zorder=5, label="baseline (P=0)")

        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("seed_pos (simulation)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Higher-order readout vs baseline  (rcut={args.rcut:g}, sgm={args.sgm:g}, "
                 f"pair_seeds={args.n_pair_seeds})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_higher_order.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  baseline_vs_higher_order.png", flush=True)


def _plot_pooled(df: pd.DataFrame, args, out_dir: Path, ylims: dict | None = None,
                 out_name: str = "baseline_vs_higher_order_pooled.png"):
    """seed(seed_pos・pair_seed)をランダムサンプルとしてプールし、
    baseline(P=0) と 高次(P=500) を 1 軸(2 カテゴリ)で比較する。

    ylims: {metric: (lo, hi)} を渡すと各パネルの縦軸範囲を固定する（複数条件の比較用）。
    """
    apply_style()
    groups = [("baseline\n(P=0)",            df[df["n_pairs"] == 0],          "black"),
              (f"higher-order\n(P={args.n_pairs})", df[df["n_pairs"] == args.n_pairs], "tab:blue")]
    metrics = [("mc_test", "MC_test"), ("nrmse_test", "NRMSE_test")]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), squeeze=False)
    for col, (metric, label) in enumerate(metrics):
        ax = axes[0][col]
        for xi, (name, g, color) in enumerate(groups):
            vals = g[metric].to_numpy()
            xs = xi + rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xs, vals, color=color, alpha=0.45, s=22, zorder=2)
            # 平均 ± 標準偏差
            m, sd = float(np.mean(vals)), float(np.std(vals))
            ax.errorbar(xi, m, yerr=sd, fmt="o", color=color, ecolor=color,
                        markersize=8, capsize=5, elinewidth=1.5, zorder=3,
                        markeredgecolor="white")
            ax.annotate(f"mean={m:.3f}\n(n={len(vals)})", (xi, m),
                        textcoords="offset points", xytext=(14, 0),
                        fontsize=8, va="center")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_xlim(-0.5, len(groups) - 0.5)
        if ylims and metric in ylims:
            ax.set_ylim(*ylims[metric])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Higher-order readout vs baseline — pooled over seeds "
                 f"(rcut={args.rcut:g}, sgm={args.sgm:g}, pair_seeds={args.n_pair_seeds})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_name}", flush=True)


if __name__ == "__main__":
    main()
