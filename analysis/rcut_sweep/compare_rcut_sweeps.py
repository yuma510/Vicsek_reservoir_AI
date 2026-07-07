"""複数の rcut_sweep run を縦軸スケールを揃えて再プロット（比較用）。

各 run ディレクトリの rcut_sweep_data.csv を読み、metric ごとに全 run 横断の
最大値から共通 ylim=(0, ymax*1.05) を求めて、各 run に *_aligned.png を出力する。
元の PNG・CSV・results_*.json は変更しない。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DIRS = [
    "analysis/rcut_sweep/20260619_204013",
    "analysis/rcut_sweep/20260623_121526",
]

# (test列, train列, ylabel, 出力ファイル名)
METRICS = [
    ("MC_test",     "MC_train",     "MC (test/train)",     "mc_vs_rcut_scatter_aligned.png"),
    ("nrmse_test",  "nrmse_train",  "NRMSE (test/train)",  "nrmse_vs_rcut_scatter_aligned.png"),
    ("nrmse2_test", "nrmse2_train", "NRMSE2 (test/train)", "nrmse2_vs_rcut_scatter_aligned.png"),
]

YMARGIN = 1.05


def plot_aligned(df: pd.DataFrame, test_key, train_key, ylabel, ylim, out_path: Path):
    """phase3_plot と同じスタイルで散布図を描き、共通 ylim を適用して保存する。"""
    rcut_sorted = sorted(df["rcut"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, rcut in enumerate(rcut_sorted):
        sub = df[df["rcut"] == rcut]
        ax.scatter([rcut] * len(sub), sub[test_key],
                   color="tab:blue", alpha=0.6, s=30, marker="o",
                   label="test" if i == 0 else "")
        ax.scatter([rcut] * len(sub), sub[train_key],
                   color="tab:red", alpha=0.6, s=30, marker="^",
                   label="train" if i == 0 else "")
    ax.set_xlabel("rcut")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="複数 rcut_sweep run を縦軸スケールを揃えて再プロット")
    p.add_argument("dirs", nargs="*", default=DEFAULT_DIRS,
                   help="rcut_sweep_data.csv を含む run ディレクトリ（デフォルト: 今回の2つ）")
    args = p.parse_args()

    dirs = [Path(d) for d in args.dirs]
    dfs = {}
    for d in dirs:
        csv = d / "rcut_sweep_data.csv"
        if not csv.exists():
            raise SystemExit(f"[ERROR] not found: {csv}")
        dfs[d] = pd.read_csv(csv)

    # metric ごとに全 run 横断の最大値から共通 ylim=(0, ymax*1.05) を算出
    ylims = {}
    for test_key, train_key, _, _ in METRICS:
        ymax = max(df[[test_key, train_key]].max().max() for df in dfs.values())
        ylims[test_key] = (0.0, ymax * YMARGIN)

    for d in dirs:
        print(f"[{d}]", flush=True)
        for test_key, train_key, ylabel, fname in METRICS:
            plot_aligned(dfs[d], test_key, train_key, ylabel,
                         ylims[test_key], d / fname)

    print("\n共通縦軸範囲 (0, ymax*1.05):", flush=True)
    for test_key, _, ylabel, _ in METRICS:
        print(f"  {ylabel:22s}: {ylims[test_key]}", flush=True)


if __name__ == "__main__":
    main()
