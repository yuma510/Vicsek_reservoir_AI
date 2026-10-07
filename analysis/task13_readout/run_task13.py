"""task_13: リードアウトの効果を確認（粒子1個で入力そのもの u(t) を予測、k=0）

粒子 N=1 のリザバーを入力 u(t)（NARMA10）で駆動し、状態 [1, sin(θ(t))] から
**遅延 0（現在の入力 u(t)）**を Ridge 回帰で予測する。MC は k ステップ前を予測するが
本タスクは k=0 のみ。リードアウト（sin θ → u）の非線形復元能力を見る。

出力（CSV、CLAUDE.md §5.2）:
  theta.csv          各時刻フレームの θ
  sin_theta.csv      各時刻フレームの sin(θ)
  prediction.csv     各時刻の u_true, u_pred（k=0）
  readout_prediction.png  予測 vs 正解の時系列
  params_used.json   使用パラメータ

パラメータは default_params.json から読む（直書き禁止）。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import (apply_style, load_position_fast, build_states,
                       ridge_gram_decomp, nrmse, mck_score, find_narma_by_seed)

apply_style()
BINARY = str(ROOT / "vicsek_dynamic")
_P = json.loads((Path(__file__).parent / "default_params.json").read_text())


def run_sim(mp, input_path, data_dir):
    """N=1 シミュレーションを実行し、出力 dir を返す。"""
    cfg = {"input_file": str(input_path), "output_base": str(data_dir), **mp}
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    before = set(Path(data_dir).glob("2*"))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)   # 1 行 1 フィールド（C パーサ要件）
        r = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    if r.returncode != 0:
        sys.exit(f"[ERROR] simulation failed: {r.stderr[:300]}")
    after = set(Path(data_dir).glob("2*"))
    new = sorted(after - before)
    return new[-1] if new else sorted(after)[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="analysis/task13_readout")
    p.add_argument("--narma-root", default="narma_data")
    p.add_argument("--skip-sim", action="store_true", help="既存の sim dir を --sim-dir で指定して評価のみ")
    p.add_argument("--sim-dir", default=None)
    args = p.parse_args()

    mp = _P["model"]
    rp = _P["reservoir"]
    narma_seed = _P["narma_seed"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    input_path, target_path = find_narma_by_seed(args.narma_root, narma_seed)
    if input_path is None:
        sys.exit(f"NARMA10 (seed={narma_seed}) が {args.narma_root}/ に見つからない。")

    # Phase 1: N=1 シミュレーション
    if args.skip_sim and args.sim_dir:
        sim_dir = Path(args.sim_dir)
    else:
        print("N=1 シミュレーション実行中 …", flush=True)
        sim_dir = run_sim(mp, input_path, args.data_dir)
    print(f"sim_dir = {sim_dir}", flush=True)
    params = json.load(open(sim_dir / "params_model.json"))

    # Phase 2: 状態構築（[1, sin θ]）と k=0 入力予測
    N, utime, ntime = params["N"], params["utime"], params["ntime"]
    n_input = ntime // utime
    data = load_position_fast(sim_dir / "position.dat")
    pre = (data.shape[0] == N * n_input)
    states = build_states(data, N, utime, pre_subsampled=pre)   # (T, N+1) = (T, 2)
    theta = data[:, 2].reshape(-1, N)                            # (frames, 1) 全ステップ
    # utime サブサンプルした θ（状態と同じフレーム）
    idx = (np.arange(0, theta.shape[0]) if pre
           else np.arange(utime - 1, theta.shape[0], utime))
    theta_sub = theta[idx, 0]
    sin_theta = np.sin(theta_sub)

    raw = np.loadtxt(input_path)
    u = (raw[:, 0] if raw.ndim > 1 else raw)
    n_eval = int(min(n_input, states.shape[0], u.shape[0]))

    wash, tn, lam = rp["washout"], rp["train_num"], rp["ridge_lambda"]
    train_end = min(wash + tn, n_eval)
    X = states[wash:train_end]                 # (train, 2)
    y = u[wash:train_end]                       # k=0: 現在の入力
    vals, vecs = ridge_gram_decomp(X)
    W = vecs @ ((vecs.T @ (X.T @ y)) / (vals + lam))
    u_pred = states[:n_eval] @ W                # 全区間予測

    # メトリクス（test 区間）
    te = slice(train_end, n_eval)
    nr_test = nrmse(u[te], u_pred[te])
    mc0_test = mck_score(u[te], u_pred[te])     # k=0 の決定係数（corr^2）
    print(f"k=0 予測: NRMSE_test={nr_test:.4f}  MC0_test(corr^2)={mc0_test:.4f}", flush=True)

    # Phase 3: 出力 CSV
    t = np.arange(n_eval)
    pd.DataFrame({"frame": t, "theta": theta_sub[:n_eval]}).to_csv(out / "theta.csv", index=False)
    pd.DataFrame({"frame": t, "sin_theta": sin_theta[:n_eval]}).to_csv(out / "sin_theta.csv", index=False)
    pd.DataFrame({"frame": t, "u_true": u[:n_eval], "u_pred": u_pred[:n_eval]}).to_csv(
        out / "prediction.csv", index=False)

    # プロット（test 区間の先頭 200 フレーム）
    s = train_end
    e = min(train_end + 200, n_eval)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(s, e), u[s:e], "k-", lw=1.5, label="u_true (input)")
    ax.plot(range(s, e), u_pred[s:e], "r--", lw=1.2, label="u_pred (readout, k=0)")
    ax.set_xlabel("frame"); ax.set_ylabel("input u")
    ax.set_title(f"Readout of current input (k=0), N=1  "
                 f"[NRMSE={nr_test:.3f}, corr$^2$={mc0_test:.3f}]")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "readout_prediction.png", dpi=150)
    plt.close(fig)

    with open(out / "params_used.json", "w") as f:
        json.dump({"model": params, "reservoir": rp, "narma_seed": narma_seed,
                   "n_eval": n_eval, "nrmse_test": nr_test, "mc0_test": mc0_test},
                  f, indent=2, ensure_ascii=False)
    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()
