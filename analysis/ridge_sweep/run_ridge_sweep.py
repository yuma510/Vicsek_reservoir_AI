"""
λ × train_num の 2D 掃引で過学習を抑える最適リッジ回帰設定を探索する。

Phase 0: NARMA10 生成（length=24000、seeds=[1,2,3,5,6]）
Phase 1: 長尺シミュレーション（ntime=220000、rcut=13、sgm=0）
Phase 2: λ × train_num 2D 評価（各 seed で状態行列を1回だけ構築）
Phase 3: ヒートマップ・折れ線プロット・CSV 出力

実行例:
    python analysis/ridge_sweep/run_ridge_sweep.py
    python analysis/ridge_sweep/run_ridge_sweep.py --skip-sim
    python analysis/ridge_sweep/run_ridge_sweep.py --skip-sim --skip-eval
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import (
    load_position_fast, build_states, delayed_input, find_exp_dir,
    nrmse, nrmse2, mck_score, memory_capacity, write_params_used,
    ridge_gram_decomp, ridge_solve_gram,
    new_narma_dir, iter_narma_dirs, save_narma_params,
)
from generate_narma10 import generate_narma10 as _narma10_gen

BINARY = str(ROOT / "vicsek_dynamic")

# ── 実験設定 ─────────────────────────────────────────────────────────────────

LAMBDAS    = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TRAIN_NUMS = [2000, 4000, 6000, 8000, 10000]
WASHOUT    = 2000
SEEDS      = [1, 2, 3, 5, 6]
RCUT       = 13.0
SGM        = 0.0
K_MAX      = 100
N_FRAMES   = 22000   # WASHOUT + 2 * max(TRAIN_NUMS) = 2000 + 20000
NTIME      = N_FRAMES * 10  # utime=10 → 220000


# ── Phase 0: NARMA10 生成 ─────────────────────────────────────────────────

def ensure_narma10(seed: int, narma_root: Path, low=0.0, high=0.5, length=24000):
    """既存の日付 dir から NARMA を再利用し、無ければ専用の新しい日付 dir に生成する
    （1 時刻 dir = 1 データセット）。"""
    prefix   = f"{low}:{high}_seed{seed}_n{length}"
    in_name  = f"narma10_input_{prefix}.dat"
    tgt_name = f"narma10_target_{prefix}.dat"

    # narma_root 以下（日付 dir を新しい順、末尾に直下）から再利用
    for d in iter_narma_dirs(narma_root):
        input_path, target_path = d / in_name, d / tgt_name
        if input_path.exists() and target_path.exists():
            y = np.loadtxt(target_path)
            if not np.all(np.isfinite(y)):
                print(f"[NARMA10] WARN seed={seed}: existing file diverged — skip", flush=True)
                return None, None
            print(f"[NARMA10] Skip seed={seed} length={length} (exists) [{d.name}]", flush=True)
            return input_path, target_path

    # 無ければ seed 専用の新しい日付 dir に生成
    u, y = _narma10_gen(length, low, high, seed)
    if not np.all(np.isfinite(y)):
        n_bad = np.sum(~np.isfinite(y))
        print(f"[NARMA10] WARN seed={seed}: {n_bad} non-finite (diverged) — skip", flush=True)
        return None, None
    gen_dir = new_narma_dir(narma_root)
    input_path, target_path = gen_dir / in_name, gen_dir / tgt_name
    np.savetxt(input_path, u)
    np.savetxt(target_path, y)
    save_narma_params(gen_dir, in_name, tgt_name, length, seed, low, high)
    print(f"[NARMA10] Generated seed={seed} length={length} [{gen_dir.name}]", flush=True)
    return input_path, target_path


# ── Phase 1: シミュレーション ─────────────────────────────────────────────

def _run_one_sim(args):
    seed, input_path, data_group_dir = args
    cfg = {
        "input_file":  str(input_path),
        "output_base": str(data_group_dir),
        "rcut":        RCUT,
        "sgm":         SGM,
        "ntime":       NTIME,
        "seed_noise":  seed + 2,
        "seed_pos":    seed + 3,
        "seed_nf":     seed + 6,
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        result = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    if result.returncode != 0:
        print(f"[ERROR] seed={seed}: {result.stderr[:200]}", flush=True)
    else:
        print(f"[SIM done] seed={seed} ntime={NTIME}", flush=True)
    return seed, result.returncode


def phase1_simulate(valid_seeds, narma_paths, data_group_dir, n_jobs):
    data_group_dir = Path(data_group_dir)
    data_group_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, narma_paths[s][0], data_group_dir) for s in valid_seeds]
    print(f"Phase 1: {len(jobs)} simulations (ntime={NTIME}, rcut={RCUT}) …", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            seed, rc = f.result()
            if rc != 0:
                failed.append(seed)
    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
    return failed


# ── Phase 2: 評価（λ × train_num 2D 掃引） ────────────────────────────────

def _eval_one_seed(seed, data_group_dir, narma_paths, output_dir):
    cache_file = output_dir / f"results_seed={seed}.json"
    if cache_file.exists():
        print(f"[CACHE] seed={seed}", flush=True)
        with open(cache_file) as f:
            return json.load(f)

    # 実験ディレクトリを検索（新形式 → 旧形式フォールバック）
    exp_dir = find_exp_dir(data_group_dir, rcut=RCUT, seed=seed+3, seed_key="seed_pos")
    if exp_dir is None:
        exp_dir = find_exp_dir(data_group_dir, rcut=RCUT, seed=seed, seed_index=0)
    if exp_dir is None:
        print(f"[MISS] seed={seed}: no exp dir found", flush=True)
        return []

    pos_path    = exp_dir / "position.dat"
    params_file = exp_dir / "params_model.json"
    if not pos_path.exists() or not params_file.exists():
        print(f"[MISS] seed={seed}: files missing in {exp_dir}", flush=True)
        return []

    with open(params_file) as f:
        params = json.load(f)

    print(f"[EVAL] seed={seed}: loading position.dat …", flush=True)
    data   = load_position_fast(pos_path)
    N      = params["N"]
    states = build_states(data, N, params["utime"], pre_subsampled=True)   # (N_FRAMES, N+1)
    total_frames = states.shape[0]
    print(f"  states shape: {states.shape}", flush=True)

    input_path, target_path = narma_paths[seed]
    y_target  = np.loadtxt(target_path)[:total_frames]
    raw_input = np.loadtxt(input_path)
    col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input
    col_input = col_input[:total_frames]

    # 全遅延入力を事前に構築（K_MAX+1 × total_frames）
    all_del = np.stack([
        np.concatenate([np.zeros(k), col_input])[:total_frames]
        for k in range(K_MAX + 1)
    ], axis=0)   # (K+1, total_frames)

    rows = []
    for T in TRAIN_NUMS:
        tr_sl  = slice(WASHOUT, WASHOUT + T)
        te_sl  = slice(WASHOUT + T, WASHOUT + 2 * T)

        X_tr   = states[tr_sl]               # (T, N+1)
        y_tr   = y_target[tr_sl]
        y_te   = y_target[te_sl]
        threshold = N / T

        vals, vecs = ridge_gram_decomp(X_tr)

        # NARMA10 評価用 XtY
        XtY_narma = X_tr.T @ y_tr
        VtY_narma = vecs.T @ XtY_narma        # (N+1,)

        # MC 評価用: Y_del_tr (T × K+1) の XtY を一括計算
        Y_del_tr  = all_del[:, WASHOUT:WASHOUT+T].T   # (T, K+1)
        XtY_mc    = X_tr.T @ Y_del_tr                  # (N+1, K+1)
        VtY_mc    = vecs.T @ XtY_mc                    # (N+1, K+1)

        for lam in LAMBDAS:
            # NARMA10
            W_n   = vecs @ (VtY_narma / (vals + lam))
            pred_n = states @ W_n
            nr_tr  = nrmse( y_tr, pred_n[tr_sl])
            nr_te  = nrmse( y_te, pred_n[te_sl])
            nr2_tr = nrmse2(y_tr, pred_n[tr_sl])
            nr2_te = nrmse2(y_te, pred_n[te_sl])

            # MC: 全遅延の重みを一括計算
            W_mc_all   = vecs @ (VtY_mc / (vals[:, None] + lam))   # (N+1, K+1)
            pred_mc_all = states @ W_mc_all                          # (total, K+1)

            mc_tr_vals, mc_te_vals = [], []
            for k in range(K_MAX + 1):
                y_del_te = all_del[k, WASHOUT+T:WASHOUT+2*T]
                y_del_tr = all_del[k, tr_sl]
                mc_tr_vals.append(mck_score(y_del_tr,  pred_mc_all[tr_sl,  k]))
                mc_te_k = mck_score(y_del_te, pred_mc_all[te_sl, k])
                mc_te_vals.append(mc_te_k)
                if mc_te_k <= threshold:
                    break

            rows.append({
                "seed":        seed,
                "lambda":      lam,
                "train_num":   T,
                "test_num":    T,
                "nrmse_train": nr_tr,   "nrmse_test":   nr_te,
                "nrmse_gap":   nr_te - nr_tr,
                "nrmse2_train": nr2_tr, "nrmse2_test":  nr2_te,
                "mc_train":    memory_capacity(mc_tr_vals, threshold, cutoff_ref=mc_te_vals),
                "mc_test":     memory_capacity(mc_te_vals, threshold),
                "mc_gap":      (memory_capacity(mc_tr_vals, threshold, cutoff_ref=mc_te_vals)
                                - memory_capacity(mc_te_vals, threshold)),
            })

        print(f"  train_num={T} done", flush=True)

    with open(cache_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[EVAL done] seed={seed}", flush=True)
    return rows


def phase2_evaluate(valid_seeds, narma_paths, data_group_dir, output_dir):
    all_rows = []
    for seed in valid_seeds:
        rows = _eval_one_seed(seed, Path(data_group_dir), narma_paths, Path(output_dir))
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


# ── Phase 3: プロット ─────────────────────────────────────────────────────

def phase3_plot(df: pd.DataFrame, output_dir: Path):
    output_dir = Path(output_dir)

    # CSV 保存
    df.to_csv(output_dir / "ridge_sweep_data.csv", index=False)
    print("  ridge_sweep_data.csv", flush=True)

    lam_vals  = sorted(df["lambda"].unique())
    T_vals    = sorted(df["train_num"].unique())
    lam_log   = np.log10(lam_vals)

    # ── ヒートマップ（seed 平均） ─────────────────────────────────────────
    metrics_hm = [
        ("nrmse_test",  "NRMSE test",        "viridis_r"),
        ("nrmse_gap",   "NRMSE gap (te−tr)", "RdYlGn_r"),
        ("mc_test",     "MC test",            "viridis"),
        ("mc_gap",      "MC gap (tr−te)",    "RdYlGn_r"),
    ]
    fig_hm, axes_hm = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title, cmap) in zip(axes_hm.flat, metrics_hm):
        grid = np.full((len(T_vals), len(lam_vals)), np.nan)
        for i, T in enumerate(T_vals):
            for j, lam in enumerate(lam_vals):
                vals_ij = df[(df["train_num"] == T) & (df["lambda"] == lam)][col]
                if len(vals_ij) > 0:
                    grid[i, j] = vals_ij.mean()
        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(lam_vals)))
        ax.set_xticklabels([f"1e{int(l):+d}" if l == int(l) else f"{l:.0f}"
                            for l in lam_log], fontsize=7, rotation=45)
        ax.set_yticks(range(len(T_vals)))
        ax.set_yticklabels(T_vals)
        ax.set_xlabel("λ (log10)")
        ax.set_ylabel("train_num")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    fig_hm.suptitle(f"λ × train_num sweep (rcut={RCUT}, seed avg, {len(df['seed'].unique())} seeds)")
    fig_hm.tight_layout()
    fig_hm.savefig(output_dir / "heatmap.png", dpi=150)
    plt.close(fig_hm)
    print("  heatmap.png", flush=True)

    # ── 折れ線プロット（λ依存、各 train_num） ──────────────────────────
    metrics_line = [
        ("nrmse_test",  "nrmse_train",  "NRMSE",          "nrmse_vs_lambda.png"),
        ("mc_test",     "mc_train",     "MC",              "mc_vs_lambda.png"),
        ("nrmse_gap",   None,           "NRMSE gap (te−tr)", "nrmse_gap_vs_lambda.png"),
        ("mc_gap",      None,           "MC gap (tr−te)",  "mc_gap_vs_lambda.png"),
    ]
    for test_col, train_col, ylabel, fname in metrics_line:
        fig, ax = plt.subplots(figsize=(7, 4))
        for T in T_vals:
            sub = df[df["train_num"] == T].groupby("lambda")
            test_m  = sub[test_col].mean().values
            test_s  = sub[test_col].std(ddof=1).values if len(df["seed"].unique()) > 1 else np.zeros_like(test_m)
            lams = sorted(sub.groups.keys())
            ax.errorbar(range(len(lams)), test_m, yerr=test_s,
                        marker="o", label=f"T={T}", capsize=3)
            if train_col:
                train_m = sub[train_col].mean().values
                ax.plot(range(len(lams)), train_m, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(lam_vals)))
        ax.set_xticklabels([f"1e{int(l):+d}" if l == int(l) else f"{l:.0f}"
                            for l in lam_log], fontsize=8, rotation=45)
        ax.set_xlabel("λ")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        if train_col:
            ax.set_title(f"{ylabel}: solid=test, dashed=train")
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {fname}", flush=True)

    # 推奨設定の提示
    grp = df.groupby(["lambda", "train_num"])[["nrmse_test", "nrmse_gap"]].mean().reset_index()
    best = grp.loc[grp["nrmse_test"].idxmin()]
    print(f"\n[推奨] nrmse_test 最小: λ={best['lambda']:.1e}, "
          f"train_num={int(best['train_num'])}, "
          f"NRMSE_test={best['nrmse_test']:.4f}, gap={best['nrmse_gap']:.4f}", flush=True)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump({
            "best_lambda":    float(best["lambda"]),
            "best_train_num": int(best["train_num"]),
            "best_test_num":  int(best["train_num"]),
            "nrmse_test":     float(best["nrmse_test"]),
            "nrmse_gap":      float(best["nrmse_gap"]),
        }, f, indent=2)
    print("  best_params.json", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="λ × train_num 2D sweep for ridge regression")
    p.add_argument("--seeds",      nargs="+", type=int, default=SEEDS)
    p.add_argument("--rcut",       type=float, default=RCUT)
    p.add_argument("--n-jobs",     type=int, default=4)
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--narma-dir",  default="narma_data")
    p.add_argument("--output-dir", default="analysis/ridge_sweep")
    p.add_argument("--skip-sim",   action="store_true")
    p.add_argument("--skip-eval",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    narma_dir = Path(ROOT / args.narma_dir)

    # Phase 0: NARMA10 生成（生成する seed ごとに専用の日付 dir。1 dir = 1 データセット。既存は再利用）
    narma_paths = {s: ensure_narma10(s, narma_dir) for s in args.seeds}
    valid_seeds = [s for s, (ip, _) in narma_paths.items() if ip is not None]
    if not valid_seeds:
        print("[ERROR] No valid seeds.", flush=True)
        return

    # グループ dir（シミュレーション出力先）
    data_group_dir = Path(ROOT / args.data_dir) / f"{ts}_ridge_sweep"

    # Phase 1: シミュレーション（--skip-sim なければ実行）
    if not args.skip_sim:
        phase1_simulate(valid_seeds, narma_paths, data_group_dir, args.n_jobs)
    else:
        # 既存データを検索（--skip-sim 時は data/ 配下全体から rcut=13 を探す）
        data_group_dir = Path(ROOT / args.data_dir)
        print(f"[INFO] --skip-sim: searching {data_group_dir}", flush=True)

    # Phase 2: 評価
    if not args.skip_eval:
        df = phase2_evaluate(valid_seeds, narma_paths, data_group_dir, output_dir)
    else:
        csv_path = output_dir / "ridge_sweep_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            print("[ERROR] --skip-eval but no CSV found. Run evaluation first.", flush=True)
            return

    if df.empty:
        print("[WARN] No results.", flush=True)
        return

    # Phase 3: プロット
    phase3_plot(df, output_dir)

    # params_used.json
    params_list = []
    for seed in valid_seeds:
        exp_dir = find_exp_dir(data_group_dir, rcut=args.rcut, seed=seed+3, seed_key="seed_pos")
        if exp_dir is None:
            exp_dir = find_exp_dir(data_group_dir, rcut=args.rcut, seed=seed, seed_index=0)
        if exp_dir and (exp_dir / "params_model.json").exists():
            with open(exp_dir / "params_model.json") as f:
                params_list.append(json.load(f))

    if params_list:
        reservoir_swept = {
            "ridge_lambda": LAMBDAS,
            "train_num":    TRAIN_NUMS,
            "test_num":     TRAIN_NUMS,
        }
        reservoir_fixed = {
            "readout":  1,
            "washout":  WASHOUT,
            "k_max":    K_MAX,
        }
        write_params_used(output_dir, params_list, reservoir_fixed, reservoir_swept)
        print("  params_used.json", flush=True)

    print(f"\nOutput: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
