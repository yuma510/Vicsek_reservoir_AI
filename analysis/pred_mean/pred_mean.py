"""task_10: 予測値平均の性能（S 掃引）

各 noise 実現で個別に ridge 予測し、先頭 S 個の予測を平均したときの NRMSE_test / MC_test を
S=1..S_max で評価する。task_06（sgm_mean_state, 状態平均）と同じデータ・同じ S 掃引の
「予測平均」版で、直接比較できる形式（mc_nrmse_vs_S.png）で出力する。

data/ の既存 noise-averaged シム（rcut=13, sgm∈…, seed_pos=13/seed_nf=16 固定, noise seed 1..S_max）を
再利用する（新規シミュレーション不要）。評価パラメータは configs/default_reservoir_params.json から
取得する（CLAUDE.md §5.1、直書き禁止）。

実行例:
    python analysis/pred_mean/pred_mean.py \
        --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
        --noise-seeds 1 2 3 4 5 6 7 8 9 10 11 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
        --rcut 13 --data-dir data --output-dir analysis/pred_mean
"""
import argparse
import json
import sys
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
    apply_style, load_position_fast, build_states,
    ridge_gram_decomp,
    nrmse, nrmse2, mck_score,
    find_narma_by_seed, load_reservoir_defaults,
    write_params_used,
)

apply_style()
_RC = load_reservoir_defaults()


# ── noise-averaged 実験の索引 ────────────────────────────────────────────────

def build_index(data_dir, rcut, seed_pos, seed_nf):
    """data/ を 1 回走査し、(sgm, seed_noise) -> dir を返す。

    noise-averaged 集合を厳密に選ぶため **seed_pos・seed_nf を固定**で絞る
    （flat な data/ には rcut_sweep 等 seed_pos≠13 の同一 seed_noise 実験も混在するため、
    seed_key だけでは別実験を拾ってしまう）。重複時は新しい dir を採用。
    """
    data_dir = Path(data_dir)
    index = {}
    for d in sorted(data_dir.iterdir()):
        pf = d / "params_model.json"
        if not d.is_dir() or not pf.exists():
            continue
        try:
            with open(pf) as f:
                p = json.load(f)
        except Exception:
            continue
        if p.get("seed_pos") != seed_pos or p.get("seed_nf") != seed_nf:
            continue
        if rcut is not None and abs(float(p.get("rcut", -1)) - rcut) >= 1e-6:
            continue
        sgm = round(float(p.get("sgm", -1)), 4)
        nz  = p.get("seed_noise")
        if nz is None:
            continue
        index[(sgm, int(nz))] = d   # sorted 昇順なので新しい dir で上書き
    return index


# ── 1 実現の予測（全 k を Gram 分解の使い回しで計算） ───────────────────────────

def realization_predictions(exp_dir, params, y_target, col_input,
                            washout, train_num, k_max, ridge_lambda, n_eval):
    """1 実現の予測を返す: NARMA 予測 y_pred (n_eval,) と各遅延 k の入力予測 mck_preds (k_max+1, n_eval)。

    reservoir_aggregate.compute_predictions と同等だが、S 平均の一貫性のため
    早期打ち切りせず k=0..k_max すべてを計算する。Gram 固有分解を 1 回だけ行い、
    NARMA と全遅延の readout を 1 度の行列積でまとめて解く。
    """
    pos_path = exp_dir / "position.dat"
    if not pos_path.exists():
        return None
    data    = load_position_fast(pos_path)
    N       = params["N"]
    utime   = params["utime"]
    n_input = params["ntime"] // utime
    pre     = (data.shape[0] == N * n_input)
    states  = build_states(data, N, utime, pre_subsampled=pre)   # (T, N+1)
    length  = states.shape[0]

    train_end_frame = washout + train_num
    train_end = min(train_end_frame + 1, length, y_target.shape[0])
    X = states[washout:train_end]                                # (train, N+1)

    # 学習ターゲット行列 Y_train: 列 = [NARMA, delay0, delay1, …, delay k_max]
    targets = [y_target]
    for k in range(k_max + 1):
        targets.append(np.concatenate([np.zeros(k), col_input]))
    Y_train = np.column_stack([t[washout:train_end] for t in targets])   # (train, 1+k_max+1)

    # Ridge を Gram 固有分解でまとめて解く: W = V ((V^T X^T Y)/(vals+λ))
    vals, vecs = ridge_gram_decomp(X)
    VtXtY = vecs.T @ (X.T @ Y_train)                             # (N+1, M)
    W = vecs @ (VtXtY / (vals + ridge_lambda)[:, None])          # (N+1, M)
    preds = states @ W                                           # (T, M)

    preds = preds[:n_eval]
    return {
        "y_pred":    preds[:, 0].copy(),
        "mck_preds": preds[:, 1:].T.copy(),   # (k_max+1, n_eval)
    }


# ── S 掃引の評価 ────────────────────────────────────────────────────────────

def evaluate_sgm(sgm, seeds, index, y_target, col_input,
                 washout, train_num, k_max, ridge_lambda, n_eval, N):
    """1 つの sgm について、S=1..（実現数）の予測平均性能を返す dict[S] = metrics。"""
    train_end_frame = washout + train_num
    threshold = N / train_num

    tr_s = slice(washout, min(train_end_frame + 1, n_eval))
    te_s = slice(min(train_end_frame, n_eval), n_eval)
    # 遅延ターゲット（評価用、n_eval にクリップ）
    delayed = [np.concatenate([np.zeros(k), col_input])[:n_eval] for k in range(k_max + 1)]

    y_sum   = np.zeros(n_eval)
    mck_sum = np.zeros((k_max + 1, n_eval))
    results = {}
    S = 0
    for seed in sorted(seeds):
        exp_dir = index.get((round(float(sgm), 4), seed))
        if exp_dir is None:
            print(f"    [MISS] sgm={sgm} seed_noise={seed}", flush=True)
            continue
        with open(exp_dir / "params_model.json") as f:
            params = json.load(f)
        rp = realization_predictions(exp_dir, params, y_target, col_input,
                                     washout, train_num, k_max, ridge_lambda, n_eval)
        if rp is None:
            print(f"    [MISS] no position.dat: {exp_dir.name}", flush=True)
            continue

        S += 1
        y_sum   += rp["y_pred"]
        mck_sum += rp["mck_preds"]
        yhat = y_sum / S
        mhat = mck_sum / S

        nr_tr  = nrmse( y_target[tr_s], yhat[tr_s])
        nr_te  = nrmse( y_target[te_s], yhat[te_s])
        nr2_tr = nrmse2(y_target[tr_s], yhat[tr_s])
        nr2_te = nrmse2(y_target[te_s], yhat[te_s])

        mc_tr_list, mc_te_list = [], []
        for k in range(k_max + 1):
            mc_tr_list.append(mck_score(delayed[k][tr_s], mhat[k][tr_s]))
            mc_te = mck_score(delayed[k][te_s], mhat[k][te_s])
            mc_te_list.append(mc_te)
            if mc_te <= threshold:   # 記憶容量の打ち切り（evaluate_reservoir と同じ）
                break

        results[S] = {
            "nrmse_train": float(nr_tr),  "nrmse_test":  float(nr_te),
            "nrmse2_train": float(nr2_tr), "nrmse2_test": float(nr2_te),
            "MC_train": float(np.sum(mc_tr_list)),
            "MC_test":  float(np.sum(mc_te_list)),
        }
        print(f"    S={S:2d} (seed={seed:2d})  MC_test={results[S]['MC_test']:.3f}  "
              f"NRMSE_test={results[S]['nrmse_test']:.4f}", flush=True)

    return results


# ── プロット ───────────────────────────────────────────────────────────────

def plot_results(all_results, output_dir):
    output_dir = Path(output_dir)

    cols = ["MC_test", "MC_train", "nrmse_test", "nrmse_train", "nrmse2_test", "nrmse2_train"]
    rows = [
        {"sgm": sgm, "S": S, **{c: res.get(c) for c in cols}}
        for sgm, by_S in sorted(all_results.items())
        for S, res in sorted(by_S.items())
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pred_mean_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for sgm, by_S in sorted(all_results.items()):
        xs = sorted(by_S)
        mc = [by_S[s]["MC_test"]    for s in xs]
        nr = [by_S[s]["nrmse_test"] for s in xs]
        axes[0].plot(xs, mc, marker="o", label=f"sgm={sgm:.1f}")
        axes[1].plot(xs, nr, marker="o", label=f"sgm={sgm:.1f}")
    axes[0].set(xlabel="S (# averaged predictions)", ylabel="MC (test)")
    axes[1].set(xlabel="S (# averaged predictions)", ylabel="NRMSE (test)")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "mc_nrmse_vs_S.png", dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {output_dir}/", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="prediction-averaged reservoir performance (S sweep)")
    p.add_argument("--sgm-values", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], metavar="S")
    p.add_argument("--noise-seeds", nargs="+", type=int,
                   default=list(range(1, 11)), metavar="NS")
    p.add_argument("--rcut", type=float, default=13)
    p.add_argument("--seed-pos", type=int, default=13, help="固定した初期位置 seed（noise-avg 集合の識別）")
    p.add_argument("--seed-nf",  type=int, default=16, help="固定した自然振動数 seed（noise-avg 集合の識別）")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--output-dir", default="analysis/pred_mean")
    p.add_argument("--narma-root", default="narma_data")
    p.add_argument("--narma-seed", type=int, default=666)
    p.add_argument("--washout",      type=int,   default=_RC["washout"])
    p.add_argument("--train-num",    type=int,   default=_RC["train_num"])
    p.add_argument("--ridge-lambda", type=float, default=_RC["ridge_lambda"])
    p.add_argument("--k-max",        type=int,   default=_RC["k_max"])
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # NARMA10 解決
    input_path, target_path = find_narma_by_seed(args.narma_root, args.narma_seed)
    if input_path is None:
        sys.exit(f"NARMA10 (seed={args.narma_seed}) が {args.narma_root}/ に見つからない。"
                 "generate_narma10.py で生成してください。")
    y_target  = np.loadtxt(target_path)
    raw_input = np.loadtxt(input_path)
    col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input

    # noise-averaged 集合の索引を 1 回の走査で構築（seed_pos/seed_nf 固定で厳密に絞る）
    index = build_index(data_dir, args.rcut, args.seed_pos, args.seed_nf)
    print(f"index: {len(index)} 実験 (rcut={args.rcut}, seed_pos={args.seed_pos}, seed_nf={args.seed_nf})",
          flush=True)

    all_results = {}
    model_params = []
    N = None
    n_eval = None

    for sgm in args.sgm_values:
        # 評価長 n_eval は最初に見つかった実験の n_input と NARMA 長の最小値
        if n_eval is None:
            probe = index.get((round(float(sgm), 4), sorted(args.noise_seeds)[0]))
            if probe is None:
                print(f"[WARN] sgm={sgm}: 実験が見つからない", flush=True)
                continue
            with open(probe / "params_model.json") as f:
                pp = json.load(f)
            N = pp["N"]
            n_input = pp["ntime"] // pp["utime"]
            n_eval = int(min(n_input, y_target.shape[0], col_input.shape[0]))
            print(f"n_eval={n_eval} (n_input={n_input}, NARMA len={y_target.shape[0]})", flush=True)

        cache = output_dir / f"results_sgm={sgm:.2f}.json"
        print(f"\nsgm={sgm:.2f} …", flush=True)
        res = evaluate_sgm(sgm, args.noise_seeds, index,
                           y_target, col_input, args.washout, args.train_num,
                           args.k_max, args.ridge_lambda, n_eval, N)
        if not res:
            print(f"[WARN] sgm={sgm}: 有効な実現なし", flush=True)
            continue
        all_results[sgm] = res
        with open(cache, "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f, indent=2)

        # params_used 用にモデル params を収集
        for seed in sorted(args.noise_seeds):
            ed = index.get((round(float(sgm), 4), seed))
            if ed and (ed / "params_model.json").exists():
                with open(ed / "params_model.json") as f:
                    model_params.append(json.load(f))

    if not all_results:
        sys.exit("有効な結果がありません。")

    plot_results(all_results, output_dir)

    # params_used.json（CLAUDE.md §5.1）
    reservoir_fixed = {
        "readout": _RC["readout"], "washout": args.washout, "train_num": args.train_num,
        "ridge_lambda": args.ridge_lambda, "k_max": args.k_max,
        "narma_seed": args.narma_seed, "n_eval": n_eval,
    }
    s_max = max((max(r) for r in all_results.values()), default=0)
    reservoir_swept = {"S": list(range(1, s_max + 1))}
    if model_params:
        write_params_used(output_dir, model_params, reservoir_fixed, reservoir_swept)

    print(f"\nOutput: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
