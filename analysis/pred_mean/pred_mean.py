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
    nrmse, nrmse2, mck_score, memory_capacity,
    find_narma_by_seed, load_reservoir_defaults,
    write_params_used,
)

apply_style()
_RC = load_reservoir_defaults()
_AP = json.loads((Path(__file__).parent / "default_params.json").read_text())


# ── noise-averaged 実験の索引 ────────────────────────────────────────────────

def build_index(data_dir, rcut, seed_pos, seed_nf, v0=None, narma_seed=None):
    """data/ を 1 回走査し、(sgm, seed_noise) -> dir を返す。

    noise-averaged 集合を厳密に選ぶため **seed_pos・seed_nf を固定**で絞る
    （flat な data/ には rcut_sweep 等 seed_pos≠13 の同一 seed_noise 実験も混在するため、
    seed_key だけでは別実験を拾ってしまう）。v0 も指定して v0=0 の実験の混入を防ぐ。
    narma_seed を指定すると、params_model.json の input_file が別 NARMA seed の実験を除外する
    （input_file 未記録の旧シムは除外しない）。重複時は新しい dir を採用。
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
        if v0 is not None and abs(float(p.get("v0", -1)) - v0) >= 1e-6:
            continue
        inf = p.get("input_file")
        if narma_seed is not None and inf and f"seed{narma_seed}.dat" not in inf:
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

def _metrics_from_avg(yhat, mhat, y_target, delayed_mat, tr_s, te_s, threshold):
    """平均予測 (yhat, mhat) から NRMSE / MC を計算する。

    MC はベクトル化して全遅延の相関を一括計算し、旧ループと同じ規則
    「テスト側 MC_k が最初に閾値以下になった k まで（その項を含めて）総和」を適用する。
    """
    nr_tr  = nrmse( y_target[tr_s], yhat[tr_s])
    nr_te  = nrmse( y_target[te_s], yhat[te_s])
    nr2_tr = nrmse2(y_target[tr_s], yhat[tr_s])
    nr2_te = nrmse2(y_target[te_s], yhat[te_s])

    def _corr2_rows(T, P):
        Tc = T - T.mean(axis=1, keepdims=True)
        Pc = P - P.mean(axis=1, keepdims=True)
        num = (Tc * Pc).sum(axis=1)
        den = np.sqrt((Tc ** 2).sum(axis=1) * (Pc ** 2).sum(axis=1))
        return (num / den) ** 2

    mc_tr = _corr2_rows(delayed_mat[:, tr_s], mhat[:, tr_s])
    mc_te = _corr2_rows(delayed_mat[:, te_s], mhat[:, te_s])
    return {
        "nrmse_train": float(nr_tr),  "nrmse_test":  float(nr_te),
        "nrmse2_train": float(nr2_tr), "nrmse2_test": float(nr2_te),
        "MC_train": memory_capacity(mc_tr, threshold, cutoff_ref=mc_te),
        "MC_test":  memory_capacity(mc_te, threshold),
    }


def load_predictions(sgm, seeds, index, y_target, col_input,
                     washout, train_num, k_max, ridge_lambda, n_eval, N):
    """1 つの sgm の全実現の予測を計算して返す（S 掃引・順列評価で再利用）。

    Returns:
        seed_list: 実際に見つかった seed_noise のリスト（昇順）
        y_preds:   (n, n_eval)          NARMA 予測
        mck_preds: (n, k_max+1, n_eval) 遅延入力予測
        per_seed:  dict[seed_noise] = metrics（各 seed 単独）
    """
    train_end_frame = washout + train_num
    threshold = N / train_num
    tr_s = slice(washout, min(train_end_frame + 1, n_eval))
    te_s = slice(min(train_end_frame, n_eval), n_eval)
    delayed_mat = np.vstack([np.concatenate([np.zeros(k), col_input])[:n_eval]
                             for k in range(k_max + 1)])

    seed_list, y_list, mck_list = [], [], []
    per_seed = {}
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
        seed_list.append(seed)
        y_list.append(rp["y_pred"])
        mck_list.append(rp["mck_preds"])
        per_seed[seed] = _metrics_from_avg(rp["y_pred"], rp["mck_preds"],
                                           y_target, delayed_mat, tr_s, te_s, threshold)
        if len(seed_list) % 10 == 0:
            print(f"    loaded {len(seed_list)} realizations …", flush=True)

    y_preds   = np.array(y_list)
    mck_preds = np.array(mck_list)
    return seed_list, y_preds, mck_preds, per_seed


def evaluate_cumulative(order, y_preds, mck_preds, y_target, col_input,
                        washout, train_num, k_max, n_eval, N, verbose=True):
    """指定した順序 order（インデックス列）で累積 S 平均を評価する。

    order=range(n)（昇順）が従来の evaluate_sgm と同じ結果を与える。
    Returns: dict[S] = metrics
    """
    train_end_frame = washout + train_num
    threshold = N / train_num
    tr_s = slice(washout, min(train_end_frame + 1, n_eval))
    te_s = slice(min(train_end_frame, n_eval), n_eval)
    delayed_mat = np.vstack([np.concatenate([np.zeros(k), col_input])[:n_eval]
                             for k in range(k_max + 1)])

    y_sum   = np.zeros(n_eval)
    mck_sum = np.zeros((k_max + 1, n_eval))
    results = {}
    for S, idx in enumerate(order, start=1):
        y_sum   += y_preds[idx]
        mck_sum += mck_preds[idx]
        results[S] = _metrics_from_avg(y_sum / S, mck_sum / S,
                                       y_target, delayed_mat, tr_s, te_s, threshold)
        if verbose:
            print(f"    S={S:3d}  MC_test={results[S]['MC_test']:.3f}  "
                  f"NRMSE_test={results[S]['nrmse_test']:.4f}", flush=True)
    return results


def evaluate_orderings(y_preds, mck_preds, n_orderings, ordering_seed,
                       y_target, col_input, washout, train_num, k_max, n_eval, N):
    """R 通りのランダム順列で S 掃引を繰り返し、S ごとの平均・標準偏差を返す。

    非復元サンプリング（順列の先頭 S 個）なので S=n では全順列が同一集合になり std=0。
    Returns: dict[S] = {"MC_mean", "MC_std", "nrmse_mean", "nrmse_std"}
    """
    n = y_preds.shape[0]
    rng = np.random.default_rng(ordering_seed)
    mc_all = np.zeros((n_orderings, n))
    nr_all = np.zeros((n_orderings, n))
    for r in range(n_orderings):
        perm = rng.permutation(n)
        res = evaluate_cumulative(perm, y_preds, mck_preds, y_target, col_input,
                                  washout, train_num, k_max, n_eval, N, verbose=False)
        mc_all[r] = [res[S]["MC_test"]    for S in range(1, n + 1)]
        nr_all[r] = [res[S]["nrmse_test"] for S in range(1, n + 1)]
        print(f"    ordering {r + 1}/{n_orderings} done", flush=True)

    return {S: {"MC_mean":    float(mc_all[:, S - 1].mean()),
                "MC_std":     float(mc_all[:, S - 1].std(ddof=1)) if n_orderings > 1 else 0.0,
                "nrmse_mean": float(nr_all[:, S - 1].mean()),
                "nrmse_std":  float(nr_all[:, S - 1].std(ddof=1)) if n_orderings > 1 else 0.0}
            for S in range(1, n + 1)}


# ── プロット ───────────────────────────────────────────────────────────────

def plot_results(all_results, output_dir, all_per_seed=None):
    output_dir = Path(output_dir)

    cols = ["MC_test", "MC_train", "nrmse_test", "nrmse_train", "nrmse2_test", "nrmse2_train"]
    rows = [
        {"sgm": sgm, "S": S, **{c: res.get(c) for c in cols}}
        for sgm, by_S in sorted(all_results.items())
        for S, res in sorted(by_S.items())
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pred_mean_data.csv", index=False)

    if all_per_seed:
        rows_indiv = [
            {"sgm": sgm, "seed_noise": seed, **{c: metrics.get(c) for c in cols}}
            for sgm, by_seed in sorted(all_per_seed.items())
            for seed, metrics in sorted(by_seed.items())
        ]
        pd.DataFrame(rows_indiv).to_csv(output_dir / "pred_individual_data.csv", index=False)
        print(f"  pred_individual_data.csv ({len(rows_indiv)} rows)")

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


def plot_stats(all_stats, n_orderings, output_dir):
    """ランダム順列 R 通りの S 掃引の平均±標準偏差（帯）をプロットし CSV 保存する。"""
    output_dir = Path(output_dir)
    rows = [
        {"sgm": sgm, "S": S, **st}
        for sgm, by_S in sorted(all_stats.items())
        for S, st in sorted(by_S.items())
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pred_mean_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for sgm, by_S in sorted(all_stats.items()):
        xs = sorted(by_S)
        mc_m = np.array([by_S[s]["MC_mean"]    for s in xs])
        mc_s = np.array([by_S[s]["MC_std"]     for s in xs])
        nr_m = np.array([by_S[s]["nrmse_mean"] for s in xs])
        nr_s = np.array([by_S[s]["nrmse_std"]  for s in xs])
        line, = axes[0].plot(xs, mc_m, label=f"sgm={sgm:.1f}")
        axes[0].fill_between(xs, mc_m - mc_s, mc_m + mc_s, color=line.get_color(), alpha=0.25)
        axes[1].plot(xs, nr_m, color=line.get_color(), label=f"sgm={sgm:.1f}")
        axes[1].fill_between(xs, nr_m - nr_s, nr_m + nr_s, color=line.get_color(), alpha=0.25)
    axes[0].set(xlabel="S (# averaged predictions)", ylabel="MC (test)")
    axes[1].set(xlabel="S (# averaged predictions)", ylabel="NRMSE (test)")
    for ax in axes:
        ax.legend(title=f"mean ± std ({n_orderings} orderings)")
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "mc_nrmse_vs_S_errorbar.png", dpi=150)
    plt.close(fig)
    print(f"  mc_nrmse_vs_S_errorbar.png / pred_mean_stats.csv", flush=True)


def plot_baseline_nolegend(all_results, output_dir, baseline_sgm=0.0, ymax=None):
    """発表用: 折れ線・凡例なし。sgm=baseline_sgm は水平ベースライン（黒破線）で描く。

    sgm=0 はノイズなしで S に依らず一定なので、他 sgm 曲線を比較する基準線として
    全 S 域に水平線で引く。マーカーなしの折れ線。縦軸は 0 起点。
    ymax: メトリクス名（"MC"/"nrmse"）→ 縦軸上限の dict（複数図で揃えたいとき）。
    """
    output_dir = Path(output_dir)
    ymax = ymax or {}
    for key, ylabel, fname, mkey in [
        ("MC_test",    "MC (test)",    "mc_vs_S_baseline_nolegend.png",    "MC"),
        ("nrmse_test", "NRMSE (test)", "nrmse_vs_S_baseline_nolegend.png", "nrmse"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for sgm, by_S in sorted(all_results.items()):
            xs = sorted(by_S)
            ys = [by_S[s][key] for s in xs]
            if abs(sgm - baseline_sgm) < 1e-9:
                ax.axhline(ys[0], color="k", ls="--", lw=1.5)   # baseline（一定値）
            else:
                ax.plot(xs, ys, lw=1.8)
        ax.set_xlabel("S (# averaged predictions)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax.get(mkey))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {fname}", flush=True)


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
    p.add_argument("--v0", type=float, default=0.5, help="v0 フィルタ（v0=0 実験の混入防止）")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--output-dir", default="analysis/pred_mean")
    p.add_argument("--narma-root", default="narma_data")
    p.add_argument("--narma-seed", type=int, default=666)
    p.add_argument("--washout",      type=int,   default=_RC["washout"])
    p.add_argument("--train-num",    type=int,   default=_RC["train_num"])
    p.add_argument("--ridge-lambda", type=float, default=_RC["ridge_lambda"])
    p.add_argument("--k-max",        type=int,   default=_RC["k_max"])
    p.add_argument("--n-orderings",  type=int,   default=_AP["n_orderings"],
                   help="ランダム順列の本数 R（0 で従来の昇順1通りのみ）")
    p.add_argument("--ordering-seed", type=int,  default=_AP["ordering_seed"],
                   help="順列生成の乱数 seed（sgm 間で同じ順列集合を使う）")
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
    index = build_index(data_dir, args.rcut, args.seed_pos, args.seed_nf, args.v0,
                        narma_seed=args.narma_seed)
    print(f"index: {len(index)} 実験 (rcut={args.rcut}, seed_pos={args.seed_pos}, "
          f"seed_nf={args.seed_nf}, v0={args.v0}, narma_seed={args.narma_seed})", flush=True)

    all_results  = {}
    all_per_seed = {}
    all_stats    = {}
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

        print(f"\nsgm={sgm:.2f} …", flush=True)
        seed_list, y_preds, mck_preds, per_seed = load_predictions(
            sgm, args.noise_seeds, index, y_target, col_input,
            args.washout, args.train_num, args.k_max, args.ridge_lambda, n_eval, N)
        if not seed_list:
            print(f"[WARN] sgm={sgm}: 有効な実現なし", flush=True)
            continue

        # 従来の昇順 1 通りの S 掃引（既存出力と互換）
        res = evaluate_cumulative(range(len(seed_list)), y_preds, mck_preds,
                                  y_target, col_input, args.washout, args.train_num,
                                  args.k_max, n_eval, N)
        all_results[sgm]  = res
        all_per_seed[sgm] = per_seed

        # ランダム順列 R 通りの平均±標準偏差
        if args.n_orderings > 0:
            print(f"    orderings: R={args.n_orderings} …", flush=True)
            all_stats[sgm] = evaluate_orderings(
                y_preds, mck_preds, args.n_orderings, args.ordering_seed,
                y_target, col_input, args.washout, args.train_num,
                args.k_max, n_eval, N)
        del y_preds, mck_preds

        # params_used 用にモデル params を収集
        for seed in sorted(args.noise_seeds):
            ed = index.get((round(float(sgm), 4), seed))
            if ed and (ed / "params_model.json").exists():
                with open(ed / "params_model.json") as f:
                    model_params.append(json.load(f))

    if not all_results:
        sys.exit("有効な結果がありません。")

    plot_results(all_results, output_dir, all_per_seed)
    plot_baseline_nolegend(all_results, output_dir)
    if all_stats:
        plot_stats(all_stats, args.n_orderings, output_dir)

    # params_used.json（CLAUDE.md §5.1）
    reservoir_fixed = {
        "readout": _RC["readout"], "washout": args.washout, "train_num": args.train_num,
        "ridge_lambda": args.ridge_lambda, "k_max": args.k_max,
        "narma_seed": args.narma_seed, "n_eval": n_eval,
        "n_orderings": args.n_orderings, "ordering_seed": args.ordering_seed,
    }
    s_max = max((max(r) for r in all_results.values()), default=0)
    reservoir_swept = {"S": list(range(1, s_max + 1))}
    if model_params:
        write_params_used(output_dir, model_params, reservoir_fixed, reservoir_swept)

    print(f"\nOutput: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
