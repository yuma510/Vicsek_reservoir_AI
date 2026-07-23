"""Task 08: Higher-order readout analysis.

拡張状態 X_ext = [1, sinθ_1..N, sinθ_i×sinθ_j × P ランダムペア] で
MC/NRMSE を評価し、rcut×sgm の 2D ヒートマップを v0 別に出力する。
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
    find_narma_by_seed, load_reservoir_defaults, write_params_used,
)
from vicsek_rc.loaders import load_position_fast, build_states
from vicsek_rc.metrics import mck_score, nrmse, memory_capacity
from vicsek_rc.plotting import apply_style

_RC = load_reservoir_defaults()
_CA = json.loads((Path(__file__).parent / "default_params.json").read_text())

RESERVOIR_DEFAULTS = {
    "readout":      _RC["readout"],
    "washout":      _RC["washout"],
    "train_num":    _RC["train_num"],
    "ridge_lambda": _RC["ridge_lambda"],
    "k_max":        _RC["k_max"],
}

# base(P=0) MC がこの値未満なら、駆動入力の取り違え等でリザバーが入力を
# 全く記憶していない退化状態とみなし、その sim を集計から除外する。
# 正常時は最小でも O(1)（例 rcut=13 sgm>0 で ~1.1）なのに対し、不一致時は ~1e-4。
MC_DEGENERATE_THRESHOLD = 0.1


# ── データ探索 ────────────────────────────────────────────────────────────────

def _scan_sims(data_dir: Path, v0: float, rcut: float, sgm: float,
               seed_pos_keep=None):
    """(seed_pos, sim_dir) リストを返す（同一条件の seed を収集）。

    seed_pos_keep（集合/リスト）を渡すと、その seed_pos のセルのみ採用し、
    全 (rcut,sgm) セルの平均 seed 数を揃える。
    """
    keep = set(seed_pos_keep) if seed_pos_keep is not None else None
    results = {}  # seed_pos → newest dir
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        pf = d / "params_model.json"
        if not pf.exists():
            continue
        with open(pf) as f:
            p = json.load(f)
        if abs(p.get("rcut", -1) - rcut) >= 1e-6:
            continue
        if abs(p.get("sgm", -1) - sgm) >= 1e-6:
            continue
        if abs(p.get("v0", -1) - v0) >= 1e-6:
            continue
        seed_pos = p.get("seed_pos")
        if seed_pos is None:
            continue
        if keep is not None and seed_pos not in keep:
            continue
        if not (d / "position.dat").exists():
            continue
        results[seed_pos] = d  # 最新 dir で上書き
    return list(results.items())


def _load_narma(sim_dir: Path, params: dict, narma_root: Path):
    """シムに使われた NARMA 入力・ターゲットを返す。

    1. params["input_file"] があればそのパスから読む（新フォーマット）
    2. sgm==0 → seed_pos - 3（rcut_sweep 慣習）
    3. sgm>0  → narma_seed=666（sgm_sweep 慣習）
    """
    n_frames = int(params["ntime"]) // int(params["utime"])
    sgm = float(params.get("sgm", 0.0))
    seed_pos = params.get("seed_pos")

    # 1. input_file フィールドあり: u と y を必ず同一ファイル対から取る
    #    （target は input のファイル名 input→target 置換で一意に決まる）
    if "input_file" in params:
        ip = Path(params["input_file"])
        if not ip.is_absolute():
            ip = ROOT / ip
        if ip.exists():
            tp = ip.with_name(ip.name.replace("narma10_input_", "narma10_target_", 1))
            if tp.exists():
                u = np.loadtxt(ip)[:n_frames]
                y = np.loadtxt(tp)[:n_frames]
                return u, y

    # 2/3. ヒューリスティック
    narma_seed = _infer_narma_seed(params, sgm)
    ip, tp = find_narma_by_seed(narma_root, narma_seed)
    if ip is None or tp is None:
        raise FileNotFoundError(
            f"NARMA seed={narma_seed} が {narma_root} に見つかりません。"
        )
    u = np.loadtxt(ip)[:n_frames]
    y = np.loadtxt(tp)[:n_frames]
    return u, y


def _infer_narma_seed(params: dict, sgm: float) -> int:
    if sgm < 1e-6:
        seed_pos = params.get("seed_pos")
        if seed_pos is not None and seed_pos >= 3:
            return seed_pos - 3
    return 666


# ── 状態構築 ──────────────────────────────────────────────────────────────────

def _build_ext(X_base: np.ndarray, n_pairs: int, pair_seed: int, N: int) -> np.ndarray:
    """X_base (T, N+1) → X_ext (T, N+1+P)。バイアス列を保持したまま積項を追加。"""
    rng = np.random.default_rng(pair_seed)
    i_arr = rng.integers(0, N, size=n_pairs)
    j_arr = rng.integers(0, N, size=n_pairs)
    sin_states = X_base[:, 1:]  # (T, N)
    pair_feats = sin_states[:, i_arr] * sin_states[:, j_arr]  # (T, P)
    return np.hstack([X_base, pair_feats])


# ── 評価 ─────────────────────────────────────────────────────────────────────

def _evaluate_ext(X: np.ndarray, u_full: np.ndarray, y_full: np.ndarray,
                  washout: int, train_num: int, ridge_lambda: float,
                  k_max: int, N: int):
    """拡張状態 X で NRMSE_test と MC_test を評価する。

    evaluate_reservoir と同じスライス慣習:
      train: [washout, washout+train_num+1)
      test:  [washout+train_num, n_frames)
    M_inv を一度だけ計算（O(D³)）し、MC ループは O(D²) で高速化。
    """
    n_frames = X.shape[0]
    D = X.shape[1]
    train_end = washout + train_num

    X_tr = X[washout:train_end + 1]
    y_tr = y_full[washout:train_end + 1]

    # M_inv を前計算（NARMA 予測・MC で共有）
    M = X_tr.T @ X_tr + ridge_lambda * np.eye(D)
    M_inv = np.linalg.inv(M)

    # NRMSE
    w_narma = M_inv @ (X_tr.T @ y_tr)
    te_slice = slice(train_end, n_frames)
    y_pred_te = X[te_slice] @ w_narma
    y_te = y_full[te_slice]
    nrmse_test = nrmse(y_te, y_pred_te)

    # MC
    threshold = N / train_num
    mc_list = []
    for delay in range(k_max + 1):
        padded = np.concatenate([np.zeros(delay), u_full])[:n_frames]
        u_tr = padded[washout:train_end + 1]
        w_k = M_inv @ (X_tr.T @ u_tr)
        mc_k = mck_score(padded[te_slice], X[te_slice] @ w_k)
        mc_list.append(mc_k)
        if mc_k <= threshold:
            break

    return float(nrmse_test), memory_capacity(mc_list, threshold)


# ── プロット ──────────────────────────────────────────────────────────────────

def _plot_metric(agg, metric, metric_label, cmap, sgm_vals, rcut_vals,
                 v0_labels, out_path):
    """1 指標を v0 パネル横並び（1×len(v0)）の 1 図として保存する。"""
    n_sgm = len(sgm_vals)
    n_rcut = len(rcut_vals)

    fig, axes = plt.subplots(1, len(v0_labels),
                             figsize=(4 * len(v0_labels), 3.2),
                             squeeze=False)
    for col, v0 in enumerate(v0_labels):
        ax = axes[0][col]
        sub = agg[np.abs(agg["v0"] - v0) < 1e-6]

        mat = np.full((n_rcut, n_sgm), np.nan)
        for ri, rc in enumerate(rcut_vals):
            for si, sg in enumerate(sgm_vals):
                row_data = sub[
                    (np.abs(sub["rcut"] - rc) < 1e-6) &
                    (np.abs(sub["sgm"]  - sg) < 1e-6)
                ]
                if not row_data.empty:
                    mat[ri, si] = float(row_data[metric].iloc[0])

        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_sgm))
        ax.set_xticklabels([f"{s:.1f}" for s in sgm_vals])
        ax.set_yticks(range(n_rcut))
        ax.set_yticklabels([str(int(r)) for r in rcut_vals])
        ax.set_xlabel("sgm")
        ax.set_ylabel("rcut")
        ax.set_title(f"v0={v0:.1f}  {metric_label}")

        for ri in range(n_rcut):
            for si in range(n_sgm):
                v = mat[ri, si]
                if not np.isnan(v):
                    ax.text(si, ri, f"{v:.3f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle(f"Higher-order readout (P=500) — {metric_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}", flush=True)


def _make_heatmap(df: pd.DataFrame, sgm_values, rcut_values, v0_values, out_dir: Path):
    """MC・NRMSE を別画像で保存する（各図 v0 パネル横並び、rcut×sgm）。"""
    apply_style()

    sgm_vals = sorted(sgm_values)
    rcut_vals = sorted(rcut_values)

    # P=500 のみ使用。seed_pos × pair_seed 平均
    df500 = df[df["n_pairs"] == 500]
    agg = (df500.groupby(["v0", "rcut", "sgm"])[["mc_test", "nrmse_test"]]
               .mean().reset_index())

    v0_labels = sorted(v0_values, reverse=True)  # 左パネルが大きい v0

    _plot_metric(agg, "mc_test", "MC_test", "Blues",
                 sgm_vals, rcut_vals, v0_labels, out_dir / "heatmap_mc.png")
    _plot_metric(agg, "nrmse_test", "NRMSE_test", "Reds_r",
                 sgm_vals, rcut_vals, v0_labels, out_dir / "heatmap_nrmse.png")


# ── メイン ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Higher-order readout: P=500 random pair analysis")
    p.add_argument("--data-dir", default="data",
                   help="シム dir 親ディレクトリ")
    p.add_argument("--v0-values", nargs="+", type=float, default=_CA.get("v0_values", [0.5]),
                   metavar="V", help="評価する v0 値（複数可、default: 0.5）")
    p.add_argument("--rcut-values", nargs="+", type=float, default=_CA.get("rcut_values", [1.0, 5.0, 13.0]),
                   metavar="R", help="評価する rcut 値（default: 1 5 13）")
    p.add_argument("--sgm-values", nargs="+", type=float,
                   default=_CA.get("sgm_values", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                   metavar="S", help="評価する sgm 値（default: default_params.json）")
    p.add_argument("--seed-pos", nargs="+", type=int,
                   default=_CA.get("seed_pos_keep"),
                   metavar="SP", help="平均に使う seed_pos 集合（default: default_params.json）。"
                                       "未指定なら全 seed_pos を使用")
    p.add_argument("--n-pairs", type=int, default=_CA["n_pairs"],
                   help=f"ランダムペア数 P（default: {_CA['n_pairs']}）")
    p.add_argument("--n-pair-seeds", type=int, default=_CA["n_pair_seeds"],
                   help=f"pair_seed の試行数（default: {_CA['n_pair_seeds']}）")
    p.add_argument("--narma-root", default="narma_data",
                   help="NARMA10 探索ルート（default: narma_data）")
    p.add_argument("--output-dir", default="analysis/higher_order_readout",
                   help="出力先（タイムスタンプサブ dir を自動生成）")
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

    rows = []
    model_params_list = []

    combos = [(v0, rcut, sgm)
              for v0   in args.v0_values
              for rcut in args.rcut_values
              for sgm  in args.sgm_values]

    for v0, rcut, sgm in combos:
        sim_list = _scan_sims(data_dir, v0, rcut, sgm, seed_pos_keep=args.seed_pos)
        if not sim_list:
            print(f"[MISS] v0={v0} rcut={rcut} sgm={sgm}: no sim found", flush=True)
            continue

        for seed_pos, sim_dir in sim_list:
            print(f"[EVAL] v0={v0} rcut={rcut} sgm={sgm} seed_pos={seed_pos} …", flush=True)

            with open(sim_dir / "params_model.json") as f:
                params = json.load(f)
            model_params_list.append(params)

            try:
                u, y = _load_narma(sim_dir, params, narma_root)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}", flush=True)
                continue

            pos_path = sim_dir / "position.dat"
            data = load_position_fast(pos_path)
            N = int(params["N"])
            utime = int(params["utime"])
            n_frames = int(params["ntime"]) // utime
            pre_subsampled = (data.shape[0] == N * n_frames)
            X_base = build_states(data, N, utime, pre_subsampled=pre_subsampled)

            # P=0 ベースライン（pair_seed 不要）
            nr0, mc0 = _evaluate_ext(
                X_base, u, y,
                args.washout, args.train_num, args.ridge_lambda, args.k_max, N,
            )
            # ガード: base MC が退化 or NRMSE が非有限 → 入力取り違え等の疑い。
            # 破損値・NaN を CSV/ヒートマップへ混入させないため、この sim を除外する。
            if mc0 < MC_DEGENERATE_THRESHOLD or not np.isfinite(nr0):
                print(f"  [SKIP] v0={v0} rcut={rcut} sgm={sgm} seed_pos={seed_pos}: "
                      f"退化 (base MC={mc0:.2e}, NRMSE={nr0}). 駆動入力の不一致の可能性 — 集計から除外",
                      flush=True)
                continue
            rows.append({"v0": v0, "rcut": rcut, "sgm": sgm,
                         "seed_pos": seed_pos, "n_pairs": 0, "pair_seed": -1,
                         "nrmse_test": nr0, "mc_test": mc0})
            print(f"  P=0   NRMSE={nr0:.4f}  MC={mc0:.3f}", flush=True)

            # P=500 × n_pair_seeds
            for ps in range(args.n_pair_seeds):
                X_ext = _build_ext(X_base, args.n_pairs, ps, N)
                nr, mc = _evaluate_ext(
                    X_ext, u, y,
                    args.washout, args.train_num, args.ridge_lambda, args.k_max, N,
                )
                rows.append({"v0": v0, "rcut": rcut, "sgm": sgm,
                             "seed_pos": seed_pos, "n_pairs": args.n_pairs,
                             "pair_seed": ps, "nrmse_test": nr, "mc_test": mc})
            avg_nr = np.mean([r["nrmse_test"] for r in rows[-args.n_pair_seeds:]])
            avg_mc = np.mean([r["mc_test"]    for r in rows[-args.n_pair_seeds:]])
            print(f"  P={args.n_pairs} NRMSE={avg_nr:.4f}  MC={avg_mc:.3f} "
                  f"(avg over {args.n_pair_seeds} pair seeds)", flush=True)

    if not rows:
        print("[ERROR] 結果が 1 件もありません。データが揃っているか確認してください。",
              flush=True)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "higher_order_data.csv", index=False)
    print(f"  higher_order_data.csv ({len(df)} rows)", flush=True)

    _make_heatmap(df, args.sgm_values, args.rcut_values, args.v0_values, out_dir)

    if model_params_list:
        reservoir_fixed = {
            "washout":      args.washout,
            "train_num":    args.train_num,
            "ridge_lambda": args.ridge_lambda,
            "k_max":        args.k_max,
            "n_pairs":      args.n_pairs,
            "n_pair_seeds": args.n_pair_seeds,
            "seed_pos":     args.seed_pos,
        }
        write_params_used(out_dir, model_params_list, reservoir_fixed)
        print(f"  params_used.json", flush=True)

    print(f"\nOutput: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
