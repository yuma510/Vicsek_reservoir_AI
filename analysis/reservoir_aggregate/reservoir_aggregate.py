#!/usr/bin/env python3
"""
Aggregate and evaluate reservoir predictions by sgm groups.

For each (sgm, seed) pair found in data/, this script:
  1. Runs ridge regression on position.dat to get raw prediction arrays
  2. Averages NARMA predictions and per-delay MCk predictions across seeds
  3. Recomputes NRMSE and truncated Memory Capacity from the averaged signals

Output goes to <out>/<sgm>/ (averaged .dat files, plots, CSV) plus
a top-level summary_all.csv.

実行例:
    python analysis/reservoir_aggregate/reservoir_aggregate.py \\
        --data-dir data \\
        --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \\
        --seeds 1 2 3 4 5 6 7 8 9 10 \\
        --rcut 13.0 \\
        --narma-seed 666 \\
        --out analysis/reservoir_aggregate
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import (
    apply_style,
    load_position_fast, build_states, ridge_predict,
    find_exp_dir, find_narma_by_seed, load_reservoir_defaults,
    nrmse, nrmse2, mck_score, corrcoef, delayed_input,
)

apply_style()
_RC = load_reservoir_defaults()


# ── prediction computation ─────────────────────────────────────────────────

def compute_predictions(exp_dir: Path, params: dict,
                        y_target: np.ndarray, col_input: np.ndarray,
                        washout: int, train_num: int, k_max: int
                        ) -> dict | None:
    """Run ridge regression and return raw prediction arrays for one experiment."""
    pos_path = exp_dir / "position.dat"
    if not pos_path.exists():
        return None

    print(f"    Loading {exp_dir.name} …", flush=True)
    data = load_position_fast(pos_path)

    N      = params["N"]
    utime  = params["utime"]
    n_input = params["ntime"] // utime
    pre_subsampled = (data.shape[0] == N * n_input)

    states = build_states(data, N, utime, pre_subsampled=pre_subsampled)
    train_end_frame = washout + train_num
    length = states.shape[0]
    total_num = n_input
    threshold = N / train_num

    y_pred = ridge_predict(states, y_target, washout, train_end_frame)

    mck_preds: dict[int, np.ndarray] = {}
    for delay in range(k_max + 1):
        padded = np.concatenate([np.zeros(delay), col_input])
        tgt    = padded[:length]
        pred   = ridge_predict(states, tgt, washout, train_end_frame)
        mck_preds[delay] = pred
        te_s = slice(min(train_end_frame, length), min(total_num, length))
        if mck_score(tgt[te_s], pred[te_s]) <= threshold:
            break

    return {
        "y_pred":          y_pred,
        "mck_preds":       mck_preds,
        "train_end_frame": train_end_frame,
        "washout":         washout,
        "train_num":       train_num,
        "total_num":       total_num,
        "N":               N,
    }


def load_runs(data_dir: Path, sgm_values: list[float], seeds: list[int],
              rcut: float, y_target: np.ndarray, col_input: np.ndarray,
              washout: int, train_num: int, k_max: int,
              seed_key: str = "seed_noise") -> list[dict]:
    """Find experiments in data_dir and compute predictions for each."""
    runs = []
    for sgm in sgm_values:
        for seed in seeds:
            exp_dir = find_exp_dir(data_dir, sgm=sgm, rcut=rcut,
                                   seed=seed, seed_key=seed_key)
            if exp_dir is None:
                print(f"  [MISS] sgm={sgm} rcut={rcut} {seed_key}={seed}", flush=True)
                continue

            with open(exp_dir / "params_model.json") as f:
                params = json.load(f)

            result = compute_predictions(exp_dir, params, y_target, col_input,
                                         washout, train_num, k_max)
            if result is None:
                continue

            runs.append({
                "sgm":    sgm,
                "seed":   seed,
                "result": result,
            })
    return runs


# ── averaging and evaluation ───────────────────────────────────────────────

def average_and_evaluate_group(sgm_key: str, group: list[dict],
                                y_target: np.ndarray, col_input: np.ndarray,
                                out_dir: Path, mc_cutoff: float = 0.0) -> dict:
    """Average predictions across seeds for one sgm value, recompute metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    narma_list: list[np.ndarray] = []
    mck_per_k: dict[int, list[np.ndarray]] = {}
    mc_meta: dict | None = None

    for r in group:
        res = r["result"]
        narma_list.append(res["y_pred"])
        for k, pred in res["mck_preds"].items():
            mck_per_k.setdefault(k, []).append(pred)
        if mc_meta is None:
            mc_meta = {k: res[k] for k in
                       ("train_end_frame", "washout", "train_num", "total_num", "N")}

    if not narma_list:
        raise RuntimeError(f"No predictions for sgm={sgm_key}")

    washout        = mc_meta["washout"]
    train_num      = mc_meta["train_num"]
    train_end      = mc_meta["train_end_frame"]
    total_num      = mc_meta["total_num"]
    N              = mc_meta["N"]

    # average NARMA prediction
    min_len_n = min(a.size for a in narma_list)
    narma_avg = np.mean(np.vstack([a[:min_len_n] for a in narma_list]), axis=0)
    np.savetxt(out_dir / "NARMA10_prediction_avg.dat", narma_avg)

    # average MCk per delay
    per_k_stats: dict[int, dict] = {}
    for k in sorted(mck_per_k):
        arrs = mck_per_k[k]
        min_len_m = min(a.size for a in arrs)
        mck_avg_k = np.mean(np.vstack([a[:min_len_m] for a in arrs]), axis=0)
        np.savetxt(out_dir / f"MCk_prediction_k={k}_avg.dat", mck_avg_k)
        per_k_stats[k] = {"n_runs": len(arrs), "length": int(min_len_m)}

    # metrics on averaged NARMA prediction
    L = min(y_target.size, narma_avg.size)
    y_t, y_p = y_target[:L], narma_avg[:L]
    tr_end = min(train_end, L)

    res_summary: dict = {"n_runs": len(group)}
    if tr_end > washout:
        res_summary.update({
            "nrmse_train":  nrmse( y_t[washout:tr_end], y_p[washout:tr_end]),
            "nrmse_test":   nrmse( y_t[tr_end:L],       y_p[tr_end:L]),
            "nrmse2_train": nrmse2(y_t[washout:tr_end], y_p[washout:tr_end]),
            "nrmse2_test":  nrmse2(y_t[tr_end:L],       y_p[tr_end:L]),
        })

    # plot averaged NARMA prediction
    plt.figure(figsize=(10, 4))
    plt.plot(y_t, label="Target")
    plt.plot(y_p, label="Averaged Prediction", alpha=0.8)
    if tr_end:
        plt.axvline(x=tr_end, color="red", linestyle="--", label="train end")
    plt.legend()
    plt.title(f"NARMA10 averaged prediction (sgm={sgm_key})")
    plt.tight_layout()
    plt.savefig(out_dir / "NARMA10_avg.png")
    plt.close()

    pd.DataFrame({
        "timestep":       np.arange(L),
        "target":         y_t,
        "prediction_avg": y_p,
    }).to_csv(out_dir / "NARMA10_avg.csv", index=False)

    # recompute MCk from averaged signals
    MC_train_sum = MC_test_sum = None
    if per_k_stats:
        delays = sorted(per_k_stats)
        mck_train_list, mck_test_list = [], []
        for k in delays:
            mck_avg_k = np.loadtxt(out_dir / f"MCk_prediction_k={k}_avg.dat")
            Lm = mck_avg_k.size
            delayed = delayed_input(col_input[:Lm], k, Lm)
            tr_s = slice(washout, min(train_end, Lm))
            te_s = slice(min(train_end, Lm), min(total_num, Lm))
            mc_tr = corrcoef(delayed[tr_s], mck_avg_k[tr_s]) if tr_s.stop > tr_s.start else None
            mc_te = corrcoef(delayed[te_s], mck_avg_k[te_s]) if te_s.stop > te_s.start else None
            mck_train_list.append(mc_tr if mc_tr is not None else 0.0)
            mck_test_list.append(mc_te if mc_te is not None else 0.0)
            per_k_stats[k].update({"MCk_train": mc_tr, "MCk_test": mc_te})

        denom = max(1, train_num - washout)
        threshold = float(mc_cutoff) if mc_cutoff > 0 else float(N) / denom

        mck_test_arr  = np.asarray([np.nan if v is None else v for v in mck_test_list])
        mck_train_arr = np.asarray([np.nan if v is None else v for v in mck_train_list])
        comp = np.where(~np.isnan(mck_test_arr), mck_test_arr <= threshold, False)
        idxs = np.where(comp)[0]
        cut  = int(idxs[0]) if idxs.size > 0 else len(delays)

        MC_train_sum = float(np.nansum(mck_train_arr[:cut]))
        MC_test_sum  = float(np.nansum(mck_test_arr[:cut]))

        plt.figure(figsize=(6, 4))
        plt.plot(delays[:cut], mck_train_list[:cut], marker="o", label="MCk_train")
        plt.plot(delays[:cut], mck_test_list[:cut],  marker="s", label="MCk_test")
        plt.xlabel("delay (k)")
        plt.ylabel("MCk")
        plt.title(f"MCk vs delay (sgm={sgm_key})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "MCk_vs_delay.png")
        plt.close()

        pd.DataFrame({
            "delay":     delays[:cut],
            "mck_train": mck_train_list[:cut],
            "mck_test":  mck_test_list[:cut],
        }).to_csv(out_dir / "MCk_vs_delay.csv", index=False)

        res_summary["MC_cutoff_used"] = float(threshold)
        res_summary["MC_cut_index"]   = int(cut)

    if MC_train_sum is not None:
        res_summary["MC_train_truncated"] = MC_train_sum
    if MC_test_sum is not None:
        res_summary["MC_test_truncated"] = MC_test_sum
    if per_k_stats:
        res_summary["per_k"] = {str(k): v for k, v in per_k_stats.items()}

    scalar_summary = {k: v for k, v in res_summary.items() if k != "per_k"}
    pd.DataFrame([scalar_summary]).to_csv(out_dir / "summary.csv", index=False)
    return res_summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data",
                   help="Directory containing simulation dirs (default: data)")
    p.add_argument("--sgm-values", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   help="sgm values to aggregate (default: 0.0 0.1 0.2 0.3 0.4 0.5)")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(range(1, 11)),
                   help="Noise seed values to aggregate (default: 1..10)")
    p.add_argument("--rcut", type=float, default=_RC.get("rcut", 13.0),
                   help="rcut to filter experiments (default: 13.0)")
    p.add_argument("--seed-key", default="seed_noise",
                   help="params_model.json key for the noise seed (default: seed_noise)")
    p.add_argument("--narma-root", default="narma_data",
                   help="Root dir for NARMA10 data files")
    p.add_argument("--narma-seed", type=int, default=666,
                   help="NARMA10 seed to use (default: 666)")
    p.add_argument("--washout",   type=int, default=_RC["washout"])
    p.add_argument("--train-num", type=int, default=_RC["train_num"])
    p.add_argument("--k-max",     type=int, default=_RC["k_max"])
    p.add_argument("--mc-cutoff", type=float, default=0.0,
                   help="Absolute MCk_test cutoff (0 = N/(train-washout))")
    p.add_argument("--only-sgm", default=None,
                   help="If set, only process this sgm (e.g. 0.2)")
    p.add_argument("--out", default="analysis/reservoir_aggregate",
                   help="Output directory (default: analysis/reservoir_aggregate)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    # resolve NARMA data
    input_path, target_path = find_narma_by_seed(args.narma_root, args.narma_seed)
    if input_path is None or target_path is None:
        sys.exit(f"NARMA10 (seed={args.narma_seed}) not found under {args.narma_root}. "
                 "Run generate_narma10.py first.")
    y_target  = np.loadtxt(target_path)
    raw_input = np.loadtxt(input_path)
    col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input

    # load and compute predictions
    print(f"Scanning {data_dir} for sgm={args.sgm_values} rcut={args.rcut} "
          f"seeds={args.seeds} …", flush=True)
    runs = load_runs(data_dir, args.sgm_values, args.seeds, args.rcut,
                     y_target, col_input,
                     args.washout, args.train_num, args.k_max,
                     seed_key=args.seed_key)

    if not runs:
        sys.exit(f"No valid experiments found in {data_dir}.")

    # group by sgm
    groups: dict[str, list[dict]] = {}
    for r in runs:
        key = f"{r['sgm']:.1f}"
        groups.setdefault(key, []).append(r)

    summary_all = {}
    for sgm_key, group in sorted(groups.items()):
        if args.only_sgm is not None and sgm_key != f"{float(args.only_sgm):.1f}":
            continue
        print(f"\nProcessing sgm={sgm_key}, {len(group)} runs …", flush=True)
        try:
            summary_all[sgm_key] = average_and_evaluate_group(
                sgm_key, group, y_target, col_input,
                out_base / sgm_key, mc_cutoff=args.mc_cutoff,
            )
        except Exception as e:
            print(f"  Failed: {e}", flush=True)

    all_rows = [
        {"sgm": sgm_key, **{k: v for k, v in res.items() if k != "per_k"}}
        for sgm_key, res in sorted(summary_all.items())
    ]
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_base / "summary_all.csv", index=False)

    print(f"\nDone. Results saved to {out_base}")


if __name__ == "__main__":
    main()
