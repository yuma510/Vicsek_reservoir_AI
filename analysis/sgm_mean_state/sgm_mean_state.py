"""
noise-averaged reservoir state: per-particle complex average over S noise realizations.

Phase 1: sgm_values × noise_seeds シミュレーション（初期位置・nf seed 固定、noise seed のみ変化）
Phase 2: 各 sgm について S=1..S_max で noise-averaged 状態を評価
Phase 3: MC/NRMSE vs S を sgm ごとに折れ線プロット

実行例:
    python analysis/sgm_mean_state/sgm_mean_state.py \\
        --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \\
        --noise-seeds 1 2 3 4 5 6 7 8 9 10 \\
        --rcut 13 --n-jobs 4 \\
        --data-dir data \\
        --output-dir analysis/sgm_mean_state \\
        --input-path  narma_data/narma10_input_0.0:0.5_seed666.dat \\
        --target-path narma_data/narma10_target_0.0:0.5_seed666.dat
"""
import argparse
import json
import subprocess
import sys
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
    nrmse, nrmse2, mck_score, ridge_predict,
    load_theta, build_noise_averaged_states, write_params_used,
    load_reservoir_defaults, find_narma_by_seed,
)

_RC = load_reservoir_defaults()

BINARY = str(ROOT / "vicsek_dynamic")


# ── Phase 1: simulation ────────────────────────────────────────────────────

def _run_one_sim(args):
    noise_seed, sgm, rcut, input_path, data_dir = args
    import json, os, tempfile
    cfg = {
        "input_file":  str(input_path),
        "output_base": str(data_dir),
        "rcut":        float(rcut),
        "sgm":         float(sgm),
        "seed_noise":  noise_seed,
        "seed_pos":    13,
        "seed_nf":     16,
        "ntime":       140000,
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        result = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    if result.returncode != 0:
        print(f"[ERROR] sgm={sgm} noise_seed={noise_seed}: {result.stderr[:200]}", flush=True)
    else:
        print(f"[SIM done] sgm={sgm} noise_seed={noise_seed}", flush=True)
    return sgm, noise_seed, result.returncode


def phase1_simulate(sgm_values, noise_seeds, rcut, input_path, data_dir, n_jobs):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(ns, sgm, rcut, input_path, data_dir) for sgm in sgm_values for ns in noise_seeds]
    print(f"Phase 1: launching {len(jobs)} simulations (n_jobs={n_jobs}) …", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            sgm, ns, rc = f.result()
            if rc != 0:
                failed.append((sgm, ns))

    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
    return failed


# ── Phase 2: noise-averaged evaluation ────────────────────────────────────

def _scan_experiments(data_dir: Path, sgm_values: list, noise_seeds: list):
    """Return dict[sgm][noise_seed] = exp_dir, sorted by noise_seed."""
    by_sgm: dict[float, dict[int, Path]] = {sgm: {} for sgm in sgm_values}

    for d in sorted(data_dir.iterdir()):
        pf = d / "params_model.json"
        if not d.is_dir() or not pf.exists():
            continue
        with open(pf) as f:
            params = json.load(f)
        sgm = round(float(params["sgm"]), 4)
        ns  = int(params["seed_noise"] if "seed_noise" in params else params["seed"][2])
        for target_sgm in sgm_values:
            if abs(sgm - round(target_sgm, 4)) < 1e-6 and ns in noise_seeds:
                by_sgm[target_sgm][ns] = d
                break

    return by_sgm


def _evaluate_noise_avg(theta_list: list, params: dict,
                        target: np.ndarray, col_input: np.ndarray,
                        s_values: list,
                        washout: int, train_num: int, k_max: int) -> dict:
    """Evaluate noise-averaged states for each S in s_values."""
    N              = params["N"]
    total_num      = params["ntime"] // params["utime"]
    train_end_frame = washout + train_num
    threshold      = N / train_num
    results        = {}

    for S in s_values:
        states = build_noise_averaged_states(theta_list[:S])
        length = states.shape[0]
        eval_total = min(total_num, target.shape[0], length)
        train_end  = min(train_end_frame, eval_total)

        y_pred    = ridge_predict(states, target, washout, train_end_frame)
        nr_train  = nrmse( target[washout:train_end],    y_pred[washout:train_end])
        nr_test   = nrmse( target[train_end:eval_total], y_pred[train_end:eval_total])
        nr2_train = nrmse2(target[washout:train_end],    y_pred[washout:train_end])
        nr2_test  = nrmse2(target[train_end:eval_total], y_pred[train_end:eval_total])

        mck_train_list, mck_test_list = [], []
        for delay in range(k_max + 1):
            padded = np.concatenate([np.zeros(delay), col_input])
            tgt    = padded[:length]
            pred   = ridge_predict(states, tgt, washout, train_end_frame)
            tr_s   = slice(washout, min(train_end_frame + 1, length))
            te_s   = slice(min(train_end_frame, length), min(total_num, length))
            mck_train_list.append(mck_score(tgt[tr_s], pred[tr_s]))
            mck_test_list.append(mck_score(tgt[te_s],  pred[te_s]))
            if mck_test_list[-1] <= threshold:
                break

        results[S] = {
            "nrmse_train": nr_train,   "nrmse_test":   nr_test,
            "nrmse2_train": nr2_train, "nrmse2_test":  nr2_test,
            "MC_train": float(np.sum(mck_train_list)),
            "MC_test":  float(np.sum(mck_test_list)),
        }
        print(f"    S={S:2d}  MC_test={results[S]['MC_test']:.3f}  NRMSE_test={results[S]['nrmse_test']:.4f}", flush=True)

    return results


def phase2_evaluate(sgm_values, noise_seeds, rcut, data_dir, output_dir,
                    target_path, input_path,
                    washout: int = _RC["washout"],
                    train_num: int = _RC["train_num"],
                    k_max: int = _RC["k_max"]):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    by_sgm = _scan_experiments(data_dir, sgm_values, noise_seeds)

    target    = np.loadtxt(target_path)
    raw_input = np.loadtxt(input_path)
    col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input

    all_results: dict[float, dict[int, dict]] = {}

    for sgm in sgm_values:
        ns_map = by_sgm[sgm]
        if not ns_map:
            print(f"[WARN] no experiments found for sgm={sgm:.2f}", flush=True)
            continue

        sorted_seeds = sorted(ns_map.keys())
        s_max = len(sorted_seeds)
        print(f"\nsgm={sgm:.2f}: found {s_max} noise seeds ({sorted_seeds})", flush=True)

        cache_file = output_dir / f"results_sgm={sgm:.2f}.json"
        if cache_file.exists():
            print(f"[CACHE] sgm={sgm:.2f}", flush=True)
            with open(cache_file) as f:
                raw = json.load(f)
            all_results[sgm] = {int(k): v for k, v in raw.items()}
            continue

        # Load all theta arrays (shape: (12000, N) each)
        exp_dir_0 = ns_map[sorted_seeds[0]]
        with open(exp_dir_0 / "params_model.json") as f:
            params = json.load(f)

        theta_list = []
        for ns in sorted_seeds:
            pos_path = ns_map[ns] / "position.dat"
            theta_list.append(load_theta(pos_path, params["N"]))

        s_values = list(range(1, s_max + 1))
        res = _evaluate_noise_avg(theta_list, params, target, col_input,
                                  s_values, washout, train_num, k_max)

        all_results[sgm] = res
        with open(cache_file, "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f, indent=2)

    return all_results


# ── Phase 3: plot ──────────────────────────────────────────────────────────

def phase3_plot(all_results: dict, output_dir: Path):
    output_dir = Path(output_dir)

    # プロット元データを long 形式 CSV で保存（(sgm, S) ごとに 1 行）
    cols = ["MC_test", "MC_train", "nrmse_test", "nrmse_train",
            "nrmse2_test", "nrmse2_train"]
    rows = [
        {"sgm": sgm, "S": S, **{c: res.get(c) for c in cols}}
        for sgm, res_by_S in sorted(all_results.items())
        for S, res in sorted(res_by_S.items())
    ]
    pd.DataFrame(rows).to_csv(output_dir / "noise_avg_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for sgm, res_by_S in sorted(all_results.items()):
        xs = sorted(res_by_S)
        mc = [res_by_S[s]["MC_test"]    for s in xs]
        nr = [res_by_S[s]["nrmse_test"] for s in xs]
        label = f"sgm={sgm:.1f}"
        axes[0].plot(xs, mc, marker="o", label=label)
        axes[1].plot(xs, nr, marker="o", label=label)

    axes[0].set(xlabel="S (# noise realizations)", ylabel="MC (test)")
    axes[1].set(xlabel="S (# noise realizations)", ylabel="NRMSE (test)")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "mc_nrmse_vs_S.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/", flush=True)
    print(f"  mc_nrmse_vs_S.png", flush=True)
    print(f"  noise_avg_data.csv", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="noise-averaged reservoir state (per-particle complex average)")
    p.add_argument("--sgm-values", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], metavar="S")
    p.add_argument("--noise-seeds", nargs="+", type=int,
                   default=list(range(1, 11)), metavar="NS")
    p.add_argument("--rcut", type=float, default=13)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--output-dir",  default="analysis/sgm_mean_state")
    p.add_argument("--narma-root",  default="narma_data",
                   help="NARMA10 探索ルート（日付 dir を自動選択）")
    p.add_argument("--narma-seed",  type=int, default=666,
                   help="使用する NARMA10 の seed（default 666）")
    p.add_argument("--input-path",  default=None,
                   help="未指定なら最新日付 dir から自動解決")
    p.add_argument("--target-path", default=None,
                   help="未指定なら最新日付 dir から自動解決")
    p.add_argument("--skip-sim",    action="store_true")
    p.add_argument("--washout",     type=int, default=2000)
    p.add_argument("--train-num",   type=int, default=_RC["train_num"])
    args = p.parse_args()

    if args.input_path is None or args.target_path is None:
        ip, tp = find_narma_by_seed(args.narma_root, args.narma_seed)
        args.input_path  = args.input_path  or (str(ip) if ip else None)
        args.target_path = args.target_path or (str(tp) if tp else None)
        if args.input_path is None or args.target_path is None:
            p.error(f"NARMA10 (seed={args.narma_seed}) が {args.narma_root}/ に見つからない。"
                    f"generate_narma10.py で生成してください。")
    return args


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sim:
        phase1_simulate(args.sgm_values, args.noise_seeds, args.rcut,
                        args.input_path, args.data_dir, args.n_jobs)

    all_results = phase2_evaluate(
        args.sgm_values, args.noise_seeds, args.rcut,
        args.data_dir, output_dir,
        args.target_path, args.input_path,
        washout=args.washout, train_num=args.train_num,
    )

    if all_results:
        phase3_plot(all_results, output_dir)
    else:
        print("[WARN] No results to plot.", flush=True)

    # 使用パラメータの記録（S=noise 平均数は解析軸として swept に記録）
    by_sgm = _scan_experiments(Path(args.data_dir), args.sgm_values, args.noise_seeds)
    model_params = []
    s_max = 0
    for ns_map in by_sgm.values():
        s_max = max(s_max, len(ns_map))
        for exp_dir in ns_map.values():
            pf = exp_dir / "params_model.json"
            if pf.exists():
                with open(pf) as f:
                    model_params.append(json.load(f))
    if model_params:
        reservoir_fixed = {
            "readout": _RC["readout"], "washout": args.washout,
            "train_num": args.train_num, "ridge_lambda": _RC["ridge_lambda"], "k_max": _RC["k_max"],
        }
        reservoir_swept = {"S": list(range(1, s_max + 1))}
        write_params_used(output_dir, model_params, reservoir_fixed, reservoir_swept)
        print(f"  params_used.json", flush=True)

    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
