"""rcut × 複数trial スイープ — NARMA10生成・シミュレーション・評価・散布図プロット一括実行"""
import argparse
import json
import sys
from collections import defaultdict
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

from vicsek_rc import (evaluate_reservoir, find_exp_dir, write_params_used, load_reservoir_defaults,
                       new_narma_dir, iter_narma_dirs, save_narma_params)
from generate_narma10 import generate_narma10 as _narma10_gen

BINARY = str(ROOT / "vicsek_dynamic")

_RC = load_reservoir_defaults()
RESERVOIR_DEFAULTS = {
    "readout":      _RC["readout"],
    "washout":      _RC["washout"],
    "train_num":    _RC["train_num"],
    "ridge_lambda": _RC["ridge_lambda"],
    "k_max":        _RC["k_max"],
}


def collect_model_params(rcut_values, trial_seeds, data_dir):
    """評価に使った各 sim dir の params_model.json を集める（キャッシュ skip 時も網羅）。"""
    data_dir = Path(data_dir)
    params_list = []
    for rcut in rcut_values:
        for seed in trial_seeds:
            exp_dir = find_exp_dir(data_dir, rcut=rcut, seed=seed+3, seed_key="seed_pos")
            if exp_dir is None:
                exp_dir = find_exp_dir(data_dir, rcut=rcut, seed=seed, seed_index=0)
            if exp_dir is None:
                continue
            pf = exp_dir / "params_model.json"
            if pf.exists():
                with open(pf) as f:
                    params_list.append(json.load(f))
    return params_list


# ── Phase 0: NARMA10 生成 ─────────────────────────────────────────────────

def ensure_narma10(seed, narma_root, low=0.0, high=0.5, length=14000):
    """既存の日付 dir から十分な長さの NARMA を再利用し、無ければ専用の新しい日付 dir に生成する
    （1 時刻 dir = 1 データセット）。"""
    prefix   = f"{low}:{high}_seed{seed}"
    in_name  = f"narma10_input_{prefix}.dat"
    tgt_name = f"narma10_target_{prefix}.dat"

    # narma_root 以下（日付 dir を新しい順、末尾に直下）から再利用
    for d in iter_narma_dirs(narma_root):
        input_path, target_path = d / in_name, d / tgt_name
        if input_path.exists() and target_path.exists() \
                and sum(1 for _ in open(target_path)) >= length:
            y = np.loadtxt(target_path)
            if not np.all(np.isfinite(y)):
                n_bad = np.sum(~np.isfinite(y))
                print(f"[NARMA10] WARN seed={seed}: existing file has {n_bad} non-finite values "
                      f"(diverged) — skip this trial", flush=True)
                return None, None
            print(f"[NARMA10] Skip seed={seed} (exists, {len(y)} rows) [{d.name}]", flush=True)
            return input_path, target_path

    # 無ければ seed 専用の新しい日付 dir に生成
    u, y = _narma10_gen(length, low, high, seed)
    if not np.all(np.isfinite(y)):
        n_bad = np.sum(~np.isfinite(y))
        print(f"[NARMA10] WARN seed={seed}: {n_bad} non-finite values (diverged) — "
              f"skip this trial", flush=True)
        return None, None
    gen_dir = new_narma_dir(narma_root)
    input_path, target_path = gen_dir / in_name, gen_dir / tgt_name
    np.savetxt(input_path, u)
    np.savetxt(target_path, y)
    save_narma_params(gen_dir, in_name, tgt_name, length, seed, low, high)
    print(f"[NARMA10] Generated seed={seed} (length={length}) [{gen_dir.name}]", flush=True)
    return input_path, target_path


# ── Phase 1: シミュレーション ─────────────────────────────────────────────

def _run_one_sim(args):
    rcut, seed, input_path, data_dir, v0 = args
    import json, os, subprocess, tempfile
    cfg = {
        "input_file":  str(input_path),
        "output_base": str(data_dir),
        "rcut":        float(rcut),
        "sgm":         0.0,
        "v0":          float(v0),
        "seed_noise":  seed + 2,
        "seed_pos":    seed + 3,
        "seed_nf":     seed + 6,
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
        print(f"[ERROR] rcut={rcut} seed={seed}: {result.stderr[:200]}", flush=True)
    else:
        print(f"[SIM done] rcut={rcut} seed={seed}", flush=True)
    return rcut, seed, result.returncode


def phase1_simulate(rcut_values, trial_seeds, narma_paths, data_dir, n_jobs, v0=0.5):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(r, s, narma_paths[s][0], data_dir, v0) for r in rcut_values for s in trial_seeds]
    print(f"Phase 1: launching {len(jobs)} simulations (n_jobs={n_jobs}) …", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            rcut, seed, rc = f.result()
            if rc != 0:
                failed.append((rcut, seed))

    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
    return failed


# ── Phase 2: リザバー評価 ─────────────────────────────────────────────────

def phase2_evaluate(rcut_values, trial_seeds, narma_paths, data_dir, output_dir):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for rcut in rcut_values:
        for seed in trial_seeds:
            key      = f"rcut={int(rcut)}_seed={seed}"
            out_file = output_dir / f"results_{key}.json"
            if out_file.exists():
                print(f"[SKIP] {key} (already evaluated)", flush=True)
                with open(out_file) as f:
                    results[key] = json.load(f)
                continue

            exp_dir = find_exp_dir(data_dir, rcut=rcut, seed=seed+3, seed_key="seed_pos")
            if exp_dir is None:
                exp_dir = find_exp_dir(data_dir, rcut=rcut, seed=seed, seed_index=0)
            if exp_dir is None:
                print(f"[MISS] no dir for {key}", flush=True)
                continue

            pos_path    = exp_dir / "position.dat"
            params_file = exp_dir / "params_model.json"
            if not pos_path.exists() or not params_file.exists():
                print(f"[MISS] files missing in {exp_dir}", flush=True)
                continue

            with open(params_file) as f:
                params = json.load(f)

            input_path, target_path = narma_paths[seed]
            print(f"[EVAL] {key} …", flush=True)
            res = evaluate_reservoir(pos_path, params, target_path, input_path)
            res["rcut"] = rcut
            res["seed"] = seed
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            results[key] = res
            print(f"       NRMSE_test={res['nrmse_test']:.4f}  MC_test={res['MC_test']:.3f}",
                  flush=True)

    return results


# ── Phase 3: 散布図プロット ───────────────────────────────────────────────

def phase3_plot(results: dict, rcut_values, output_dir: Path):
    output_dir = Path(output_dir)

    by_rcut: dict[float, list] = defaultdict(list)
    for v in results.values():
        by_rcut[v["rcut"]].append(v)

    # プロット元データ（seed ごとの生データ点）を long 形式 CSV で保存
    cols = ["MC_test", "MC_train", "nrmse_test", "nrmse_train",
            "nrmse2_test", "nrmse2_train"]
    rows = [
        {"rcut": e["rcut"], "seed": e["seed"], **{c: e.get(c) for c in cols}}
        for e in sorted(results.values(), key=lambda x: (x["rcut"], x["seed"]))
    ]
    pd.DataFrame(rows).to_csv(output_dir / "rcut_sweep_data.csv", index=False)

    rcut_sorted = sorted(by_rcut)
    metrics = [
        ("MC_test",     "MC_train",     "MC (test/train)",     "mc_vs_rcut_scatter.png"),
        ("nrmse_test",  "nrmse_train",  "NRMSE (test/train)",  "nrmse_vs_rcut_scatter.png"),
        ("nrmse2_test", "nrmse2_train", "NRMSE2 (test/train)", "nrmse2_vs_rcut_scatter.png"),
    ]
    for test_key, train_key, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, rcut in enumerate(rcut_sorted):
            test_vals  = [e[test_key]  for e in by_rcut[rcut]]
            train_vals = [e[train_key] for e in by_rcut[rcut]]
            ax.scatter([rcut]*len(test_vals),  test_vals,
                       color="tab:blue", alpha=0.6, s=30, marker="o",
                       label="test"  if i == 0 else "")
            ax.scatter([rcut]*len(train_vals), train_vals,
                       color="tab:red",  alpha=0.6, s=30, marker="^",
                       label="train" if i == 0 else "")
        ax.set_xlabel("rcut")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {fname}", flush=True)

    print(f"\nPlots saved to {output_dir}/", flush=True)
    print(f"  rcut_sweep_data.csv", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="rcut × trial sweep with scatter plot")
    p.add_argument("--rcut-values", nargs="+", type=float,
                   default=list(range(1, 14)),
                   metavar="R", help="rcut values to sweep (default: 1..13)")
    p.add_argument("--trial-seeds", nargs="+", type=int,
                   default=list(range(1, 11)),
                   metavar="S",
                   help="各trialのseed。NARMA10入力（narma10 seed）と自然振動数（seed_array[6]）の両方に使用。デフォルト: 1..10")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="parallel simulation workers")
    p.add_argument("--data-dir", default="data",
                   help="directory for simulation output")
    p.add_argument("--output-dir", default="analysis/rcut_sweep",
                   help="directory for evaluation results and plots")
    p.add_argument("--narma-dir", default="narma_data",
                   help="NARMA10 入力・正解ファイルの保存先（trial seedごとに自動生成）")
    p.add_argument("--skip-sim", action="store_true",
                   help="skip simulation phase, only evaluate existing data")
    p.add_argument("--v0", type=float, default=0.5,
                   help="particle speed (default: 0.5; use 0.0 for static network)")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: 各 trial seed の NARMA10 ファイルを生成（なければ）。発散したシードは除外
    # 生成する seed ごとに専用の日付 dir を作る（1 dir = 1 データセット。既存は再利用）
    narma_paths = {
        s: ensure_narma10(s, args.narma_dir)
        for s in args.trial_seeds
    }
    valid_seeds = [s for s, (ip, _) in narma_paths.items() if ip is not None]
    if len(valid_seeds) < len(args.trial_seeds):
        skipped = set(args.trial_seeds) - set(valid_seeds)
        print(f"[WARN] Skipping diverged seeds: {sorted(skipped)}", flush=True)

    if not args.skip_sim:
        phase1_simulate(args.rcut_values, valid_seeds,
                        narma_paths, args.data_dir, args.n_jobs, args.v0)

    results = phase2_evaluate(args.rcut_values, valid_seeds,
                              narma_paths, args.data_dir, output_dir)

    if results:
        phase3_plot(results, args.rcut_values, output_dir)
    else:
        print("[WARN] No results to plot.", flush=True)

    model_params = collect_model_params(args.rcut_values, valid_seeds, args.data_dir)
    if model_params:
        write_params_used(output_dir, model_params, RESERVOIR_DEFAULTS)
        print(f"  params_used.json", flush=True)

    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
