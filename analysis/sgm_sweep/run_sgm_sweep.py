"""sgm × 複数seed スイープ — シミュレーション・評価・エラーバープロット一括実行"""
import argparse
import json
import subprocess
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

from vicsek_rc.seeds import trial_seeds as seeds_for, set_seed_scheme  # 試行 b → seed（seed_policy.json）
from vicsek_rc import (evaluate_reservoir, find_exp_dir, write_params_used, load_reservoir_defaults,
                       find_narma_by_seed)

BINARY = str(ROOT / "vicsek_dynamic")

_RC = load_reservoir_defaults()
RESERVOIR_DEFAULTS = {
    "readout":      _RC["readout"],
    "washout":      _RC["washout"],
    "train_num":    _RC["train_num"],
    "ridge_lambda": _RC["ridge_lambda"],
    "k_max":        _RC["k_max"],
}


def collect_model_params(sgm_values, seeds, rcut, data_dir, v0=None, narma_seed=None):
    """評価に使った各 sim dir の params_model.json を集める（キャッシュ skip 時も網羅）。"""
    data_dir = Path(data_dir)
    params_list = []
    for sgm in sgm_values:
        for seed in seeds:
            exp_dir = find_exp_dir(data_dir, rcut=rcut, sgm=sgm, seed=seeds_for(seed)["seed_pos"],
                                   seed_key="seed_pos", v0=v0, ntime=140000, narma_seed=narma_seed)
            if exp_dir is None:
                exp_dir = find_exp_dir(data_dir, rcut=rcut, sgm=sgm, seed=seed,
                                       seed_index=0, v0=v0, ntime=140000, narma_seed=narma_seed)
            if exp_dir is None:
                continue
            pf = exp_dir / "params_model.json"
            if pf.exists():
                with open(pf) as f:
                    params_list.append(json.load(f))
    return params_list


# ── Phase 1: simulation ────────────────────────────────────────────────────

def _run_one_sim(args):
    sgm, seed, rcut, input_path, data_dir, v0 = args
    import json, os, tempfile
    cfg = {
        "input_file":  str(input_path),
        "output_base": str(data_dir),
        "rcut":        float(rcut),
        "sgm":         float(sgm),
        "seed_noise":  seeds_for(seed)["seed_noise"],
        "seed_pos":    seeds_for(seed)["seed_pos"],
        "seed_nf":     seeds_for(seed)["seed_nf"],
        "ntime":       140000,
    }
    if v0 is not None:
        cfg["v0"] = float(v0)
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        result = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    if result.returncode != 0:
        print(f"[ERROR] sgm={sgm} seed={seed}: {result.stderr[:200]}", flush=True)
    else:
        print(f"[SIM done] sgm={sgm} seed={seed}", flush=True)
    return sgm, seed, result.returncode


def phase1_simulate(sgm_values, seeds, rcut, input_path, data_dir, n_jobs, v0=None, narma_seed=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 既存 sim はスキップ（v0 フィルタ付き）
    jobs = []
    for s in sgm_values:
        for sd in seeds:
            if find_exp_dir(data_dir, rcut=rcut, sgm=s, seed=seeds_for(sd)["seed_pos"],
                            seed_key="seed_pos", v0=v0, ntime=140000, narma_seed=narma_seed) is not None:
                print(f"[SKIP SIM] sgm={s} seed_pos={seeds_for(sd)["seed_pos"]} (already exists)", flush=True)
                continue
            jobs.append((s, sd, rcut, input_path, data_dir, v0))
    print(f"Phase 1: launching {len(jobs)} simulations (n_jobs={n_jobs}) …", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            sgm, seed, rc = f.result()
            if rc != 0:
                failed.append((sgm, seed))

    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
    return failed


# ── Phase 2: reservoir evaluation ─────────────────────────────────────────

def phase2_evaluate(sgm_values, seeds, rcut, data_dir, output_dir,
                    target_path, input_path, v0=None, narma_seed=None):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for sgm in sgm_values:
        for seed in seeds:
            key = f"sgm={sgm:.2f}_seed={seed}"
            out_file = output_dir / f"results_{key}.json"
            if out_file.exists():
                print(f"[SKIP] {key} (already evaluated)", flush=True)
                with open(out_file) as f:
                    results[key] = json.load(f)
                continue

            exp_dir = find_exp_dir(data_dir, rcut=rcut, sgm=sgm, seed=seeds_for(seed)["seed_pos"],
                                   seed_key="seed_pos", v0=v0, ntime=140000, narma_seed=narma_seed)
            if exp_dir is None:
                exp_dir = find_exp_dir(data_dir, rcut=rcut, sgm=sgm, seed=seed,
                                       seed_index=0, v0=v0, ntime=140000, narma_seed=narma_seed)
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

            print(f"[EVAL] {key} …", flush=True)
            res = evaluate_reservoir(pos_path, params, Path(target_path), Path(input_path))
            res["sgm"] = sgm
            res["seed"] = seed
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            results[key] = res
            print(f"       NRMSE_test={res['nrmse_test']:.4f}  MC_test={res['MC_test']:.3f}", flush=True)

    return results


# ── Phase 3: aggregate and plot ────────────────────────────────────────────

def phase3_plot(results: dict, output_dir: Path):
    output_dir = Path(output_dir)

    by_sgm: dict[float, list] = defaultdict(list)
    for v in results.values():
        by_sgm[v["sgm"]].append(v)

    # プロット元データ（seed ごとの生データ点）を long 形式 CSV で保存
    cols = ["MC_test", "MC_train", "nrmse_test", "nrmse_train",
            "nrmse2_test", "nrmse2_train"]
    rows = [
        {"sgm": e["sgm"], "seed": e["seed"], **{c: e.get(c) for c in cols}}
        for e in sorted(results.values(), key=lambda x: (x["sgm"], x["seed"]))
    ]
    pd.DataFrame(rows).to_csv(output_dir / "sgm_sweep_data.csv", index=False)

    sgm_sorted = sorted(by_sgm)

    metrics = [
        ("MC_test",     "MC_train",     "MC (test/train)",     "mc_vs_sgm_scatter.png"),
        ("nrmse_test",  "nrmse_train",  "NRMSE (test/train)",  "nrmse_vs_sgm_scatter.png"),
        ("nrmse2_test", "nrmse2_train", "NRMSE2 (test/train)", "nrmse2_vs_sgm_scatter.png"),
    ]
    for test_key, train_key, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, sgm in enumerate(sgm_sorted):
            test_vals  = [e[test_key]  for e in by_sgm[sgm]]
            train_vals = [e[train_key] for e in by_sgm[sgm]]
            lbl_test  = "test"  if i == 0 else ""
            lbl_train = "train" if i == 0 else ""
            ax.scatter([sgm] * len(test_vals),  test_vals,  color="tab:blue", alpha=0.6, s=30, label=lbl_test)
            ax.scatter([sgm] * len(train_vals), train_vals, color="tab:red",  alpha=0.6, s=30, marker="^", label=lbl_train)
        ax.set_xlabel("sgm")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)

    plot_errorbar(rows, output_dir)

    print(f"\nPlots saved to {output_dir}/", flush=True)
    for _, _, _, fname in metrics:
        print(f"  {fname}", flush=True)
    print(f"  sgm_sweep_data.csv", flush=True)


def plot_errorbar(rows, output_dir, ymax=None):
    """seed 間の平均±標準偏差を errorbar でプロット（rows は phase3_plot の long 形式）。

    ymax: メトリクス名（"MC"/"nrmse"/"nrmse2"）→ 縦軸上限の dict。
          複数スイープで縦軸を揃えたいとき指定する（None なら自動）。
    """
    output_dir = Path(output_dir)
    ymax = ymax or {}
    df = pd.DataFrame(rows)
    metrics = [
        ("MC_test",     "MC_train",     "MC (test/train)",     "mc_vs_sgm_errorbar.png",     "MC"),
        ("nrmse_test",  "nrmse_train",  "NRMSE (test/train)",  "nrmse_vs_sgm_errorbar.png",  "nrmse"),
        ("nrmse2_test", "nrmse2_train", "NRMSE2 (test/train)", "nrmse2_vs_sgm_errorbar.png", "nrmse2"),
    ]
    for test_key, train_key, ylabel, fname, mkey in metrics:
        g = df.groupby("sgm")
        xs = sorted(df["sgm"].unique())
        # 凡例あり（fname）と凡例なし（発表用、_nolegend）の 2 枚を出力
        for legend in (True, False):
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, color, marker, lbl in [(test_key, "tab:blue", "o", "test"),
                                            (train_key, "tab:red", "^", "train")]:
                mean = g[key].mean().reindex(xs)
                std  = g[key].std(ddof=1).reindex(xs)
                ax.errorbar(xs, mean.values, yerr=std.values, color=color, marker=marker,
                            capsize=3, ls="-", label=lbl)
            ax.set_xlabel("sgm")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, ymax.get(mkey))
            if legend:
                ax.legend(title="mean ± std (seeds)")
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            out = fname if legend else fname.replace(".png", "_nolegend.png")
            fig.savefig(output_dir / out, dpi=150)
            plt.close(fig)
            print(f"  {out}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="sgm × seed sweep with error bars")
    p.add_argument("--sgm-values", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   metavar="S", help="sgm values to sweep (default: 0.0..0.5)")
    p.add_argument("--rcut", type=float, default=13,
                   help="fixed rcut value (default: 13)")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[10, 11, 12, 13, 14],
                   metavar="SD", help="noise seeds (seed_array[2])")
    p.add_argument("--seed-scheme", choices=["random64", "legacy"], default=None,
                   help="試行の seed の規則（configs/seed_policy.json）。既定は random64。既存データの再評価は legacy")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="parallel simulation workers")
    p.add_argument("--data-dir", default="data",
                   help="directory for simulation output")
    p.add_argument("--output-dir", default="analysis/sgm_sweep",
                   help="directory for evaluation results and plots")
    p.add_argument("--narma-root", default="narma_data",
                   help="NARMA10 探索ルート（日付 dir を自動選択）")
    p.add_argument("--narma-seed", type=int, default=666,
                   help="使用する NARMA10 の seed（default 666）")
    p.add_argument("--input-path", default=None,
                   help="NARMA10 input signal（未指定なら最新日付 dir から自動解決）")
    p.add_argument("--target-path", default=None,
                   help="NARMA10 target signal（未指定なら最新日付 dir から自動解決）")
    p.add_argument("--skip-sim", action="store_true",
                   help="skip simulation, only evaluate existing data")
    p.add_argument("--v0", type=float, default=0.5,
                   help="粒子速度 v0（default 0.5; v0=0 実験の誤マッチ防止フィルタにも使用）")
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
    set_seed_scheme(args.seed_scheme)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sim:
        phase1_simulate(args.sgm_values, args.seeds, args.rcut,
                        args.input_path, args.data_dir, args.n_jobs, v0=args.v0, narma_seed=args.narma_seed)

    results = phase2_evaluate(args.sgm_values, args.seeds, args.rcut,
                              args.data_dir, output_dir,
                              args.target_path, args.input_path, v0=args.v0, narma_seed=args.narma_seed)

    if results:
        phase3_plot(results, output_dir)
    else:
        print("[WARN] No results to plot.", flush=True)

    model_params = collect_model_params(args.sgm_values, args.seeds, args.rcut,
                                        args.data_dir, v0=args.v0, narma_seed=args.narma_seed)
    if model_params:
        write_params_used(output_dir, model_params, RESERVOIR_DEFAULTS)
        print(f"  params_used.json", flush=True)

    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
