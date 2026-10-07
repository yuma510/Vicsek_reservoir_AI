"""rcut × utime スイープ — v0=0, sgm=0 での MC/NRMSE ヒートマップ生成"""
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

from vicsek_rc import (evaluate_reservoir, write_params_used, load_reservoir_defaults,
                       new_narma_dir, iter_narma_dirs, save_narma_params)
from vicsek_rc.catalog import append_row
from vicsek_rc.seeds import trial_seeds as seeds_for, set_seed_scheme  # 試行 b → seed（seed_policy.json）
from generate_narma10 import generate_narma10 as _narma10_gen

BINARY = str(ROOT / "vicsek_dynamic")
NARMA_LENGTH = 14000  # washout(2000) + train(6000) + test(6000)

_CA = json.loads((Path(__file__).parent / "default_params.json").read_text())
_RC = load_reservoir_defaults()
RESERVOIR_DEFAULTS = {k: _RC[k] for k in ("readout", "washout", "train_num", "ridge_lambda", "k_max")}


# ── utime 対応の dir 検索 ────────────────────────────────────────────────────

def _find_exp_utime(data_dir, rcut, utime, seed_pos, v0=0.0, sgm=0.0):
    """index.csv を直接フィルタして (rcut, utime, seed_pos, v0, sgm) に一致する dir を返す。"""
    index_path = Path(data_dir) / "index.csv"
    if not index_path.exists():
        return None
    df = pd.read_csv(index_path, dtype={"dir": str})
    mask = (
        (df["rcut"]     == float(rcut))   &
        (df["utime"]    == int(utime))    &
        (df["seed_pos"] == int(seed_pos)) &
        (df["v0"]       == float(v0))     &
        (df["sgm"]      == float(sgm))    &
        (df["has_position"] == True)
    )
    hits = df[mask]
    if hits.empty:
        return None
    return Path(data_dir) / hits.iloc[-1]["dir"]


# ── Phase 0: NARMA10 生成 ─────────────────────────────────────────────────

def ensure_narma10(seed, narma_root, low=0.0, high=0.5, length=NARMA_LENGTH):
    prefix   = f"{low}:{high}_seed{seed}"
    in_name  = f"narma10_input_{prefix}.dat"
    tgt_name = f"narma10_target_{prefix}.dat"

    for d in iter_narma_dirs(narma_root):
        inp, tgt = d / in_name, d / tgt_name
        if inp.exists() and tgt.exists() and sum(1 for _ in open(tgt)) >= length:
            y = np.loadtxt(tgt)
            if not np.all(np.isfinite(y)):
                print(f"[NARMA10] WARN seed={seed}: diverged — skip", flush=True)
                return None, None
            print(f"[NARMA10] Reuse seed={seed} [{d.name}]", flush=True)
            return inp, tgt

    u, y = _narma10_gen(length, low, high, seed)
    if not np.all(np.isfinite(y)):
        print(f"[NARMA10] WARN seed={seed}: diverged — skip", flush=True)
        return None, None
    gen_dir = new_narma_dir(narma_root)
    inp, tgt = gen_dir / in_name, gen_dir / tgt_name
    np.savetxt(inp, u)
    np.savetxt(tgt, y)
    save_narma_params(gen_dir, in_name, tgt_name, length, seed, low, high)
    print(f"[NARMA10] Generated seed={seed} (length={length}) [{gen_dir.name}]", flush=True)
    return inp, tgt


# ── Phase 1: シミュレーション ─────────────────────────────────────────────

def _run_one_sim(args):
    rcut, utime, seed, input_path, data_dir = args
    cfg = {
        "input_file":  str(input_path),
        "output_base": str(data_dir),
        "rcut":        float(rcut),
        "sgm":         0.0,
        "v0":          0.0,
        "utime":       int(utime),
        "ntime":       NARMA_LENGTH * int(utime),
        "seed_noise":  seeds_for(seed)["seed_noise"],
        "seed_pos":    seeds_for(seed)["seed_pos"],
        "seed_nf":     seeds_for(seed)["seed_nf"],
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        result = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    if result.returncode != 0:
        print(f"[ERROR] rcut={rcut} utime={utime} seed={seed}: {result.stderr[:200]}", flush=True)
    else:
        # 新規 dir を index.csv に追記
        new_dirs = sorted(Path(str(data_dir)).glob("*/params_model.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if new_dirs:
            append_row(str(data_dir), new_dirs[0].parent)
        print(f"[SIM done] rcut={rcut} utime={utime} seed={seed}", flush=True)
    return rcut, utime, seed, result.returncode


def phase1_simulate(rcut_values, utime_values, trial_seeds, narma_paths, data_dir, n_jobs):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r in rcut_values:
        for u in utime_values:
            for s in trial_seeds:
                if _find_exp_utime(data_dir, r, u, seeds_for(s)["seed_pos"]) is not None:
                    print(f"[SIM skip] rcut={r} utime={u} seed={s} (exists)", flush=True)
                    continue
                jobs.append((r, u, s, narma_paths[s][0], data_dir))

    if not jobs:
        print("Phase 1: all simulations already exist, skipping.", flush=True)
        return []

    print(f"Phase 1: launching {len(jobs)} simulations (n_jobs={n_jobs}) …", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            rcut, utime, seed, rc = f.result()
            if rc != 0:
                failed.append((rcut, utime, seed))
    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
    return failed


# ── Phase 2: リザバー評価 ─────────────────────────────────────────────────

def _eval_one(args):
    key, rcut, utime, seed, pos_path, params, input_path, target_path = args
    res = evaluate_reservoir(pos_path, params, target_path, input_path)
    res["rcut"], res["utime"], res["seed"] = rcut, utime, seed
    return key, res


def phase2_evaluate(rcut_values, utime_values, trial_seeds, narma_paths, data_dir, output_dir,
                    eval_jobs=1):
    """各 (rcut, utime, seed) を evaluate_reservoir で評価する。

    /mnt/d からの position.dat 読み込みが律速（1 本約 2 分）なので eval_jobs 並列で評価する。
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rcut in rcut_values:
        for utime in utime_values:
            for seed in trial_seeds:
                key = f"rcut={rcut:g}_utime={int(utime)}_seed={seed}"

                exp_dir = _find_exp_utime(data_dir, rcut, utime, seeds_for(seed)["seed_pos"])
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
                jobs.append((key, rcut, utime, seed, pos_path, params, input_path, target_path))

    results = {}
    with ProcessPoolExecutor(max_workers=eval_jobs) as ex:
        futures = [ex.submit(_eval_one, j) for j in jobs]
        for f in as_completed(futures):
            key, res = f.result()
            results[key] = res
            print(f"[EVAL {len(results)}/{len(jobs)}] {key}  NRMSE_test={res['nrmse_test']:.4f}  "
                  f"MC_test={res['MC_test']:.3f}", flush=True)

    return results


# ── Phase 3: ヒートマップ ────────────────────────────────────────────────

def phase3_plot(results, rcut_values, utime_values, output_dir):
    output_dir = Path(output_dir)

    rows = [
        {"rcut": v["rcut"], "utime": v["utime"], "seed": v["seed"],
         "MC_test": v["MC_test"], "NRMSE_test": v["nrmse_test"]}
        for v in sorted(results.values(), key=lambda x: (x["rcut"], x["utime"], x["seed"]))
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "heatmap_data.csv", index=False)
    print("  heatmap_data.csv", flush=True)

    df_mean = df.groupby(["rcut", "utime"])[["MC_test", "NRMSE_test"]].mean().reset_index()

    plot_metric_vs_utime(df, output_dir)
    plot_metric_vs_utime_scatter(df, output_dir)

    for metric, fname, cmap, fmt in [
        ("MC_test",    "heatmap_mc.png",    "Blues",   ".2f"),
        ("NRMSE_test", "heatmap_nrmse.png", "Reds_r",  ".3f"),
    ]:
        pivot = df_mean.pivot(index="utime", columns="rcut", values=metric)
        pivot = pivot.reindex(index=sorted(utime_values), columns=sorted(rcut_values))

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       origin="lower",
                       extent=[-0.5, len(rcut_values) - 0.5,
                               -0.5, len(utime_values) - 0.5])
        ax.set_xticks(range(len(rcut_values)))
        ax.set_xticklabels([str(r) for r in sorted(rcut_values)])
        ax.set_yticks(range(len(utime_values)))
        ax.set_yticklabels([str(u) for u in sorted(utime_values)])
        ax.set_xlabel("rcut (interaction radius)")
        ax.set_ylabel("utime (simulation steps per input value)")
        ax.set_title(f"{metric} (mean over trial seeds, v0=0, sgm=0)")
        fig.text(0.01, 0.005, "MC_test: memory capacity on test interval,  NRMSE_test: NARMA10 normalized RMSE,  "
                 "v0: self-propulsion speed,  sgm: noise strength", fontsize=7, color="#52514e")
        plt.colorbar(im, ax=ax)

        for i, utime in enumerate(sorted(utime_values)):
            for j, rcut in enumerate(sorted(rcut_values)):
                val = pivot.at[utime, rcut]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center",
                            fontsize=7, color="black")

        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {fname}", flush=True)


def plot_metric_vs_utime(df, output_dir):
    """rcut ごとに、横軸 utime・縦軸 MC_test / NRMSE_test（試行平均 ± 標準偏差）の線グラフを描く。

    元データは heatmap_data.csv（試行ごとの値）。集計値は metric_vs_utime.csv に保存する。
    """
    agg = (df.groupby(["rcut", "utime"])
             .agg(MC_mean=("MC_test", "mean"), MC_std=("MC_test", "std"),
                  NRMSE_mean=("NRMSE_test", "mean"), NRMSE_std=("NRMSE_test", "std"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    agg.to_csv(output_dir / "metric_vs_utime.csv", index=False)

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for col, fname, ylabel in [
        ("MC", "mc_vs_utime.png", "MC_test (memory capacity on test interval)"),
        ("NRMSE", "nrmse_vs_utime.png", "NRMSE_test (NARMA10 normalized RMSE on test interval)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for i, (rcut, g) in enumerate(agg.groupby("rcut")):
            g = g.sort_values("utime")
            ax.errorbar(g["utime"], g[f"{col}_mean"], yerr=g[f"{col}_std"].fillna(0),
                        color=colors[i % len(colors)], marker="o", ms=5, lw=2, capsize=2.5,
                        mec="white", mew=0.8, label=f"rcut = {rcut:g}")
        ax.set_xlabel("utime (simulation steps per input value)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        n = int(agg["n_seeds"].max())
        ax.set_title(f"v0=0, sgm=0 (mean ± std over {n} trial seeds)", loc="left", fontsize=10)
        ax.grid(True, color="0.9", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.text(0.01, 0.005, "rcut: interaction radius,  v0: self-propulsion speed,  sgm: noise strength",
                 fontsize=7.5, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {fname}", flush=True)


def plot_metric_vs_utime_scatter(df, output_dir):
    """エラーバーの代わりに、試行ごとの値を点で描く（rcut ごとに 1 枚、MC と NRMSE）。

    点の色と形で試行（trial seed）を区別し、灰色の線で試行平均を示す。元データは heatmap_data.csv。
    """
    seed_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for rcut, g in df.groupby("rcut"):
        for col, label, fname in [
            ("MC_test", "MC_test (memory capacity on test interval)", "mc_vs_utime_scatter"),
            ("NRMSE_test", "NRMSE_test (NARMA10 normalized RMSE on test interval)", "nrmse_vs_utime_scatter"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            mean = g.groupby("utime")[col].mean()
            ax.plot(mean.index, mean.values, color="0.6", lw=1.2, zorder=1, label="mean over trials")
            for i, (seed, gs) in enumerate(g.groupby("seed")):
                gs = gs.sort_values("utime")
                ax.scatter(gs["utime"], gs[col], s=28, color=seed_colors[i % len(seed_colors)],
                           marker=markers[i % len(markers)], edgecolors="white", linewidths=0.6,
                           zorder=2, label=f"trial seed b = {seed}")
            ax.set_xlabel("utime (simulation steps per input value)")
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.set_title(f"rcut = {rcut:g}, v0=0, sgm=0 (each point = one trial)", loc="left", fontsize=10)
            ax.grid(True, color="0.9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8.5)
            fig.text(0.01, 0.005,
                     "rcut: interaction radius,  v0: self-propulsion speed,  sgm: noise strength\n"
                     "trial seed b: simulation seeds from b (configs/seed_policy.json), NARMA input seed = b",
                     fontsize=7, color="#52514e")
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            suffix = "" if df["rcut"].nunique() == 1 else f"_rcut{rcut:g}"
            fig.savefig(output_dir / f"{fname}{suffix}.png", dpi=150)
            plt.close(fig)
            print(f"  {fname}{suffix}.png", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="rcut × utime sweep heatmap (v0=0, sgm=0)")
    p.add_argument("--rcut-values",  nargs="+", type=float, default=_CA["rcut_values"])
    p.add_argument("--utime-values", nargs="+", type=int,   default=_CA["utime_values"])
    p.add_argument("--trial-seeds",  nargs="+", type=int,   default=_CA["trial_seeds"])
    p.add_argument("--seed-scheme", choices=["random64", "legacy"], default=None,
                   help="試行の seed の規則（configs/seed_policy.json）。既定は random64。既存データの再評価は legacy")
    p.add_argument("--n-jobs",    type=int, default=4)
    p.add_argument("--data-dir",  default="data")
    p.add_argument("--output-dir", default="analysis/rcut_utime_heatmap")
    p.add_argument("--narma-dir", default="narma_data")
    p.add_argument("--skip-sim",  action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="シミュレーションだけ行い評価・作図をしない")
    p.add_argument("--eval-jobs", type=int, default=4, help="評価の並列数（I/O 律速）")
    p.add_argument("--replot", default=None, metavar="OUTPUT_DIR",
                   help="既存の出力ディレクトリの heatmap_data.csv から、試行ごとの散布図だけを描き直す")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed_scheme(args.seed_scheme)
    if args.replot:
        # 既存の heatmap_data.csv から図だけ描き直す（評価はやり直さない）
        out = Path(args.replot)
        df = pd.read_csv(out / "heatmap_data.csv")
        results = {f"{r.rcut}_{r.utime}_{r.seed}": {"rcut": r.rcut, "utime": r.utime, "seed": r.seed,
                                                    "MC_test": r.MC_test, "nrmse_test": r.NRMSE_test}
                   for r in df.itertuples()}
        phase3_plot(results, sorted(df.rcut.unique()), sorted(df.utime.unique()), out)
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts

    narma_paths = {s: ensure_narma10(s, args.narma_dir) for s in args.trial_seeds}
    valid_seeds = [s for s, (ip, _) in narma_paths.items() if ip is not None]
    if len(valid_seeds) < len(args.trial_seeds):
        skipped = set(args.trial_seeds) - set(valid_seeds)
        print(f"[WARN] Skipping diverged seeds: {sorted(skipped)}", flush=True)

    if not args.skip_sim:
        phase1_simulate(args.rcut_values, args.utime_values, valid_seeds,
                        narma_paths, args.data_dir, args.n_jobs)

    if args.skip_eval:
        print(f"--skip-eval: simulations only. Rebuild the catalog, then rerun with --skip-sim.", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = phase2_evaluate(args.rcut_values, args.utime_values, valid_seeds,
                              narma_paths, args.data_dir, output_dir, args.eval_jobs)

    if results:
        phase3_plot(results, args.rcut_values, args.utime_values, output_dir)
    else:
        print("[WARN] No results to plot.", flush=True)

    # params_used.json — モデルパラメータを index.csv から収集
    index_path = Path(args.data_dir) / "index.csv"
    if index_path.exists():
        df_idx = pd.read_csv(index_path, dtype={"dir": str})
        params_list = []
        for rcut in args.rcut_values:
            for utime in args.utime_values:
                for seed in valid_seeds:
                    exp_dir = _find_exp_utime(
                        Path(args.data_dir), rcut, utime, seeds_for(seed)["seed_pos"])
                    if exp_dir and (exp_dir / "params_model.json").exists():
                        with open(exp_dir / "params_model.json") as f:
                            params_list.append(json.load(f))
        if params_list:
            write_params_used(output_dir, params_list, RESERVOIR_DEFAULTS)
            print("  params_used.json", flush=True)

    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
