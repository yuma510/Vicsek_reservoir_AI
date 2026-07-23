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
                if _find_exp_utime(data_dir, r, u, s + 3) is not None:
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

def phase2_evaluate(rcut_values, utime_values, trial_seeds, narma_paths, data_dir, output_dir):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for rcut in rcut_values:
        for utime in utime_values:
            for seed in trial_seeds:
                key = f"rcut={int(rcut)}_utime={int(utime)}_seed={seed}"

                exp_dir = _find_exp_utime(data_dir, rcut, utime, seed + 3)
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
                res["rcut"]  = rcut
                res["utime"] = utime
                res["seed"]  = seed
                results[key] = res
                print(f"       NRMSE_test={res['nrmse_test']:.4f}  MC_test={res['MC_test']:.3f}",
                      flush=True)

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
        ax.set_xlabel("rcut")
        ax.set_ylabel("utime")
        ax.set_title(f"{metric} (seed mean, v0=0, sgm=0)")
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


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="rcut × utime sweep heatmap (v0=0, sgm=0)")
    p.add_argument("--rcut-values",  nargs="+", type=float, default=_CA["rcut_values"])
    p.add_argument("--utime-values", nargs="+", type=int,   default=_CA["utime_values"])
    p.add_argument("--trial-seeds",  nargs="+", type=int,   default=_CA["trial_seeds"])
    p.add_argument("--n-jobs",    type=int, default=4)
    p.add_argument("--data-dir",  default="data")
    p.add_argument("--output-dir", default="analysis/rcut_utime_heatmap")
    p.add_argument("--narma-dir", default="narma_data")
    p.add_argument("--skip-sim",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    narma_paths = {s: ensure_narma10(s, args.narma_dir) for s in args.trial_seeds}
    valid_seeds = [s for s, (ip, _) in narma_paths.items() if ip is not None]
    if len(valid_seeds) < len(args.trial_seeds):
        skipped = set(args.trial_seeds) - set(valid_seeds)
        print(f"[WARN] Skipping diverged seeds: {sorted(skipped)}", flush=True)

    if not args.skip_sim:
        phase1_simulate(args.rcut_values, args.utime_values, valid_seeds,
                        narma_paths, args.data_dir, args.n_jobs)

    results = phase2_evaluate(args.rcut_values, args.utime_values, valid_seeds,
                              narma_paths, args.data_dir, output_dir)

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
                        Path(args.data_dir), rcut, utime, seed + 3)
                    if exp_dir and (exp_dir / "params_model.json").exists():
                        with open(exp_dir / "params_model.json") as f:
                            params_list.append(json.load(f))
        if params_list:
            write_params_used(output_dir, params_list, RESERVOIR_DEFAULTS)
            print("  params_used.json", flush=True)

    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
