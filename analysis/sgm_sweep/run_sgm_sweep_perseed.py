"""sgm スイープ（per-seed NARMA 版）— rcut_sweep と同じ seed の取り方に合わせる。

従来の run_sgm_sweep.py は全 seed を単一 NARMA（seed666）で駆動していたため、
seed666 固有の外れ値（sgm=0 で test 汎化失敗）が出た。本スクリプトは rcut_sweep と同様に
**各 trial seed を自分の NARMA（seed=trial）で駆動**する。

- trial seeds = [1,2,3,5,6,7,8,9,10]（rcut_sweep と同一、trial=4 欠番）
- 固定: rcut=13, v0=0.5, ntime=140000
- sgm=0 は rcut_sweep の rcut=13 シムを流用（既存）。sgm>0 のみ新規生成。
- 生成後に catalog 再構築 → 評価 → プロット（run_sgm_sweep の phase3_plot / plot_errorbar 再利用）
"""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/sgm_sweep"))

import numpy as np
import pandas as pd

from vicsek_rc.seeds import trial_seeds as seeds_for, set_seed_scheme  # 試行 b → seed（seed_policy.json）
from vicsek_rc import (evaluate_reservoir, find_exp_dir, find_narma_by_seed,
                       load_reservoir_defaults, write_params_used)
from vicsek_rc.catalog import write_catalog
import run_sgm_sweep as base

BINARY = str(ROOT / "vicsek_dynamic")
DATA = ROOT / "data"
_RC = load_reservoir_defaults()

TRIALS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SGM_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
RCUT, V0, NTIME = 13.0, 0.5, 140000
N_JOBS = 24
YMAX = {"MC": 15.0, "nrmse": 0.85}

# trial seed -> NARMA (input, target)
NARMA = {}
for s in TRIALS:
    ip, tp = find_narma_by_seed(str(ROOT / "narma_data"), s)
    NARMA[s] = (str(ip), str(tp))


def _run_one(args):
    sgm, s = args
    cfg = {"input_file": NARMA[s][0], "output_base": str(DATA), "rcut": RCUT, "v0": V0,
           "sgm": float(sgm), **seeds_for(s),
           "ntime": NTIME}
    fd, p = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)   # 1行1フィールド
        r = subprocess.run([BINARY, p], capture_output=True, text=True)
    finally:
        os.unlink(p)
    return sgm, s, r.returncode, r.stderr[:150]


def main():
    import argparse
    global V0, TRIALS, N_JOBS, NARMA, DATA
    p = argparse.ArgumentParser(description="per-seed sgm sweep")
    p.add_argument("--v0", type=float, default=0.5)
    p.add_argument("--trials", nargs="+", type=int, default=[1, 2, 3, 5, 6, 7, 8, 9, 10])
    p.add_argument("--seed-scheme", choices=["random64", "legacy"], default=None,
                   help="試行の seed の規則（configs/seed_policy.json）。既定は random64。既存データの再評価は legacy")
    p.add_argument("--n-jobs", type=int, default=N_JOBS)
    p.add_argument("--data-dir", default=None, help="データ dir（既定 data。修正後は data_newK 等）")
    args = p.parse_args()
    set_seed_scheme(args.seed_scheme)
    V0 = args.v0
    TRIALS = args.trials
    N_JOBS = args.n_jobs
    if args.data_dir:
        DATA = ROOT / args.data_dir
    NARMA = {}
    for s in TRIALS:
        ip, tp = find_narma_by_seed(str(ROOT / "narma_data"), s)
        NARMA[s] = (str(ip), str(tp))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "analysis/sgm_sweep" / f"{ts}_perseed_v0={V0:g}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"output → {out}  (v0={V0}, trials={TRIALS})", flush=True)

    # Phase 1: sgm>0 の欠損シムを生成（per-seed NARMA 駆動）
    jobs = []
    for sgm in SGM_VALS:
        for s in TRIALS:
            d = find_exp_dir(DATA, rcut=RCUT, sgm=sgm, v0=V0, ntime=NTIME,
                             narma_seed=s, seed=seeds_for(s)["seed_pos"], seed_key="seed_pos")
            ok = d is not None and (json.load(open(d / "params_model.json")).get("input_file", "") or "").endswith(f"seed{s}.dat")
            if not ok:
                jobs.append((sgm, s))
    print(f"Phase1: 生成 {len(jobs)} 件（sgm=0 と既存は流用）", flush=True)
    if jobs:
        failed = []
        done = 0
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            futs = {ex.submit(_run_one, j): j for j in jobs}
            for f in as_completed(futs):
                sgm, s, rc, err = f.result()
                done += 1
                if rc != 0:
                    failed.append((sgm, s)); print(f"[ERR {done}/{len(jobs)}] sgm={sgm} seed={s}: {err}", flush=True)
                elif done % 10 == 0:
                    print(f"[{done}/{len(jobs)}] done", flush=True)
        print("rebuilding catalog …", flush=True)
        write_catalog(str(DATA))
        if failed:
            print(f"[WARN] failed: {failed}", flush=True)

    # Phase 2: 評価（per-seed NARMA で照合・評価）
    results = {}
    model_params = []
    for sgm in SGM_VALS:
        for s in TRIALS:
            d = find_exp_dir(DATA, rcut=RCUT, sgm=sgm, v0=V0, ntime=NTIME,
                             narma_seed=s, seed=seeds_for(s)["seed_pos"], seed_key="seed_pos")
            if d is None:
                print(f"[MISS] sgm={sgm} seed={s}", flush=True); continue
            params = json.load(open(d / "params_model.json"))
            ip, tp = NARMA[s]
            res = evaluate_reservoir(d / "position.dat", params, tp, ip)
            res["sgm"] = sgm; res["seed"] = s
            results[f"sgm={sgm}_seed={s}"] = res
            model_params.append(params)
            print(f"  sgm={sgm} seed={s}: MC_test={res['MC_test']:.3f} NRMSE_test={res['nrmse_test']:.4f}", flush=True)

    # Phase 3: プロット（run_sgm_sweep の関数を再利用）+ 共通軸 errorbar
    base.phase3_plot(results, out)
    rows = pd.read_csv(out / "sgm_sweep_data.csv").to_dict("records")
    base.plot_errorbar(rows, out, ymax=YMAX)

    reservoir_fixed = {"readout": _RC["readout"], "washout": _RC["washout"],
                       "train_num": _RC["train_num"], "ridge_lambda": _RC["ridge_lambda"],
                       "k_max": _RC["k_max"], "narma": "per-seed (seed=trial)"}
    write_params_used(out, model_params, reservoir_fixed, {"sgm": SGM_VALS, "trial_seed": TRIALS})
    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()
