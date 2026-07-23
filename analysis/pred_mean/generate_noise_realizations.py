"""task_10 拡張: noise-averaged 集合の追加シミュレーション生成

pred_mean.py の S 掃引を S=100 まで伸ばすため、既存の noise-averaged 集合
（rcut=13, seed_pos=13, seed_nf=16, v0=0.5, ntime=140000）に不足している
seed_noise の実現を追加生成する。パラメータは default_params.json から読む
（CLAUDE.md §5.1、直書き禁止）。

- 既存シムは data/index.csv で確認してスキップ（再実行安全）
- sgm=0 は noise 項が消え全実現が同一になるため対象外（default_params.json 参照）
- index.csv への追記は行わず、最後に vicsek_rc.catalog で全再構築する
  （並列 append の競合による取りこぼしを避ける）

実行例:
    python analysis/pred_mean/generate_noise_realizations.py
    python analysis/pred_mean/generate_noise_realizations.py --n-jobs 8
"""
import argparse
import json
import subprocess
import sys
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from vicsek_rc import find_narma_by_seed
from vicsek_rc.catalog import write_catalog

BINARY = str(ROOT / "vicsek_dynamic")
_GP = json.loads((Path(__file__).parent / "default_params.json").read_text())


def existing_keys(data_dir, gp):
    """index.csv から既存の (sgm, seed_noise) ペア集合を返す。"""
    idx_path = Path(data_dir) / "index.csv"
    if not idx_path.exists():
        return set()
    df = pd.read_csv(idx_path)
    sub = df[(df.seed_pos == gp["seed_pos"]) & (df.seed_nf == gp["seed_nf"])
             & (abs(df.rcut - gp["rcut"]) < 1e-6) & (abs(df.v0 - gp["v0"]) < 1e-6)
             & (df.ntime == gp["ntime"]) & (df.has_position == True)]  # noqa: E712
    return {(round(float(s), 4), int(n)) for s, n in zip(sub.sgm, sub.seed_noise)}


def _run_one_sim(args):
    sgm, seed_noise, input_path, data_dir, gp = args
    cfg_lines = {
        "input_file":  str(input_path),
        "output_base": str(data_dir),
        "rcut":        float(gp["rcut"]),
        "v0":          float(gp["v0"]),
        "sgm":         float(sgm),
        "seed_noise":  int(seed_noise),
        "seed_pos":    int(gp["seed_pos"]),
        "seed_nf":     int(gp["seed_nf"]),
        "ntime":       int(gp["ntime"]),
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg_lines, f, indent=2)   # indent=2 → 1 行 1 フィールド（パーサ要件）
        result = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    finally:
        os.unlink(cfg_path)
    return sgm, seed_noise, result.returncode, result.stderr[:200]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-jobs", type=int, default=_GP["n_jobs"])
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    input_path, _ = find_narma_by_seed("narma_data", _GP["narma_seed"])
    if input_path is None:
        sys.exit(f"NARMA10 (seed={_GP['narma_seed']}) が narma_data/ に見つからない。")
    print(f"input: {input_path}", flush=True)

    seeds = range(_GP["seed_noise_start"], _GP["seed_noise_end"] + 1)
    have = existing_keys(data_dir, _GP)
    jobs = [(sgm, sn, input_path, data_dir, _GP)
            for sgm in _GP["sgm_values"] for sn in seeds
            if (round(float(sgm), 4), int(sn)) not in have]
    n_skip = len(_GP["sgm_values"]) * len(seeds) - len(jobs)
    print(f"jobs: {len(jobs)} (skip existing: {n_skip}, n_jobs={args.n_jobs})", flush=True)

    failed = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(_run_one_sim, j): j for j in jobs}
        for f in as_completed(futures):
            sgm, sn, rc, err = f.result()
            done += 1
            if rc != 0:
                failed.append((sgm, sn))
                print(f"[ERROR {done}/{len(jobs)}] sgm={sgm} seed_noise={sn}: {err}", flush=True)
            else:
                print(f"[done {done}/{len(jobs)}] sgm={sgm} seed_noise={sn}", flush=True)

    # index.csv を全再構築（並列 append の競合回避、CLAUDE.md メモ参照）
    print("rebuilding catalog …", flush=True)
    write_catalog(str(data_dir))

    if failed:
        print(f"[WARN] {len(failed)} simulations failed: {failed}", flush=True)
        sys.exit(1)
    print("all simulations finished.", flush=True)


if __name__ == "__main__":
    main()
