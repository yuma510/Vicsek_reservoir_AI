"""入力なし（F=0）の相転移（task_16）のシミュレーション実行。

条件 = K × Ω × 試行 b。他のパラメータは default_params.json の "base"（段階 1 は論文のセットアップ）。
Ω（= ω/D_r）から自然周波数の平均を nf_mean = Ω·D_r/(2π)、D_r = sgm²/2 で決める。
seed は試行 b から `vicsek_rc.seeds.trial_seeds(b, seed_scheme)` で決める（seed_scheme は JSON。既定は configs/seed_policy.json）。

既に同じ条件のシム（params_model.json が全キー一致）がある場合はスキップする（再開可能）。
`--smoke` で smoke 設定（短い ntime、別の出力先）を使う。`--params` で段階 2 の別 JSON を指定できる。
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "theta_fluctuation"))

# task_14 と同じ「全キー一致」の判定を使う。このファイルも run_sims.py なので、パスを指定して読み込む
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("theta_fluctuation_run_sims",
                                     ROOT / "analysis" / "theta_fluctuation" / "run_sims.py")
_tf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tf)
find_existing = _tf.find_existing

from vicsek_rc.seeds import trial_seeds as seeds_for

BINARY = str(ROOT / "vicsek_dynamic")
PARAMS_PATH = Path(__file__).parent / "default_params.json"


def build_config(P, K, Omega, b, output_base, ntime) -> dict:
    cfg = dict(P["base"])
    cfg["ntime"] = int(ntime)
    cfg["K"] = float(K)
    D_r = cfg["sgm"] ** 2 / 2.0
    cfg["nf_mean"] = round(float(Omega) * D_r / (2.0 * math.pi), 8)
    cfg.update(seeds_for(b, P.get("seed_scheme")))   # configs/seed_policy.json の規則（JSON で legacy も指定可）
    cfg["input_file"] = str((ROOT / P["input_file"]).resolve())
    cfg["output_base"] = str(output_base)
    return cfg


def conditions(P, smoke=False):
    if smoke:
        S = P["smoke"]
        return [(K, Om, S["trial_seed"]) for K in S["K_values"] for Om in S["Omega_values"]]
    return [(K, Om, b) for K in P["K_values"] for Om in P["Omega_values"] for b in P["trial_seeds"]]


def _run_one(cfg: dict):
    fd, p = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)  # 1 行 1 フィールド（vicsek_dynamic の sscanf 仕様）
        r = subprocess.run([BINARY, p], capture_output=True, text=True)
    finally:
        os.unlink(p)
    return r.returncode, r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())

    if args.smoke:
        out_base, ntime = ROOT / P["smoke"]["output_base"], P["smoke"]["ntime"]
    else:
        out_base, ntime = ROOT / P["data_dir"], P["base"]["ntime"]
    out_base.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], 0
    for K, Om, b in conditions(P, args.smoke):
        cfg = build_config(P, K, Om, b, out_base, ntime)
        if find_existing(out_base, cfg) is not None:
            skipped += 1
            continue
        jobs.append(((K, Om, b), cfg))
    print(f"[{datetime.now():%F %T}] {skipped + len(jobs)} runs: {skipped} done, {len(jobs)} to run -> {out_base}",
          flush=True)

    b0 = P["base"]
    need_gb = len(jobs) * ntime // b0["utime"] * b0["N"] * 40 / 1e9
    free_gb = shutil.disk_usage(out_base).free / 1e9
    print(f"disk: free {free_gb:.0f} GB, need ~{need_gb:.1f} GB", flush=True)
    if free_gb - need_gb < P["min_free_gb"]:
        sys.exit("not enough disk space")
    if args.dry_run or not jobs:
        return

    failed = []
    with ProcessPoolExecutor(max_workers=args.n_jobs or P["n_jobs"]) as ex:
        futs = {ex.submit(_run_one, cfg): key for key, cfg in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            rc, err = fut.result()
            if rc != 0:
                failed.append(key)
            print(f"[{datetime.now():%T}] {i}/{len(jobs)} K={key[0]} Ω={key[1]} b={key[2]} "
                  f"{'ok' if rc == 0 else 'FAIL ' + err.strip()}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
