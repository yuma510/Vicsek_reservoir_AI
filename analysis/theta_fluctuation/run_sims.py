"""θ ゆらぎ比較（task_14）のシミュレーション実行。

条件 = 入力（const: 定数 ū_b ／ narma: NARMA seed b）× sgm × 試行 b。
seed は試行 b から `vicsek_rc.seeds.trial_seeds(b, seed_scheme)` で決める（条件間で共通、試行間ですべて変化）。

既に同じ条件のシム（params_model.json が全キー一致）がある場合はスキップする（再開可能）。
`--smoke` で default_params.json の smoke 設定（短い ntime、別の出力先）を使う。
"""
import argparse
import json
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
sys.path.insert(0, str(Path(__file__).parent))

from vicsek_rc import find_narma_by_seed
from vicsek_rc.seeds import trial_seeds as seeds_for
from make_constant_input import find_const_inputs

BINARY = str(ROOT / "vicsek_dynamic")
PARAMS_PATH = Path(__file__).parent / "default_params.json"


def input_files(P) -> dict:
    """{(mode, b): 入力ファイルの絶対パス}"""
    narma_root = ROOT / P["narma_root"]
    const = find_const_inputs(narma_root, P["trial_seeds"])
    out = {}
    for b in P["trial_seeds"]:
        if b not in const:
            raise FileNotFoundError(f"constant input for b={b} missing: run make_constant_input.py")
        ip, _ = find_narma_by_seed(str(narma_root), b)
        if ip is None:
            raise FileNotFoundError(f"NARMA input seed {b} missing")
        out[("const", b)] = str(Path(const[b]).resolve())
        out[("narma", b)] = str(Path(ip).resolve())
    return out


def build_config(P, mode, sgm, b, inputs, output_base, ntime) -> dict:
    cfg = dict(P["model"])
    cfg["ntime"] = int(ntime)
    cfg["sgm"] = float(sgm)
    cfg.update(seeds_for(b, P.get("seed_scheme")))   # configs/seed_policy.json の規則（JSON で legacy も指定可）
    cfg["input_file"] = inputs[(mode, b)]
    cfg["output_base"] = str(output_base)
    return cfg


def _matches(p: dict, cfg: dict) -> bool:
    for k, v in cfg.items():
        if k == "output_base":
            continue
        if k == "input_file":
            if Path(str(p.get(k, ""))).resolve() != Path(v).resolve():
                return False
        elif isinstance(v, float):
            if abs(float(p.get(k, float("nan"))) - v) >= 1e-6:
                return False
        elif p.get(k) != v:
            return False
    return True


def find_existing(data_dir: Path, cfg: dict, check_complete: bool = True) -> Path | None:
    """cfg と全キー一致し、position.dat が完走している最新 dir を返す。

    check_complete=False では行数を数えない（/mnt/d では wc -l が 1 本約 2 分かかるため、
    解析側は読み込んだ配列の形で完走を確認する）。
    """
    if not data_dir.exists():
        return None
    n_lines = cfg["ntime"] // cfg["utime"] * cfg["N"]
    for d in sorted(data_dir.iterdir(), reverse=True):
        pf = d / "params_model.json"
        if not pf.exists():
            continue
        try:
            p = json.loads(pf.read_text())
        except json.JSONDecodeError:
            continue
        if not _matches(p, cfg):
            continue
        pos = d / "position.dat"
        # 1 行は約 40 byte。完走判定は行数で厳密に行う（解析側でも再確認する）
        if pos.exists() and pos.stat().st_size > 0 and (not check_complete or _count_lines(pos) == n_lines):
            return d
    return None


def _count_lines(path: Path) -> int:
    r = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0])


def _run_one(cfg: dict):
    fd, p = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)  # 1 行 1 フィールド（vicsek_dynamic の sscanf 仕様）
        r = subprocess.run([BINARY, p], capture_output=True, text=True)
    finally:
        os.unlink(p)
    return r.returncode, r.stdout[-300:], r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())
    inputs = input_files(P)

    if args.smoke:
        S = P["smoke"]
        out_base = ROOT / S["output_base"]
        ntime = S["ntime"]
        runs = [(m, s, S["trial_seed"]) for m, s in S["runs"]]
    else:
        out_base = ROOT / P["data_dir"]
        ntime = P["model"]["ntime"]
        runs = [(m, s, b) for m in P["input_modes"] for s in P["sgm_values"] for b in P["trial_seeds"]]
    out_base.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], 0
    for m, s, b in runs:
        cfg = build_config(P, m, s, b, inputs, out_base, ntime)
        if find_existing(out_base, cfg) is not None:
            skipped += 1
            continue
        jobs.append(((m, s, b), cfg))
    print(f"[{datetime.now():%F %T}] {len(runs)} runs: {skipped} done, {len(jobs)} to run -> {out_base}",
          flush=True)

    # 空き容量の確認（CLAUDE.md）: 1 本あたり ntime/utime*N 行 × 約 40 byte
    need_gb = len(jobs) * ntime // P["model"]["utime"] * P["model"]["N"] * 40 / 1e9
    free_gb = shutil.disk_usage(out_base).free / 1e9
    print(f"disk: free {free_gb:.0f} GB, need ~{need_gb:.1f} GB (min_free_gb={P['min_free_gb']})", flush=True)
    if free_gb - need_gb < P["min_free_gb"]:
        sys.exit("not enough disk space")
    if args.dry_run or not jobs:
        for key, _ in jobs:
            print("  todo:", key)
        return

    n_jobs = args.n_jobs or P["n_jobs"]
    failed = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_run_one, cfg): key for key, cfg in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            rc, out, err = fut.result()
            status = "ok" if rc == 0 else f"FAIL rc={rc} {err.strip()}"
            if rc != 0:
                failed.append(key)
            print(f"[{datetime.now():%F %T}] {i}/{len(jobs)} {key} {status}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
