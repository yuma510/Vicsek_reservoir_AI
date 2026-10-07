"""定数入力ファイルの生成（task_14）。

試行 b ごとに NARMA 入力（seed b）の実測平均 ū_b を求め、長さ ntime/utime の定数入力ファイルを
`narma_data/<YYYYMMDD_HHMMSS>/` に保存する。

ファイル名は `const_input_mean-of-narma-s<b>_u<ū>.dat` とし、`seed<b>.dat` を含めない
（find_exp_dir の narma_seed フィルタ・find_narma_by_seed に NARMA 入力として誤って拾われないため）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from vicsek_rc import find_narma_by_seed, new_narma_dir

PARAMS_PATH = Path(__file__).parent / "default_params.json"


def const_input_name(b: int, u_mean: float) -> str:
    return f"const_input_mean-of-narma-s{b}_u{u_mean:.6f}.dat"


def find_const_inputs(narma_root: Path, trial_seeds) -> dict:
    """既存の定数入力ファイルを試行 seed ごとに返す（最新 dir 優先）。無ければ欠ける。"""
    found = {}
    for d in sorted(Path(narma_root).iterdir(), reverse=True):
        if not d.is_dir():
            continue
        for b in trial_seeds:
            if b in found:
                continue
            hits = sorted(d.glob(f"const_input_mean-of-narma-s{b}_u*.dat"))
            if hits:
                found[b] = hits[0]
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())

    n_input = P["model"]["ntime"] // P["model"]["utime"]
    narma_root = ROOT / P["narma_root"]
    existing = find_const_inputs(narma_root, P["trial_seeds"])
    todo = [b for b in P["trial_seeds"] if b not in existing]
    if not todo:
        print("all constant inputs exist:")
        for b, p in existing.items():
            print(f"  b={b}: {p}")
        return

    out_dir = new_narma_dir(str(narma_root))
    meta = {"length": n_input, "trials": {}}
    for b in todo:
        in_path, _ = find_narma_by_seed(str(narma_root), b)
        if in_path is None:
            raise FileNotFoundError(f"NARMA input for seed {b} not found under {narma_root}")
        u = np.loadtxt(in_path)
        if u.size < n_input:
            raise ValueError(f"{in_path} has {u.size} < {n_input} samples")
        u_mean = float(u[:n_input].mean())
        name = const_input_name(b, u_mean)
        np.savetxt(out_dir / name, np.full(n_input, u_mean), fmt="%.10f")
        meta["trials"][str(b)] = {"source_narma_input": str(in_path.relative_to(ROOT)),
                                  "u_mean": u_mean, "file": name}
        print(f"b={b}: u_mean={u_mean:.6f} (source {in_path.name}) -> {out_dir / name}")
    (out_dir / "const_input_params.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
