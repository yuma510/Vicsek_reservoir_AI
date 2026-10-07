"""task_15 段階 1: 予測平均（task_10）の各ノイズ実現の予測をキャッシュする。

予測は task_10 の `pred_mean.realization_predictions` で計算し（同じ readout・同じ ridge）、
NARMA 予測 y_pred と、指定した遅延 k の入力予測だけを float32 で 1 実現 1 ファイルに保存する。
sgm=0 は全実現が同じ予測になるので、先頭の seed 1 本だけ計算する。

`--smoke`: smoke 設定の sgm・先頭 n_seeds 実現を計算し、task_10 の per-seed CSV と
NRMSE / MC が一致するかを確認する（キャッシュは書かない）。
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from vicsek_rc import find_narma_by_seed, load_reservoir_defaults
from pred_mean import _metrics_from_avg, build_index, realization_predictions

PARAMS_PATH = Path(__file__).parent / "pred_distribution_params.json"


def load_narma(P):
    ip, tp = find_narma_by_seed(str(ROOT / P["narma_root"]), P["narma_seed"])
    y_target = np.loadtxt(tp)
    raw = np.loadtxt(ip)
    col_input = raw[:, 0] if raw.ndim > 1 else raw
    return y_target, col_input


def cache_path(P, sgm, seed) -> Path:
    return ROOT / P["cache_dir"] / f"sgm{sgm:.2f}_seed{seed:03d}.npz"


def _work(args):
    """1 実現の予測を計算。smoke では指標を、通常はキャッシュを書いてパスを返す。"""
    exp_dir, sgm, seed, P, RC, y_target, col_input, n_eval, smoke = args
    params = json.loads((exp_dir / "params_model.json").read_text())
    rp = realization_predictions(exp_dir, params, y_target, col_input, RC["washout"],
                                 RC["train_num"], RC["k_max"], RC["ridge_lambda"], n_eval)
    if rp is None:
        raise FileNotFoundError(f"no position.dat in {exp_dir}")
    if smoke:
        N, tr_end = params["N"], RC["washout"] + RC["train_num"]
        delayed = np.vstack([np.concatenate([np.zeros(k), col_input])[:n_eval]
                             for k in range(RC["k_max"] + 1)])
        m = _metrics_from_avg(rp["y_pred"], rp["mck_preds"], y_target, delayed,
                              slice(RC["washout"], min(tr_end + 1, n_eval)),
                              slice(min(tr_end, n_eval), n_eval), N / RC["train_num"])
        # 旧 MC 定義（k=0 を含む、2026-07-23 以前）との比較用に MC_0 も返す
        te = slice(min(tr_end, n_eval), n_eval)
        m["MC0_test"] = float(np.corrcoef(delayed[0, te], rp["mck_preds"][0, te])[0, 1] ** 2)
        return sgm, seed, exp_dir.name, m
    ks = np.array(P["k_values"])
    out = cache_path(P, sgm, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, y_pred=rp["y_pred"].astype(np.float32),
             k_pred=rp["mck_preds"][ks].astype(np.float32), k_values=ks,
             exp_dir=np.array(exp_dir.name))
    tmp.rename(out)  # 書きかけのファイルを完成扱いにしない
    return sgm, seed, exp_dir.name, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())
    RC = load_reservoir_defaults()
    y_target, col_input = load_narma(P)

    index = build_index(ROOT / P["data_dir"], P["rcut"], P["seed_pos"], P["seed_nf"],
                        P["v0"], narma_seed=P["narma_seed"])
    probe = json.loads((next(iter(index.values())) / "params_model.json").read_text())
    n_eval = int(min(probe["ntime"] // probe["utime"], len(y_target), len(col_input)))

    if args.smoke:
        S = P["smoke"]
        jobs = [(S["sgm"], s) for s in P["noise_seeds"][:S["n_seeds"]]]
    else:
        jobs = []
        for sgm in P["sgm_values"]:
            seeds = P["noise_seeds"][:1] if sgm == 0 else P["noise_seeds"]
            jobs += [(sgm, s) for s in seeds if not cache_path(P, sgm, s).exists()]
    missing = [j for j in jobs if (round(j[0], 4), j[1]) not in index]
    if missing:
        sys.exit(f"simulations not found: {missing}")
    print(f"[{datetime.now():%F %T}] n_eval={n_eval}, {len(jobs)} realizations to compute", flush=True)

    tasks = [(index[(round(sgm, 4), s)], sgm, s, P, RC, y_target, col_input, n_eval, args.smoke)
             for sgm, s in jobs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.n_jobs or P["n_jobs"]) as ex:
        futs = [ex.submit(_work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            sgm, s, d, m = fut.result()
            print(f"[{datetime.now():%T}] {i}/{len(tasks)} sgm={sgm} seed={s} <- {d}", flush=True)
            if m is not None:
                rows.append(dict(sgm=sgm, seed_noise=s, exp_dir=d, **m))

    if args.smoke:
        new = pd.DataFrame(rows)
        ref = pd.read_csv(ROOT / P["smoke"]["reference_csv"])
        if "S" in ref.columns:
            # pred_mean_data.csv 形式: S=1 は昇順で先頭の seed 単独の値
            ref = ref[ref.S == 1].assign(seed_noise=P["noise_seeds"][0])
        cmp = new.merge(ref, on=["sgm", "seed_noise"], suffixes=("", "_ref"))
        cmp["MC_test_oldDef"] = cmp["MC_test"] + cmp["MC0_test"]
        cols = ["sgm", "seed_noise", "nrmse_test", "nrmse_test_ref",
                "MC_test", "MC_test_oldDef", "MC_test_ref"]
        print(cmp[cols].to_string(index=False))
        ok = np.allclose(cmp["nrmse_test"], cmp["nrmse_test_ref"], rtol=1e-6)
        print("NRMSE match:", ok)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
