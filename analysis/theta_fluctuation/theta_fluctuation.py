"""θ ゆらぎ比較（task_14）の解析と作図。

定数入力 ū_b（ノイズのみ）と NARMA 入力（入力＋ノイズ）について、sgm ごとに

    D_θ   = sqrt( (1/T) Σ_t (1/N) Σ_i wrap(θ_i(t) − ⟨θ_i⟩)^2 ),  ⟨θ_i⟩ = arg mean_t e^{iθ_i}
    D_sin = sqrt( (1/T) Σ_t (1/N) Σ_i (sinθ_i(t) − ⟨sinθ_i⟩)^2 )

を求める（⟨·⟩ は粒子ごとの時間平均、区間はフレーム [washout, n_frames)）。
粒子群 all / locked / unlocked ごとに集計する。locked は同じ試行の定数入力・sgm=0 で
θ の時間方向 2 乗ずれが lock_threshold 以下の粒子。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import apply_style, load_reservoir_defaults, load_theta, write_params_used
from run_sims import build_config, find_existing, input_files

PARAMS_PATH = Path(__file__).parent / "default_params.json"
# dataviz リファレンスパレットの categorical slot 1, 2
STYLE = {"const": dict(color="#2a78d6", marker="o", label="constant input ū (noise only)"),
         "narma": dict(color="#eb6834", marker="s", label="NARMA input u(t) (input + noise)")}
QUANT = {"theta": r"$D_\theta$ [rad]", "sin": r"$D_{\sin}$"}


def particle_msd(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, N) の θ から粒子ごとの時間方向 2 乗ずれ（θ は円周、sinθ は通常）を返す。"""
    z = np.exp(1j * theta)
    mean_ang = np.angle(z.mean(axis=0))
    d = np.angle(z * np.exp(-1j * mean_ang))          # wrap(θ − ⟨θ⟩) ∈ (−π, π]
    msd_theta = (d ** 2).mean(axis=0)
    msd_sin = np.sin(theta).var(axis=0)
    return msd_theta, msd_sin


def sim_msd(args):
    """1 本のシムについて、区間ごとの粒子 2 乗ずれを返す（キャッシュがあれば読む）。"""
    pos_path, N, n_frames_expected, washout, cache_path = args
    if cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}
    theta = load_theta(pos_path, N)
    if theta.shape[0] != n_frames_expected:
        raise ValueError(f"{pos_path}: {theta.shape[0]} frames != {n_frames_expected} (incomplete run?)")
    mid = (washout + n_frames_expected) // 2
    res = {}
    for w, (t0, t1) in {"full": (washout, n_frames_expected), "first": (washout, mid),
                        "second": (mid, n_frames_expected)}.items():
        res[f"{w}_theta"], res[f"{w}_sin"] = particle_msd(theta[t0:t1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())
    RC = load_reservoir_defaults()
    washout = RC["washout"]
    N = P["model"]["N"]
    inputs = input_files(P)

    if args.smoke:
        S = P["smoke"]
        data_dir, ntime = ROOT / S["output_base"], S["ntime"]
        runs = [(m, s, S["trial_seed"]) for m, s in S["runs"]]
        trials = [S["trial_seed"]]
    else:
        data_dir, ntime = ROOT / P["data_dir"], P["model"]["ntime"]
        runs = [(m, s, b) for m in P["input_modes"] for s in P["sgm_values"] for b in P["trial_seeds"]]
        trials = P["trial_seeds"]

    # ── 粒子ごとの 2 乗ずれ（I/O 律速なので並列に読み、1 本ごとにキャッシュ） ──────
    cache_dir = ROOT / P["cache_dir"]
    n_frames = ntime // P["model"]["utime"]
    tasks, model_params = [], []
    for m, s, b in runs:
        cfg = build_config(P, m, s, b, inputs, data_dir, ntime)
        d = find_existing(data_dir, cfg, check_complete=False)
        if d is None:
            sys.exit(f"missing simulation: {(m, s, b)}")
        model_params.append(json.loads((d / "params_model.json").read_text()))
        tasks.append(((m, s, b), d, (d / "position.dat", N, n_frames, washout,
                                      cache_dir / data_dir.name / f"{d.name}.npz")))
    results = {}
    with ProcessPoolExecutor(max_workers=args.n_jobs or P["n_jobs"]) as ex:
        futs = {ex.submit(sim_msd, t[2]): t for t in tasks}
        for fut in as_completed(futs):
            key, d, _ = futs[fut]
            results[key] = (d, fut.result())
            print(f"[{datetime.now():%T}] {len(results)}/{len(tasks)} {key} <- {d.name}", flush=True)
    pp_rows = []
    for key, d, _ in tasks:
        _, res = results[key]
        for w in ("full", "first", "second"):
            for i in range(N):
                pp_rows.append((*key, w, i, res[f"{w}_theta"][i], res[f"{w}_sin"][i], d.name))
    pp = pd.DataFrame(pp_rows, columns=["input_mode", "sgm", "trial_seed", "window", "particle",
                                        "msd_theta", "msd_sin", "exp_dir"])

    # ── ロック判定（試行ごと、定数入力・sgm=0・全区間） ───────────────────────
    ref = pp[(pp.input_mode == "const") & (pp.sgm == 0.0) & (pp.window == "full")]
    lock = ref[["trial_seed", "particle", "msd_theta"]].rename(columns={"msd_theta": "msd_theta_const_sgm0"})
    lock["locked"] = lock["msd_theta_const_sgm0"] <= P["lock_threshold"]
    pp = pp.merge(lock[["trial_seed", "particle", "locked"]], on=["trial_seed", "particle"])
    for b in trials:
        print(f"trial {b}: locked {int(lock[lock.trial_seed == b].locked.sum())}/{N}")

    # ── D の計算（群ごとに 2 乗ずれを平均してから平方根） ─────────────────────
    rows = []
    u_const = {b: float(np.loadtxt(inputs[("const", b)], max_rows=1)) for b in trials}
    seeds = {b: {k: b + o for k, o in P["seed_offsets"].items()} for b in trials}
    groups = {"all": lambda df: df, "locked": lambda df: df[df.locked], "unlocked": lambda df: df[~df.locked]}
    for (m, s, b, w), g in pp.groupby(["input_mode", "sgm", "trial_seed", "window"]):
        for gname, sel in groups.items():
            gg = sel(g)
            rows.append(dict(input_mode=m, sgm=s, trial_seed=b, **seeds[b],
                             u_const=u_const[b] if m == "const" else np.nan,
                             particle_group=gname, window=w, n_particles=len(gg),
                             D_theta=np.sqrt(gg.msd_theta.mean()) if len(gg) else np.nan,
                             D_sin=np.sqrt(gg.msd_sin.mean()) if len(gg) else np.nan))
    fd = pd.DataFrame(rows)
    summ = (fd.groupby(["input_mode", "sgm", "particle_group", "window"])
              .agg(D_theta_mean=("D_theta", "mean"), D_theta_std=("D_theta", "std"),
                   D_sin_mean=("D_sin", "mean"), D_sin_std=("D_sin", "std"),
                   n_trials=("D_theta", "size"))
              .reset_index())

    out = Path(args.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    fd.to_csv(out / "fluct_data.csv", index=False)
    summ.to_csv(out / "fluct_summary.csv", index=False)
    lock.to_csv(out / "locked_mask.csv", index=False)
    pp[pp.window == "full"].drop(columns="window").to_csv(out / "per_particle.csv", index=False)
    write_params_used(out, model_params,
                      reservoir_fixed={"washout": washout, "lock_threshold": P["lock_threshold"],
                                       "u_const": u_const, "data_dir": str(data_dir)})

    # ── 作図 ────────────────────────────────────────────────────────────────
    apply_style()
    full = summ[summ.window == "full"]
    for q, ylabel in QUANT.items():
        for gname in ["all", "locked"]:
            plot_one(full[full.particle_group == gname], q, ylabel, gname,
                     out / f"fluct_{q}_vs_sgm{'' if gname == 'all' else '_locked'}.png")
    print(f"output -> {out}")


def plot_one(df, q, ylabel, gname, path):
    """横軸・縦軸とも線形（等間隔の目盛り）。縦軸は 0 から。"""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for m in ["const", "narma"]:
        d = df[df.input_mode == m].sort_values("sgm")
        y, e = d[f"D_{q}_mean"].to_numpy(), d[f"D_{q}_std"].fillna(0).to_numpy()
        st = STYLE[m]
        ax.errorbar(d.sgm, y, yerr=e, color=st["color"], marker=st["marker"], ms=6,
                    lw=2, capsize=3, label=st["label"], mec="white", mew=1)
    ref = df[(df.input_mode == "narma") & (df.sgm == 0.0)][f"D_{q}_mean"]
    if len(ref):
        ax.axhline(ref.iloc[0], color=STYLE["narma"]["color"], ls="--", lw=1,
                   label=r"NARMA input, $\sigma=0$ (input only)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"noise strength $\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"particles: {gname}", loc="left")
    ax.grid(True, which="major", color="0.9", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
