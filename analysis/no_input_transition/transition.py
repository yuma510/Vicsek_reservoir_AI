"""入力なし（F=0）の相転移（task_16）の解析と作図。

各シムについて、秩序変数 ψ(t) = |(1/N) Σ_j e^{iθ_j(t)}| を計算し、定常区間（後ろから steady_frac）で
  ψ_ss = ⟨ψ⟩、χ = N(⟨ψ²⟩ − ⟨ψ⟩²)
を求める。χ は E2（Vicsek_rotate/analysis/fig3_sm.py）と同じく、3 試行の定常サンプルをまとめて計算する（pooled）。
試行ごとの χ も CSV に残す。χ（pooled）が最大になる K を転移点 K_c とする。

論文の量との対応（近傍平均の結合）: gρ0 ≈ K/(π D_r)、D_r = sgm²/2。
"""
import argparse
import json
import math
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

from vicsek_rc import apply_style, load_theta, write_params_used
from run_sims import PARAMS_PATH, build_config, conditions, find_existing

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz リファレンス slot 1〜5
C_REF = "#52514e"


def psi_series(args):
    """1 本のシムの ψ(t) を返す（キャッシュがあれば読む）。"""
    pos_path, N, n_frames, cache_path = args
    if cache_path.exists():
        return np.load(cache_path)
    theta = load_theta(pos_path, N)
    if theta.shape[0] != n_frames:
        raise ValueError(f"{pos_path}: {theta.shape[0]} frames != {n_frames} (incomplete run?)")
    psi = np.abs(np.exp(1j * theta).mean(axis=1))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, psi)
    return psi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())
    b0 = P["base"]
    N = b0["N"]
    D_r = b0["sgm"] ** 2 / 2.0

    if args.smoke:
        data_dir, ntime = ROOT / P["smoke"]["output_base"], P["smoke"]["ntime"]
    else:
        data_dir, ntime = ROOT / P["data_dir"], b0["ntime"]
    n_frames = ntime // b0["utime"]
    n_ss = max(1, int(n_frames * P["steady_frac"]))

    tasks, model_params = [], []
    for K, Om, b in conditions(P, args.smoke):
        cfg = build_config(P, K, Om, b, data_dir, ntime)
        d = find_existing(data_dir, cfg, check_complete=False)
        if d is None:
            sys.exit(f"missing simulation: K={K} Ω={Om} b={b}")
        model_params.append(json.loads((d / "params_model.json").read_text()))
        cache = ROOT / P["cache_dir"] / data_dir.name / f"{d.name}.npy"
        tasks.append(((K, Om, b), d, (d / "position.dat", N, n_frames, cache)))

    psis = {}
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(psi_series, t[2]): t for t in tasks}
        for fut in as_completed(futs):
            key, d, _ = futs[fut]
            psis[key] = (d.name, fut.result())
    print(f"[{datetime.now():%T}] loaded {len(psis)} simulations", flush=True)

    # ── 試行ごとの値 ──
    rows = []
    for (K, Om, b), (dname, psi) in sorted(psis.items()):
        ss = psi[-n_ss:]
        half = len(psi) // 2
        rows.append(dict(K=K, Omega=Om, trial_seed=b, grho0=K / (math.pi * D_r), exp_dir=dname,
                         psi_ss=ss.mean(), psi_ss_std=ss.std(), chi_trial=N * ss.var(),
                         psi_first_half=psi[:half].mean(), psi_second_half=psi[half:].mean(),
                         n_samples=len(ss)))
    df = pd.DataFrame(rows)

    # ── 試行をまとめた値（χ は E2 と同じく定常サンプルをまとめて計算） ──
    summ = []
    for (K, Om), g in df.groupby(["K", "Omega"]):
        pooled = np.concatenate([psis[(K, Om, b)][1][-n_ss:] for b in g.trial_seed])
        summ.append(dict(K=K, Omega=Om, grho0=K / (math.pi * D_r), n_trials=len(g),
                         psi_ss_mean=g.psi_ss.mean(), psi_ss_std=g.psi_ss.std(),
                         chi_pooled=N * pooled.var(), chi_trial_mean=g.chi_trial.mean(),
                         chi_trial_std=g.chi_trial.std()))
    sm = pd.DataFrame(summ)
    kc = (sm.loc[sm.groupby("Omega").chi_pooled.idxmax(), ["Omega", "K", "grho0", "chi_pooled"]]
            .rename(columns={"K": "K_c", "grho0": "grho0_c", "chi_pooled": "chi_max"}))

    out = Path(args.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "transition_data.csv", index=False)
    sm.to_csv(out / "transition_summary.csv", index=False)
    kc.to_csv(out / "transition_point.csv", index=False)
    pd.DataFrame({"t": np.arange(n_frames) * b0["utime"] * b0["h1"],
                  **{f"psi_K{K:g}_Om{Om:g}_b{b}": v[1] for (K, Om, b), v in sorted(psis.items())}}
                 ).to_csv(out / "psi_timeseries.csv", index=False)
    write_params_used(out, model_params,
                      reservoir_fixed={"steady_frac": P["steady_frac"], "D_r": D_r, "data_dir": str(data_dir),
                                       "grho0_conversion": "K/(pi*D_r)"})
    print(kc.to_string(index=False))

    # ── 図 ──
    ref = None
    ref_path = (ROOT / P["e2_reference_csv"]).resolve()
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
    apply_style()
    for q, ylabel, fname in [("psi", "ψ_ss  (order parameter, steady-state mean)", "psi_vs_K.png"),
                             ("chi", "χ = N(⟨ψ²⟩ − ⟨ψ⟩²)  (susceptibility)", "chi_vs_K.png")]:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for i, (Om, g) in enumerate(sm.groupby("Omega")):
            g = g.sort_values("K")
            y = g.psi_ss_mean if q == "psi" else g.chi_pooled
            e = g.psi_ss_std.fillna(0) if q == "psi" else None
            ax.errorbar(g.K, y, yerr=e, color=COLORS[i % len(COLORS)], marker="o", ms=5, lw=2, capsize=2.5,
                        mec="white", mew=0.8, label=f"this model, Ω = {Om:g}")
            if ref is not None:
                r = ref[np.isclose(ref.omega, Om)].sort_values("grho0")
                if len(r):
                    ax.plot(r.grho0 * math.pi * D_r, r.P if q == "psi" else r.chi, color=COLORS[i % len(COLORS)],
                            ls="--", lw=1.2, marker="x", ms=5, label=f"paper model (E2), Ω = {Om:g}")
        ax.set_xlabel("K  (coupling × neighbour mean)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        top = ax.secondary_xaxis("top", functions=(lambda k: k / (math.pi * D_r), lambda g: g * math.pi * D_r))
        top.set_xlabel("gρ0 = K / (π D_r)  (paper's control parameter)")
        ax.grid(True, color="0.9", lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
        note = (f"Ω = ω/D_r: rotation frequency / rotational diffusion.  D_r = σ²/2 = {D_r:g}.  "
                f"N={N}, L={b0['boxsize']:g} (ρ0 = N rcut²/L² = {N * b0['rcut']**2 / b0['boxsize']**2:g}), rcut={b0['rcut']:g}, "
                f"v0={b0['v0']:g}, nf_sigma={b0['nf_sigma']:g} (same ω for all), F=0 (no input).  "
                f"Error bars: std over trials.  Dashed: paper's CAP model (sum coupling), E2.")
        fig.text(0.01, 0.005, note, fontsize=7, color=C_REF, wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
    print(f"output -> {out}")


if __name__ == "__main__":
    main()
