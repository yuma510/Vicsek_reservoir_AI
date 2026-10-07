"""task_15 段階 2: 予測平均（task_10）の予測の、実現間の分布を図にする。

`cache_predictions.py` が作ったキャッシュだけを読む（position.dat は読まない）。

- 図 A: テスト区間の一部で、正解 y(t)・個々の予測 ŷ_a(t)・平均 μ(t) を重ねた時系列
- 図 B: 代表時刻での ŷ_a(t) のヒストグラム（正解・平均・σ=0 の予測を縦線）と Q-Q プロット

対象タスクは NARMA10 と遅延 k の入力予測 u(t−k)。
"""
import argparse
import json
import sys
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
from scipy import stats

from vicsek_rc import apply_style, load_reservoir_defaults, write_params_used
from cache_predictions import PARAMS_PATH, cache_path, load_narma

# dataviz リファレンスパレット: categorical slot 1, 2 と text tokens
C_INDIV, C_MEAN = "#2a78d6", "#eb6834"
C_TRUE, C_MUTED = "#0b0b0b", "#52514e"


def task_names(P):
    return ["narma"] + [f"k{k:02d}" for k in P["k_values"]]


def task_label(task):
    return "NARMA10" if task == "narma" else f"u(t−{int(task[1:])})"


def true_series(task, y_target, col_input, n_eval):
    if task == "narma":
        return y_target[:n_eval]
    k = int(task[1:])
    return np.concatenate([np.zeros(k), col_input])[:n_eval]


def load_cached(P, sgm):
    """{task: (S, n_eval) float64}, seed list, exp_dir names"""
    seeds = P["noise_seeds"][:1] if sgm == 0 else P["noise_seeds"]
    preds = {t: [] for t in task_names(P)}
    dirs = []
    for s in seeds:
        z = np.load(cache_path(P, sgm, s))
        assert list(z["k_values"]) == P["k_values"], "cache k_values differ from params"
        preds["narma"].append(z["y_pred"])
        for j, k in enumerate(P["k_values"]):
            preds[f"k{k:02d}"].append(z["k_pred"][j])
        dirs.append(str(z["exp_dir"]))
    return {t: np.array(v, dtype=np.float64) for t, v in preds.items()}, seeds, dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()
    P = json.loads(Path(args.params).read_text())
    RC = load_reservoir_defaults()
    y_target, col_input = load_narma(P)
    apply_style()

    sgms = P["sgm_values"]
    data = {s: load_cached(P, s) for s in sgms}
    n_eval = data[sgms[0]][0]["narma"].shape[1]
    te0 = RC["washout"] + RC["train_num"]
    w0, w1 = P["timeseries_window"]
    noisy = [s for s in sgms if s > 0]
    ref = 0.0 if 0.0 in sgms else None

    out = Path(args.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_dist")
    out.mkdir(parents=True, exist_ok=True)
    ts_rows, hist_rows, summ_rows = [], [], []

    for task in task_names(P):
        y = true_series(task, y_target, col_input, n_eval)

        # 代表時刻: 指定がなければテスト区間で y の分位点に最も近い時刻
        if P["hist_times"]:
            times = [(None, int(t)) for t in P["hist_times"]]
        else:
            yt = y[te0:n_eval]
            times = [(q, te0 + int(np.argmin(np.abs(yt - np.quantile(yt, q)))))
                     for q in P["hist_quantiles"]]

        # ── CSV ──
        for s in sgms:
            pr, seeds, _ = data[s]
            X = pr[task]
            mu = X.mean(axis=0)
            for a, sd in enumerate(seeds):
                for t in range(w0, w1):
                    ts_rows.append((task, s, t, y[t], mu[t], sd, X[a, t]))
            for q, t in times:
                y0 = data[ref][0][task][0, t] if ref is not None else np.nan
                for a, sd in enumerate(seeds):
                    hist_rows.append((task, s, t, q, y[t], mu[t], y0, sd, X[a, t]))
                col = X[:, t]
                summ_rows.append(dict(task=task, sgm=s, t=t, quantile=q, S=len(seeds), y_true=y[t],
                                      mean=col.mean(), std=col.std(ddof=1) if len(col) > 1 else 0.0,
                                      skew=stats.skew(col) if len(col) > 2 else np.nan,
                                      excess_kurtosis=stats.kurtosis(col) if len(col) > 3 else np.nan,
                                      mu_minus_y=col.mean() - y[t], y_sgm0=y0))
            # テスト区間全体の要約（時刻ごとの値を平均）
            Xt, yt_ = X[:, te0:n_eval], y[te0:n_eval]
            mu_t = Xt.mean(axis=0)
            multi = Xt.shape[0] > 1
            summ_rows.append(dict(task=task, sgm=s, t="test_all", quantile=np.nan, S=len(seeds),
                                  y_true=np.nan, mean=np.nan,
                                  std=float(np.sqrt((Xt.var(axis=0, ddof=1)).mean())) if multi else 0.0,
                                  skew=float(np.nanmean(stats.skew(Xt, axis=0))) if multi else np.nan,
                                  excess_kurtosis=float(np.nanmean(stats.kurtosis(Xt, axis=0))) if multi else np.nan,
                                  mu_minus_y=float(np.sqrt(((mu_t - yt_) ** 2).mean())),
                                  y_sgm0=np.nan))

        # ── 図 A ──
        fig, axes = plt.subplots(len(sgms), 1, figsize=(7.5, 1.9 * len(sgms) + 0.8),
                                 sharex=True, sharey=True)
        tt = np.arange(w0, w1)
        for ax, s in zip(np.atleast_1d(axes), sgms):
            X = data[s][0][task]
            if X.shape[0] > 1:
                ax.plot(tt, X[:, w0:w1].T, color=C_INDIV, lw=0.5, alpha=0.12)
                ax.plot([], [], color=C_INDIV, lw=1.5, alpha=0.5, label=f"each realization (S={X.shape[0]})")
            ax.plot(tt, X[:, w0:w1].mean(axis=0), color=C_MEAN, lw=2,
                    label="mean μ(t)" if X.shape[0] > 1 else "prediction (σ=0)")
            ax.plot(tt, y[w0:w1], color=C_TRUE, lw=1.2, ls="--", label="target")
            ax.set_title(f"σ = {s:g}", loc="left", fontsize=10)
            ax.grid(True, color="0.92", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ax_l = np.atleast_1d(axes)[-1]
        ax_l.set_xlabel("time step t (test interval)")
        h, l = ax_l.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
        fig.supylabel(task_label(task))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"A_timeseries_{task}.png")
        plt.close(fig)

        # ── 図 B: ヒストグラム と Q-Q ──
        for kind in ("hist", "qq"):
            fig, axes = plt.subplots(len(noisy), len(times), figsize=(3.2 * len(times), 2.5 * len(noisy)),
                                     squeeze=False, sharex=(kind == "hist"))
            for r, s in enumerate(noisy):
                X = data[s][0][task]
                for c, (q, t) in enumerate(times):
                    ax, col = axes[r, c], X[:, t]
                    if kind == "hist":
                        ax.hist(col, bins=P["hist_bins"], color=C_INDIV, alpha=0.75,
                                edgecolor="white", linewidth=0.8)
                        ax.axvline(y[t], color=C_TRUE, lw=1.5, ls="--", label="target")
                        ax.axvline(col.mean(), color=C_MEAN, lw=2, label="mean μ")
                        if ref is not None:
                            ax.axvline(data[ref][0][task][0, t], color=C_MUTED, lw=1.2, ls=":",
                                       label="σ=0 prediction")
                        ax.text(0.02, 0.95, f"sd={col.std(ddof=1):.3g}", transform=ax.transAxes,
                                va="top", fontsize=7.5, color=C_MUTED)
                    else:
                        (osm, osr), (slope, icpt, _) = stats.probplot(col, dist="norm")
                        ax.plot(osm, osr, "o", ms=3.5, color=C_INDIV, mec="white", mew=0.5)
                        ax.plot(osm, slope * osm + icpt, color=C_MUTED, lw=1)
                    if r == 0:
                        qs = f"q={q:g}, " if q is not None else ""
                        ax.set_title(f"{qs}t={t}", fontsize=9)
                    if c == 0:
                        ax.set_ylabel(f"σ = {s:g}\n" + ("count" if kind == "hist" else "ordered values"))
                    if r == len(noisy) - 1:
                        ax.set_xlabel("prediction" if kind == "hist" else "normal quantiles")
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
            if kind == "hist":
                h, l = axes[0, 0].get_legend_handles_labels()
                fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8,
                           bbox_to_anchor=(0.5, 0.035))
            fig.suptitle(f"{task_label(task)}: distribution over noise realizations", fontsize=10)
            # グラフ内で使った文字の説明（CLAUDE.md のルール）
            note = ("q: quantile of the target over the test interval; the column shows the time t where the target "
                    "is closest to its q-quantile (q=0.1: low end, 0.5: middle, 0.9: high end).  "
                    "σ: angular noise strength.")
            if kind == "hist":
                note += "  sd: standard deviation of the predictions across noise realizations."
            else:
                note += "  Points on the line = normally distributed predictions."
            fig.text(0.01, 0.005, note, fontsize=7, color=C_MUTED, wrap=True)
            fig.tight_layout(rect=(0, 0.09 if kind == "hist" else 0.05, 1, 1))
            fig.savefig(out / (f"B_hist_{task}.png" if kind == "hist" else f"B_qq_{task}.png"))
            plt.close(fig)
        print(f"[{datetime.now():%T}] {task} done", flush=True)

    pd.DataFrame(ts_rows, columns=["task", "sgm", "t", "y_true", "mu", "seed_noise", "y_pred"]) \
        .to_csv(out / "timeseries_window.csv", index=False)
    pd.DataFrame(hist_rows, columns=["task", "sgm", "t", "quantile", "y_true", "mu", "y_sgm0",
                                     "seed_noise", "y_pred"]).to_csv(out / "hist_samples.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(out / "dist_summary.csv", index=False)

    # 平均予測 μ がテスト区間で正解にどれだけ追従するか（図は作らない）
    mv_rows = []
    for task in task_names(P):
        yt = true_series(task, y_target, col_input, n_eval)[te0:n_eval]
        for s in sgms:
            X = data[s][0][task][:, te0:n_eval]
            mu = X.mean(axis=0)
            mv_rows.append(dict(task=task, sgm=s, S=X.shape[0],
                                slope_mu_on_y=float(np.polyfit(yt, mu, 1)[0]),
                                corr2_mu_y=float(np.corrcoef(yt, mu)[0, 1] ** 2),
                                corr2_single_y_mean=float(np.mean([np.corrcoef(yt, x)[0, 1] ** 2 for x in X])),
                                rms_mu_minus_y=float(np.sqrt(((mu - yt) ** 2).mean()))))
    pd.DataFrame(mv_rows).to_csv(out / "mean_vs_target.csv", index=False)

    model_params = []
    for s in sgms:
        for d in data[s][2]:
            model_params.append(json.loads((ROOT / P["data_dir"] / d / "params_model.json").read_text()))
    write_params_used(out, model_params,
                      reservoir_fixed={k: RC[k] for k in ("readout", "washout", "train_num",
                                                          "ridge_lambda")} |
                      {"narma_seed": P["narma_seed"], "n_eval": int(n_eval),
                       "timeseries_window": P["timeseries_window"], "hist_quantiles": P["hist_quantiles"]},
                      reservoir_swept={"task": task_names(P)})
    print(f"output -> {out}")


if __name__ == "__main__":
    main()
