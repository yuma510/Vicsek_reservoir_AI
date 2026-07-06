"""seed=2, train_num=4000 の異常を調べる 3 種類の解析

① r(t) 秩序変数の時系列（seed=1 と比較）
② sin(θ_i) の粒子間分散 σ²(t)（seed=1 と比較）
③ ridge 予測 vs NARMA10 正解の重ね描き（train/test 両方、seed=1 と比較）
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import load_theta, ridge_predict, find_narma_by_seed

# ── パラメータ ───────────────────────────────────────────────────────────
WASHOUT   = 2000
TRAIN_NUM = 4000
TRAIN_END = WASHOUT + TRAIN_NUM   # 6000
TEST_END  = TRAIN_END + TRAIN_NUM # 10000
LAM       = 1e-9

DATA_DIR  = ROOT / "data/20260619_165214_ridge_sweep"
DIR_S2    = DATA_DIR / "20260619_165214_1"   # seed=2 (seed_pos=5)
DIR_S1    = DATA_DIR / "20260619_165214_2"   # seed=1 (seed_pos=4)
# narma_data/ 以下の最新日付 dir から seed666 を自動解決
NARMA_INPUT, NARMA_TARGET = find_narma_by_seed(ROOT / "narma_data", 666)
if NARMA_INPUT is None:
    sys.exit("NARMA10 (seed=666) が narma_data/ に見つからない。generate_narma10.py で生成してください。")

out_dir = Path(__file__).parent / datetime.now().strftime("%Y%m%d_%H%M%S_seed2_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

# ── データ読み込み ───────────────────────────────────────────────────────
print("Loading theta (seed=2) …", flush=True)
with open(DIR_S2 / "params_model.json") as f:
    params = json.load(f)
N = params["N"]

theta2 = load_theta(DIR_S2 / "position.dat", N)   # (22000, 500)
print("Loading theta (seed=1) …", flush=True)
theta1 = load_theta(DIR_S1 / "position.dat", N)   # (22000, 500)

T = theta2.shape[0]
frames = np.arange(T)
print(f"Loaded: {T} frames × {N} particles", flush=True)

target = np.loadtxt(NARMA_TARGET)   # (14000,) ← ridge sweep の ntime=22000 より短い
u_raw  = np.loadtxt(NARMA_INPUT)
u = u_raw[:, 0] if u_raw.ndim > 1 else u_raw

# ── 共通プロット設定 ─────────────────────────────────────────────────────
COLORS = {"seed=2": "tab:red", "seed=1": "tab:blue"}
VLINES = {
    "washout end\n(train start)": WASHOUT,
    "train end\n(test start)":    TRAIN_END,
    "test end":                   TEST_END,
}

def add_vlines(ax):
    for label, x in VLINES.items():
        ax.axvline(x, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.text(x + 30, ax.get_ylim()[1] * 0.97, label,
                fontsize=7, va="top", color="gray")

# ── ① 秩序変数 r(t) ────────────────────────────────────────────────────
print("① Computing r(t) …", flush=True)
r2 = np.abs(np.exp(1j * theta2).mean(axis=1))
r1 = np.abs(np.exp(1j * theta1).mean(axis=1))

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(frames, r1, lw=0.8, color=COLORS["seed=1"], label="seed=1", alpha=0.9)
ax.plot(frames, r2, lw=0.8, color=COLORS["seed=2"], label="seed=2", alpha=0.9)
ax.set_xlim(0, T)
ax.set_ylim(0, 1)
add_vlines(ax)
ax.set_xlabel("Frame")
ax.set_ylabel("r(t) = |⟨e^{iθ}⟩|")
ax.set_title("Order parameter r(t)  (rcut=13, sgm=0, ntime=22000)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "order_parameter.png", dpi=150)
plt.close(fig)
pd.DataFrame({"frame": frames, "r_seed1": r1, "r_seed2": r2}).to_csv(
    out_dir / "order_parameter.csv", index=False)
print("  order_parameter.png", flush=True)

# ── ② 粒子間分散 σ²(t) ──────────────────────────────────────────────────
print("② Computing sin(θ) variance …", flush=True)
var2 = np.var(np.sin(theta2), axis=1)
var1 = np.var(np.sin(theta1), axis=1)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(frames, var1, lw=0.8, color=COLORS["seed=1"], label="seed=1", alpha=0.9)
ax.plot(frames, var2, lw=0.8, color=COLORS["seed=2"], label="seed=2", alpha=0.9)
ax.set_xlim(0, T)
ax.set_ylim(bottom=0)
add_vlines(ax)
ax.set_xlabel("Frame")
ax.set_ylabel("Var_i[sin(θ_i(t))]")
ax.set_title("Particle-wise variance of sin(θ_i)  — reservoir state diversity")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "sin_theta_variance.png", dpi=150)
plt.close(fig)
pd.DataFrame({"frame": frames, "var_seed1": var1, "var_seed2": var2}).to_csv(
    out_dir / "sin_theta_variance.csv", index=False)
print("  sin_theta_variance.png", flush=True)

# ── ③ ridge 予測 vs 正解 ────────────────────────────────────────────────
print("③ Running ridge regression …", flush=True)

def build_states(theta):
    """sin(θ) + bias → (T, N+1)"""
    s = np.sin(theta)
    return np.hstack([np.ones((s.shape[0], 1)), s])

states2 = build_states(theta2)
states1 = build_states(theta1)

pred2 = ridge_predict(states2, target, WASHOUT, TRAIN_END, LAM)
pred1 = ridge_predict(states1, target, WASHOUT, TRAIN_END, LAM)

# 表示範囲: washout〜test_end（ただし target の長さ以内）
plot_end = min(TEST_END, len(target))

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, pred, seed_label, color in [
    (axes[0], pred2, "seed=2", COLORS["seed=2"]),
    (axes[1], pred1, "seed=1", COLORS["seed=1"]),
]:
    t_range = np.arange(WASHOUT, plot_end)
    ax.plot(t_range, target[WASHOUT:plot_end], lw=0.8, color="black",
            label="target", alpha=0.8)
    ax.plot(t_range, pred[WASHOUT:plot_end], lw=0.8, color=color,
            label=f"prediction ({seed_label})", alpha=0.8)
    ax.axvspan(WASHOUT, TRAIN_END, alpha=0.08, color="green", label="train")
    ax.axvspan(TRAIN_END, plot_end, alpha=0.08, color="orange", label="test")
    ax.axvline(TRAIN_END, color="gray", lw=1, ls="--")
    ax.set_ylabel("NARMA10")
    ax.set_title(f"{seed_label}  (λ={LAM:.0e}, train_num={TRAIN_NUM})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
axes[1].set_xlabel("Frame")
fig.suptitle("Ridge prediction vs NARMA10 target  (train=green, test=orange)")
fig.tight_layout()
fig.savefig(out_dir / "ridge_prediction.png", dpi=150)
plt.close(fig)

t_range = np.arange(WASHOUT, plot_end)
pd.DataFrame({
    "frame":    t_range,
    "target":   target[WASHOUT:plot_end],
    "pred_s2":  pred2[WASHOUT:plot_end],
    "pred_s1":  pred1[WASHOUT:plot_end],
}).to_csv(out_dir / "ridge_prediction.csv", index=False)
print("  ridge_prediction.png", flush=True)

# ── ④ θ_i(t) ヒートマップ ─────────────────────────────────────────────────
print("④ Plotting θ heatmap …", flush=True)

# 全時刻（22000 フレーム）: seed=1 / seed=2 を上下に並べる
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, theta, seed_label in [(axes[0], theta1, "seed=1"), (axes[1], theta2, "seed=2")]:
    im = ax.imshow(
        (theta % (2 * np.pi)).T,          # (N, T) で粒子を y 軸
        aspect="auto", cmap="hsv",
        extent=[0, T, 0, N],
        origin="lower", vmin=0, vmax=2 * np.pi,
    )
    for x in [WASHOUT, TRAIN_END, TEST_END]:
        ax.axvline(x, color="white", lw=1, ls="--", alpha=0.8)
    ax.set_ylabel("Particle index")
    ax.set_title(f"θ_i(t) mod 2π  —  {seed_label}")
plt.colorbar(im, ax=axes, label="θ mod 2π", shrink=0.6)
axes[1].set_xlabel("Frame")
fig.suptitle("θ_i(t) heatmap  (white dashed: washout / train / test boundaries)")
fig.tight_layout()
fig.savefig(out_dir / "theta_heatmap_full.png", dpi=150)
plt.close(fig)
print("  theta_heatmap_full.png", flush=True)

# 拡大: 同期が起きる可能性がある frames 3000–11000 に絞る
ZOOM_START, ZOOM_END = 3000, 11000
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, theta, seed_label in [(axes[0], theta1, "seed=1"), (axes[1], theta2, "seed=2")]:
    im = ax.imshow(
        (theta[ZOOM_START:ZOOM_END] % (2 * np.pi)).T,
        aspect="auto", cmap="hsv",
        extent=[ZOOM_START, ZOOM_END, 0, N],
        origin="lower", vmin=0, vmax=2 * np.pi,
    )
    for x in [WASHOUT, TRAIN_END, TEST_END]:
        if ZOOM_START <= x <= ZOOM_END:
            ax.axvline(x, color="white", lw=1.2, ls="--", alpha=0.9)
    ax.set_ylabel("Particle index")
    ax.set_title(f"θ_i(t) mod 2π  —  {seed_label}  [frames {ZOOM_START}–{ZOOM_END}]")
plt.colorbar(im, ax=axes, label="θ mod 2π", shrink=0.6)
axes[1].set_xlabel("Frame")
fig.suptitle("θ_i(t) heatmap (zoomed)  —  white dashed: train/test boundary at 6000")
fig.tight_layout()
fig.savefig(out_dir / "theta_heatmap_zoom.png", dpi=150)
plt.close(fig)
print("  theta_heatmap_zoom.png", flush=True)

print(f"\nOutput: {out_dir}", flush=True)
