"""デフォルトパラメータでシム実行 → θ_i(t) ヒートマップ + 秩序変数 + 選択粒子プロット"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import find_exp_dir, load_reservoir_defaults, load_theta

# ── Step 1: シミュレーション実行 ──────────────────────────────────────────
with open(ROOT / "configs/default_params.json") as f:
    cfg = json.load(f)

fd, cfg_path = tempfile.mkstemp(suffix=".json")
try:
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    result = subprocess.run([str(ROOT / "vicsek_dynamic"), cfg_path],
                            capture_output=True, text=True)
finally:
    os.unlink(cfg_path)

if result.returncode != 0:
    print(f"[ERROR] simulation failed:\n{result.stderr}", flush=True)
    sys.exit(1)
print("[SIM done]", flush=True)

# ── Step 2: データ取得 ────────────────────────────────────────────────────
exp_dir = find_exp_dir(ROOT / "data", rcut=cfg["rcut"], sgm=cfg["sgm"],
                       seed_key="seed_pos", seed=cfg["seed_pos"])
if exp_dir is None:
    print("[ERROR] simulation output not found", flush=True)
    sys.exit(1)

with open(exp_dir / "params_model.json") as f:
    params = json.load(f)

print(f"Loading theta from {exp_dir} …", flush=True)
theta = load_theta(exp_dir / "position.dat", params["N"])  # (T, N)
T, N = theta.shape
print(f"theta shape: {T} frames × {N} particles", flush=True)

out_dir = Path(__file__).parent / datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Step 3a: ヒートマップ（全粒子・全時刻） ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(
    (theta % (2 * np.pi)).T,
    aspect="auto", cmap="hsv",
    extent=[0, T, 0, N],
    origin="lower",
    vmin=0, vmax=2 * np.pi,
)
plt.colorbar(im, ax=ax, label="θ mod 2π")
ax.set_xlabel("Frame")
ax.set_ylabel("Particle index")
ax.set_title(f"θ_i(t)  v0={cfg['v0']} rcut={cfg['rcut']} sgm={cfg['sgm']}")
fig.tight_layout()
fig.savefig(out_dir / "theta_heatmap.png", dpi=150)
plt.close(fig)
print("  theta_heatmap.png", flush=True)

# ── Step 3b: 秩序変数 r(t) ────────────────────────────────────────────────
r = np.abs(np.exp(1j * theta).mean(axis=1))  # (T,)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(r, lw=0.8)
ax.set_xlabel("Frame")
ax.set_ylabel("r(t)")
ax.set_title("Order parameter  r(t) = |⟨e^{iθ}⟩|")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(out_dir / "order_parameter.png", dpi=150)
plt.close(fig)
print("  order_parameter.png", flush=True)

# ── Step 3c/3d: 選択 5 粒子の θ と sin(θ)（test 期間冒頭 100 フレーム） ──
_RC = load_reservoir_defaults()
test_start = _RC["washout"] + _RC["train_num"]  # 2000 + 6000 = 8000
N_SHOW = 5
N_FRAMES = 100

rng = np.random.default_rng(0)
sel = sorted(rng.choice(N, N_SHOW, replace=False).tolist())
t_slice = theta[test_start: test_start + N_FRAMES]      # (100, N)
t_axis = np.arange(test_start, test_start + N_FRAMES)   # 入力サンプルインデックス

for fname, ylabel, values in [
    ("theta_selected_particles.png",   "θ (rad)", t_slice),
    ("sintheta_selected_particles.png", "sin θ",  np.sin(t_slice)),
]:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in sel:
        ax.plot(t_axis, values[:, i], label=f"particle {i}")
    ax.set_xlabel("Input sample index t")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel}_i(t) for {N_SHOW} particles"
        f"  (test period: t={test_start}–{test_start + N_FRAMES - 1})"
    )
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  {fname}", flush=True)

# ── CSV 保存 ───────────────────────────────────────────────────────────────
pd.DataFrame({"frame": np.arange(T), "r": r}).to_csv(
    out_dir / "order_parameter.csv", index=False)

rows = [
    {"frame": int(t_axis[t]), "particle": int(i),
     "theta": float(t_slice[t, i]), "sin_theta": float(np.sin(t_slice[t, i]))}
    for t in range(N_FRAMES) for i in sel
]
pd.DataFrame(rows).to_csv(out_dir / "theta_selected.csv", index=False)
print("  order_parameter.csv  theta_selected.csv", flush=True)

print(f"\nOutput: {out_dir}", flush=True)
