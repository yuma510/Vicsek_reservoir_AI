"""Shared matplotlib style and common plot helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_style():
    """Apply the project's default matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 10, "axes.linewidth": 0.8,
        "lines.linewidth": 1.2, "figure.dpi": 150, "savefig.dpi": 150,
    })


def plot_correlation_decay_all(summaries: list[dict], out_path):
    """Overlay correlation decay curves for all rcut values."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    rcuts = sorted(set(s["rcut"] for s in summaries))
    colors = {r: cmap(i / max(len(rcuts) - 1, 1)) for i, r in enumerate(rcuts)}

    for s in sorted(summaries, key=lambda x: x["rcut"]):
        ax.plot(s["dts"], s["correlations"],
                label=f"rcut={s['rcut']:.0f}", color=colors[s["rcut"]])
    ax.set_xlabel("Time lag Δt (frames)")
    ax.set_ylabel("Normalised correlation")
    ax.set_title("Network Correlation Decay — all rcut")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_vs_rcut(summaries: list[dict], target_dts: list[int], out_path):
    """Correlation value at selected Δt values vs rcut."""
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, dt in enumerate(target_dts):
        xs, ys = [], []
        for s in sorted(summaries, key=lambda x: x["rcut"]):
            if dt < len(s["dts"]):
                xs.append(s["rcut"])
                ys.append(s["correlations"][dt])
        ax.plot(xs, ys, marker=markers[i % len(markers)], label=f"Δt = {dt}")
    ax.set_xlabel("rcut")
    ax.set_ylabel("Normalised correlation")
    ax.set_title("Network Correlation vs rcut")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_nrmse_vs_rcut(summaries: list[dict], out_path):
    """NRMSE (train & test) vs rcut."""
    xs = sorted([s["rcut"] for s in summaries if "reservoir" in s])
    if not xs:
        print("  No reservoir results found — skipping NRMSE vs rcut plot.")
        return
    ordered = [s for s in sorted(summaries, key=lambda x: x["rcut"]) if "reservoir" in s]
    ys_tr  = [s["reservoir"]["nrmse_train"]  for s in ordered]
    ys_te  = [s["reservoir"]["nrmse_test"]   for s in ordered]
    ys2_tr = [s["reservoir"]["nrmse2_train"] for s in ordered]
    ys2_te = [s["reservoir"]["nrmse2_test"]  for s in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, tr, te, title in zip(
        axes, [ys_tr, ys2_tr], [ys_te, ys2_te],
        ["NRMSE vs rcut", "NRMSE2 vs rcut"],
    ):
        ax.plot(xs, tr, marker="o", label="train")
        ax.plot(xs, te, marker="s", label="test")
        ax.set_xlabel("rcut")
        ax.set_ylabel(title.split(" ")[0])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_by_rcut_single_seed(summaries: list[dict], seed: int, out_path):
    """Overlay C(Δt) curves for all rcut values at a single seed."""
    filtered = [s for s in summaries if s.get("seed") == seed]
    if not filtered:
        print(f"  seed={seed} not found — skipping single-seed plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    rcuts = sorted(s["rcut"] for s in filtered)
    colors = {r: cmap(i / max(len(rcuts) - 1, 1)) for i, r in enumerate(rcuts)}
    for s in sorted(filtered, key=lambda x: x["rcut"]):
        ax.plot(s["dts"], s["correlations"],
                label=f"rcut={s['rcut']:.0f}", color=colors[s["rcut"]])
    ax.set_xlabel("Time lag Δt (frames)")
    ax.set_ylabel("Normalised correlation")
    ax.set_title(f"Network Correlation Decay  (seed={seed})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_mc_vs_rcut(summaries: list[dict], out_path):
    """MC (train & test) vs rcut."""
    data = [(s["rcut"], s["reservoir"]["MC_train"], s["reservoir"]["MC_test"])
            for s in summaries if "reservoir" in s]
    if not data:
        print("  No reservoir results found — skipping MC vs rcut plot.")
        return
    data.sort()
    xs, tr, te = zip(*data)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, tr, marker="o", label="MC train")
    ax.plot(xs, te, marker="s", label="MC test")
    ax.set_xlabel("rcut")
    ax.set_ylabel("Memory Capacity")
    ax.set_title("MC vs rcut")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
