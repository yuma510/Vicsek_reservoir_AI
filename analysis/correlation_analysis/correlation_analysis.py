"""
Network correlation analysis for Vicsek reservoir experiments.

Computes how particle interaction networks (adjacency matrices) decorrelate
over time, and evaluates reservoir performance (NRMSE, MC) per rcut value.

Usage — single experiment:
    python analysis/correlation_analysis/correlation_analysis.py \\
        --input-dir data/20260611_120500_adj_rcut_changed/20260522_095150 \\
        --output-dir analysis/correlation_analysis/20260614

Usage — batch (all subdirs under a parent):
    python analysis/correlation_analysis/correlation_analysis.py \\
        --batch-dir data/20260611_120500_adj_rcut_changed \\
        --output-dir analysis/correlation_analysis/20260614 \\
        --dt-max 50 \\
        --run-reservoir \\
        --target-path tmp/narma10_target_0:0.5_seed666.dat \\
        --input-path  tmp/narma10_input_0:0.5_seed666.dat
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from vicsek_rc import (
    apply_style, load_adjacency_dir, compute_correlation_decay,
    evaluate_reservoir,
    plot_correlation_decay_all, plot_correlation_vs_rcut,
    plot_nrmse_vs_rcut, plot_mc_vs_rcut,
)

apply_style()


# ── single-experiment analysis ─────────────────────────────────────────────

def analyse_experiment(exp_dir: Path, out_dir: Path, dt_max: int,
                       run_reservoir: bool = False,
                       target_path: Path | None = None,
                       input_path:  Path | None = None) -> dict:
    """Full analysis for one experiment directory. Returns a summary dict."""
    params_path = exp_dir / "params_model.json"
    with open(params_path) as f:
        params = json.load(f)
    rcut = params["rcut"]
    tag  = f"rcut={rcut:.0f}"

    print(f"  [{tag}] Loading adjacency matrices …", flush=True)
    timesteps, matrices = load_adjacency_dir(exp_dir / "adjacency")

    print(f"  [{tag}] Computing correlation decay (dt_max={dt_max}) …", flush=True)
    dts, corrs = compute_correlation_decay(matrices, dt_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"correlation_{tag}.csv"
    pd.DataFrame({"dt": dts, "correlation": corrs}).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(dts, corrs, marker="o", markersize=3)
    ax.set_xlabel("Time lag Δt (frames)")
    ax.set_ylabel("Normalised correlation")
    ax.set_title(f"Network Correlation Decay  ({tag})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_decay_{tag}.png")
    plt.close()

    summary = {
        "rcut":         rcut,
        "exp_dir":      str(exp_dir),
        "n_frames":     len(matrices),
        "timesteps":    [timesteps[0], timesteps[-1]],
        "dts":          dts,
        "correlations": corrs,
    }

    if run_reservoir and target_path and input_path:
        pos_path = exp_dir / "position.dat"
        if pos_path.exists():
            print(f"  [{tag}] Running reservoir evaluation …", flush=True)
            res = evaluate_reservoir(pos_path, params, target_path, input_path)
            summary["reservoir"] = res
            with open(out_dir / f"reservoir_{tag}.json", "w") as f:
                json.dump({"params_model": params, "results": res}, f, indent=2)
            print(f"  [{tag}] NRMSE_test={res['nrmse_test']:.4f}  MC_test={res['MC_test']:.4f}")

    with open(out_dir / f"analysis_{tag}.json", "w") as f:
        json.dump({
            "params_model": params,
            "n_frames": len(matrices),
            "timesteps": [timesteps[0], timesteps[-1]],
            "dt_max": dt_max,
        }, f, indent=2)

    return summary


def save_summary(summaries: list[dict], out_path: Path):
    """Save CSV summary of all experiments (correlation-vs-rcut の元データ)."""
    rows = []
    for s in sorted(summaries, key=lambda x: x["rcut"]):
        row = {
            "rcut": s["rcut"],
            "exp_dir": s["exp_dir"],
            "n_frames": s["n_frames"],
            "timestep_first": s["timesteps"][0],
            "timestep_last":  s["timesteps"][-1],
            "corr_dt1":  s["correlations"][1]  if len(s["correlations"]) > 1  else None,
            "corr_dt5":  s["correlations"][5]  if len(s["correlations"]) > 5  else None,
            "corr_dt10": s["correlations"][10] if len(s["correlations"]) > 10 else None,
        }
        if "reservoir" in s:
            row.update(s["reservoir"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-dir", type=Path,
                     help="Single experiment directory (must contain adjacency/).")
    grp.add_argument("--batch-dir", type=Path,
                     help="Parent directory; all subdirs with adjacency/ are processed.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory where results are saved.")
    p.add_argument("--dt-max", type=int, default=50,
                   help="Maximum time lag for correlation computation (default: 50).")
    p.add_argument("--target-dts", type=int, nargs="+", default=[1, 5, 10],
                   help="Δt values for the 'Correlation vs rcut' plot (default: 1 5 10).")
    p.add_argument("--run-reservoir", action="store_true",
                   help="Also run reservoir evaluation (NRMSE, MC) on each experiment.")
    p.add_argument("--target-path", type=Path,
                   default=Path("tmp/narma10_target_0:0.5_seed666.dat"))
    p.add_argument("--input-path",  type=Path,
                   default=Path("tmp/narma10_input_0:0.5_seed666.dat"))
    p.add_argument("--washout",  type=int, default=2000)
    p.add_argument("--train-num", type=int, default=7000)
    p.add_argument("--k-max",    type=int, default=100)
    return p.parse_args()


def collect_exp_dirs(args) -> list[Path]:
    if args.input_dir:
        return [args.input_dir]
    dirs = sorted(
        d for d in args.batch_dir.iterdir()
        if d.is_dir() and (d / "adjacency").exists() and (d / "params_model.json").exists()
    )
    if not dirs:
        sys.exit(f"No valid experiment directories found under {args.batch_dir}")
    return dirs


def main():
    args = parse_args()
    exp_dirs = collect_exp_dirs(args)
    out_dir  = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(exp_dirs)} experiment(s). Output → {out_dir}")

    summaries = []
    for exp_dir in exp_dirs:
        print(f"\nProcessing: {exp_dir.name}")
        s = analyse_experiment(
            exp_dir, out_dir,
            dt_max=args.dt_max,
            run_reservoir=args.run_reservoir,
            target_path=args.target_path,
            input_path=args.input_path,
        )
        summaries.append(s)

    print("\nGenerating aggregate plots …")
    plot_correlation_decay_all(summaries, out_dir / "correlation_decay_all.png")
    plot_correlation_vs_rcut(summaries, args.target_dts, out_dir / "correlation_vs_rcut.png")

    if args.run_reservoir:
        plot_nrmse_vs_rcut(summaries, out_dir / "nrmse_vs_rcut.png")
        plot_mc_vs_rcut(   summaries, out_dir / "mc_vs_rcut.png")

    save_summary(summaries, out_dir / "analysis_summary.csv")
    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
