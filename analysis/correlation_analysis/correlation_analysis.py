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
        --target-path narma_data/narma10_target_0:0.5_seed666.dat \\
        --input-path  narma_data/narma10_input_0:0.5_seed666.dat
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from vicsek_rc import (
    apply_style, load_adjacency_dir, compute_correlation_decay,
    load_position_dat, compute_adjacency_from_positions,
    evaluate_reservoir, load_reservoir_defaults, find_narma_by_seed,
    split_fixed_varied,
    plot_correlation_decay_all, plot_correlation_vs_rcut,
    plot_correlation_by_rcut_single_seed,
    plot_nrmse_vs_rcut, plot_mc_vs_rcut,
)

apply_style()
_RC = load_reservoir_defaults()
_CA = json.loads((Path(__file__).parent / "default_params.json").read_text())


# ── single-experiment analysis ─────────────────────────────────────────────

def analyse_experiment(exp_dir: Path, out_dir: Path, dt_max: int,
                       frame_start: int = 2000, frame_end: int = 2500,
                       run_reservoir: bool = False,
                       target_path: Path | None = None,
                       input_path:  Path | None = None) -> dict:
    """Full analysis for one experiment directory. Returns a summary dict."""
    params_path = exp_dir / "params_model.json"
    with open(params_path) as f:
        params = json.load(f)
    rcut    = params["rcut"]
    N       = params["N"]
    boxsize = params["boxsize"]
    # 新フォーマット: seed_pos スカラ  /  旧フォーマット: seed 配列
    raw_seed = params.get("seed_pos", params.get("seed", 0))
    seed = raw_seed[0] if isinstance(raw_seed, list) else raw_seed
    tag  = f"rcut={rcut:.0f}_seed={seed}"

    adj_dir = exp_dir / "adjacency"
    if adj_dir.is_dir() and any(adj_dir.iterdir()):
        print(f"  [{tag}] Loading adjacency matrices from files …", flush=True)
        timesteps, matrices = load_adjacency_dir(adj_dir)
        ts_first, ts_last = timesteps[0], timesteps[-1]
    else:
        print(f"  [{tag}] Computing adjacency from position.dat "
              f"(frames {frame_start}–{frame_end}) …", flush=True)
        positions = load_position_dat(exp_dir / "position.dat", N,
                                      frame_start, frame_end)
        matrices  = [compute_adjacency_from_positions(positions[i], rcut, boxsize)
                     for i in range(len(positions))]
        ts_first, ts_last = frame_start, frame_end - 1

    print(f"  [{tag}] Computing correlation decay (dt_max={dt_max}) …", flush=True)
    dts, corrs = compute_correlation_decay(matrices, dt_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"correlation_{tag}.csv"
    pd.DataFrame({"dt": dts, "correlation": corrs}).to_csv(csv_path, index=False)

    summary = {
        "rcut":         rcut,
        "seed":         seed,
        "exp_dir":      str(exp_dir),
        "n_frames":     len(matrices),
        "timesteps":    [ts_first, ts_last],
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
            "timesteps": [ts_first, ts_last],
            "dt_max": dt_max,
        }, f, indent=2)

    return summary


def save_summary(summaries: list[dict], out_path: Path):
    """Save CSV summary of all experiments (correlation-vs-rcut の元データ)."""
    rows = []
    for s in sorted(summaries, key=lambda x: (x["rcut"], x.get("seed", 0))):
        row = {
            "rcut":    s["rcut"],
            "seed":    s.get("seed", 0),
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
                     help="Single experiment directory.")
    grp.add_argument("--batch-dir", type=Path,
                     help="Parent directory; all subdirs with position.dat are processed.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory where results are saved.")
    p.add_argument("--dt-max", type=int, default=_CA["dt_max"],
                   help="Maximum time lag for correlation computation (default: 50).")
    p.add_argument("--target-dts", type=int, nargs="+", default=[1, 5, 10],
                   help="Δt values for the 'Correlation vs rcut' plot (default: 1 5 10).")
    p.add_argument("--run-reservoir", action="store_true",
                   help="Also run reservoir evaluation (NRMSE, MC) on each experiment.")
    p.add_argument("--narma-root", default="narma_data",
                   help="NARMA10 探索ルート（日付 dir を自動選択）。")
    p.add_argument("--narma-seed", type=int, default=666,
                   help="使用する NARMA10 の seed（default 666）。")
    p.add_argument("--target-path", type=Path, default=None,
                   help="未指定なら narma_data/ 最新日付 dir から自動解決（--run-reservoir 時のみ必要）。")
    p.add_argument("--input-path",  type=Path, default=None,
                   help="未指定なら narma_data/ 最新日付 dir から自動解決（--run-reservoir 時のみ必要）。")
    p.add_argument("--frame-start", type=int, default=_RC["washout"],
                   help="First frame to use for adjacency analysis (default: 2000, after washout).")
    p.add_argument("--frame-end",   type=int, default=_CA["frame_end"],
                   help="One-past-last frame for adjacency analysis (default: 2500).")
    p.add_argument("--washout",  type=int, default=_RC["washout"])
    p.add_argument("--plot-seed", type=int, default=None,
                   help="Seed for single-seed rcut overlay plot (default: min seed).")
    p.add_argument("--filter-rcut", type=float, nargs="+", default=None,
                   help="Only process experiments with these rcut values.")
    p.add_argument("--filter-sgm", type=float, default=None,
                   help="Only process experiments with this sgm value.")
    p.add_argument("--filter-ntime", type=int, default=None,
                   help="Only process experiments with this ntime value.")
    p.add_argument("--filter-v0", type=float, default=None,
                   help="Only process experiments with this v0 value.")
    p.add_argument("--train-num", type=int, default=_RC["train_num"])
    p.add_argument("--k-max",    type=int, default=100)
    args = p.parse_args()

    # リザバー評価を行う場合のみ NARMA を最新日付 dir から解決
    if args.run_reservoir and (args.input_path is None or args.target_path is None):
        ip, tp = find_narma_by_seed(args.narma_root, args.narma_seed)
        args.input_path  = args.input_path  or (Path(ip) if ip else None)
        args.target_path = args.target_path or (Path(tp) if tp else None)
        if args.input_path is None or args.target_path is None:
            p.error(f"NARMA10 (seed={args.narma_seed}) が {args.narma_root}/ に見つからない。"
                    f"generate_narma10.py で生成してください。")
    return args


def _params_match(d: Path, args) -> bool:
    try:
        with open(d / "params_model.json") as f:
            p = json.load(f)
    except Exception:
        return False
    if args.filter_rcut is not None:
        if not any(abs(p.get("rcut", -1) - r) < 1e-6 for r in args.filter_rcut):
            return False
    if args.filter_sgm is not None:
        if abs(p.get("sgm", -1) - args.filter_sgm) >= 1e-6:
            return False
    if args.filter_ntime is not None:
        if p.get("ntime") != args.filter_ntime:
            return False
    if args.filter_v0 is not None:
        if abs(p.get("v0", -1) - args.filter_v0) >= 1e-6:
            return False
    return True


def collect_exp_dirs(args) -> list[Path]:
    if args.input_dir:
        return [args.input_dir]
    dirs = sorted(
        d for d in args.batch_dir.iterdir()
        if d.is_dir()
        and (d / "position.dat").exists()
        and (d / "params_model.json").exists()
        and _params_match(d, args)
    )
    # (rcut, seed_pos) ペアが重複する場合は最新 dir のみ使用
    seen: dict[tuple, Path] = {}
    for d in dirs:
        with open(d / "params_model.json") as f:
            p = json.load(f)
        key = (p.get("rcut"), p.get("seed_pos", p.get("seed", None)))
        seen[key] = d  # sorted は古い→新しい順なので上書きで最新が残る
    dirs = sorted(seen.values())
    if not dirs:
        sys.exit(f"No valid experiment directories found under {args.batch_dir}")
    return dirs


def main():
    args = parse_args()
    exp_dirs = collect_exp_dirs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(exp_dirs)} experiment(s). Output → {out_dir}")

    summaries = []
    all_params = []
    for exp_dir in exp_dirs:
        print(f"\nProcessing: {exp_dir.name}")
        s = analyse_experiment(
            exp_dir, out_dir,
            dt_max=args.dt_max,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            run_reservoir=args.run_reservoir,
            target_path=args.target_path,
            input_path=args.input_path,
        )
        summaries.append(s)
        with open(exp_dir / "params_model.json") as f:
            all_params.append(json.load(f))

    print("\nGenerating aggregate plots …")
    plot_correlation_decay_all(summaries, out_dir / "correlation_decay_all.png")
    seeds_in_results = sorted(set(s["seed"] for s in summaries))
    plot_seed = args.plot_seed if args.plot_seed is not None else seeds_in_results[0]
    plot_correlation_by_rcut_single_seed(
        summaries, plot_seed,
        out_dir / f"correlation_by_rcut_seed={plot_seed}.png",
    )
    plot_correlation_vs_rcut(summaries, args.target_dts, out_dir / "correlation_vs_rcut.png")

    if args.run_reservoir:
        plot_nrmse_vs_rcut(summaries, out_dir / "nrmse_vs_rcut.png")
        plot_mc_vs_rcut(   summaries, out_dir / "mc_vs_rcut.png")

    save_summary(summaries, out_dir / "analysis_summary.csv")

    model_fixed, model_swept = split_fixed_varied(all_params)
    params_used = {
        "model":    {"fixed": model_fixed, "swept": model_swept},
        "analysis": {"frame_start": args.frame_start, "frame_end": args.frame_end,
                     "dt_max": args.dt_max},
    }
    with open(out_dir / "params_used.json", "w") as f:
        json.dump(params_used, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
