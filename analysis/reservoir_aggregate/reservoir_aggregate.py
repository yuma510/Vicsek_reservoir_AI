#!/usr/bin/env python3
"""
Aggregate and evaluate reservoir predictions by sgm groups.

For each sgm value found in run folders under a given reservoir_data folder,
this averages NARMA10_prediction.dat across runs with the same sgm and the
per-delay MCk_prediction files, then recomputes NRMSE (train/test) and the
truncated Memory Capacity from the averaged signals.

Outputs go to <out>/<sgm>/ (averaged .dat files, plots, プロット元データ CSV と
summary.csv) plus a top-level summary_all.csv.

実行例:
    python analysis/reservoir_aggregate/reservoir_aggregate.py \\
        --reservoir-data reservoir_data/20260603_102700_seed2_change \\
        --out analysis/reservoir_aggregate
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import nrmse, nrmse2, corrcoef, delayed_input


class Analyzer:
    def __init__(self, reservoir_data: Path, out_base: Path, workspace_root: Path,
                 mc_cutoff: float = 0.0):
        self.reservoir_data = reservoir_data
        self.out_base = out_base
        self.workspace_root = workspace_root
        self.mc_cutoff = mc_cutoff

    def load_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for child in sorted(self.reservoir_data.iterdir()):
            if not child.is_dir():
                continue
            results = child / "results_reservoir.json"
            narma = child / "NARMA10_prediction.dat"
            mc_folder = child / "MCk_prediction"
            if not results.exists() or not narma.exists() or not mc_folder.exists():
                continue
            try:
                with open(results, "r") as f:
                    info = json.load(f)
            except Exception:
                continue
            sgm = info.get("params_model", {}).get("sgm")
            runs.append({
                "path": child,
                "sgm": float(sgm) if sgm is not None else None,
                "results": info,
            })
        return runs

    def group_by_sgm(self, runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in runs:
            s = r["sgm"]
            key = "unknown" if s is None else f"{s:.1f}"
            groups.setdefault(key, []).append(r)
        return groups

    def find_path(self, pth_str: Optional[str]) -> Optional[Path]:
        if not pth_str:
            return None
        candidates = [
            self.workspace_root / pth_str,
            Path(pth_str),
            self.reservoir_data / pth_str,
            self.reservoir_data.parent / pth_str,
            Path("./tmp") / Path(pth_str).name,
        ]
        for c in candidates:
            try:
                c = c.resolve()
            except Exception:
                continue
            if c.exists():
                return c
        for base in [self.workspace_root, self.reservoir_data,
                     self.reservoir_data.parent, Path(".")]:
            for p in base.rglob("narma10_target*"):
                return p.resolve()
        return None

    def average_and_evaluate_group(self, sgm_key: str,
                                   group: List[Dict[str, Any]]) -> Dict[str, Any]:
        out_dir = self.out_base / sgm_key
        out_dir.mkdir(parents=True, exist_ok=True)

        k_map = None if sgm_key == "unknown" else int(round(float(sgm_key) * 10))

        narma_list: List[np.ndarray] = []
        mck_per_k: Dict[int, List[np.ndarray]] = {}
        mc_train_vals, mc_test_vals = [], []
        params_res = params_model = None
        target_path = input_path = None

        for r in group:
            p = r["path"]
            try:
                narma = np.loadtxt(p / "NARMA10_prediction.dat")
            except Exception:
                print("Failed to load", p / "NARMA10_prediction.dat")
                continue
            narma_list.append(narma)

            mc_folder = p / "MCk_prediction"
            if mc_folder.exists():
                for mc_file in mc_folder.iterdir():
                    if mc_file.is_file() and mc_file.name.startswith("k="):
                        try:
                            k_idx = int(mc_file.name.split("=")[1].split(".")[0])
                            mck = np.loadtxt(mc_file)
                        except Exception:
                            continue
                        mck_per_k.setdefault(k_idx, []).append(mck)

            rt = r["results"].get("results", {})
            if "MC_train" in rt:
                mc_train_vals.append(float(rt["MC_train"]))
            if "MC_test" in rt:
                mc_test_vals.append(float(rt["MC_test"]))

            if params_res is None:
                params_res = r["results"].get("params_reservoir", {})
                params_model = r["results"].get("params_model", {})
                target_path = r["results"].get("data_path", {}).get("target_path")
                input_path = r["results"].get("data_path", {}).get("input_path")

        if not narma_list:
            raise RuntimeError(f"No NARMA predictions for group {sgm_key}")

        # average NARMA prediction
        min_len_n = min(arr.size for arr in narma_list)
        narma_avg = np.mean(np.vstack([a[:min_len_n] for a in narma_list]), axis=0)
        np.savetxt(out_dir / "NARMA10_prediction_avg.dat", narma_avg)

        # average MCk per delay
        per_k_stats: Dict[int, Dict[str, Any]] = {}
        for k_idx in sorted(mck_per_k):
            arrs = mck_per_k[k_idx]
            min_len_m = min(arr.size for arr in arrs)
            mck_avg_k = np.mean(np.vstack([a[:min_len_m] for a in arrs]), axis=0)
            np.savetxt(out_dir / f"MCk_prediction_k={k_idx}_avg.dat", mck_avg_k)
            per_k_stats[k_idx] = {"n_runs": len(arrs), "length": int(min_len_m)}

        target_fp = self.find_path(target_path)
        input_fp = self.find_path(input_path)
        if input_fp is None:
            for base in [self.workspace_root, self.reservoir_data,
                         self.reservoir_data.parent, Path(".")]:
                for p in base.rglob("narma10_input*"):
                    input_fp = p.resolve()
                    break
                if input_fp:
                    break

        res_summary: Dict[str, Any] = {
            "n_runs": len(group), "k_mapped": k_map, "mc_cutoff": float(self.mc_cutoff),
        }

        washout = int(params_res.get("washout", 0)) if params_res else 0
        train_num = int(params_res.get("train_num", 0)) if params_res else 0

        if target_fp and target_fp.exists():
            y_target = np.loadtxt(target_fp)
            L = min(y_target.size, narma_avg.size)
            y_target, y_pred = y_target[:L], narma_avg[:L]
            w = min(washout, L - 1)
            train_end = min(train_num, L)

            res_summary.update({
                "nrmse_train":  nrmse(y_target[w:train_end], y_pred[w:train_end]) if train_end > w else None,
                "nrmse_test":   nrmse(y_target[train_end:L], y_pred[train_end:L]) if L > train_end else None,
                "nrmse2_train": nrmse2(y_target[w:train_end], y_pred[w:train_end]) if train_end > w else None,
                "nrmse2_test":  nrmse2(y_target[train_end:L], y_pred[train_end:L]) if L > train_end else None,
            })

            plt.figure(figsize=(10, 4))
            plt.plot(y_target, label="Target")
            plt.plot(y_pred, label="Averaged Prediction", alpha=0.8)
            if train_end:
                plt.axvline(x=train_end, color="red", linestyle="--", label="train end")
            plt.legend()
            plt.title(f"NARMA10 averaged prediction (sgm={sgm_key})")
            plt.tight_layout()
            plt.savefig(out_dir / "NARMA10_avg.png")
            plt.close()

            # NARMA10 平均予測プロットの元データを CSV で保存
            pd.DataFrame({
                "timestep":       np.arange(L),
                "target":         y_target,
                "prediction_avg": y_pred,
            }).to_csv(out_dir / "NARMA10_avg.csv", index=False)

        # recompute MCk from averaged signals
        MC_train_sum = MC_test_sum = None
        if per_k_stats and params_model is not None and params_res is not None:
            delays = sorted(per_k_stats)
            mck_train_list, mck_test_list = [], []
            for k_idx in delays:
                mc_train = mc_test = None
                try:
                    mck_avg_k = np.loadtxt(out_dir / f"MCk_prediction_k={k_idx}_avg.dat")
                    Lm = mck_avg_k.size
                    if input_fp and input_fp.exists():
                        raw_input = np.loadtxt(input_fp, delimiter=",")
                        col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input
                        delayed = delayed_input(col_input[:Lm], k_idx, Lm)
                        train_end = min(train_num, Lm)
                        if train_end > washout:
                            mc_train = corrcoef(delayed[washout:train_end], mck_avg_k[washout:train_end])
                        if Lm > train_end:
                            mc_test = corrcoef(delayed[train_end:Lm], mck_avg_k[train_end:Lm])
                except Exception:
                    pass
                mck_train_list.append(mc_train if mc_train is not None else 0.0)
                mck_test_list.append(mc_test if mc_test is not None else 0.0)
                per_k_stats[k_idx].update({"MCk_train": mc_train, "MCk_test": mc_test})

            N = int(params_model.get("N", 0))
            denom = max(1, train_num - washout)
            default_threshold = float(N) / denom
            threshold = float(self.mc_cutoff) if self.mc_cutoff and self.mc_cutoff > 0.0 else default_threshold

            mck_test_arr = np.asarray([np.nan if v is None else v for v in mck_test_list])
            mck_train_arr = np.asarray([np.nan if v is None else v for v in mck_train_list])
            comp = np.where(~np.isnan(mck_test_arr), mck_test_arr <= threshold, False)
            idxs = np.where(comp)[0]
            cut = int(idxs[0]) if idxs.size > 0 else len(delays)

            MC_train_sum = float(np.nansum(mck_train_arr[:cut]))
            MC_test_sum = float(np.nansum(mck_test_arr[:cut]))

            plt.figure(figsize=(6, 4))
            plt.plot(delays[:cut], mck_train_list[:cut], marker="o", label="MCk_train")
            plt.plot(delays[:cut], mck_test_list[:cut], marker="s", label="MCk_test")
            plt.xlabel("delay (k)")
            plt.ylabel("MCk")
            plt.title(f"MCk vs delay (sgm={sgm_key})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "MCk_vs_delay.png")
            plt.close()

            # MCk vs delay プロットの元データを CSV で保存
            pd.DataFrame({
                "delay":     delays[:cut],
                "mck_train": mck_train_list[:cut],
                "mck_test":  mck_test_list[:cut],
            }).to_csv(out_dir / "MCk_vs_delay.csv", index=False)

            res_summary["MC_cutoff_used"] = float(threshold)
            res_summary["MC_cut_index"] = int(cut)

        if mc_train_vals:
            res_summary["MC_train_avg_across_runs"] = float(np.mean(mc_train_vals))
        if mc_test_vals:
            res_summary["MC_test_avg_across_runs"] = float(np.mean(mc_test_vals))
        if MC_train_sum is not None:
            res_summary["MC_train_truncated"] = MC_train_sum
        if MC_test_sum is not None:
            res_summary["MC_test_truncated"] = MC_test_sum
        if per_k_stats:
            res_summary["per_k"] = {str(k): v for k, v in per_k_stats.items()}

        # スカラ指標を 1 行 CSV で保存（per_k のカーブは MCk_vs_delay.csv に対応）
        scalar_summary = {k: v for k, v in res_summary.items() if k != "per_k"}
        pd.DataFrame([scalar_summary]).to_csv(out_dir / "summary.csv", index=False)
        return res_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and evaluate Vicsek reservoir results by sgm group."
    )
    parser.add_argument("--reservoir-data",
                        default="reservoir_data/20260603_102700_seed2_change",
                        help="reservoir_data subfolder containing run folders")
    parser.add_argument("--out", default="analysis/reservoir_aggregate",
                        help="Output analysis_data folder (created if missing)")
    parser.add_argument("--root", default=str(ROOT),
                        help="Workspace root for resolving relative target/input paths")
    parser.add_argument("--only-sgm", default=None,
                        help="If set, only process this sgm (e.g. 0.2)")
    parser.add_argument("--mc-cutoff", type=float, default=0.0,
                        help="Absolute MCk_test cutoff for truncation. 0 = use N/(train-washout).")
    return parser.parse_args()


def main():
    args = parse_args()
    reservoir_data = Path(args.reservoir_data).resolve()
    out_base = Path(args.out).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    analyzer = Analyzer(reservoir_data, out_base, Path(args.root),
                        mc_cutoff=float(args.mc_cutoff))
    runs = analyzer.load_runs()
    if not runs:
        print("No valid runs found in", reservoir_data)
        return
    groups = analyzer.group_by_sgm(runs)

    summary_all = {}
    for sgm_key, group in sorted(groups.items()):
        if args.only_sgm is not None and sgm_key != f"{float(args.only_sgm):.1f}":
            continue
        print(f"Processing sgm={sgm_key}, {len(group)} runs")
        try:
            summary_all[sgm_key] = analyzer.average_and_evaluate_group(sgm_key, group)
        except Exception as e:
            print("Failed processing", sgm_key, e)

    # sgm × スカラ指標を CSV で保存（per_k のネストは除外）
    all_rows = [
        {"sgm": sgm_key, **{k: v for k, v in res.items() if k != "per_k"}}
        for sgm_key, res in sorted(summary_all.items())
    ]
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_base / "summary_all.csv", index=False)


if __name__ == "__main__":
    main()
