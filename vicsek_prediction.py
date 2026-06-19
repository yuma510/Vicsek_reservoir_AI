"""
Vicsek リザバーの単発評価パイプライン。

1 実験ディレクトリ（position.dat）を読み、NARMA10 予測と Memory Capacity を計算して
reservoir_data/<timestamp>/ に results_reservoir.json・各種プロット・予測データを保存する。

実行例:
    python analysis/vicsek_prediction.py --data-folder data --date-folder <YYYYMMDD_HHMMSS>
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vicsek_rc import (
    nrmse, nrmse2, mck_score, load_position_fast, build_states, ridge_predict, delayed_input,
)


def _setup_matplotlib():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 12, "figure.titlesize": 12,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def truncate_mck(delays, mck_train, mck_test, threshold):
    """Drop delays where mck_test first falls at or below threshold."""
    arr = np.asarray(mck_test)
    idxs = np.where(arr <= threshold)[0]
    if idxs.size == 0:
        return delays, mck_train, mck_test
    cut = int(idxs[0])
    return delays[:cut], mck_train[:cut], mck_test[:cut]


def plot_narma10(target, pred, washout, train_num, total_num, out_path):
    total_num = min(total_num, target.shape[0], pred.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(target[washout:total_num], label="Target (NARMA10)")
    plt.plot(pred[washout:total_num],   label="Vicsek model Prediction", alpha=0.7)
    plt.axvline(x=min(train_num, total_num) - washout, color="red", linestyle="--")
    plt.title("Vicsek model Prediction of NARMA10")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_mck(delays, mck_train, mck_test, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(delays, mck_train, marker="o", label="MCk train")
    plt.plot(delays, mck_test,  marker="s", label="MCk test")
    plt.xlabel("delay")
    plt.ylabel("MCk")
    plt.title(f"MCk vs delay (0..{len(delays) - 1})")
    plt.ylim(0, None)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_memory_capacity(states, raw_input, washout, train_num, total_num,
                            k_max, n_nodes, out_folder):
    """Return (delays, mck_train, mck_test) after computing MC for delays 0..k_max."""
    mck_pred_folder = out_folder / "MCk_prediction"
    mck_pred_folder.mkdir(parents=True, exist_ok=True)

    length          = states.shape[0]
    washout_local   = min(washout,    length - 1)
    train_num_local = min(train_num,  length - 1)
    total_num_local = min(total_num,  length)
    train_slice     = slice(washout_local, train_num_local + 1)
    test_slice      = slice(train_num_local, total_num_local)

    delays, mck_train_list, mck_test_list = [], [], []
    for delay in range(k_max + 1):
        target = delayed_input(raw_input, delay, length)
        pred   = ridge_predict(states, target, washout_local, train_num_local, 1e-11)
        np.savetxt(mck_pred_folder / f"k={delay}.dat", pred)
        delays.append(delay)
        mck_train_list.append(mck_score(target[train_slice], pred[train_slice]))
        mck_test_list.append(mck_score(target[test_slice],   pred[test_slice]))

    threshold = n_nodes / (train_num - washout)
    return truncate_mck(delays, mck_train_list, mck_test_list, threshold)


class VicsekReservoir:
    """Vicsek reservoir computing pipeline: load → build states → evaluate."""

    def __init__(self, data_folder, date_folder, output_data_folder,
                 target_path, input_path,
                 washout, train_num, ridge_lambda, readout, k_max):
        self.folder_path        = Path(data_folder) / date_folder
        self.date_folder        = Path(date_folder)
        self.output_data_folder = Path(output_data_folder)
        self.target_path        = Path(target_path)
        self.input_path         = Path(input_path)

        self.washout      = washout
        self.train_num    = train_num
        self.ridge_lambda = ridge_lambda
        self.readout      = readout
        self.k_max        = k_max

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.output_data_folder / now
        self.output_folder.mkdir(parents=True, exist_ok=True)

        _setup_matplotlib()

    def run(self):
        """Execute the full reservoir computing pipeline."""
        data = load_position_fast(self.folder_path / "position.dat")
        with open(self.folder_path / "params_model.json") as f:
            params_model = json.load(f)

        n         = params_model["N"]
        utime     = params_model["utime"]
        ntime     = params_model["ntime"]
        total_num = ntime // utime
        pre_subsampled = (data.shape[0] == n * total_num)

        states = build_states(data, n, utime, readout=self.readout,
                              pre_subsampled=pre_subsampled)

        y_target = np.loadtxt(self.target_path)
        y_pred   = ridge_predict(states, y_target,
                                 self.washout, self.train_num, self.ridge_lambda)
        np.savetxt(self.output_folder / "NARMA10_prediction.dat", y_pred)
        plot_narma10(y_target, y_pred,
                     self.washout, self.train_num, total_num,
                     self.output_folder / "narma10_prediction.png")

        eval_total = min(total_num, y_target.shape[0], y_pred.shape[0])
        train_end  = min(self.train_num, eval_total)

        # NARMA10 予測プロットの元データ（時系列）を CSV で保存
        pd.DataFrame({
            "timestep":   np.arange(eval_total),
            "target":     y_target[:eval_total],
            "prediction": y_pred[:eval_total],
        }).to_csv(self.output_folder / "narma10_prediction.csv", index=False)

        nrmse_train  = nrmse( y_target[self.washout:train_end], y_pred[self.washout:train_end])
        nrmse_test   = nrmse( y_target[train_end:eval_total],   y_pred[train_end:eval_total])
        nrmse2_train = nrmse2(y_target[self.washout:train_end], y_pred[self.washout:train_end])
        nrmse2_test  = nrmse2(y_target[train_end:eval_total],   y_pred[train_end:eval_total])

        raw_input = np.loadtxt(self.input_path, delimiter=",")
        col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input

        delays, mck_train, mck_test = compute_memory_capacity(
            states, col_input,
            self.washout, self.train_num, total_num,
            self.k_max, n, self.output_folder,
        )
        MC_train = float(np.sum(mck_train))
        MC_test  = float(np.sum(mck_test))
        plot_mck(delays, mck_train, mck_test, self.output_folder / "MCk.png")

        # MCk プロットの元データ（遅延ごとの MC_k）を CSV で保存
        pd.DataFrame({
            "delay":     delays,
            "mck_train": mck_train,
            "mck_test":  mck_test,
        }).to_csv(self.output_folder / "MCk.csv", index=False)

        results = {
            "data_path": {
                "position_path": str(self.date_folder),
                "target_path":   str(self.target_path),
                "input_path":    str(self.input_path),
            },
            "params_model": params_model,
            "params_reservoir": {
                "readout":      int(self.readout),
                "washout":      int(self.washout),
                "train_num":    int(self.train_num),
                "total_num":    int(eval_total),
                "ridge_lambda": float(self.ridge_lambda),
                "k_max":        int(self.k_max),
            },
            "results": {
                "nrmse_train":  float(nrmse_train),
                "nrmse_test":   float(nrmse_test),
                "nrmse2_train": float(nrmse2_train),
                "nrmse2_test":  float(nrmse2_test),
                "MC_train":     MC_train,
                "MC_test":      MC_test,
            },
        }
        with open(self.output_folder / "results_reservoir.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("Saved result to", self.output_folder)


def latest_data_folder(data_folder):
    """Return the name of the newest child folder containing Vicsek model outputs."""
    data_folder = Path(data_folder)
    candidates = [
        p for p in data_folder.iterdir()
        if p.is_dir()
        and (p / "position.dat").exists()
        and (p / "params_model.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No model output folders found in {data_folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Vicsek reservoir readout for NARMA10 prediction and memory capacity."
    )
    parser.add_argument("--data-folder",   default="data",
                        help="Base folder containing model output folders.")
    parser.add_argument("--date-folder",   default=None,
                        help="Model output folder name. Defaults to newest under --data-folder.")
    parser.add_argument("--output-folder", default="reservoir_data",
                        help="Folder for reservoir results.")
    parser.add_argument("--target-path",   default="tmp/narma10_target_0:0.5_seed666.dat")
    parser.add_argument("--input-path",    default="tmp/narma10_input_0:0.5_seed666.dat")
    parser.add_argument("--washout",       type=int,   default=2000)
    parser.add_argument("--train-num",     type=int,   default=7000)
    parser.add_argument("--ridge-lambda",  type=float, default=1e-11)
    parser.add_argument("--readout",       type=int,   choices=[0, 1], default=1,
                        help="0: theta, 1: sin(theta)")
    parser.add_argument("--k-max",         type=int,   default=100)
    return parser.parse_args()


def main():
    args        = parse_args()
    date_folder = args.date_folder or latest_data_folder(args.data_folder)
    VicsekReservoir(
        data_folder=args.data_folder,
        date_folder=date_folder,
        output_data_folder=args.output_folder,
        target_path=args.target_path,
        input_path=args.input_path,
        washout=args.washout,
        train_num=args.train_num,
        ridge_lambda=args.ridge_lambda,
        readout=args.readout,
        k_max=args.k_max,
    ).run()


if __name__ == "__main__":
    main()
