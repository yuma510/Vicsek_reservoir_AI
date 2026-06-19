"""
vicsek_rc — Vicsek リザバー解析の共有ライブラリ。

解析スクリプト（analysis/<解析名>/）から共通処理を import する。
新しい共通処理はこのパッケージに追加し、スクリプト側で再定義しないこと。
"""
from .metrics import nrmse, nrmse2, corrcoef, mck_score
from .loaders import (
    load_position_fast, build_states, load_theta,
    load_adjacency_dir, delayed_input, find_exp_dir,
)
from .evaluate import ridge_predict, compute_theta_mean_states, evaluate_reservoir, build_noise_averaged_states
from .params_io import split_fixed_varied, write_params_used
from .correlation import correlation_at_lag, compute_correlation_decay
from .plotting import (
    apply_style,
    plot_correlation_decay_all, plot_correlation_vs_rcut,
    plot_nrmse_vs_rcut, plot_mc_vs_rcut,
)

__all__ = [
    "nrmse", "nrmse2", "corrcoef", "mck_score",
    "load_position_fast", "build_states", "load_theta",
    "load_adjacency_dir", "delayed_input", "find_exp_dir",
    "ridge_predict", "compute_theta_mean_states", "evaluate_reservoir", "build_noise_averaged_states",
    "split_fixed_varied", "write_params_used",
    "correlation_at_lag", "compute_correlation_decay",
    "apply_style",
    "plot_correlation_decay_all", "plot_correlation_vs_rcut",
    "plot_nrmse_vs_rcut", "plot_mc_vs_rcut",
]
