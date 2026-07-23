"""
vicsek_rc — Vicsek リザバー解析の共有ライブラリ。

解析スクリプト（analysis/<解析名>/）から共通処理を import する。
新しい共通処理はこのパッケージに追加し、スクリプト側で再定義しないこと。
"""
from .metrics import nrmse, nrmse2, corrcoef, mck_score, memory_capacity
from .loaders import (
    load_position_fast, build_states, load_theta,
    load_adjacency_dir, delayed_input, find_exp_dir,
)
from .evaluate import (ridge_predict, ridge_gram_decomp, ridge_solve_gram,
                       compute_theta_mean_states, evaluate_reservoir, build_noise_averaged_states)
from .params_io import split_fixed_varied, write_params_used, load_reservoir_defaults
from .catalog import build_catalog, write_catalog, load_catalog, append_row
from .narma_io import new_narma_dir, iter_narma_dirs, find_narma_by_seed, save_narma_params
from .correlation import (correlation_at_lag, compute_correlation_decay,
                          load_position_dat, compute_adjacency_from_positions)
from .plotting import (
    apply_style,
    plot_correlation_decay_all, plot_correlation_vs_rcut,
    plot_correlation_by_rcut_single_seed,
    plot_nrmse_vs_rcut, plot_mc_vs_rcut,
)

__all__ = [
    "nrmse", "nrmse2", "corrcoef", "mck_score", "memory_capacity",
    "load_position_fast", "build_states", "load_theta",
    "load_adjacency_dir", "delayed_input", "find_exp_dir",
    "ridge_predict", "ridge_gram_decomp", "ridge_solve_gram",
    "compute_theta_mean_states", "evaluate_reservoir", "build_noise_averaged_states",
    "split_fixed_varied", "write_params_used", "load_reservoir_defaults",
    "build_catalog", "write_catalog", "load_catalog", "append_row",
    "new_narma_dir", "iter_narma_dirs", "find_narma_by_seed", "save_narma_params",
    "correlation_at_lag", "compute_correlation_decay",
    "load_position_dat", "compute_adjacency_from_positions",
    "apply_style",
    "plot_correlation_decay_all", "plot_correlation_vs_rcut",
    "plot_correlation_by_rcut_single_seed",
    "plot_nrmse_vs_rcut", "plot_mc_vs_rcut",
]
