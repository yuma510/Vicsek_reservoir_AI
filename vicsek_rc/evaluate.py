"""Ridge-regression readout and reservoir evaluation (NRMSE, MC)."""
import numpy as np

from .loaders import load_position_fast, build_states
from .metrics import nrmse, nrmse2, mck_score
from .params_io import load_reservoir_defaults

_RC = load_reservoir_defaults()


def ridge_gram_decomp(X_train: np.ndarray) -> tuple:
    """Eigendecompose X_train.T @ X_train for efficient λ-sweep.

    Returns (vals, vecs) from eigh. Use with ridge_solve_gram to avoid
    recomputing the Gram matrix for each λ.
    """
    XtX = X_train.T @ X_train
    vals, vecs = np.linalg.eigh(XtX)
    return vals, vecs


def ridge_solve_gram(X_train: np.ndarray, vals: np.ndarray, vecs: np.ndarray,
                     y_train: np.ndarray, lam: float) -> np.ndarray:
    """Compute output weights W using precomputed Gram eigendecomposition.

    Efficient when sweeping over λ with fixed X_train.
    Returns W (D,) where D = X_train.shape[1].
    """
    XtY = X_train.T @ y_train
    VtY = vecs.T @ XtY
    return vecs @ (VtY / (vals + lam))


def ridge_predict(states: np.ndarray, target: np.ndarray,
                  washout: int, train_num: int,
                  ridge_lambda: float = _RC["ridge_lambda"]) -> np.ndarray:
    """Train ridge regression on [washout:train_num] and predict over all states."""
    train_end = min(train_num + 1, states.shape[0], target.shape[0])
    X = states[washout:train_end]
    Y = target[washout:train_end]
    Wout = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(X.shape[1]), X.T @ Y)
    return states @ Wout


def compute_theta_mean_states(theta_all: np.ndarray, utime: int,
                              add_bias: bool = True) -> np.ndarray:
    """
    Collective mean-direction state: θ_mean(t) = angle(mean(exp(i·θ))).
    Returns (T, 2) sin/cos components, with a leading bias column when add_bias.
    """
    sampled = theta_all[::utime]
    cplx    = np.exp(1j * sampled)
    mean_c  = cplx.mean(axis=1)
    ang     = np.angle(mean_c)
    feat    = np.column_stack([np.sin(ang), np.cos(ang)])
    if add_bias:
        feat = np.hstack([np.ones((feat.shape[0], 1)), feat])
    return feat


def build_noise_averaged_states(theta_list: list,
                                add_bias: bool = True) -> np.ndarray:
    """Per-particle noise-realization average.

    theta_list: list of S arrays each shape (T, N), already subsampled.
    θ_i^avg(t) = arg( Σ_a exp(i·θ_i^a(t)) / S )
    Returns sin(θ_avg): shape (T, N+1) with bias.
    """
    stacked   = np.stack(theta_list, axis=0)        # (S, T, N)
    cplx_avg  = np.exp(1j * stacked).mean(axis=0)  # (T, N)
    theta_avg = np.angle(cplx_avg)                  # (T, N)
    states    = np.sin(theta_avg)                   # (T, N)
    if add_bias:
        states = np.hstack([np.ones((states.shape[0], 1)), states])
    return states


def evaluate_reservoir(pos_path, params: dict,
                       target_path, input_path,
                       washout: int = _RC["washout"],
                       train_num: int = _RC["train_num"],
                       k_max: int = _RC["k_max"]) -> dict:
    """Run ridge regression on position.dat and return NRMSE / MC results."""
    print("    Loading position.dat …", flush=True)
    data = load_position_fast(pos_path)

    N       = params["N"]
    utime   = params["utime"]
    n_input = params["ntime"] // utime
    pre_subsampled = (data.shape[0] == N * n_input)
    total_num = n_input

    states   = build_states(data, N, utime, pre_subsampled=pre_subsampled)
    y_target = np.loadtxt(target_path)
    # train_num = number of training samples (after washout); frame index = washout + train_num
    train_end_frame = washout + train_num
    y_pred   = ridge_predict(states, y_target, washout, train_end_frame)

    eval_total = min(total_num, y_target.shape[0], y_pred.shape[0])
    train_end  = min(train_end_frame, eval_total)

    nr_train  = nrmse( y_target[washout:train_end],  y_pred[washout:train_end])
    nr_test   = nrmse( y_target[train_end:eval_total], y_pred[train_end:eval_total])
    nr2_train = nrmse2(y_target[washout:train_end],  y_pred[washout:train_end])
    nr2_test  = nrmse2(y_target[train_end:eval_total], y_pred[train_end:eval_total])

    # MC
    raw_input = np.loadtxt(input_path)
    col_input = raw_input[:, 0] if raw_input.ndim > 1 else raw_input
    length    = states.shape[0]
    threshold = N / train_num

    mck_train_list, mck_test_list = [], []
    for delay in range(k_max + 1):
        padded = np.concatenate([np.zeros(delay), col_input])
        tgt    = padded[:length]
        pred   = ridge_predict(states, tgt, washout, train_end_frame)
        tr_s   = slice(washout, min(train_end_frame + 1, length))
        te_s   = slice(min(train_end_frame, length), min(total_num, length))
        mck_train_list.append(mck_score(tgt[tr_s], pred[tr_s]))
        mck_test_list.append(mck_score(tgt[te_s],  pred[te_s]))
        if mck_test_list[-1] <= threshold:  # early stopping
            break

    return {
        "nrmse_train": nr_train,   "nrmse_test":   nr_test,
        "nrmse2_train": nr2_train, "nrmse2_test":  nr2_test,
        "MC_train": float(np.sum(mck_train_list)),
        "MC_test":  float(np.sum(mck_test_list)),
    }
