"""Reservoir performance metrics (NRMSE, MC_k)."""
import numpy as np


def nrmse(y_true, y_pred) -> float:
    """Normalized RMSE: sqrt(MSE / mean(y_true^2))."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse / np.mean(y_true ** 2)))


def nrmse2(y_true, y_pred) -> float:
    """Normalized MSE by variance: MSE / Var(y_true)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def corrcoef(a, b) -> float:
    """Pearson correlation with size guard (returns nan for empty input)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size != b.size:
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
    return float(np.corrcoef(a, b)[0, 1])


def mck_score(y_true, y_pred) -> float:
    """Memory Capacity at delay k: squared Pearson correlation."""
    return float(corrcoef(y_true, y_pred) ** 2)


def memory_capacity(mck, threshold, cutoff_ref=None) -> float:
    """遅延ごとの MC_k 列から合計 Memory Capacity を返す（MC の唯一の定義）。

    mck[k] = 遅延 k の MC_k（先頭が k=0）。**標準定義に従い遅延 k=0 は総和に含めない**
    （k=0 は現在入力の自明な再構成で MC_0≈1 になるため）。

    総和は k=1 から、テスト側 MC_k が最初に `threshold` 以下になる遅延（その項を含む）まで。
    cutoff 判定は `cutoff_ref`（通常はテスト MC_k 列）で行い、未指定なら `mck` 自身で行う。
    MC_train を求めるときは cutoff_ref にテスト MC_k 列を渡す（打ち切り位置をテストと揃える）。

    早期打ち切りで cutoff まで truncate 済みの list を渡しても、full 配列を渡しても同じ結果になる。
    """
    import numpy as np
    mck = np.asarray(mck, dtype=float)
    ref = np.asarray(mck if cutoff_ref is None else cutoff_ref, dtype=float)
    kstop = len(ref) - 1
    for k in range(1, len(ref)):
        if ref[k] <= threshold:
            kstop = k
            break
    return float(mck[1:kstop + 1].sum())
