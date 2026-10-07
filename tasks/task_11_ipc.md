# #11 デフォルトパラメータのレザバー IPC 測定

**状態**: `[x]` 完了（2026-07-01、出力: `analysis/ipc/20260701_194332/`）

## 目標

Dambre et al. 2012「Information Processing Capacity of Dynamical Systems」の手法で、
デフォルトパラメータ（rcut=13, sgm=0）の Vicsek レザバーの **情報処理容量 (IPC)** を測定する。

IPC は「リザバーが入力の過去をどれだけ多様な形で記憶しているか」を定量化する。
- 1次（線形記憶）: 標準 MC と一致
- 2次（2乗・交差積記憶）: 非線形状態 sin(θ) がどれほど非線形演算を実行できるかを示す
- 理論上限: C_total ≤ N = 500（独立状態数）

## 理論

### 容量の定義（Dambre 2012）

任意の目標関数 z_l(t) に対して:

```
C[X, z_l] = corr²(z_l_test,  X_test @ w*)
```

ここで `w*` は学習区間で線形リードアウトを Ridge 回帰で求めたもの。
直交正規化された基底関数のセット `{z_l}` を使えば C_total = Σ_l C_l ≤ N が成立。

### 基底関数（Uniform[0, 0.5] 用正規化 Legendre 多項式積）

変換: x(t) = 4u(t) - 1 ∈ [-1, 1]（u ~ Uniform[0,0.5] → x ~ Uniform[-1,1]）

| 種別 | 基底関数 | 期待値 | 分散 |
|---|---|---|---|
| degree-1 (d1) | P̃₁(u(t-k)) = √3·x(t-k) | 0 | 1 |
| degree-2a (d2a) | P̃₂(u(t-k)) = √5·(3x²-1)/2 at lag k | 0 | 1 |
| degree-2b (d2b) | P̃₁(u(t-k₁))·P̃₁(u(t-k₂)), k₁<k₂ | 0 | 1 |

d1 の合計容量 Σ_k C_d1(k) は標準の MC（Memory Capacity）と等価。

### 効率的計算

全基底関数で共通の行列を前計算:

```python
M_inv = inv(X_tr.T @ X_tr + λI)          # O(N³)、一度だけ
w_l   = M_inv @ (X_tr.T @ z_l_tr)        # O(N²)、各基底関数ごと
C_l   = mck_score(z_l_te, X_te @ w_l)    # O(T_te × N)
```

## データ

ntime=220000（22000 フレーム）の実験を使用（14000 フレーム版より多い T_test が得られる）:

- rcut=13, sgm=0, v0=0.5
- seed_pos = 4, 5, 6, 8, 9（5 seed）
- T_test = 22000 - 2000（washout） - 6000（train） = **14000 フレーム**

### NARMA 入力について

Vicsek シミュレーションは外部トルク `F·sin(c·u(t)−θ_i)` で駆動されている（F=14.3, c=0.1）。
IPC 基底関数の構築には **同一の u(t) シーケンス** を用いなければならない（異なる系列を使うと
C=0 になる）。

実験は `run_ridge_sweep.py` で生成され、各 seed_pos に対して narma_seed = seed_pos − 3 の
長さ 24000 の NARMA 入力が使われている（ファイル: `narma10_input_0.0:0.5_seed{narma_seed}_n24000.dat`）:

| seed_pos | narma_seed | 入力ファイル |
|---|---|---|
| 4 | 1 | `narma10_input_0.0:0.5_seed1_n24000.dat` |
| 5 | 2 | `narma10_input_0.0:0.5_seed2_n24000.dat` |
| 6 | 3 | `narma10_input_0.0:0.5_seed3_n24000.dat` |
| 8 | 5 | `narma10_input_0.0:0.5_seed5_n24000.dat` |
| 9 | 6 | `narma10_input_0.0:0.5_seed6_n24000.dat` |

## パラメータ

| パラメータ | 値 | 出典 |
|---|---|---|
| washout | 2000 | `configs/default_reservoir_params.json` |
| train_num | 6000 | `configs/default_reservoir_params.json` |
| ridge_lambda | 1e-9 | `configs/default_reservoir_params.json` |
| k_max_d1 | 100 | `analysis/ipc/default_params.json` |
| k_max_d2a | 50 | `analysis/ipc/default_params.json` |
| k_max_d2b | 20 | `analysis/ipc/default_params.json` |

## スクリプト

`analysis/ipc/ipc.py`

主要な依存関数（`vicsek_rc/`）:

| 関数 | 役割 |
|---|---|
| `load_position_fast(path)` | `position.dat` → raw data |
| `build_states(data, N, utime, ...)` | (T, N+1) 状態行列（bias 付き） |
| `load_reservoir_defaults()` | washout/train_num/λ を JSON から取得 |
| `mck_score(y_true, y_pred)` | Pearson 相関の 2 乗（= C_l の経験推定） |
| `write_params_used(...)` | params_used.json を保存 |
| `apply_style()` | matplotlib スタイル統一 |

## 実行コマンド

```bash
python analysis/ipc/ipc.py \
    --data-dir data \
    --rcut 13 \
    --sgm 0.0 \
    --seeds 4 5 6 8 9 \
    --narma-seeds 1 2 3 5 6 \
    --ntime 220000 \
    --output-dir analysis/ipc
```

| 引数 | 意味 |
|---|---|
| `--data-dir` | シム dir を含む親 dir |
| `--rcut` | フィルタする rcut 値 |
| `--sgm` | フィルタする sgm 値 |
| `--seeds` | seed_pos 値リスト（複数 seed を平均） |
| `--narma-seeds` | 各 seed_pos に対応する NARMA 入力シード（`--seeds` と同順・同数） |
| `--narma-dir` | NARMA データ dir（default: `narma_data`） |
| `--tmp-dir` | NARMA 入力ファイルのフォールバック dir（default: `tmp`） |
| `--ntime` | フィルタする ntime（省略時は最新の一致 dir を使用） |
| `--output-dir` | 出力先（タイムスタンプサブ dir を自動生成） |
| `--washout` / `--train-num` / `--ridge-lambda` | リザバーパラメータ（default: JSON） |
| `--k-max-d1` / `--k-max-d2a` / `--k-max-d2b` | 各次数の最大ラグ（default: JSON） |

## 出力ファイル

```
analysis/ipc/<YYYYMMDD_HHMMSS>/
    ipc_data.csv        # per-seed per-basis: seed, type, k1, k2, capacity
    ipc_avg.csv         # seed 平均後の per-basis: type, k1, k2, capacity
    ipc_summary.csv     # 次数別合計: metric, value（C_d1/C_d2a/C_d2b/C_d2/C_total）
    ipc_vs_delay.png    # d1・d2a の容量 vs 遅延 k（2段）
    ipc_by_degree.png   # 次数別合計の棒グラフ
    ipc_heatmap_d2.png  # degree-2 IPC の 2D ヒートマップ（対角=d2a, 非対角=d2b）
    params_used.json    # 使用パラメータ（model fixed/swept + reservoir fixed）
```

## 結果（2026-07-01）

- 出力: `analysis/ipc/20260701_194332/`
- ntime=220000（T_test=14000）、5 seed（seed_pos=4,5,6,8,9）平均

| 指標 | 値 |
|---|---|
| C_d1（線形記憶） | **9.53** |
| C_d2a（単一遅延 2 乗記憶） | **1.50** |
| C_d2b（交差遅延積記憶） | **2.88** |
| C_d2 = C_d2a + C_d2b | **4.38** |
| **C_total（次数 2 まで）** | **13.91** |
| N（理論上限） | 500 |

### seed ごとの内訳

| seed_pos | C_d1 | C_d2a | C_d2b |
|---|---|---|---|
| 4 | 11.12 | 1.54 | 3.18 |
| 5 | 10.40 | 1.49 | 2.94 |
| 6 | 11.14 | 1.75 | 3.49 |
| 8 | **4.39** | **1.07** | **1.17** |
| 9 | 10.62 | 1.66 | 3.59 |

### 考察

- **C_d1 ≈ 9.5** は標準 MC（≈8.1、task_04）よりやや高い。これは T_test が 14000 と長く
  統計的に正確な推定ができたためと考えられる。
- **seed_pos=8 が顕著な外れ値**（C_d1=4.39）。初期配置の偶然により、この実現では
  粒子系が記憶容量の低い動的相に落ち込んだ可能性がある。
- **C_d2 ≈ 4.4** は有意な 2 次容量を示す。レザバー状態 sin(θ_i) が非線形であり、
  線形リードアウトで 2 次演算を実行できることを示す。
- **C_d2b（交差遅延）≈ 2.9 > C_d2a（同一遅延 2 乗）≈ 1.5**:
  過去の異なる時刻の入力積（u(t-k₁)·u(t-k₂)）を記憶する能力が高い。
- **C_total ≈ 13.9 ≪ N = 500**: 次数 2 までで全容量の 2.8% 程度。残りの大部分は
  高次（3 次以上）の非線形記憶容量として存在するか、あるいは独立状態の大部分が
  未活用の状態にある。

---

## 追記: 旧 `analysis/README.md`「実行結果メモ」から移した記録（2026-10-06）

2026-10-06 の文書再編で `analysis/README.md` の実行結果メモを削除した。そのうち、このファイルになかった記述を原文のまま移す。

### IPC と MC の定義変更

- ※ IPC は独自定義（`analysis/ipc/ipc.py`）で、MC の定義変更の影響を受けない
  （旧メモではパスが `vicsek_rc/ipc.py` と誤記されていたので直した）
