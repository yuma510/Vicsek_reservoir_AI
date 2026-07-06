# コード詳細リファレンス

> prompt.md の Phase 0 から切り出した技術詳細。コードを変更した場合はここも更新すること。

---

## モデルとリザバー計算の対応

このプロジェクトでは **Vicsek モデル（結合振動子系）をリザバーとして使用**し、NARMA10 タスクへの性能を評価する。

### リザバー計算の概念とコードの対応

| リザバー計算の概念 | 本プロジェクトでの実体 | 対応するコード / ファイル |
|---|---|---|
| リザバー（動的システム） | Vicsek モデル（N=500 粒子の結合振動子） | `vicsek_dynamic.c` |
| リザバー状態 x(t) | 各粒子の角度 θ_m（utime=10 ステップごとのスナップショット） | `position.dat` 3列目、`build_states` in `vicsek_rc/loaders.py` |
| 入力信号 u(t) | NARMA10 入力時系列（長さ 12000、Uniform(0, 0.5)） | `narma_data/<YYYYMMDD_HHMMSS>/narma10_input_*.dat` |
| 入力注入 | 角度更新式の外力項 `F·sin(c·v[t] - θ_m)`（v[t] = 4·(u[t]-0.25) でスケーリング） | `vicsek_dynamic.c` 更新式 |
| リードアウト状態 φ(x(t)) | `sin(θ_m)` の N 次元ベクトル + バイアス 1 → (N+1) 次元 | `build_states(readout=1, add_bias=True)` |
| 出力重み W_out | Ridge 回帰係数 | `ridge_predict` in `vicsek_rc/evaluate.py` |
| 正解信号 y(t) | NARMA10 ターゲット時系列 | `narma_data/<YYYYMMDD_HHMMSS>/narma10_target_*.dat` |
| ウォームアップ（washout） | 最初の 2000 ステップを除外 | `evaluate_reservoir` パラメータ `washout=2000` |
| 訓練区間 | [2000, 8000) の 6000 サンプル | `train_num=6000`（サンプル数。フレームインデックス = washout + train_num = 8000） |
| テスト区間 | [8000, 14000) の 6000 サンプル | 等長分割（ntime=140000 = 14000フレーム が必要） |
| 性能指標 | NRMSE（予測精度）・MC（記憶容量） | `vicsek_rc/metrics.py` |

### 入力注入の詳細

u(t) → スケーリング → v(t) → 角度への外力 という流れ:

```
u[t] ∈ [0, 0.5]  →  v[t] = 4·(u[t] - 0.25) ∈ [-1, 1]
θ_m(t+1) += h1·F·sin(c·v[t] - θ_m(t))   （入力注入項のみ抜粋）
```

F=14.3（注入強度）、c=0.1（位相スケール）はデフォルト固定値。

### リードアウト状態の構築

```
position.dat の theta 列
  → reshape: (12000フレーム, N=500粒子)
  → sin(θ) 適用
  → バイアス列 [1, 1, ..., 1] を先頭に追加
  → 状態行列 X: shape (12000, N+1) = (12000, 501)
```

訓練は `X[washout:washout+train_num]` と `y[washout:washout+train_num]` で Ridge 回帰を解く（train_num はサンプル数）。

---

## コード全体で行っていること
このコードではVicsekモデルを使用した物理レザバーを行っている。そのためにもとのVicsekモデルに入力を加えて、その状態を読み取って学習（予測）を行っている。vicsek_dynamic.cでは、入力が加えられているVicsekモデルのシミュレーションを行っている。そして、時間変化する各粒子の状態をデータとして出力している。vicsek_rc/ では、出力された粒子の状態の読み出し（リードアウト）とリッジ回帰（評価）の共通関数を提供している。analysis/ では、このレザバーの予測精度に影響を与える要因について調べている。

---

## データフロー

```
generate_narma10.py
  └─→ narma_data/<YYYYMMDD_HHMMSS>/narma10_input_<low>:<high>_seed<seed>.dat   (入力信号 u)
  └─→ narma_data/<YYYYMMDD_HHMMSS>/narma10_target_<low>:<high>_seed<seed>.dat  (正解信号 y)
  └─→ narma_data/<YYYYMMDD_HHMMSS>/narma10_params_<low>:<high>_seed<seed>.json (使用パラメータ)
  （読込側は narma_data/ 以下の最新日付 dir を自動選択: vicsek_rc.find_narma_by_seed）

# 個別実行
vicsek_dynamic.c  ←  narma_data/<YYYYMMDD_HHMMSS>/narma10_input_*.dat
  └─→ data/<YYYYMMDD_HHMMSS>/position.dat
  └─→ data/<YYYYMMDD_HHMMSS>/params_model.json
  └─→ data/<YYYYMMDD_HHMMSS>/adjacency/adjacency_<frame>.dat   (write_adjacency=1 のとき)

# グループ実行（スイープスクリプトによる複数シミュレーション）
vicsek_dynamic.c (×N)  ←  narma_data/<YYYYMMDD_HHMMSS>/narma10_input_*.dat
  └─→ data/<YYYYMMDD_HHMMSS>/{position.dat, params_model.json}

# 解析
analysis/<解析名>/<script>.py  ←  data/
  └─→ analysis/<解析名>/<YYYYMMDD_HHMMSS>/*_data.csv      (プロット元データ、long 形式)
  └─→ analysis/<解析名>/<YYYYMMDD_HHMMSS>/*.png
  └─→ analysis/<解析名>/<YYYYMMDD_HHMMSS>/params_used.json (使用パラメータ、JSON)
  └─→ analysis/<解析名>/<YYYYMMDD_HHMMSS>/results_*.json  (キャッシュ、JSON)
```

プロット元データは CSV（long 形式）で保存する（サマリー JSON は廃止）。パラメータ/メタデータは JSON のまま。
詳細は CLAUDE.md §3 を参照。各スクリプトの CSV: `rcut_sweep_data.csv` / `sgm_sweep_data.csv` /
`noise_avg_data.csv` / `MCk.csv` / `narma10_prediction.csv` / `NARMA10_avg.csv` / `MCk_vs_delay.csv` /
`summary.csv` / `summary_all.csv` / `correlation_<tag>.csv` / `analysis_summary.csv`。

---

## generate_narma10.py

NARMA10 の入力・正解信号を生成する。

**漸化式：**
```
y(t+1) = 0.3·y(t) + 0.05·y(t)·Σy(t-9:t) + 1.5·u(t-9)·u(t) + 0.1
u(t) ~ Uniform(low, high)
```

**デフォルト引数：**

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--length` | 12000 | 時系列長 |
| `--seed` | 666 | 乱数シード |
| `--low` / `--high` | 0.0 / 0.5 | 入力の範囲 |
| `--output-dir` | `tmp` | 出力先 |

**出力先：** `--output-dir`（default `narma_data`）配下に日付 dir
`narma_data/<YYYYMMDD_HHMMSS>/` を自動生成する（`vicsek_rc.new_narma_dir`）。
**1 日付 dir = 1 データセット**（下記 3 ファイル）で、
同一秒に複数生成する場合は `_1`, `_2`, … を付与して衝突を避ける。

| ファイル | 内容 |
|---|---|
| `narma10_input_<low>:<high>_seed<seed>.dat` | 入力信号 u |
| `narma10_target_<low>:<high>_seed<seed>.dat` | 正解信号 y |
| `narma10_params_<low>:<high>_seed<seed>.json` | 使用パラメータ（`length` / `seed` / `low` / `high` / `input_file` / `target_file`）。再現用メタデータ（CLAUDE.md §5、`vicsek_rc.save_narma_params`） |

読込側（sgm・rcut・ridge・correlation・reservoir_aggregate 等の解析スクリプト）は
`--input-path`/`--target-path` 未指定時に `vicsek_rc.find_narma_by_seed` で
`narma_data/` 以下の**最新日付 dir**（無ければ直下）を seed で自動解決する。
`--narma-root` / `--narma-seed`（default 666）で変更可能。
生成側（`ensure_narma10`）は既存日付 dir を再利用し、無い seed だけ**その seed 専用の
新しい日付 dir**に生成する（1 dir = 1 データセット）。

---

## vicsek_dynamic.c

### 入力注入

`u[t]`（12000 点）をファイルから読み込み `v[t] = 4·(u[t] - 0.25)` にスケーリングして角度更新式に注入。

### 更新式

```
x_m  += h1 · v0 · cos(θ_m)
y_m  += h1 · v0 · sin(θ_m)
θ_m  += h1·nf[m] + h1·(K/N)·ft[m] + h1·F·sin(c·v[t] - θ_m) + sqrt(h1)·sgm·ξ
```

| 変数 | 意味 |
|---|---|
| `nf[m]` | 自然振動数: `2π·(N(0,1)+1)` |
| `ft[m]` | 平均場力: `mean(sin(θ_j - θ_m))` (cutoff 内) |
| `K` | 相互作用結合強度 |
| `F` | 入力強制振幅 |
| `c` | 入力位相スケール |
| `sgm` | ノイズ振幅 |

### パラメータ

| パラメータ | デフォルト値 | 意味 |
|---|---|---|
| `N` | 500 | 粒子数 |
| `boxsize` | 15.8 | 領域一辺の長さ（周期境界） |
| `rho` | 2.0 | 粒子密度 |
| `h1` | 0.01 | 時間刻み |
| `v0` | 0.5 | 粒子の速さ |
| `K` | 1.0 | 結合強度 |
| `F` | 14.3 | 入力強制振幅 |
| `c` | 0.1 | 入力位相スケール |
| `sgm` | 0.0 | ノイズ振幅 |
| `utime` | 10 | 入力1点あたりのステップ数 |
| `ntime` | 120000 | 総ステップ数（= 12000 × utime） |
| `rcut` | 1.0 | 相互作用カットオフ（単一値、CLI で指定） |
| `write_adjacency` | 0 | 隣接行列を `adjacency/` に出力するか（0=off, 1=on） |

### RNG 設計

乱数には **xoshiro256\*\*** + **Box-Muller** を使用。用途ごとに独立したインスタンス（`xrng_t`）を持ち、異なるシードから独立したストリームを生成する。

| インスタンス | JSON キー | デフォルト | 用途 | 備考 |
|---|---|---|---|---|
| `rng_pos` | `seed_pos` | 13 | 初期位置 (x, y, θ) | sgm=0 でも必ず有効 |
| `rng_nf` | `seed_nf` | 16 | 自然振動数 `nf[i]` | sgm=0 でも必ず有効 |
| `rng_noise` | `seed_noise` | 12 | シミュレーション中のノイズ `ξ` | **sgm=0.0 のとき動力学に影響しない** |

3インスタンスはすべて異なる整数で初期化されるため独立したストリームになる。

### シード設計・スイープルール

rcut/sgm スイープで trial 間のエラーバーをとるときは `seed_noise/pos/nf` をすべて変化させる。スクリプトでは trial seed（base）を `b` として `seed_noise=b+2, seed_pos=b+3, seed_nf=b+6` で設定する。

- rcut スイープ（sgm=0 固定）: `seed_noise` は動力学に影響しないが、`seed_pos`・`seed_nf` が試行間で結果を変える。
- sgm スイープ（sgm>0）: ノイズを含む 3 つの seed すべてが変わる。
- **実験ディレクトリの識別キー**: `seed_pos`（`find_exp_dir` の `seed_key="seed_pos"` で検索）。

### CLI 引数 / JSON 設定ファイル

```
./vicsek_dynamic [config.json]
```

`argv[1]` に JSON ファイルパスを渡す（省略時はすべてデフォルト値で動作）。

JSON フォーマット（`configs/default_params.json` 参照 / `params_model.json` と同一形式）:

```json
{
  "input_file":  "narma_data/<YYYYMMDD_HHMMSS>/narma10_input_0.0:0.5_seed666.dat",
  "output_base": "data",
  "N": 500,
  "boxsize": 15.8,
  "ntime": 120000,
  "utime": 10,
  "h1": 0.01,
  "v0": 0.5,
  "sgm": 0.0,
  "K": 1.0,
  "F": 14.3,
  "c": 0.1,
  "rcut": 13.0,
  "rho": 2.0,
  "seed_noise": 12,
  "seed_pos": 13,
  "seed_nf": 16,
  "write_adjacency": 0
}
```

`params_model.json`（出力フォーマット。`input_file` と `write_adjacency` が追加される）:

```json
{
  "model": 1,
  "N": 500,
  "boxsize": 15.800000,
  "ntime": 120000,
  "utime": 10,
  ...
  "seed_noise": 12,
  "seed_pos": 13,
  "seed_nf": 16,
  "write_adjacency": 0,
  "input_file": "narma_data/<YYYYMMDD_HHMMSS>/narma10_input_0.0:0.5_seed10_n24000.dat"
}
```

出力ディレクトリ名: `<YYYYMMDD_HHMMSS>`（例: `20260615_120000`）。パラメータは `params_model.json` に記録される（入出力フォーマット同一）。

### 出力

**`position.dat`** — `x y theta` の3列テキスト。utime ステップごとに 1 フレーム（各入力期間の末尾ステップ）を N 行出力。合計 N×(ntime/utime) 行（デフォルト: N×12000 = 6,000,000 行）。

**`params_model.json`** — 上記パラメータ + seed + `write_adjacency` + `input_file`（使用した NARMA 入力ファイルのパス）をすべて記録。`input_file` により position.dat の再現に必要な入力シーケンスが自明になる。

**`adjacency/adjacency_<frame>.dat`** — `write_adjacency=1` のときのみ生成。N×N の 0/1 行列（スペース区切り、対称行列）。`frame` は 0 始まりの utime フレームインデックス（6 桁ゼロ埋め）。1 ファイルあたり約 500 KB（N=500 時）。

リザバー状態として使われるのは3列目の `theta`（粒子の向き）。

---

## vicsek_rc/evaluate.py — 主要関数

| 関数 | 入力 | 出力 | 説明 |
|---|---|---|---|
| `ridge_predict(states, target, washout, train_num)` | (T, D) 状態行列・正解 | (T,) 予測値 | Ridge 回帰で訓練後に全区間を予測 |
| `evaluate_reservoir(pos_path, params, target_path, input_path)` | 単一 position.dat | dict（NRMSE/MC 6指標） | 通常 500D リザバー評価（task_03〜05 で使用） |
| `compute_theta_mean_states(theta_all, utime)` | (total_frames, N) θ 行列 | (T, 3) 状態行列 | 空間平均: N 粒子の複素平均方向 → 3D 状態 |
| `build_noise_averaged_states(theta_list)` | S × (T, N) θ 配列のリスト | (T, N+1) 状態行列 | **noise 実現平均**: 粒子ごとに S 個の noise 実現を複素平均 → 501D 状態（task_06 で使用） |

### `build_noise_averaged_states` の詳細

```python
# theta_list: list of S arrays, each shape (T, N), already subsampled (T=12000)
stacked  = np.stack(theta_list, axis=0)        # (S, T, N)
cplx_avg = np.exp(1j * stacked).mean(axis=0)  # (T, N) — S 実現の複素平均
theta_avg = np.angle(cplx_avg)                 # (T, N)
states    = np.sin(theta_avg)                  # (T, N)
# + bias → (T, N+1)
```

`compute_theta_mean_states`（空間平均、3D）とは別実装。混同しないこと。

---

## vicsek_rc/loaders.py — `find_exp_dir`

```python
find_exp_dir(data_dir, *, rcut=None, sgm=None,
             seed=None, seed_index=None, seed_key=None)
```

| パラメータ | 説明 |
|---|---|
| `seed_key` | **新形式**（JSON named seeds）: `params[seed_key] == seed` で照合。例: `seed_key="seed_pos"` |
| `seed_index` | **旧形式**（8要素配列）: `params["seed"][seed_index] == seed` で照合（後方互換のために存在） |

新しいシミュレーション（JSON設定で実行）では `seed_key="seed_pos"` を使用する。
既存データ（旧 argv 形式で生成）は `seed_index=0` にフォールバックする。

---

## vicsek_rc/params_io.py — 解析プロットのパラメータ記録

| 関数 | 入力 | 出力 | 説明 |
|---|---|---|---|
| `split_fixed_varied(param_dicts)` | param dict のリスト | `(fixed, varied)` | 全 dict で一致するキーを `fixed`（スカラ）、異なるキーを `varied`（distinct 値リスト）に自動分離。list 値（`seed`）も対応 |
| `write_params_used(output_dir, model_param_dicts, reservoir_fixed, reservoir_swept=None)` | モデル param dict 群・レザバー param | `params_used.json` のパス | model は `split_fixed_varied` で自動分離、reservoir は呼び出し側が fixed/swept を明示して書き出す |

各 sweep スクリプト（`run_rcut_sweep.py` / `run_sgm_sweep.py` / `sgm_mean_state.py` / `run_ridge_sweep.py`）は
出力 dir に `params_used.json` を保存する（CLAUDE.md §3）。掃引軸（rcut/sgm/seed/λ/train_num/S 等）は
固定値と区別して**範囲（値リスト）でコンパクトに**記録され、全 sim の丸ごとダンプはしない。

```json
{
  "model":     { "fixed": { "N":500, "boxsize":15.8, ... }, "swept": { "rcut":[1,...,13], "seed_pos":[...] } },
  "reservoir": { "fixed": { "readout":1, "washout":2000, "train_num":7000, "ridge_lambda":1e-11, "k_max":100 },
                 "swept": {} }
}
```

---

## コード変更時のルール

変更後は必ず関連コードを再読し、以下の影響がないか確認して **このファイルの該当箇所を更新**すること。

- 入出力形式・ディレクトリ構成
- 状態ベクトルの構築方法・学習方法・評価方法
- パラメータの意味・NARMA10 および MC の計算方法
- グラフ生成方法

仕様変更を検出した場合は実験を継続する前に変更点を要約すること。
