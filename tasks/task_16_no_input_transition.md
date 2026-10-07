# #16 入力なし（F=0）での本線のモデルの相転移

**状態**: 段階 1 完了（2026-10-07）。段階 2（1 つずつ本線の値へずらす）は未着手
計画: [../../plan/20261006_193348_no_input_transition.md](../../plan/20261006_193348_no_input_transition.md)

## 目標

リザバーで使っている本線のモデル（`vicsek_dynamic.c`、2026-10-02 修正後）で、入力を切ったとき（F=0）の秩序転移を測る。
論文（Liebchen & Levis 2017、SM）のセットアップから出発し、1 つずつ本線の値へずらして、転移点がどう動くかを追う。
研究の本来の問い（HYPOTHESES E1）の前提。

## コードの変更（2026-10-06）

- `vicsek_dynamic.c` に `nf_sigma` を追加: `nf = 2π(nf_sigma·N(0,1) + nf_mean)`。既定 1.0 で従来と同一（乱数は常に引く）
- セルリストを作り直した: `M = floor(boxsize/rcut) ≥ 3` のとき半殻法のセルリスト（距離は常に最小イメージ）、それ以外は全ペア
- 隣接行列 `A` は `write_adjacency=1` のときだけ確保・更新する
- 検証: rcut=1, 2, 13・v0=0, 0.5・N=500, 2000（L=10）で、変更前のバイナリ（全ペア）と position.dat が 2000 ステップの間完全一致。
  `nf_sigma=0` で全粒子の位相の進みが一致（10⁻⁶）。N=2000・L=10・rcut=1 で約 3 倍速い（1000 ステップ 2.9 s → 0.9 s）
- `vicsek_rc/catalog.py` の `SCALAR_COLS` に `nf_mean`, `nf_sigma`, `n_driver` を追加（TODO C-2 の解消）

## 段階 1: 論文のセットアップ

### 条件

| 量 | 値 |
|---|---|
| N / boxsize（ρ0 = N rcut²/L²） | 2000 / 10.0（20） |
| rcut / v0 / sgm（D_r = sgm²/2） | 1.0 / 0.5 / 1.0（0.5） |
| nf_sigma / Ω（nf_mean = Ω D_r / 2π） | 0 / 0, 0.4, 0.8 |
| F | 0（入力なし。入力ファイルは task_14 の定数入力を流用、値は使われない） |
| K | 0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0, 2.5, 3.0 |
| ntime / utime / h1 | 50000 / 100 / 0.01（実時間 500、500 フレーム。後半 250 フレームを定常区間） |
| 試行 | trial_seeds=[1,2,3]、`seed_noise=b+2, seed_pos=b+3, seed_nf=b+6` |

135 本、`data_newK/`。1 本 1 分弱（24 並列で全体 13 分）。

### 解析

- ψ(t) = |(1/N) Σ e^{iθ_j}|、ψ_ss = 定常区間の平均、χ = N(⟨ψ²⟩ − ⟨ψ⟩²)
- χ は E2 と同じく 3 試行の定常サンプルをまとめて計算（pooled）。試行ごとの χ も CSV に残した
- 論文の量への換算: gρ0 = K/(π D_r)。E2 の論文モデルの結果（`Vicsek_rotate/figures/fig3sm_20260724_174754/fig3sm_data.csv`）を重ねた

### コマンド

```bash
python3 analysis/no_input_transition/run_sims.py
python -m vicsek_rc.catalog --data-dir data_newK
python3 analysis/no_input_transition/transition.py
# 定常性の確認（転移付近を実時間 2000 で）
python3 analysis/no_input_transition/run_sims.py --params analysis/no_input_transition/stage1_long_params.json
python3 analysis/no_input_transition/transition.py --params analysis/no_input_transition/stage1_long_params.json
```

### 出力

- `analysis/no_input_transition/20261007_120724/`（段階 1 本体）: `psi_vs_K.png`, `chi_vs_K.png`, `transition_data.csv`, `transition_summary.csv`,
  `transition_point.csv`, `psi_timeseries.csv`, `params_used.json`
- `analysis/no_input_transition/20261007_121558/`（定常性の確認、Ω=0・K=0.9〜1.6・実時間 2000、21 本）

### 結果

ψ_ss（3 試行平均）と χ（pooled）:

| K | 0 | 0.6 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.6 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ψ_ss（Ω=0） | 0.020 | 0.031 | 0.043 | 0.057 | 0.079 | 0.294 | 0.502 | 0.598 | 0.662 | 0.739 | 0.819 | 0.893 |
| χ（Ω=0） | 0.24 | 0.52 | 1.02 | 1.37 | 3.82 | **8.25** | 1.09 | 0.62 | 0.40 | 0.28 | 0.12 | 0.04 |

- 転移点（χ 最大）: Ω=0 で K=1.1、Ω=0.4 で 1.1、Ω=0.8 で 1.0（gρ0 = 0.70, 0.70, 0.64）。E2 の論文モデル（gρ0≈0.7）と一致
- ψ_ss は Ω=0〜0.8 でほぼ同じ（差 0.02 以内）。転移は回転に不変
- 立ち上がりは論文モデルより緩やか（K=1.1 で 0.29 対 約 0.7）
- 定常性: 実時間 2000 でも K=1.0: 0.088, 1.1: 0.296, 1.2: 0.495, 1.4: 0.658 と実時間 500 の値（0.079, 0.294, 0.502, 0.662）とほぼ同じ。
  実時間 500 の前半と後半の差（最大 0.15）は、初期状態からの立ち上がりが前半の平均に入っていたため

### 考察

- 本線のモデル（近傍の平均）は、論文のセットアップでは論文のモデル（近傍の和）と同じ位置で転移する。平均場の見積もり K_c = 2D_r とも合う
- 立ち上がりの緩やかさは「和」と「平均」の違いと考えられる。和の形では局所密度が高いほど整列が強く、密度と整列が結びついて急な（Vicsek 型の）転移になりやすい。
  平均の形では密度が効かず、蔵本モデル型の連続的な転移になる
- 本線の K=1 は、このセットアップでは転移のすぐ下にあたる
- 段階 2 で本線の値へずらすと、特に ω のばらつき（nf_sigma=1）で転移点が大きく上がると見込む（大域結合の蔵本モデルの見積もりで K_c≈10）
