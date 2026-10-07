# #1 ネットワーク相関減衰

**状態**: `[×]` 完了

## 目標
レザバーの性能と粒子の相互作用する相手の変化度合いとの関係を調べる。

## 手法

### 隣接行列 A(t)

- N×N の 0/1 行列（粒子 i と j が距離 rcut 以内なら 1、対角成分は 0）
- `position.dat` から `compute_adjacency_from_positions()` で各フレームをオンザフライ計算（周期境界条件込み）

### 正規化相関 C(Δt)

ノートブック（`adjacency_matrix.ipynb`）では Pearson 相関を試みたが、最終的に **正規化内積** を採用した。
「時刻 t に存在したエッジのうち Δt フレーム後も残っている割合」の全時刻平均：

```
C(Δt) = mean_t [ Σᵢⱼ A(t)ᵢⱼ · A(t+Δt)ᵢⱼ / Σᵢⱼ A(t)ᵢⱼ ]
```

- Δt=0 のとき C(0)=1
- A(t) がすべて 0 のフレーム（孤立粒子のみ）はスキップ
- **サンプル数の均一化**: 全 Δt で同じ `n_base = T - dt_max` 個の時刻を使用（Δt ごとにサンプル数が変わると比較がフェアでないため）
- 実装: `vicsek_rc/correlation.py` の `correlation_at_lag(n_base=...)` / `compute_correlation_decay()`

### プロット

| 軸 | 内容 |
|---|---|
| x 軸 | 時間ラグ Δt（フレーム数、1フレーム = 10 ステップ） |
| y 軸 | 正規化相関 C(Δt) |

2 種類の集計プロットを生成する:
- **全体重ね描き** (`plot_correlation_decay_all()`): 全 rcut × seed のカーブを 1 枚に重ね描き
- **単一 seed プロット** (`plot_correlation_by_rcut_single_seed()`): seed を 1 つ選び、rcut ごとのカーブを 1 枚に描く（`--plot-seed` で指定、未指定時は最小 seed を使用）

rcut が大きいほど接続が多く固定的 → C(Δt) の減衰が遅い。

---

## 実装方法

### データ

- `data/` の `position.dat`（既存 117 本）から adjacency をオンザフライで計算
- re-simulation 不要（`write_adjacency` は不使用）
- フレーム範囲: washout 後の定常状態 **frame 2000–2500**（500 フレーム、dt_max=50 で統計が十分）

### スクリプト

`analysis/correlation_analysis/correlation_analysis.py`

主要な依存関数（`vicsek_rc/`）:

| 関数 | 役割 |
|---|---|
| `load_position_dat(path, N, frame_start, frame_end)` | `position.dat` の指定範囲 → `(T, N, 3)` |
| `compute_adjacency_from_positions(pos_frame, rcut, boxsize)` | 1 フレームの位置から N×N bool 行列（周期境界込み） |
| `compute_correlation_decay(matrices, dt_max)` | Δt=0..dt_max の C(Δt) リストを返す |
| `plot_correlation_decay_all(summaries, path)` | 全 rcut カーブの重ね描き |
| `plot_correlation_vs_rcut(summaries, dts, path)` | 指定 Δt での C vs rcut 散布図 |

### 実行コマンド

```bash
python analysis/correlation_analysis/correlation_analysis.py \
    --batch-dir data \
    --filter-rcut 1 2 3 4 5 6 7 8 9 10 11 12 13 \
    --filter-sgm 0.0 \
    --filter-ntime 140000 \
    --filter-v0 0.5 \
    --output-dir analysis/correlation_analysis \
    --plot-seed 4
```

| 引数 | 意味 |
|---|---|
| `--batch-dir` | シム dir を含む親 dir（`data/` 直下に並んだ全 sim dir を検索） |
| `--filter-rcut` | 絞り込む rcut 値（複数可）。省略時は全 rcut |
| `--filter-sgm` | 絞り込む sgm 値。省略時は全 sgm |
| `--filter-ntime` | 絞り込む ntime 値（例: 140000）。省略時は全 ntime |
| `--filter-v0` | 絞り込む v0 値（例: 0.5）。省略時は全 v0 |
| `--output-dir` | 出力先（スクリプトがタイムスタンプサブ dir を自動生成） |
| `--frame-start` | 解析フレームの開始（デフォルト: `configs/default_reservoir_params.json` の `washout`） |
| `--frame-end` | 解析フレームの終了（デフォルト: `default_params.json` の `frame_end`） |
| `--dt-max` | 最大ラグ（フレーム数）。デフォルト: `default_params.json` の `dt_max`） |
| `--target-dts` | C vs rcut プロットに使う Δt 値（デフォルト: 1 5 10） |
| `--plot-seed` | 単一 seed プロットに使う seed 番号（未指定時は結果内の最小 seed） |

### 出力ファイル

```
analysis/correlation_analysis/<YYYYMMDD_HHMMSS>/
    correlation_decay_all.png             # 全 rcut × seed のカーブ重ね描き
    correlation_by_rcut_seed=S.png        # 1 seed の rcut 別カーブ（--plot-seed で指定）
    correlation_vs_rcut.png               # Δt=1,5,10 での C(Δt) vs rcut 散布図
    correlation_rcut=N_seed=S.csv         # 数値データ（dt, correlation）
    analysis_rcut=N_seed=S.json           # メタデータ（params_model, n_frames, dt_max）
    analysis_summary.csv                  # 全 rcut × seed のまとめ
    params_used.json                      # 使用パラメータ（fixed/swept + 解析設定）
```

---

## 実行結果（2026-07-01）

- 出力: `analysis/correlation_analysis/20260701_162825/`
- 118 実験（rcut=1~13 × 9〜10 seeds）、ntime=140000・v0=0.5・sgm=0
- frame 2000–2500（500 フレーム）、dt_max=50、サンプル数均一化（n_base=450）

| rcut | C(Δt=1) 平均 | C(Δt=10) 平均 | 傾向 |
|---|---|---|---|
| 1 | 0.981 | 0.822 | 接続が疎でネットワーク変化が速い |
| 5 | 0.996 | 0.964 | 中程度の安定性 |
| 10 | 1.000 | 0.997 | ほぼ固定 |
| 12–13 | 1.000 | 1.000 | 完全連結・固定（全粒子が rcut 内） |

rcut が大きいほど C(Δt) の減衰が遅くなる（ネットワークが安定）ことを確認。
詳細プロットは `correlation_by_rcut_seed=4.png`・`correlation_decay_all.png`・`correlation_vs_rcut.png` を参照。

---

## 追記: 旧 `analysis/README.md`「実行結果メモ」から移した記録（2026-10-06）

2026-10-06 の文書再編で `analysis/README.md` の実行結果メモを削除した。そのうち、このファイルになかった記述を原文のまま移す。

### correlation_analysis 初回 — 2026-06-30

- 出力: `analysis/correlation_analysis/20260630_202345/`、データ: `data/rcut_sweep/`（117 本）
- position.dat の frame 2000–2500（washout 後の定常状態 500 フレーム）から adjacency を計算
- dt_max=50、rcut=1~13 × 9 seeds（117 実験）
- rcut=1: C(Δt=1)≈0.98, C(Δt=10)≈0.82（ネットワーク変化が速い）
- rcut=13: C(Δt=1)=1.0, C(Δt=10)=1.0（完全連結・固定ネットワーク）
- rcut が大きいほどネットワークが安定し C(Δt) の減衰が遅くなる傾向を確認

### correlation_analysis（dt_max=2000, 訓練区間全体）— 2026-07-15

- 出力: `analysis/correlation_analysis/20260715_192338/`、データ: `data/`（v0=0.5, sgm=0, ntime=140000 の 118 実験）
- フレーム 2000–8000（訓練区間 6001 フレーム、n_base=4001）、dt_max=2000
- `compute_correlation_decay` を FFT 相互相関に高速化（推定量は旧実装と同値、~86 s/実験）
- 各 rcut とも Δt≈500 までにプラトーへ到達し、以後ほぼ一定（周期的な小振動あり）
- プラトー値（Δt=2000, seed 平均）: rcut=1: 0.02 / rcut=4: 0.21 / rcut=7: 0.62 / rcut=9: 0.92 / rcut≥11: 1.00
- rcut≤3 は長時間でほぼ完全にネットワークが再編される一方、rcut≥8 は初期構造の大部分が残存
