# #4 rcut vs MC/NRMSE（複数trial、散布図）

**状態**: `[x]` 完了

---

複数の trial seed を使い rcut ごとに散布図で MC / NRMSE を可視化する。

## データ・スクリプト

- **データ**: `data/rcut_sweep/`（スクリプトが自動生成）
- **スクリプト**: `analysis/rcut_sweep/run_rcut_sweep.py`

## 前提条件

1. `vicsek_dynamic` がコンパイル済みであること（`cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm`）
2. ディスク空き容量を確認: `df -h /mnt/d/`（13 rcut × 10 trial = 130 シミュレーション、各 ~230MB → 計 ~30GB）

## 実行コマンド

```bash
python analysis/rcut_sweep/run_rcut_sweep.py \
  --rcut-values 1 2 3 4 5 6 7 8 9 10 11 12 13 \
  --trial-seeds 1 2 3 4 5 6 7 8 9 10 \
  --n-jobs 4 \
  --data-dir data/rcut_sweep \
  --output-dir analysis/rcut_sweep
```

## 各フェーズ

- **Phase 0**: 各 trial seed で NARMA10 入力・正解を生成（`narma_data/<YYYYMMDD_HHMMSS>/narma10_{input|target}_0.0:0.5_seed{s}.dat`。既存の日付 dir にあれば再利用）
- **Phase 1**: `vicsek_dynamic` を `--n-jobs` プロセスで並列実行（`--trial-seeds` を NARMA10 seed と argv[7]（base seed）に使用 → `seed_array[i] = trial_seed + i` で全8要素を一括設定。seed_array[2]=noise, seed_array[3]=位置, seed_array[6]=自然振動数 がすべて異なる値で初期化される）
- **Phase 2**: 各 `position.dat` を `evaluate_reservoir` で評価（`results_rcut=*_seed=*.json` にキャッシュ、再実行時スキップ）
- **Phase 3**: rcut ごとに全 trial の値を散布図でプロット

**注意**: `--skip-sim` を付けると既存の `data/rcut_sweep/` を使って評価のみ実行。

## プロット内容

X 軸は rcut 値。各グラフに test（青 ●）と train（赤 △）を同時プロット（9 seed 分の散点、seed=4 は NARMA10 発散のためスキップ）。

| プロット | test キー | train キー |
|---|---|---|
| `mc_vs_rcut_scatter.png` | `MC_test` | `MC_train` |
| `nrmse_vs_rcut_scatter.png` | `nrmse_test` | `nrmse_train` |
| `nrmse2_vs_rcut_scatter.png` | `nrmse2_test` | `nrmse2_train` |

サマリー JSON には test/train 両方の平均・標本標準偏差（`np.nanstd(..., ddof=1)`）を記録。

## 出力

`analysis/rcut_sweep/<YYYYMMDD_HHMMSS>/mc_vs_rcut_scatter.png`, `nrmse_vs_rcut_scatter.png`, `nrmse2_vs_rcut_scatter.png`, `rcut_sweep_summary.json`

## 実行後の後処理

（CLAUDE.md Section 2a/2c より）

- `data/README.md` の「グループ実験一覧」に `data/rcut_sweep/` を追記（目的・変化パラメータ・固定パラメータ）
- `analysis/README.md` の解析一覧に実行結果メモを追記

## 結果

### 第2回（ntime=140000, 14000 フレーム, train_num=6000 サンプル, λ=1e-9）
出力: `analysis/rcut_sweep/20260619_204013/`、データ: `data/rcut_sweep_v2/`

**変更内容**: `evaluate_reservoir` のバグ修正（train_num=6000 をフレームインデックスではなく訓練サンプル数として解釈）と ntime=140000 への変更（等長 train/test: 各 6000 サンプル）。

rcut=13 での 9 seed 平均（seed=4 は NARMA10 発散のためスキップ）:

| 指標 | 平均 | 標準偏差 |
|---|---|---|
| MC_test | **8.142** | 0.302 |
| NRMSE_test | **0.1641** | 0.0078 |

- MC_test 最大の rcut: **12**（平均 8.3 付近）
- rcut が小さい（＜5）と MC が急激に低下

### 考察
- 第1回（ntime=120000, 旧デフォルト λ=1e-11, train_num=7000 フレームインデックス=5000 サンプル）より MC_test が向上
- 正則化 (λ=1e-9) と適切な訓練サンプル数 (6000) により test 性能が改善

---

### 第3回 — v0=0（静的ネットワーク、ntime=140000, λ=1e-9）

出力: `analysis/rcut_sweep/20260623_121526/`、データ: `data/rcut_sweep_v0/`

```bash
python analysis/rcut_sweep/run_rcut_sweep.py \
  --rcut-values 1 2 3 4 5 6 7 8 9 10 11 12 13 \
  --trial-seeds 1 2 3 4 5 6 7 8 9 10 \
  --n-jobs 4 --v0 0.0 \
  --data-dir data/rcut_sweep_v0 \
  --output-dir analysis/rcut_sweep
```

rcut 別 MC_test 平均±標準偏差（9 seed、seed=4 は NARMA10 発散でスキップ）:

| rcut | MC_test 平均 | MC_test std | NRMSE_test 平均 |
|---|---|---|---|
| 1 | **10.409** | 0.925 | 0.1633 |
| 2 | **10.471** | 1.058 | 0.1630 |
| 3 | 10.324 | 0.918 | 0.1626 |
| 4 | 10.200 | 0.969 | 0.1624 |
| 5 | 10.037 | 0.835 | 0.1625 |
| 6 | 9.871 | 0.704 | 0.1624 |
| 7 | 9.761 | 0.662 | 0.1623 |
| 8 | 9.388 | 0.419 | 0.1626 |
| 9 | 9.007 | 0.462 | 0.1627 |
| 10 | 8.717 | 0.370 | 0.1630 |
| 11 | 8.232 | 0.344 | 0.1636 |
| 12 | 8.214 | 0.338 | 0.1637 |
| 13 | 8.194 | 0.345 | 0.1637 |

- MC_test 最大の rcut: **2**（平均 10.47）
- rcut が大きいほど MC 単調減少（v0=0.5 とは逆傾向）
- NRMSE は rcut によらずほぼ一定（0.162〜0.164）

### v0=0 vs v0=0.5 比較（rcut=13 で比較）

| v0 | MC_test | NRMSE_test |
|---|---|---|
| 0.0（静的） | 8.19 ± 0.34 | 0.164 ± 0.008 |
| 0.5（動的） | 8.14 ± 0.30 | 0.164 ± 0.008 |

rcut=13 では v0 によらずほぼ同一性能。ただし v0=0 は rcut が小さいほど高 MC（最大 10.47@rcut=2）。

### 考察（v0=0）
- v0=0 では粒子が空間移動せず相互作用ネットワークが**静的**（初期位置のみで決定）
- rcut が小さい → 粒子間結合が疎 → 各粒子が独立したオシレータに近い → 多様な状態が保存 → 高 MC
- rcut が大きい → 全粒子が連結 → 同期（synchronization）が起きやすい → 状態の多様性が消える → 低 MC
- v0=0.5（動的）では rcut=12 付近で MC にピークがあったが、v0=0 では単調減少 → 動的ネットワーク（粒子の移動）が中程度の rcut での MC 改善に貢献している
