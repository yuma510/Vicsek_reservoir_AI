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

- **Phase 0**: 各 trial seed で NARMA10 入力・正解を生成（`tmp/narma10_{input|target}_0.0:0.5_seed{s}.dat`、既存ならスキップ）
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
