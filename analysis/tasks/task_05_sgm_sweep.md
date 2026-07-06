# #5 sgm vs MC/NRMSE（複数seed、散布図）

## やりたいこと

ノイズ強度を変化させて、そのときのレザバーの性能を調べたい。具体的には、sgmを0, 0.1, ..., 0.5の範囲で変化させて、それぞれのMCとNRMSE、NRMSE2をプロットしたい。このとき、MCやNRMSEの信頼性を確保するため、各sgmに対して10個のseedでレザバー性能を測りたい。

**状態**: `[x]` 完了（2026-06-18, 出力: `analysis/sgm_sweep/20260618_122652/`）

---

## データ・スクリプト

- **データ**: `data/sgm_sweep/`（スクリプトが自動生成）
- **スクリプト**: `analysis/sgm_sweep/run_sgm_sweep.py`

## 前提条件

1. `vicsek_dynamic` がコンパイル済みであること（`cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm`）
2. NARMA10 ファイルが存在すること（`python generate_narma10.py` で生成、`narma_data/<YYYYMMDD_HHMMSS>/narma10_{input|target}_0.0:0.5_seed666.dat`）。
   `--input-path`/`--target-path` は省略可（`narma_data/` 以下の最新日付 dir を自動解決。下記コマンドの明示指定は任意）

## 実行コマンド

```bash
python analysis/sgm_sweep/run_sgm_sweep.py \
  --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
  --rcut 13 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --n-jobs 4 \
  --input-path narma_data/narma10_input_0.0:0.5_seed666.dat \
  --target-path narma_data/narma10_target_0.0:0.5_seed666.dat \
  --output-dir analysis/sgm_sweep
```

## 各フェーズ

- **Phase 1**: 6 sgm × 10 seed = 60 シミュレーションを n_jobs=4 で並列実行。各 trial seed は `argv[7]`（base seed）として `vicsek_dynamic` に渡し、`seed_array[i] = seed + i` で全 8 要素を設定する
- **Phase 2**: 各 `position.dat` を `evaluate_reservoir` で評価し `results_sgm=*_seed=*.json` にキャッシュ（再実行時スキップ）
- **Phase 3**: sgm ごとに平均・標準偏差を計算し、エラーバープロットを生成

## seed 設計

各 trial seed `s` に対して `seed_array[i] = s + i` で全 8 要素を設定する。

| seed_array インデックス | 値 | 用途 |
|---|---|---|
| 0 | s | 識別キー（find_exp_dir で seed_index=0 を使用） |
| 2 | s+2 | ノイズ RNG（`rng_noise`）|
| 3 | s+3 | 初期位置 RNG（`rng_pos`）|
| 6 | s+6 | 自然振動数 RNG（`rng_nf`）|

3 つの RNG インスタンスはすべて異なる整数で初期化されるため独立したストリームになる。

## プロット内容

X 軸は sgm 値。各グラフに test（青 ●）と train（赤 △）を同時プロット（10 seed 分の散点）。

| プロット | test キー | train キー |
|---|---|---|
| `mc_vs_sgm_scatter.png` | `MC_test` | `MC_train` |
| `nrmse_vs_sgm_scatter.png` | `nrmse_test` | `nrmse_train` |
| `nrmse2_vs_sgm_scatter.png` | `nrmse2_test` | `nrmse2_train` |

サマリー JSON には test/train 両方の平均・標本標準偏差（`np.nanstd(..., ddof=1)`）を記録。

## 出力

`analysis/sgm_sweep/<YYYYMMDD_HHMMSS>/mc_vs_sgm_scatter.png`, `nrmse_vs_sgm_scatter.png`, `nrmse2_vs_sgm_scatter.png`, `sgm_sweep_summary.json`

## 注意・制限事項

- **sgm=0 のときのノイズ seed**: `seed_array[2]`（ノイズ）は sgm=0 では動力学に影響しない。ただし `seed_array[3]`（初期位置）と `seed_array[6]`（自然振動数）は sgm=0 でも有効なため、10 seed 間で異なる結果が得られる
- **`--skip-sim`**: 既存の `data/sgm_sweep/` を使って評価のみ実行する場合に付ける

## 結果

### 第2回（ntime=140000, train_num=6000 サンプル, λ=1e-9）
出力: `analysis/sgm_sweep/20260620_070251/`、データ: `data/sgm_sweep_v2/`

**変更内容**: `evaluate_reservoir` バグ修正 + ntime=140000 への変更（等長 train/test: 各 6000 サンプル）。

seed 平均±標準偏差（10 seed）:

| sgm | MC_test | NRMSE_test |
|---|---|---|
| 0.0 | 6.856 ± 2.513 | 0.3592 ± 0.4226 |
| 0.1 | 1.965 ± 0.038 | 0.2355 ± 0.0029 |
| 0.2 | 1.616 ± 0.025 | 0.2497 ± 0.0025 |
| 0.3 | 1.408 ± 0.027 | 0.2616 ± 0.0028 |
| 0.4 | 1.242 ± 0.034 | 0.2690 ± 0.0023 |
| 0.5 | 1.140 ± 0.019 | 0.2741 ± 0.0019 |

## 考察
- sgm=0（ノイズなし）で MC が高く（6.9）、ノイズが増えるにつれて急速に低下
- sgm=0 の標準偏差が大きい（2.5）のは、ノイズなしでは初期位置・振動数の差異が動力学に大きく影響するため
- sgm ≥ 0.1 では標準偏差が小さく（～0.03）再現性が高い
