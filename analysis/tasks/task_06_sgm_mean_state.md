# #6 sgm ごとの粒子向き複素平均評価

**状態**: `[x]` 完了（2026-06-18, 出力: `analysis/sgm_mean_state/20260618_152006/`） 

## 目標
現在使用しているモデルでは状態の更新にノイズが加わっている。そこで、ノイズの影響を軽減するためノイズの平均化を試したい。ノイズの平均化には2つの方法がある。一つは状態の平均、もう一つは予測値の平均。このタスクでは状態の平均を行いたい。具体的には、同じ入力、同じ初期値から異なるsgmによって更新された粒子の向きの複素平均をとる。

## 目的の出力
横軸が平均をとる数、縦軸が平均した状態の予測精度。

## 前提条件

1. 各 sgm 値について、**初期位置・自然振動数 seed を固定**し、noise seed のみ 1, 2, ..., S_max と変えた S_max 回のシミュレーションが `data/noise_avg_sweep/` に存在すること
   - seed_array[3]（初期位置）= デフォルト 13（argv[7] は使わない）
   - seed_array[6]（自然振動数）= デフォルト 16（argv[7] は使わない）
   - seed_array[2]（ノイズ）= 1, 2, ..., S_max（argv[3] を変化）
2. NARMA10 ファイルが `tmp/narma10_{input|target}_0.0:0.5_seed666.dat` に存在すること

## データ・スクリプト

- **データ**: `data/noise_avg_sweep/`（新規作成。各 sgm × S_max noise seed のシミュレーション）
- **スクリプト**: `analysis/sgm_mean_state/sgm_mean_state.py`
- **新関数**: `build_noise_averaged_states` in `vicsek_rc/evaluate.py`（実装済み）
- **NARMA10**: `tmp/narma10_{input|target}_0.0:0.5_seed666.dat`

> **注意**: `data/sgm_sweep/`（task_05）は位置・振動数・ノイズ seed がすべて変わるため流用不可。

## 実行コマンド

```bash
python analysis/sgm_mean_state/sgm_mean_state.py \
  --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
  --noise-seeds 1 2 3 4 5 6 7 8 9 10 \
  --rcut 13 \
  --n-jobs 4 \
  --data-dir data/noise_avg_sweep \
  --output-dir analysis/sgm_mean_state \
  --input-path  tmp/narma10_input_0.0:0.5_seed666.dat \
  --target-path tmp/narma10_target_0.0:0.5_seed666.dat
```

`--skip-sim` を付けると既存の `data/noise_avg_sweep/` を使って評価のみ実行。

## 手法

各粒子 i について、S 個のノイズ実現を複素平均して noise を抑制した状態を得る:

```
θ_i^avg(t) = arg( Σ_{a=1}^{S} exp(i·θ_i^a(t)) / S )
```

1. 同じ初期値（seed_array[3], seed_array[6] 固定）で noise seed のみ 1..S_max に変えた S_max 回のシミュレーションを実行
2. S_max 個の `position.dat` から各粒子 i の θ_i^a(t) を読み込む（各ファイル: 形状 `(frame_count, N)`）
3. S=1, 2, ..., S_max それぞれについて、先頭 S 個を複素平均: `θ_i^avg(t) = arg( mean_a(exp(i·θ_i^a(t))) )`
4. リードアウト状態: `[1, sin(θ_1^avg(t)), ..., sin(θ_N^avg(t))]`（N+1=501 次元）
5. Ridge 回帰で NARMA10 を予測 → NRMSE・MC を評価
6. S ごとの性能をプロット（複数 sgm を重ねて表示）

> **注意**: `vicsek_rc/evaluate.py` の `compute_theta_mean_states` は空間平均（同時刻の N 粒子を平均）の別実装。混同しないこと。

## 通常レザバーとの比較

| 手法 | 状態次元 | ノイズ低減 |
|---|---|---|
| 通常リザバー（task_05） | 501（sin(θ_m) × 500 + バイアス） | なし（S=1 に相当） |
| noise-avg リザバー（task_06） | 501（sin(θ_i^avg) × 500 + バイアス） | S 回平均で noise を抑制 |

S=1 の場合は通常リザバーと同等。S を増やすにつれ性能が向上するかを確認する。

## 出力

```
analysis/sgm_mean_state/<YYYYMMDD_HHMMSS>/
    mc_nrmse_vs_S.png              # MC (test) と NRMSE (test) を横並び（sgm ごとに折れ線）
    noise_avg_summary.json         # sgm × S の全評価結果
    results_sgm={sgm:.2f}.json     # sgm ごとのキャッシュ（再実行時スキップ）
```

## 結果

（実行後に数値と所感を記入）

## 考察

（実行後に記入）
