# #10 予測値平均の性能（S 掃引）

**状態**: `[x]` 完了（2026-07-01、出力: `analysis/pred_mean/20260701_190915/`）

## 目標
モデルの状態更新にはノイズが加わっている。ノイズの影響を軽減する平均化には 2 つの方法があり、
task_06 では **状態の平均**を評価した。本タスクではもう一方の **予測値の平均**を調べる。

具体的には、同じ入力・同じ初期値（seed_pos・seed_nf 固定）で noise seed のみ変えた S 個のシミュレーションに対し、
**各実現ごとに個別にリザバー予測を行い、S 個の予測値を平均**したときの予測精度を評価する。
task_06（状態平均）と同じデータ・同じ S 掃引で行い、両手法を直接比較する。

## 目的の出力
横軸が平均をとる数 S、縦軸が平均した予測値の予測精度（NRMSE_test / MC_test）。sgm ごとに折れ線で重ねる。
task_06 の `mc_nrmse_vs_S.png` と同形式で、状態平均カーブと比較できるようにする。

## 前提条件

1. task_06 と同一の noise-averaged シミュレーションが `data/` に存在すること（**現存確認済み**）:
   - rcut=13.0、sgm ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}、各 sgm に noise seed 1〜30（計 30 実現）
   - **初期位置・自然振動数 seed を固定**: seed_pos=13、seed_nf=16
   - noise seed のみ変化: seed_noise = 1, 2, …, 30、ntime=140000
   - 取得は `find_exp_dir(data_dir, sgm=…, rcut=13, seed=<noise>, seed_key="seed_noise")`
2. NARMA10（seed=666）が `narma_data/` 以下に存在すること（`find_narma_by_seed("narma_data", 666)` で自動解決）

> **注意**: このデータは task_06 と共有（`data/` は共有の軌道倉庫、CLAUDE.md §4.1）。新規シミュレーションは不要。

## 手法

各 noise 実現 a（a=1..30）について:

1. `position.dat` から通常リザバー状態 `[1, sin(θ_1), …, sin(θ_N)]`（N+1=501 次元）を構築。
2. Ridge 回帰で NARMA10 を予測 → `y_pred^a(t)`。各遅延 k（0..k_max）の入力 `u(t-k)` も予測 → `mck_pred^a_k(t)`。
   （この 1 実現分の計算は `analysis/reservoir_aggregate/reservoir_aggregate.py` の `compute_predictions()` と同じ）

S=1, 2, …, 30 それぞれについて:

3. 先頭 S 実現の予測を平均: `ŷ_S(t) = mean_{a=1..S} y_pred^a(t)`、`m̂ck_{S,k}(t) = mean_{a=1..S} mck_pred^a_k(t)`。
4. `ŷ_S` から NRMSE_test を、各遅延 k の `m̂ck_{S,k}` と `u(t-k)` の二乗相関から MC_k を求め打ち切って MC_test を算出。
5. S ごとの (NRMSE_test, MC_test) を sgm 別に折れ線プロット。

S=1 のとき通常リザバー（単一実現）と同等。S を増やすほど予測平均でノイズが抑制され性能が向上するかを確認する。

## task_06（状態平均）との違い

| 手法 | 平均する対象 | readout 回数 | 予測 |
|---|---|---|---|
| 状態平均（task_06） | 粒子向き θ の複素平均 `arg(mean_a e^{iθ})` | 1 回（平均状態で 1 回学習） | 平均状態から 1 つ |
| 予測平均（task_10） | 各実現の**予測値** `y_pred^a` | S 回（実現ごとに学習） | S 個を平均 |

どちらも S=1 は通常リザバーに一致。同一データ・同一 S 掃引で NRMSE/MC を比較する。

## データ・スクリプト（実装時の想定）

- **データ**: 既存 `data/`（task_06 と共有の noise-averaged 30 実現）を再利用。
- **スクリプト**: `analysis/pred_mean/pred_mean.py`（新規・未実装）。
- **流用**:
  - `analysis/reservoir_aggregate/reservoir_aggregate.py` の `compute_predictions()`（各実現の予測計算）。
  - `vicsek_rc`: `ridge_predict` / `nrmse` / `mck_score` / `corrcoef` / `delayed_input` /
    `find_exp_dir` / `find_narma_by_seed` / `load_reservoir_defaults` / `build_states` / `load_position_fast`。
- **評価パラメータ**: `configs/default_reservoir_params.json`（washout=2000, train_num=6000, ridge_lambda=1e-9,
  k_max=100）を `load_reservoir_defaults()` で取得（CLAUDE.md §5.1、直書き禁止）。

## 出力（実装時の想定）

```
analysis/pred_mean/<YYYYMMDD_HHMMSS>/
    mc_nrmse_vs_S.png       # MC_test と NRMSE_test を横並び（sgm ごとに折れ線）
    pred_mean_data.csv      # long 形式（sgm, S, nrmse_test, MC_test, …）
    params_used.json        # 使用パラメータ（モデル固定/掃引・リザバー）
```

## 実行コマンド（実施）

```bash
python analysis/pred_mean/pred_mean.py \
  --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
  --noise-seeds 1 2 3 4 5 6 7 8 9 10 11 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --rcut 13 --data-dir data --output-dir analysis/pred_mean
```

- 評価長 `n_eval=12000`（NARMA seed=666 長 12000 と sim フレーム 14000 の最小）。test 区間 = [8000, 12000)。
- noise-avg 集合は `seed_pos=13 / seed_nf=16` 固定で厳密に選択（flat な `data/` には同一 seed_noise で
  seed_pos≠13 の別実験も混在するため、`build_index` で seed_pos/seed_nf を絞る）。

## 結果

出力: `analysis/pred_mean/20260701_190915/`（`mc_nrmse_vs_S.png` / `pred_mean_data.csv` / `params_used.json`）

S=1（単一実現）と S=30（30 実現の予測平均）の比較:

| sgm | S=1 MC_test | S=30 MC_test | MC 改善 | S=1 NRMSE_test | S=30 NRMSE_test |
|---|---|---|---|---|---|
| 0.0 | 8.612 | 8.620 | +0.008 | 0.1592 | 0.1592 |
| 0.1 | 2.004 | 6.795 | +4.791 | 0.2387 | 0.2199 |
| 0.2 | 1.632 | 5.355 | +3.723 | 0.2515 | 0.2285 |
| 0.3 | 1.399 | 4.047 | +2.648 | 0.2648 | 0.2397 |
| 0.4 | 1.249 | 3.300 | +2.051 | 0.2712 | 0.2494 |
| 0.5 | 1.133 | 2.747 | +1.614 | 0.2773 | 0.2564 |

## 考察

- **sgm=0（ノイズなし）は S に依らず一定**（MC≈8.62）。ノイズ項が 0 で全実現が同一のため予測平均の効果なし（対照群）。
- **sgm≥0.1 では S を増やすほど MC_test が大きく向上し NRMSE_test が低下**。予測値の平均はノイズ抑制に有効。
- 改善量は **sgm=0.1 で最大（+4.79）**、sgm が大きいほど改善幅は縮小する。
- **task_06（状態平均）との比較**: S=1 は両手法ほぼ一致（＝単一の通常リザバー）。S=30 の MC 回復は
  予測平均のほうが大きい（例 sgm=0.1: 状態平均 +1.74 に対し予測平均 +4.79）。
  ただし本タスクは NARMA 長 12000（test=[8000,12000)）で task_06 とテスト区間長が異なる可能性があり、
  絶対値の直接比較には注意（S 依存の向上傾向は確実）。
- いずれの手法も sgm=0 の MC≈8.6 には到達しない（ノイズによる情報損失を平均化では完全には回復できない）。
