# #7 適切な訓練時間、リッジ回帰係数の設定

**状態**: `[x]`（完了: 2026-06-19）

## 目標
現在のデフォルト設定だと train と test の MC・NRMSE が大きく乖離し、過学習していると考えられる。
リッジ回帰係数 λ と訓練時間ステップ train_num を変化させることで過学習を抑える適切なセットアップを求める。

### 過学習の実測（rcut=13, seed=1, 現行デフォルト）
- NRMSE: train **0.147** vs test **0.20**
- MC: train **7.89** vs test **6.21**

### 原因（readout 側）
- リザバー状態は **N+1 = 501 次元**（N=500 粒子の sin θ ＋ bias 列）
- 訓練サンプルは washout=2000〜train_num=7000 の **わずか 5000 サンプル**
- `ridge_lambda = 1e-11` は実質**正則化なし**（`vicsek_prediction.py:233`, `vicsek_rc/evaluate.py:10`）
- → 501 次元 × λ≈0 でほぼ補間に近い当てはめになり、train だけ良く test が落ちる

## 目的の出力
- **λ × train_num の 2D 掃引**ヒートマップ（各セルは seed 平均値）:
  - NRMSE_test、NRMSE gap（test − train）
  - MC_test、MC gap（train − test）
- 固定 train_num での train vs test 折れ線（λ 依存、seed 平均 ± 標準偏差のエラーバー付き）
- summary JSON（seed 平均/標準偏差）
- **推奨セットアップ**（test NRMSE 最小かつ gap 小の (λ, train_num)）の提示

## 手法

### 設定（確定事項）
- 掃引: **λ × train_num の 2D 掃引**
- **train_num = test_num（等長分割）**: 長さが違うと NRMSE/MC の意味が変わるため、各 T で
  `washout(2000) + train(T) + test(T)` を取る。既存 12000 フレームでは不足するため
  **rcut=13 で長尺の再シミュレーション**を行う。
- 最大 train_num **T_max = 10000** → 総フレーム = 2000 + 2×10000 = **22000**（ntime=220000, utime=10、約 420MB/sim）
- seed 数 **5**（seed 1,2,3,5,6）、rcut=13 固定、sgm=0
- λ レンジ: `1e-10 1e-8 1e-6 1e-4 1e-2 1 1e2 1e4`（8 点・対数）
- train_num レンジ: `2000 4000 6000 8000 10000`

### 実装ステップ

1. **C シミュレータの長尺化** (`vicsek_dynamic.c`)
   - `ntime` がソース内固定（`vicsek_dynamic.c:291`）。後方互換の CLI 引数を追加:
     ```c
     const int n_frames = (argc > 8) ? atoi(argv[8]) : 12000;
     const int ntime    = n_frames * utime;
     ```
   - 引数 7 個までの既存呼び出しは従来どおり 12000 フレーム（後方互換、`run_rcut_sweep.py` に影響なし）
   - 再コンパイル: `cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm`

2. **長尺 NARMA10 生成**: `generate_narma10.py` で length=24000（≥22000）を seed ごとに生成
   （`run_rcut_sweep.py:ensure_narma10` を length 指定で流用）

3. **長尺シミュレーション**: rcut=13・sgm=0 で 5 seed 分を `data/<ts>_ridge_sweep/` グループ dir に出力
   `vicsek_dynamic <narma_input> data/<group> 12 13 0.0 16 <seed> 22000`

4. **評価スクリプト** `analysis/ridge_sweep/run_ridge_sweep.py`（出力 `analysis/ridge_sweep/<ts>/`）
   - seed ごとに `find_exp_dir(rcut=13, seed=s)` → `build_states` で **状態行列を 1 回だけ構築**
   - (λ, train_num) 全組合せを同一 states に対し評価:
     - 窓: `w=2000`, train=`[w:w+T]`, test=`[w+T:w+2T]`（等長）
     - NARMA: `ridge_predict(states, y_target, w, w+T, λ)` → NRMSE/NRMSE2
     - MC: 遅延 0..k_max で `delayed_input` ＋ ridge → `mck_score` 合算
     - gap = NRMSE_test − NRMSE_train、MC_train − MC_test
   - 集約: summary JSON（seed 平均/標準偏差、キャッシュ skip）、ヒートマップ、折れ線
   - **効率化**: Gram `XᵀX` は λ・遅延に依らず train_num のみ依存 → `evaluate.py` に
     Gram/`XᵀY` キャッシュヘルパー（例 `ridge_solve_cached`）を追加して再利用

### 再利用する既存資産
- `vicsek_rc/evaluate.py:ridge_predict`、`vicsek_rc/loaders.py`（`find_exp_dir`/`build_states`/`delayed_input`/`load_position_fast`）
- `vicsek_rc/metrics.py`（`nrmse`/`nrmse2`/`mck_score`）
- `analysis/rcut_sweep/run_rcut_sweep.py`（`ensure_narma10`・`phase3_plot` のスタイル）

## 通常レザバーとの比較
今回の task07 では扱わない（Vicsek レザバーの過学習解消に集中）。ESN（`ESN_reservoir/`）との
train/test ギャップ比較は別タスク／後続で扱う。

## 出力
- `analysis/ridge_sweep/20260619_165214/` … CSV・ヒートマップ・折れ線 PNG・best_params.json・params_used.json
- `data/20260619_165214_ridge_sweep/` … 長尺シミュレーション（rcut=13, ntime=220000, 5 seed）

## 結果

### 第1回（λ 8点: 1e-10〜1e4） — `analysis/ridge_sweep/20260619_165214/`

seed 5 個（1,2,3,5,6）の平均（λ × train_num 上位）:

| λ | train_num | NRMSE_test | NRMSE_gap | MC_test | MC_gap |
|---|---|---|---|---|---|
| **1e-8** | **6000** | **0.1642** | **0.0018** | **7.87** | **0.43** |
| 1e-10 | 6000 | 0.1675 | 0.0147 | 7.62 | 1.64 |
| 1e-6 | 6000 | 0.1685 | 0.0012 | 6.79 | 0.32 |

### 第2回（λ 10点: 1e-12, 1e-9 追加） — `analysis/ridge_sweep/20260619_193558/`

| λ | train_num | NRMSE_test | NRMSE_gap | MC_test | MC_gap |
|---|---|---|---|---|---|
| **1e-9** | **6000** | **0.1628** | **0.0061** | **8.20** | **0.98** |
| 1e-8 | 6000 | 0.1642 | 0.0018 | 7.87 | 0.43 |
| 1e-9 | 2000 | 0.1671 | 0.0132 | 6.00 | 0.42 |
| 1e-10 | 6000 | 0.1675 | 0.0147 | 7.62 | 1.64 |
| 1e-12 | 6000 | 0.1757 | 0.0192 | 7.06 | 1.69 |

**推奨設定: λ=1e-9, train_num=6000**
- NRMSE_test=0.163（第1回 0.164 からさらに改善）
- MC_test=8.20（現行 6.21 から大幅改善）
- gap はわずかに拡大（0.006）するがまだ十分小さい
- λ=1e-12 は実質無正則化に近くギャップが大きい（1e-10 と同様）

## 考察
- 現行デフォルト（λ=1e-11、train_num=7000）は実質正則化なしで過学習していた
- λ=1e-8〜1e-9 程度の正則化で train/test gap が大幅に抑制され、test 性能も向上した
- λ=1e-9 が NRMSE_test 最小（0.163）だが gap は 0.006（λ=1e-8 の 0.002 より大きい）
- train_num=6000 が最適（2000 では学習不足、10000 では過学習傾向）
- λ=1e-12 は 1e-10 と同様に実質無正則化域で競争力なし

---

## 追記: 第3回（λ を 13 点に細分化）— 2026-06-19

出力: `analysis/ridge_sweep/20260619_194250/`

λ = 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1（13 点）
× train_num = 2000, 4000, 6000, 8000, 10000 × 5 seed（1,2,3,5,6）= 325 点。

**最良は第2回と完全に同一**: λ=1e-9, train_num=6000, test_num=6000,
NRMSE_test=0.16284, nrmse_gap=0.00608。

→ λ の格子を 10 点から 13 点に細かくしても推奨設定は動かない。**この掃引はこれで打ち止めでよい。**

## 追記: seed=2 の不安定性の確認 — 2026-06-26

出力: `analysis/ridge_sweep/20260626_113945_seed2_analysis/`
（`20260626_114503_seed2_analysis/` は内容が完全に同一の重複。前者を正とする）

掃引で seed=2 だけ挙動が違ったため、seed=1 と並べて比較した。

| 出力ファイル | 内容 |
|---|---|
| `order_parameter.csv` | 22000 フレームの秩序変数（`r_seed1`, `r_seed2`） |
| `sin_theta_variance.csv` | 22000 フレームの `sinθ` の粒子間分散（`var_seed1`, `var_seed2`） |
| `ridge_prediction.csv` | 8000 フレームの `target` / `pred_s2` / `pred_s1` |

- **seed=2 の予測は振れ幅が過大**。予測の標準偏差は seed=1 で 0.016 だが seed=2 では 0.261
  （正解の標準偏差は 0.110）。つまり seed=2 の予測は正解より 2 倍以上大きく振れている。
- リッジ解が不安定になっている状態で、これが λ を大きめ（1e-9）に取り、
  train_num=6000 に留める根拠のひとつになった。
