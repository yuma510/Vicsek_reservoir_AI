# #7 適切な訓練時間、リッジ回帰係数の設定

**状態**: `[ ]`（計画確定・実装前）

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
- **λ × train_num の 2D 掃引**ヒートマップ:
  - NRMSE_test、NRMSE gap（test − train）
  - MC_test、MC gap（train − test）
- 固定 train_num での train vs test 折れ線（λ 依存。λ 増でギャップが縮む様子）
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
（実行後に記入）
- `analysis/ridge_sweep/<YYYYMMDD_HHMMSS>/` … summary JSON・ヒートマップ・折れ線 PNG
- `data/<YYYYMMDD_HHMMSS>_ridge_sweep/` … 長尺シミュレーション（rcut=13, 5 seed）

## 結果
（実行後に記入）

## 考察
（実行後に記入）
