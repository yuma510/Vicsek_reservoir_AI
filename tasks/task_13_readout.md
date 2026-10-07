# #13 リードアウトの効果検証（粒子1個で k=0 の入力復元、nf_mean 依存）

**状態**: `[x]` 完了（2026-07-31, 出力: `analysis/task13_readout/20260731_164419/` と
`analysis/task13_readout/20260731_171338_nfmean/`）

## 目標

リードアウト（`sinθ` から入力を線形復元する部分）が、どれだけの復元能力を持っているのかを
系のダイナミクスから切り離して確認する。Memory Capacity は k ステップ前の入力を予測するが、
本タスクは **k=0（現在の入力そのもの）だけ**を予測対象にする。粒子を 1 個にすることで
粒子間相互作用も消え、「入力注入 → 位相 → sinθ → リッジ回帰」の経路だけが残る。

あわせて、自然周波数の平均 `nf_mean` を上げたときに復元能力がどう変わるかを見る。

## 手法

粒子 N=1、駆動粒子 n_driver=1 のリザバーを NARMA10 入力 u(t) で駆動し、状態
`[1, sin(θ(t))]`（2 次元）からリッジ回帰で u(t) を予測する。

自然周波数は 1 粒子でも

```
nf = 2π (N(0,1) + nf_mean)
```

で引かれる（`seed_nf` で固定）。`nf_mean` は 2026-07-31 に `vicsek_dynamic.c` へ追加した
パラメータで、既定 1.0 が従来の挙動に一致する。`nf_mean` を上げると粒子の自転が速くなり、
入力によるトルクに対して位相が絶えず回り続ける。

評価は NRMSE_test と MC_0（遅延 0 の決定係数 corr²）。**MC_0 は MC の総和には含まれない**
（2026-07-23 以降の標準定義では遅延 k=0 を総和から除く）ので、ここで見ているのは
「MC には数えない自明な項そのものの大きさ」である。

さらに、読み出しを `sinθ` ではなく θ そのものにした場合と比較して、`sinθ` を使うことによる
損失があるかを確認した。

## データ・スクリプト

- `analysis/task13_readout/run_task13.py` — N=1 シミュレーション実行 → 状態構築 → k=0 予測。
  パラメータは `analysis/task13_readout/default_params.json` から読む（直書き禁止）
- 入力: `narma_data/20260724_141722/narma10_input_0.0:0.5_seed666.dat`（NARMA10, seed=666）
- シミュレーション出力: `20260731_164419` は `data/` 直下、`20260731_171338_nfmean` は
  出力 dir 配下の `sim_nf{1.0,2.0,3.0}/<ts>/` に置かれている

## 前提条件

| パラメータ | 値 |
|---|---|
| N / n_driver | 1 / 1 |
| boxsize / rho | 15.8 / 2.0 |
| ntime / utime | 140000 / 10（＝14000 フレーム） |
| h1 / v0 / sgm | 0.01 / 0.5 / 0.0 |
| K / rcut | 1.0 / 13.0（1 粒子なので相互作用は効かない） |
| F / c | 14.3 / 0.1 |
| nf_mean | 1.0（既定）、2.0、3.0 |
| seed_noise / seed_pos / seed_nf | 1 / 1 / 1 |
| washout / train_num / λ | 2000 / 6000 / 1e-9 |
| n_eval | 14000 |

## 実行コマンド

```bash
# 本体（nf_mean は default_params.json の値を使う）
python analysis/task13_readout/run_task13.py

# 既存の sim を使って評価だけやり直す場合
python analysis/task13_readout/run_task13.py --skip-sim --sim-dir data/<YYYYMMDD_HHMMSS>
```

## プロット内容

- `readout_prediction.png` — 予測 u_pred と正解 u_true の時系列（テスト区間）
- `csv_timeseries.png` — θ と sin θ の時系列
- `nfmean_readout.png` — nf_mean=1,2,3 の予測を並べた比較
- `readout_theta_vs_sin_nf{1,2,3}.png` — 読み出しを θ にした場合と `sinθ` の場合の比較

## 出力

```
analysis/task13_readout/20260731_164419/
├── theta.csv                    各フレームの θ
├── sin_theta.csv                各フレームの sin θ
├── prediction.csv               u_true, u_pred（k=0）
├── readout_prediction.png
├── csv_timeseries.png
└── params_used.json             使用パラメータ＋NRMSE_test, MC_0

analysis/task13_readout/20260731_171338_nfmean/
├── sim_nf{1.0,2.0,3.0}/<ts>/    各 nf_mean の N=1 シミュレーション本体
├── theta_nf{1,2,3}.csv          各 nf_mean の θ
├── sin_theta_nf{1,2,3}.csv      各 nf_mean の sin θ
├── prediction_nf{1,2,3}.csv     各 nf_mean の u_true, u_pred
├── nfmean_readout_summary.csv   nf_mean, nrmse_test, mc0_test
├── readout_theta_vs_sin_summary.csv   θ 読み出しと sinθ 読み出しの比較
├── nfmean_readout.png / csv_timeseries_nf{1,2,3}.png
├── readout_theta_vs_sin_nf{1,2,3}.png
└── params_used.json             ※通常の形式ではなく {NF, results} の結果ダンプ
```

## 結果

### 単一粒子での k=0 復元（nf_mean 依存）

`20260731_171338_nfmean/nfmean_readout_summary.csv` より:

| nf_mean | NRMSE_test | MC_0（corr²） |
|---|---|---|
| **1.0**（従来の既定） | **0.1358** | **0.9250** |
| 2.0 | 0.3403 | 0.5287 |
| 3.0 | 0.4954 | 0.0013 |

`20260731_164419/`（nf_mean=1.0 の単独ラン）も NRMSE_test=0.1358, MC_0=0.9250 で一致する。

### 読み出しを θ にした場合との比較

`readout_theta_vs_sin_summary.csv` より:

| nf_mean | NRMSE（θ 読み出し） | corr²（θ） | NRMSE（sinθ 読み出し） | corr²（sinθ） |
|---|---|---|---|---|
| 1.0 | 0.1355 | 0.9253 | 0.1358 | 0.9250 |
| 2.0 | 0.3380 | 0.5350 | 0.3403 | 0.5287 |
| 3.0 | 0.4955 | 0.0007 | 0.4954 | 0.0013 |

## 考察

1. **1 粒子・2 次元状態でも現在入力はほぼ復元できる**（nf_mean=1 で corr²=0.925）。
   つまり MC の k=0 項は系のダイナミクスをほとんど必要としない自明な項であり、
   これを総和に含めていた 2026-07-23 以前の MC が約 1.0 だけ大きかったことの直接の裏付けになる。
2. **自然周波数の平均を上げると復元能力が単調に失われ、nf_mean=3 で完全に消える**
   （corr² 0.925 → 0.529 → 0.001）。位相が入力トルクより速く回ってしまい、
   入力の情報が `sinθ` に載らなくなると解釈できる。
3. **`sinθ` を読み出しに使うことによる損失はない**。θ を直接読み出しても NRMSE の差は
   3 桁目以降（0.1355 vs 0.1358）。周期性を潰す `sinθ` が不利になるのではないかという
   懸念は、少なくともこの条件では当たらない。
4. **派生プロジェクトの結果との関係は未整理**。`Vicsek_input_methods` では自然周波数の項を
   消す（`use_nf=0`）と入力マスク併用で MC が 5.82 → 9.78 に伸びる。本タスクの
   「nf_mean を上げると壊れる」と向きは一致するが、`use_nf=0`（項ごと消す＝粒子の異質性も失う）と
   `nf_mean` を下げる（平均だけ下げ、分散は残す）は別の操作である。本線には `nf_mean=0` の条件が
   まだないので、そこが切り分けの鍵になる → `../../HYPOTHESES.md` D2

## 記録上の注意（2026-09-09 追記）

- `nf_mean` を追加した `vicsek_dynamic.c` の変更は**まだコミットされていない**
  → `../../TODO.md` C-1
- **`nf_mean` の掃引と θ vs sinθ 比較を実行したスクリプトが残っていない**。`run_task13.py` は
  `nf_mean` の掃引にも θ 読み出しにも対応しておらず（`--nf-mean` 相当の引数がない）、
  `sim_nf*/` を作る処理もない。出力の CSV は全部残っているので数値は失われていないが、
  **同じ比較を再実行するにはスクリプトを書き直す必要がある** → `../../TODO.md` C-4
- 直置きの `mck_decay_rcut1_13*.png`（4 本, 2026-08-10 作成）は発表用の MC_k 減衰図で、
  本タスクの出力ではない。これも生成スクリプトが残っていない
- `20260731_171338_nfmean/params_used.json` は他の解析と形式が違い、パラメータではなく
  `{NF: [...], results: [...]}` の結果ダンプになっている。固定パラメータは各
  `sim_nf*/<ts>/params_model.json` を参照すること
