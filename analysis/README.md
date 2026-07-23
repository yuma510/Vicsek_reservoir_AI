# analysis/ — 解析スクリプト一覧

各解析はサブディレクトリに分かれており、スクリプトと出力データが同じ場所に置かれる。

## スクリプト・データ対応表

| サブディレクトリ | スクリプト | 入力データ | 処理 | 出力データ |
|---|---|---|---|---|
| `rcut_sweep/` | `run_rcut_sweep.py` | `data/rcut_sweep/`（自動生成） | rcut × seed スイープ（シミュレーション→評価→プロット） | `rcut_sweep/<YYYYMMDD_HHMMSS>/` |
| `sgm_sweep/` | `run_sgm_sweep.py` | `data/sgm_sweep/`（自動生成） | sgm × seed スイープ（シミュレーション→評価→プロット） | `sgm_sweep/<YYYYMMDD_HHMMSS>/` |
| `sgm_mean_state/` | `sgm_mean_state.py` | `data/noise_avg_sweep/` | noise 実現を粒子ごとに複素平均した状態で S vs MC/NRMSE を評価 | `sgm_mean_state/<YYYYMMDD_HHMMSS>/` |
| `reservoir_aggregate/` | `reservoir_aggregate.py` | `data/` | `data/` から直接 ridge 予測し、sgm ごとに seeds 間平均して NRMSE・MC を再計算 | `reservoir_aggregate/<sgm_value>/` |
| `correlation_analysis/` | `correlation_analysis.py` | `data/rcut_sweep/`（position.dat から adjacency を計算） | 隣接行列の時間相関減衰を計算・プロット | `correlation_analysis/<YYYYMMDD_HHMMSS>/` |
| `ridge_sweep/` | `run_ridge_sweep.py` | `data/<ts>_ridge_sweep/`（自動生成） | λ × train_num 2D 掃引で最適リッジ回帰設定を探索 | `ridge_sweep/<YYYYMMDD_HHMMSS>/` |
| `pred_mean/` | `pred_mean.py` | `data/`（noise-avg 集合を seed_pos=13/seed_nf=16/v0=0.5 で選択） | 各 noise 実現の予測を S 個平均し S vs MC/NRMSE を評価（task_06 の予測平均版、task_10） | `pred_mean/<YYYYMMDD_HHMMSS>/` |
| `pred_mean/` | `generate_noise_realizations.py` | `data/index.csv`（既存確認） | noise-avg 集合の不足 seed_noise 実現を追加シミュレーション（S=100 拡張用、sgm=0 は全実現同一のため対象外）。終了後に index.csv を全再構築 | `data/<YYYYMMDD_HHMMSS>/`（シム本体） |
| `ipc/` | `ipc.py` | `data/`（rcut=13, sgm=0, ntime=220000 の複数 seed） | 正規化 Legendre 多項式積を基底にした IPC を次数 2 まで計算（task_11） | `ipc/<YYYYMMDD_HHMMSS>/` |
| `higher_order_readout/` | `higher_order_readout.py` | `data/`（v0=0.5/0.0, rcut=1/4/7/10/13, sgm=0〜0.4 × 5 seed=seed_pos{4,5,6,8,9}） | P=500 ランダムペア積項を追加した拡張リードアウトで MC/NRMSE を評価。**MC・NRMSE を別画像**で rcut×sgm ヒートマップ出力（`heatmap_mc.png`/`heatmap_nrmse.png`、各 v0=0.5/0.0 の2パネル、task_08）。結果: sgm=0 のみ rcut/v0 依存、sgm>0 は MC が小さく collapse | `higher_order_readout/<YYYYMMDD_HHMMSS>/` |
| `rcut_utime_heatmap/` | `rcut_utime_heatmap.py` | `data/`（v0=0, sgm=0, rcut=1〜13 × utime=10/20/30/40/50 × 3 seed） | rcut × utime の MC/NRMSE ヒートマップ生成（task_12）。結果: MC は **utime=20 でピーク**、rcut 小ほど良（20260710_004410） | `rcut_utime_heatmap/<YYYYMMDD_HHMMSS>/` |

## データフロー

```
data/<sim_dir>/  ──────────────────────→ correlation_analysis.py → correlation_analysis/<date>/
data/            ──────────────────────→ sgm_mean_state.py       → sgm_mean_state/<date>/
data/            ←(自動生成)─ rcut_sweep/run_rcut_sweep.py       → rcut_sweep/<date>/
data/            ←(自動生成)─ sgm_sweep/run_sgm_sweep.py         → sgm_sweep/<date>/
data/            ──────────────────────→ reservoir_aggregate.py  → reservoir_aggregate/<sgm>/
data/            ──────────────────────→ ipc.py                  → ipc/<date>/
```

## 実行例（プロジェクトルートから）

```bash
# rcut スイープ（シミュレーション + 評価 + プロット）
python analysis/rcut_sweep/run_rcut_sweep.py --rcut-values 1 2 3 4 5 6 7 8 9 10 11 12 13 --seeds 10 11 12 13 14 --n-jobs 4

# sgm スイープ
python analysis/sgm_sweep/run_sgm_sweep.py --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 --rcut 13 --seeds 10 11 12 13 14

# noise-avg 状態評価
python analysis/sgm_mean_state/sgm_mean_state.py --data-dir data/noise_avg_sweep

# sgm ごとに seeds 間平均して NRMSE・MC を再計算
python analysis/reservoir_aggregate/reservoir_aggregate.py \
    --data-dir data \
    --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --rcut 13.0

# ネットワーク相関解析（position.dat から adjacency をオンザフライ計算）
python analysis/correlation_analysis/correlation_analysis.py \
    --batch-dir data/rcut_sweep \
    --output-dir analysis/correlation_analysis \
    --frame-start 2000 --frame-end 2500 --dt-max 50
```

## 共通出力: プロット元データ CSV ＋ `params_used.json`

各スクリプトはプロットの元データ（数値系列・散布点）を **CSV（long 形式）** で保存する
（`rcut_sweep_data.csv` / `sgm_sweep_data.csv` / `noise_avg_data.csv` / `MCk.csv` /
`narma10_prediction.csv` / `summary.csv` / `analysis_summary.csv` 等）。サマリーを JSON では保存しない。
パラメータ/結果メタデータ（`params_used.json` / `results_reservoir.json` 等）は JSON のまま。詳細は CLAUDE.md §3。


各 sweep スクリプト（`rcut_sweep` / `sgm_sweep` / `sgm_mean_state` / `ridge_sweep`）は、出力 dir
（`analysis/<解析名>/<YYYYMMDD_HHMMSS>/`）に **`params_used.json`** を保存する。プロットがどのモデル
パラメータ・レザバー計算パラメータで作られたかを記録するためのもので、掃引軸（rcut/sgm/seed/λ/train_num/S 等）は
固定値と区別して範囲（値リスト）でコンパクトに記録される。共通実装は `vicsek_rc/params_io.py`
（`split_fixed_varied` / `write_params_used`）。詳細は CLAUDE.md §3 と `code_reference.md` を参照。

タスク管理は `analysis_tasks.md` を参照。

### ipc（task_11）— 2026-07-01

- 出力: `analysis/ipc/20260701_194332/`
- rcut=13, sgm=0, ntime=220000, 5 seed（seed_pos=4,5,6,8,9）、次数 2 まで
- C_d1=9.53（線形記憶）、C_d2a=1.50（2乗記憶）、C_d2b=2.88（交差遅延積記憶）
- **C_total=13.91**（次数 2 まで）≪ N=500（理論上限）
- seed_pos=8 は顕著な外れ値（C_d1=4.39）、他は 10〜11

## 実行結果メモ

### rcut_sweep（task_04 v2）— 2026-06-19

- 出力: `analysis/rcut_sweep/20260619_204013/`、データ: `data/rcut_sweep_v2/`
- ntime=140000（14000フレーム）、等長 train/test=6000 サンプル、λ=1e-9
- rcut=13: MC_test=8.14±0.30, NRMSE_test=0.164±0.008（9 seed 平均）
- MC_test 最大の rcut: 12
- **エラーバー版**（2026-07-20 追記）: `mc_vs_rcut_errorbar.png` 等（seed 間 平均±std）。
  MC は rcut=1〜4 で平坦（≈4.7）→ rcut=6〜10 でシグモイド状に急増 → rcut≥11 で飽和（≈8.1）

### rcut_sweep（task_04 v3, v0=0）— 2026-06-23

- 出力: `analysis/rcut_sweep/20260623_121526/`、データ: `data/rcut_sweep_v0/`
- v0=0（静的ネットワーク）、ntime=140000、等長 train/test=6000 サンプル、λ=1e-9
- MC_test 最大: rcut=2（10.47±1.06）、rcut が大きいほど単調減少（rcut=13: 8.19±0.35）
- v0=0.5 と rcut=13 での性能はほぼ同一だが、v0=0 は低 rcut で高 MC
- **エラーバー版**（2026-07-20 追記）: `mc_vs_rcut_errorbar.png` 等（v0=0.5 とは逆の単調減少傾向）

### sgm_sweep（task_05 v2）— 2026-06-20

- 出力: `analysis/sgm_sweep/20260620_070251/`、データ: `data/sgm_sweep_v2/`
- ntime=140000（14000フレーム）、等長 train/test=6000 サンプル、λ=1e-9
- sgm が大きいほど MC 低下・NRMSE 上昇（sgm=0: MC=6.9±2.5, sgm=0.5: MC=1.14±0.02）
- sgm=0 で seed 間ばらつき大（初期条件依存が顕在化）
- **エラーバー版**（2026-07-20 追記）: `mc_vs_sgm_errorbar.png` 等（10 seed 平均±std）。
  sgm=0 の大きなエラーバー（±2.5）と sgm≥0.1 での急落・ばらつき縮小が明瞭

### pred_mean / sgm_mean_state S=100 拡張（task_10 / task_06）— 2026-07-18

- データ: noise-avg 集合を各 sgm（0.1〜0.5）**100 実現**に拡張（seed_noise=1〜102 のうち 100 個、
  `pred_mean/generate_noise_realizations.py` で 350 シム追加生成、約 98 GB）
- 予測平均（`pred_mean/20260718_104927/`）: **sgm=0.1 は S≈65 で sgm=0 の MC=8.43 を超え、S=100 で MC=10.52**。
  NRMSE は 0.215 でプラトー（sgm=0 の 0.158 には届かない）
- 状態平均（`sgm_mean_state/20260718_111815/`）: MC は S=100 でも 4.32 と鈍い。NRMSE は 0.202 と予測平均より良い
- MC vs S の段差は打ち切り閾値の離散性（`mck_decay_S60_S70.png` 参照）
- ランダム順列 R=20 の平均±std（`pred_mean/20260720_130529/mc_nrmse_vs_S_errorbar.png`）:
  平均曲線は滑らか、段差は順序依存のゆらぎ（std 最大 ±0.5）。sgm=0.1 は S≈65〜70 で確実に sgm=0 超え
- 両スクリプトの実験選択に厳密フィルタ（seed_pos/seed_nf/rcut/v0/input_file）を追加
  — フラット data/ での別実験混入（実例: NARMA seed=10 入力シムが sgm=0 に混入）を防止

### sgm_mean_state（task_06 v2）— 2026-06-20

- 出力: `analysis/sgm_mean_state/20260620_135259/`、データ: `data/noise_avg_sweep_v2/`
- ntime=140000、noise seeds=30個（1〜12,14〜15,17〜32）、等長 train/test=6000 サンプル、λ=1e-9
- sgm=0 は S に依らず一定（MC=8.41）
- sgm=0.1〜0.5 で S=30 時に S=1 比 +1.3〜+1.8 の MC 改善
- S=30 でも sgm=0 の MC=8.41 には届かない

### correlation_analysis（task_01）— 2026-06-30

- 出力: `analysis/correlation_analysis/20260630_202345/`、データ: `data/rcut_sweep/`（117 本）
- position.dat の frame 2000–2500（washout 後の定常状態 500 フレーム）から adjacency を計算
- dt_max=50、rcut=1~13 × 9 seeds（117 実験）
- rcut=1: C(Δt=1)≈0.98, C(Δt=10)≈0.82（ネットワーク変化が速い）
- rcut=13: C(Δt=1)=1.0, C(Δt=10)=1.0（完全連結・固定ネットワーク）
- rcut が大きいほどネットワークが安定し C(Δt) の減衰が遅くなる傾向を確認
- 詳細: `analysis/tasks/task_01_correlation_decay.md`

### correlation_analysis（dt_max=2000, 訓練区間全体）— 2026-07-15

- 出力: `analysis/correlation_analysis/20260715_192338/`、データ: `data/`（v0=0.5, sgm=0, ntime=140000 の 118 実験）
- フレーム 2000–8000（訓練区間 6001 フレーム、n_base=4001）、dt_max=2000
- `compute_correlation_decay` を FFT 相互相関に高速化（推定量は旧実装と同値、~86 s/実験）
- 各 rcut とも Δt≈500 までにプラトーへ到達し、以後ほぼ一定（周期的な小振動あり）
- プラトー値（Δt=2000, seed 平均）: rcut=1: 0.02 / rcut=4: 0.21 / rcut=7: 0.62 / rcut=9: 0.92 / rcut≥11: 1.00
- rcut≤3 は長時間でほぼ完全にネットワークが再編される一方、rcut≥8 は初期構造の大部分が残存

### ridge_sweep（task_07）— 2026-06-19

- 第1回出力: `analysis/ridge_sweep/20260619_165214/`（λ 8点）
- 第2回出力: `analysis/ridge_sweep/20260619_193558/`（λ 10点、1e-9・1e-12 追加）
- rcut=13、sgm=0、ntime=220000（22000フレーム）、5 seed（1,2,3,5,6）
- **推奨設定: λ=1e-9, train_num=6000**（NRMSE_test=0.163, MC_test=8.20, gap=0.006）
- λ=1e-12 は実質無正則化域で競争力なし（gap=0.019）
