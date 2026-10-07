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

| `sgm_sweep/` | `run_sgm_sweep_perseed.py` | `data/`（自動生成） | sgm × seed スイープ。**各 trial seed を自分専用の NARMA 入力で駆動**（共通の seed666 駆動だと sgm=0 で外れ値が出るため） | `sgm_sweep/<YYYYMMDD_HHMMSS>_perseed/` |
| `task13_readout/` | `run_task13.py` | `data/`（自動生成、N=1） | 粒子 1 個で入力そのもの（遅延 k=0）を復元し、リードアウトの効き方と `nf_mean` 依存を確認（task_13） | `task13_readout/<YYYYMMDD_HHMMSS>/` |
| `schematic/` | **スクリプトなし** | — | 発表用の説明図（NARMA 入力・予測、`_bare` 版つき）。2026-08-10 作成。**生成スクリプトが残っていないため再生成できない** | `schematic/`（直置き） |

## データフロー

```
data/<sim_dir>/  ──────────────────────→ correlation_analysis.py → correlation_analysis/<date>/
data/            ──────────────────────→ sgm_mean_state.py       → sgm_mean_state/<date>/
data/            ←(自動生成)─ rcut_sweep/run_rcut_sweep.py       → rcut_sweep/<date>/
data/            ←(自動生成)─ sgm_sweep/run_sgm_sweep.py         → sgm_sweep/<date>/
data/            ──────────────────────→ reservoir_aggregate.py  → reservoir_aggregate/<sgm>/
data/            ──────────────────────→ ipc.py                  → ipc/<date>/
data/            ←(自動生成)─ sgm_sweep/run_sgm_sweep_perseed.py  → sgm_sweep/<date>_perseed/
data/            ←(自動生成)─ task13_readout/run_task13.py        → task13_readout/<date>/
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
パラメータ/結果メタデータ（`params_used.json` / `results_reservoir.json` 等）は JSON のまま。
根拠は CLAUDE.md の「プロットの元データは CSV で保存する」ルール。


各 sweep スクリプト（`rcut_sweep` / `sgm_sweep` / `sgm_mean_state` / `ridge_sweep`）は、出力 dir
（`analysis/<解析名>/<YYYYMMDD_HHMMSS>/`）に **`params_used.json`** を保存する。プロットがどのモデル
パラメータ・レザバー計算パラメータで作られたかを記録するためのもので、掃引軸（rcut/sgm/seed/λ/train_num/S 等）は
固定値と区別して範囲（値リスト）でコンパクトに記録される。共通実装は `vicsek_rc/params_io.py`
（`split_fixed_varied` / `write_params_used`）。詳細は `code_reference.md` の
「vicsek_rc/params_io.py — 解析プロットのパラメータ記録」節を参照。

タスク管理は `analysis_tasks.md` を参照。

結果と考察は `../tasks/task_NN_*.md` に記録する（ラン単位の「実行結果メモ」は 2026-10-06 に廃止し、task にない記述は各 task の末尾に移した）。
