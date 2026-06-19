# analysis/ — 解析スクリプト一覧

各解析はサブディレクトリに分かれており、スクリプトと出力データが同じ場所に置かれる。

## スクリプト・データ対応表

| サブディレクトリ | スクリプト | 入力データ | 処理 | 出力データ |
|---|---|---|---|---|
| `rcut_sweep/` | `run_rcut_sweep.py` | `data/rcut_sweep/`（自動生成） | rcut × seed スイープ（シミュレーション→評価→プロット） | `rcut_sweep/<YYYYMMDD_HHMMSS>/` |
| `sgm_sweep/` | `run_sgm_sweep.py` | `data/sgm_sweep/`（自動生成） | sgm × seed スイープ（シミュレーション→評価→プロット） | `sgm_sweep/<YYYYMMDD_HHMMSS>/` |
| `sgm_mean_state/` | `sgm_mean_state.py` | `data/noise_avg_sweep/` | noise 実現を粒子ごとに複素平均した状態で S vs MC/NRMSE を評価 | `sgm_mean_state/<YYYYMMDD_HHMMSS>/` |
| `reservoir_aggregate/` | `reservoir_aggregate.py` | `reservoir_data/<group>/` | sgm ごとに予測を平均し NRMSE・MC を再計算 | `reservoir_aggregate/<sgm_value>/` |
| `correlation_analysis/` | `correlation_analysis.py` | `data/<adj_dir>/adjacency/` | 隣接行列の時間相関減衰を計算・プロット | `correlation_analysis/<YYYYMMDD>/` |

## データフロー

```
data/<sim_dir>/          ──────────────────────→ correlation_analysis.py → correlation_analysis/<date>/
data/noise_avg_sweep/    ──────────────────────→ sgm_mean_state.py       → sgm_mean_state/<date>/
data/rcut_sweep/         ←(自動生成)─ rcut_sweep/run_rcut_sweep.py      → rcut_sweep/<date>/
data/sgm_sweep/          ←(自動生成)─ sgm_sweep/run_sgm_sweep.py        → sgm_sweep/<date>/
reservoir_data/<group>/  ──────────────────────→ reservoir_aggregate.py  → reservoir_aggregate/<sgm>/
```

## 実行例（プロジェクトルートから）

```bash
# rcut スイープ（シミュレーション + 評価 + プロット）
python analysis/rcut_sweep/run_rcut_sweep.py --rcut-values 1 2 3 4 5 6 7 8 9 10 11 12 13 --seeds 10 11 12 13 14 --n-jobs 4

# sgm スイープ
python analysis/sgm_sweep/run_sgm_sweep.py --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 --rcut 13 --seeds 10 11 12 13 14

# noise-avg 状態評価
python analysis/sgm_mean_state/sgm_mean_state.py --data-dir data/noise_avg_sweep

# reservoir_data 集約
python analysis/reservoir_aggregate/reservoir_aggregate.py --reservoir-data reservoir_data/20260603_102700_seed2_change

# ネットワーク相関解析
python analysis/correlation_analysis/correlation_analysis.py \
    --batch-dir data/20260611_120500_adj_rcut_changed \
    --output-dir analysis/correlation_analysis/20260614 \
    --run-reservoir
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

## 実行結果メモ

### sgm_sweep（task_05）— 2026-06-18

- 出力: `analysis/sgm_sweep/20260618_122652/`
- sgm が大きいほど MC 低下・NRMSE 上昇（sgm=0.0: MC≈5.4, sgm=0.5: MC≈1.12）
- sgm=0.0 で seed 間のばらつきが大きい（seed=2,4 で MC が 2〜3 台）

### sgm_mean_state（task_06）— 2026-06-18

- 出力: `analysis/sgm_mean_state/20260618_152006/`
- S（noise 平均数）を増やすほど MC が向上・NRMSE が低下
- sgm=0.0 は S に依らず一定（サニティチェック OK）
- sgm=0.5 で S=1→10 の MC が 1.11→1.90（+71%）と最大改善
