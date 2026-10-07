# 解析タスク一覧（prompt.md の Phase 1）

タスク詳細は `../tasks/task_NN_*.md` を参照（**2026-07 に `analysis/tasks/` から
リポジトリルートの `tasks/` へ移動した**）。新しいタスクを追加するときはタスクファイルを
作成し、この表に 1 行追加すること。

**状態の意味**:

| 状態 | 意味 |
|---|---|
| 完了 | 実行し、タスクファイルに結果と考察が書かれている |
| 完了（結果未記載） | 実行して出力は残っているが、タスクファイルに結果・考察が書かれていない |
| 未着手 | `[ ]`。prompt.md Phase 1 はこの行を順に実行する |

---

| # | タイトル | 状態 | 詳細 |
|---|---|---|---|
| 1 | ネットワーク相関減衰 | 完了 | [task_01](../tasks/task_01_correlation_decay.md) |
| 2 | Correlation vs rcut | **完了（結果未記載）** | [task_02](../tasks/task_02_correlation_vs_rcut.md) |
| 3 | NRMSE / MC vs rcut | **完了（結果未記載）** | [task_03](../tasks/task_03_nrmse_mc_vs_rcut.md) |
| 4 | rcut vs MC/NRMSE（散布図） | 完了 | [task_04](../tasks/task_04_rcut_sweep.md) |
| 5 | sgm vs MC/NRMSE（散布図） | 完了 | [task_05](../tasks/task_05_sgm_sweep.md) |
| 6 | noise-averaged 状態評価 | 完了 | [task_06](../tasks/task_06_sgm_mean_state.md) |
| 7 | λ × train_num 掃引（ridge sweep） | 完了 | [task_07](../tasks/task_07_ridge.md) |
| 8 | 高次リードアウト | 完了 | [task_08](../tasks/task_08_higher_order_readout.md) |
| 9 | θ_i(t) 時間変化プロット | 完了（可視化タスクのため考察なし） | [task_09](../tasks/task_09_theta_plot.md) |
| 10 | 予測値平均の性能（S 掃引） | 完了 | [task_10](../tasks/task_10_pred_mean.md) |
| 11 | IPC 測定（デフォルトパラメータ、次数 2 まで） | 完了 | [task_11](../tasks/task_11_ipc.md) |
| 12 | rcut × utime ヒートマップ（v0=0, MC/NRMSE） | 完了 | [task_12](../tasks/task_12_heatmap_rcut_utime.md) |
| 13 | リードアウトの効果検証（N=1 の k=0 復元、nf_mean 依存） | 完了 | [task_13](../tasks/task_13_readout.md) |
| 14 | θ のゆらぎ：ノイズのみ vs 入力（D_θ, D_sin vs σ） | 完了 | [task_14](../tasks/task_14_theta_fluctuation.md) |
| 15 | 予測平均の予測の実現間分布（時系列・ヒストグラム） | 完了 | [task_15](../tasks/task_15_pred_distribution.md) |
| 16 | 入力なし（F=0）での本線のモデルの相転移（論文のセットアップから） | 段階 1 完了 | [task_16](../tasks/task_16_no_input_transition.md) |

未着手（`[ ]`）のタスクは現在ない。次にやることの候補は
[../../TODO.md](../../TODO.md) の B 節（短期の作業キュー）と
[メモ（解析アイデア・候補）](../tasks/memo.md) を参照。

[メモ（解析アイデア・候補）](../tasks/memo.md)
