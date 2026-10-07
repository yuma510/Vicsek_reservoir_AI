# #3 NRMSE / MC vs rcut

**状態**: `[×]` 完了

---

各 rcut で `position.dat` からリザバー評価を実施（NRMSE, NRMSE2, MC）。

- **データ**: `data/20260611_120500_adj_rcut_changed/`（#1 と同じ）
- **スクリプト**: `analysis/correlation_analysis/correlation_analysis.py`（`--run-reservoir` フラグ付き）
- **出力**: `analysis/correlation_analysis/20260614/nrmse_vs_rcut.png`, `mc_vs_rcut.png`, `reservoir_rcut=*.json`
