# #12 rcut × utime ヒートマップ（v0=0, MC/NRMSE）

**状態**: `[x]`

**結果**: `analysis/rcut_utime_heatmap/20260710_004410/`（90行フルカバレッジ）。
MC_test は **utime=20 でピーク**（rcut=1, utime=20 で MC=24.44）、utime=10→20 で増加し 20→50 で単調減少。
rcut は小さいほど MC が高い（全 utime 共通）。utime=20 周辺の優位性を確認。
※ 生成時、別ジョブと同時実行して index.csv 追記が競合し 10 セル取りこぼしが発生 →
`vicsek_rc.catalog` で index.csv 全再構築後に `--skip-sim` で再評価して解消（シムは全て健在）。

---
## 目標
v0=0のとき、rcutが小さいほど性能が良くなった。このような結果になったことの仮説として、rcutが小さくなったことで入力に対しての応答が全体に広がる期間が性能に影響を与えたと考えている。そこで、v0=0のケースでutimeとrcutを変化させて、MCとNRMSEのヒートマップを作成したい。

初回スイープ（utime=[1, 2, 5, 10, 20, 50]）の結果、**utime=20 付近で性能が良い**ことが分かった。
そこで utime=20 の周辺を細かく取り直す（utime=[10, 20, 30, 40, 50]、rcut・seed は据え置き）。

## 手法

### 使用するパラメータ

| パラメータ | 値 |
|---|---|
| v0 | 0.0（固定） |
| sgm | 0.0（固定） |
| N | 500（固定） |
| boxsize | 15.8（固定） |
| h1 | 0.01（固定） |
| K | 1.0（固定） |
| ntime | 14000 × utime（出力フレーム数を14000に固定） |
| washout | 2000（デフォルト） |
| train_num | 6000（デフォルト） |
| ridge_lambda | 1e-9（デフォルト） |
| k_max | 100（デフォルト） |

### rcutとutimeの変化範囲

- **rcut**: [1, 3, 5, 7, 10, 13]（6値）
- **utime**: [10, 20, 30, 40, 50]（5値、utime=20 周辺にズーム）
- **trial_seeds**: [1, 2, 3] → seed_pos=[4, 5, 6]（3 seeds）
- utime=10, 20, 50 は既存シム（v0=0, sgm=0）を流用可（各 rcut×seed で充足済み）
- 新規シミュレーションは **utime=30, 40 のみ**: 6 rcut × 2 utime × 3 seeds = **36 シム**（≈10 GB）
- 総評価数（Phase2）: 6 rcut × 5 utime × 3 seeds = **90**

## 実装

スクリプト: `analysis/rcut_utime_heatmap/rcut_utime_heatmap.py`  
（`run_rcut_sweep.py` を参考にした4フェーズ構成）

- **Phase 0**: `ensure_narma10` — `narma_data/` の既存 NARMA10 を再利用（なければ生成）
- **Phase 1**: `phase1_simulate` — (rcut, utime, seed) を並列シム
  - cfg dict に `"utime": utime`, `"ntime": 14000 * utime` を追加（`run_rcut_sweep.py` にはない）
  - 既存データ確認: `data/index.csv` を直接 pandas でフィルタ（rcut, utime, seed_pos, v0, sgm 列）  
    ※ `find_exp_dir()` は utime フィルタ未対応のため
- **Phase 2**: `phase2_evaluate` — `evaluate_reservoir()` で MC/NRMSE 計算・キャッシュ
- **Phase 3**: `phase3_plot` — rcut × utime の imshow ヒートマップ（MC/NRMSE 各1枚）

### 出力ファイル

```
analysis/rcut_utime_heatmap/<YYYYMMDD_HHMMSS>/
    heatmap_data.csv    # long形式: rcut, utime, seed, MC_test, NRMSE_test
    heatmap_mc.png      # MC_test の rcut × utime ヒートマップ（seeds 平均、x=rcut, y=utime）
    heatmap_nrmse.png   # NRMSE_test の rcut × utime ヒートマップ（同上）
    params_used.json    # 使用パラメータ（model fixed/swept + reservoir fixed）
```