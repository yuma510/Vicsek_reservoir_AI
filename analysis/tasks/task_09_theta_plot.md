# #9 θ_i(t) 時間変化プロット

**状態**: `[x]` 完了（2026-06-24）

## 目標

デフォルトパラメータ（v0=0.0, rcut=13, sgm=0.0）で Vicsek シミュレーションを実行し、
粒子の角度 θ_i(t) の時間変化を可視化する。

## データ・スクリプト

- **スクリプト**: `analysis/theta_plot/plot_theta.py`
- **設定**: `configs/default_params.json`（v0=0.0, rcut=13, sgm=0.0, N=500, ntime=140000）
- **出力**: `analysis/theta_plot/<YYYYMMDD_HHMMSS>/`

## 実行コマンド

```bash
python analysis/theta_plot/plot_theta.py
```

## プロット内容

| ファイル | 内容 | 範囲 |
|---|---|---|
| `theta_heatmap.png` | θ_i(t) ヒートマップ（色=θ mod 2π） | 全粒子・全 14000 フレーム |
| `order_parameter.png` | 秩序変数 r(t) = \|<e^{iθ}>\| | 全 14000 フレーム |
| `theta_selected_particles.png` | 選択 5 粒子の θ_i(t) | test 期間冒頭 100 フレーム（t=8000〜8099） |
| `sintheta_selected_particles.png` | 同 5 粒子の sin(θ_i(t)) | 同上 |

test 開始 = washout + train_num = 2000 + 6000 = 8000 フレーム目。

## 結果

- シムデータ: `data/20260624_123449/`
- 出力: `analysis/theta_plot/20260624_125627/`

選択粒子（seed=0 で np.random.default_rng(0) による）: particle 81, 135, 227, 275, 358

**状態**: `[x]` 完了（2026-06-24）
