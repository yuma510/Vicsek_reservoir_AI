# #1 ネットワーク相関減衰

**状態**: `[×]` 完了

## 目標


---

隣接行列の時間差 Δt に対する正規化相関 `C(Δt)` を計算し、rcut ごとに Decay カーブを描画。

- **データ**: `data/20260611_120500_adj_rcut_changed/`（rcut = 1〜13、各 500 フレームの隣接行列あり）
- **スクリプト**: `analysis/correlation_analysis/correlation_analysis.py`
- **出力**: `analysis/correlation_analysis/20260614/correlation_decay_*.png`, `correlation_*.csv`
