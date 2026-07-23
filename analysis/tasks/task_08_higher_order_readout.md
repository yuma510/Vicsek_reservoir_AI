# #8 高次のリードアウト

**状態**: `[x]`

**結果**: `analysis/higher_order_readout/20260710_071936/`（rcut=1,4,7,10,13 × sgm=0〜0.4 ×
v0=0.5/0.0、各5 seed × 5 pair_seed、退化スキップ0）。
- **sgm=0** のみ rcut/v0 が強く効く（例 v0=0 rcut=1 で MC=15.5、v0=0.5 rcut=13 で 7.8）。
- **sgm>0 では MC が rcut・v0 にほぼ非依存で小さく collapse**（sgm=0.1→約1.83, 0.2→1.51,
  0.3→1.29, 0.4→1.16）、sgm 増で単調減少。NRMSE も同傾向（0.245→0.282）。
  → ノイズが rcut 依存の短期記憶を打ち消すことを示唆。
- 高次項(P=500)は sgm>0 で base(P=0, 例 ~1.97) をやや下回るセルもあり、明確な改善は見られない。
- 重複セル（rcut=1,13）は前回ラン(20260707_105939)と MC 完全一致で整合性確認済み。
- 不足していた rcut=4,7,10 × sgm=0.1-0.4 × v0=2 は run_sgm_sweep.py で 120 シム新規生成。

---
## 目標
粒子間の相互作用の中に短期記憶能力があるのではないかという仮説について調べたい。そのために現在は(1, sinθ)でリードアウトしているが、(1, sinθ, sinθ_i * sinθ_j)というリードアウトを試したい。また、別のリードアウトを試すことでノイズも減少することを試したい。

## 出力したいプロット
sinθ_i * sinθ_jのiとjの組み合わせをランダムで500個与えて、そのときのMemory CapacityとNARMA10とのNRMSEを調べたい。また、ノイズの強度のsgmを変えたときの結果の変化も確認したい。

# 手法

## 拡張状態の構築

現在のリードアウト状態は `X(t) = [1, sin θ_1(t), …, sin θ_N(t)]`（501 次元）。
これに P=500 個のランダムペア積項 `sin θ_i(t) × sin θ_j(t)` を追加した拡張状態を使う:

```
X_ext(t) = [1, sin θ_1, …, sin θ_N,  sin θ_{i_1} × sin θ_{j_1},  …,  sin θ_{i_500} × sin θ_{j_500}]
```

次元数: N + 1 + 500 = 1001（ベースライン P=0 は従来の 501 次元と同一）。

## ランダムペアの選択

- N=500 粒子から 500 ペア (i, j) を一様ランダムにサンプリング（i ≠ j を許容）
- ランダムシード `pair_seed` で再現可能
- `n_pair_seeds=5` 通りの異なるペア集合を試し、結果の分散を推定

## 評価手順

1. `position.dat` → `build_states()` で通常状態 X（501 次元）を取得
2. 500 ランダムペアを選択し `X_ext`（1001 次元）を構築
3. `ridge_predict()` で NARMA10（seed=666）を予測 → **NRMSE_test**
4. 遅延 k=0..50 の入力 u(t-k) を予測 → **MC_test = Σ MC_k**
5. P=0（ベースライン）と P=500 を sgm ごとに比較

## プロット内容

**前提**: rcut=1, 4, 7, 10, 13 × sgm=0..0.4 の全組み合わせについて高次リードアウト（P=500）を評価する。
そのため **事前に rcut×sgm スイープシミュレーションが必要**（§データ要件 参照）。

**MC と NRMSE を別画像**で出力する。各画像は v0=0.5 / v0=0.0 の 2 パネルを横並び（rcut×sgm）:

| 画像 | 内容 |
|---|---|
| `heatmap_mc.png`    | 左: v0=0.5 の MC_test、右: v0=0.0 の MC_test（各 rcut×sgm） |
| `heatmap_nrmse.png` | 左: v0=0.5 の NRMSE_test、右: v0=0.0 の NRMSE_test（各 rcut×sgm） |

- 横軸: sgm（0.0, 0.1, 0.2, 0.3, 0.4）
- 縦軸: rcut（1, 4, 7, 10, 13）
- 色: MC_test（高いほど良）または NRMSE_test（低いほど良）
- 各セルの値は **5 seed（seed_pos=4,5,6,8,9）× pair_seeds 平均**（全セルで seed 数を統一）

## データ要件

下記のシミュレーションを事前に実行すること:

seed は 5 個（trial 1,2,3,5,6 → seed_pos=4,5,6,8,9）で統一。

| v0 | rcut | sgm | 備考 |
|---|---|---|---|
| 0.5, 0.0 | 4, 7, 10 | 0.1, 0.2, 0.3, 0.4 | **新規シミュレーション必要**（run_sgm_sweep.py、計 120 シム） |
| 0.5, 0.0 | 1, 13 | 0.1, 0.2, 0.3, 0.4 | 既存データ利用可 |
| 0.5, 0.0 | 1, 4, 7, 10, 13 | 0.0 | 既存データ利用可 |

## 出力

```
analysis/higher_order_readout/<YYYYMMDD_HHMMSS>/
    higher_order_data.csv        # long 形式: v0, sgm, rcut, seed_pos, n_pairs, pair_seed, nrmse_test, mc_test
    heatmap_mc.png               # MC_test ヒートマップ（v0=0.5/0.0 の 2 パネル、rcut×sgm）
    heatmap_nrmse.png            # NRMSE_test ヒートマップ（同上）
    params_used.json             # 使用パラメータ（model fixed/swept + reservoir fixed）
```
