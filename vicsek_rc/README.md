# vicsek_rc — 共有ライブラリ

Vicsek リザバー解析の共通処理を集約したパッケージ。解析スクリプト（`analysis/`）や
各解析スクリプト（`analysis/<解析名>/`）はここから import して使う。

| モジュール | 内容 |
|---|---|
| `metrics.py` | `nrmse`, `nrmse2`, `corrcoef`, `mck_score` |
| `loaders.py` | `load_position_fast`, `build_states`, `load_theta`, `load_adjacency_dir`, `delayed_input`, `find_exp_dir` |
| `evaluate.py` | `ridge_predict`, `evaluate_reservoir`, `compute_theta_mean_states` |
| `correlation.py` | `correlation_at_lag`, `compute_correlation_decay` |
| `plotting.py` | `apply_style`, `plot_*`（相関減衰・vs rcut など） |

使い方:

```python
from vicsek_rc import evaluate_reservoir, nrmse, find_exp_dir
```

- 共通処理を**スクリプト側で再定義しない**。
- 新しい共通処理は各モジュールに追加し、`__init__.py` で re-export する。
- 関数の詳細仕様は `../code_reference.md` を参照。
