"""解析プロットの使用パラメータ記録（params_used.json）とデフォルト設定ロード。

掃引（rcut/sgm/seed/λ/train_num 等）した解析では、全 sim の params_model.json を
丸ごと保存すると冗長なので、固定パラメータと掃引パラメータを自動判別して
コンパクトに記録する。CLAUDE.md §3「メタデータの管理」を参照。
"""
import json
from pathlib import Path

_RC_CONFIG = Path(__file__).parent.parent / "configs" / "default_reservoir_params.json"


def load_reservoir_defaults() -> dict:
    """configs/default_reservoir_params.json からレザバーデフォルト値を読み込む。"""
    with open(_RC_CONFIG) as f:
        return json.load(f)


def _hashable(value):
    """list/dict を比較・重複除去できるよう hashable 化する。"""
    if isinstance(value, list):
        return ("__list__", tuple(_hashable(v) for v in value))
    if isinstance(value, dict):
        return ("__dict__", tuple((k, _hashable(v)) for k, v in sorted(value.items())))
    return value


def _try_sorted(values: list):
    """可能なら昇順ソート、混在型などで失敗したら出現順のまま返す。"""
    try:
        return sorted(values)
    except TypeError:
        return values


def split_fixed_varied(param_dicts: list) -> tuple[dict, dict]:
    """複数の param dict を (fixed, varied) に分離する。

    - 全 dict で値が一致するキー → fixed にスカラ値（その共通値）。
    - 値が異なるキー             → varied に distinct な値のリスト（可能なら昇順）。
    - あるキーが一部の dict に欠ける場合も varied 扱い。
    - list 値（例 'seed'）は要素ごとに比較し、異なれば distinct なリストを列挙。
    """
    if not param_dicts:
        return {}, {}

    all_keys = set()
    for d in param_dicts:
        all_keys.update(d.keys())

    fixed, varied = {}, {}
    sentinel = object()
    for key in sorted(all_keys):
        raw_values = [d.get(key, sentinel) for d in param_dicts]
        present_all = sentinel not in raw_values

        seen, uniques = set(), []
        for v in raw_values:
            if v is sentinel:
                continue
            h = _hashable(v)
            if h not in seen:
                seen.add(h)
                uniques.append(v)

        if present_all and len(uniques) == 1:
            fixed[key] = uniques[0]
        else:
            varied[key] = _try_sorted(uniques)

    return fixed, varied


def write_params_used(output_dir, model_param_dicts: list,
                      reservoir_fixed: dict,
                      reservoir_swept: dict | None = None) -> Path:
    """params_used.json を output_dir に書き出してパスを返す。

    model パラメータは split_fixed_varied で自動分離。reservoir パラメータは
    呼び出し側が fixed/swept を明示する（washout/train_num/λ 等は params_model.json に
    無いため）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_fixed, model_swept = split_fixed_varied(model_param_dicts)
    payload = {
        "model": {"fixed": model_fixed, "swept": model_swept},
        "reservoir": {"fixed": dict(reservoir_fixed),
                      "swept": dict(reservoir_swept or {})},
    }

    out_path = output_dir / "params_used.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path
