"""NARMA10 ファイルの日付ディレクトリ管理。

NARMA10 の入力・正解・params は `narma_data/<YYYYMMDD_HHMMSS>/` 配下に置く
（`data/`・`reservoir_data/` と同じタイムスタンプ dir 規約）。生成側は `new_narma_dir`
で実行ごとの dir を作り、読込側は `find_narma_by_seed` で最新 dir を自動解決する。
後方互換として `narma_data/` 直下のファイルもフォールバックで探索する。
"""
import json
from datetime import datetime
from pathlib import Path


def save_narma_params(out_dir, input_name, target_name, length, seed, low, high) -> Path:
    """1 データセットのメタデータ `narma10_params_<prefix>.json` を out_dir に書き出す。

    `input_name` は `narma10_input_<prefix>.dat` 形式を前提とし、そこから prefix を得る。
    """
    prefix = input_name[len("narma10_input_"):-len(".dat")]
    params = {
        "length": length, "seed": seed, "low": low, "high": high,
        "input_file": input_name, "target_file": target_name,
    }
    path = Path(out_dir) / f"narma10_params_{prefix}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def new_narma_dir(root="narma_data") -> Path:
    """1 データセット用の `root/<YYYYMMDD_HHMMSS>/` を作成して返す。

    同一秒に複数回呼ばれても衝突しないよう、既存なら `_1`, `_2`, … を付与する
    （1 時刻 dir = 1 データセットの規約を保つ）。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(root) / ts
    out, i = base, 1
    while out.exists():
        out = base.with_name(f"{ts}_{i}")
        i += 1
    out.mkdir(parents=True, exist_ok=True)
    return out


def iter_narma_dirs(root) -> list[Path]:
    """探索対象 dir を新しい順に返す（末尾に root 直下フォールバックを含む）。

    - `root` 直下のサブディレクトリ（= タイムスタンプ dir）を名前の降順（新しい順）で。
    - 最後に `root` 自身を加える（日付 dir 化前の直下ファイルとの後方互換）。
    """
    root = Path(root)
    if not root.is_dir():
        return []
    subdirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    return [*subdirs, root]


def _target_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def find_narma_by_seed(root, seed, min_length=None) -> tuple[Path | None, Path | None]:
    """seed 一致の NARMA input/target 対を最新の dir から解決する。

    `iter_narma_dirs` を新しい順に走査し、`narma10_input_*_seed{seed}.dat` と
    対応する target が揃う最初の dir から返す。`min_length` 指定時は target の
    行数がそれ以上のものだけ採用する。複数の low/high 候補があれば `0.0:0.5` を
    優先し、無ければ辞書順先頭。見つからなければ `(None, None)`。
    """
    for d in iter_narma_dirs(root):
        inputs = sorted(d.glob(f"narma10_input_*_seed{seed}.dat"))
        if not inputs:
            continue
        # 0.0:0.5 を優先（それ以外は辞書順先頭）
        inputs.sort(key=lambda p: (f"_0.0:0.5_seed{seed}.dat" not in p.name, p.name))
        for in_path in inputs:
            tgt_path = in_path.with_name(in_path.name.replace("narma10_input_", "narma10_target_", 1))
            if not tgt_path.exists():
                continue
            if min_length is not None and _target_rows(tgt_path) < min_length:
                continue
            return in_path, tgt_path
    return None, None
