"""data/ ディレクトリのカタログ（index.csv）生成・読み込みユーティリティ。

`data/` はフラットに多数のタイムスタンプ dir を並べる方針（CLAUDE.md §4.1）。
目視で見通せないため、各 dir の `params_model.json` を 1 表にまとめた `index.csv` を作る。
`find_exp_dir`（loaders.py）はこの index があればそれを引いて高速化する。

CLI:
    python -m vicsek_rc.catalog [--data-dir data] [--out data/index.csv]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# index.csv の列順（既知の params_model.json スキーマに対応）。
# スカラモデルパラメータ + seed（新旧フォーマット両対応）+ dir メタデータ。
SCALAR_COLS = ["model", "N", "boxsize", "ntime", "utime", "h1",
               "v0", "sgm", "K", "F", "c", "rcut", "rho"]
SEED_COLS   = ["seed_pos", "seed_noise", "seed_nf"]  # 新フォーマット
META_COLS   = ["dir", "mtime", "size_MB", "has_position", "has_params"]

COLUMNS = META_COLS + SCALAR_COLS + SEED_COLS + ["seed", "write_adjacency", "input_file"]

INDEX_NAME = "index.csv"


def _row_from_dir(d: Path) -> dict:
    """1 つの実験 dir から index の 1 行分の dict を作る。"""
    params_file = d / "params_model.json"
    pos_file    = d / "position.dat"
    has_params  = params_file.exists()
    has_pos     = pos_file.exists()

    row = {c: None for c in COLUMNS}
    row["dir"]          = d.name
    row["mtime"]        = pd.to_datetime(d.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    row["size_MB"]      = round(pos_file.stat().st_size / 1e6, 1) if has_pos else 0.0
    row["has_position"] = has_pos
    row["has_params"]   = has_params

    if not has_params:
        return row

    with open(params_file) as f:
        p = json.load(f)

    for c in SCALAR_COLS + SEED_COLS + ["write_adjacency", "input_file"]:
        if c in p:
            row[c] = p[c]
    # 旧フォーマットの seed 配列は連結文字列で 1 列に保持（seed_index 参照を index からも再現できる）
    if isinstance(p.get("seed"), list):
        row["seed"] = ",".join(str(int(s)) for s in p["seed"])
    return row


def build_catalog(data_dir="data") -> pd.DataFrame:
    """`data_dir` 直下の各実験 dir をスキャンし、1 dir = 1 行の DataFrame を返す。

    dir 名（タイムスタンプ）昇順にソートする（`find_exp_dir` の走査順と一致させるため）。
    """
    data_dir = Path(data_dir)
    rows = [_row_from_dir(d) for d in sorted(data_dir.iterdir()) if d.is_dir()]
    return pd.DataFrame(rows, columns=COLUMNS)


def write_catalog(data_dir="data", out=None) -> Path:
    """カタログを構築し `<data_dir>/index.csv` に書き出す。書き出し先 Path を返す。"""
    data_dir = Path(data_dir)
    out = Path(out) if out is not None else data_dir / INDEX_NAME
    df = build_catalog(data_dir)
    df.to_csv(out, index=False)
    return out


def load_catalog(data_dir="data") -> pd.DataFrame | None:
    """`<data_dir>/index.csv` があれば読んで返す。無ければ None。"""
    data_dir = Path(data_dir)
    idx = data_dir / INDEX_NAME
    if not idx.exists():
        return None
    return pd.read_csv(idx, dtype={"seed": str})


def append_row(data_dir, exp_dir) -> None:
    """新規に生成した 1 実験 dir を index.csv に追記する（全スキャン不要）。

    index.csv が無い場合は何もしない（初回は build/write_catalog を使う）。
    """
    data_dir = Path(data_dir)
    idx = data_dir / INDEX_NAME
    if not idx.exists():
        return
    row = _row_from_dir(Path(exp_dir))
    pd.DataFrame([row], columns=COLUMNS).to_csv(idx, mode="a", header=False, index=False)


def main():
    ap = argparse.ArgumentParser(description="data/ のカタログ index.csv を生成する")
    ap.add_argument("--data-dir", default="data", help="スキャン対象（default: data）")
    ap.add_argument("--out", default=None, help="出力先（default: <data-dir>/index.csv）")
    args = ap.parse_args()
    out = write_catalog(args.data_dir, args.out)
    df = pd.read_csv(out)
    print(f"[catalog] {len(df)} 行を {out} に書き出しました")


if __name__ == "__main__":
    main()
