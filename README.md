# Vicsek Reservoir AI

Vicsek モデルのシミュレーションと、NARMA10 に対するリザバー計算の評価を行うプロジェクトです。

## ファイル構成

- `vicsek_dynamic.c`: Vicsek 粒子ダイナミクスを計算し、`data/YYYYMMDD_HHMMSS/`（`position.dat`, `params_model.json`）に保存します。
- `generate_narma10.py`: シミュレーションと評価で使う入力データ・教師データを生成します。
- `vicsek_rc/`: 共有ライブラリ（メトリクス・ローダ・評価・プロット）。解析スクリプトはここから import します（→ `vicsek_rc/README.md`）。
- `analysis/`: 解析スクリプトとデータの対応表（→ `analysis/README.md`）、解析タスク表（→ `analysis/analysis_tasks.md`）。
  スイープ実行スクリプトも各解析サブディレクトリ内（`analysis/<解析名>/run_*.py`）に置きます。
- `tasks/`: タスクごとの詳細記録（目標・手法・実行コマンド・結果・考察）。`task_01`〜（番号の一覧は `analysis/analysis_tasks.md`）。結果と考察の一次記録はここ。
- `data/`: シミュレーション出力と実験カタログ（→ `data/README.md`）。
- `for.sh`: データ生成、シミュレーション、評価までの標準ワークフローをまとめて実行します。

ドキュメントはコードの近くに分散配置しています。横断的な文書はルート直下:
`prompt.md`（研究ワークフロー）, `code_reference.md`（コード技術リファレンス）, `CLAUDE.md`（プロジェクトルール）。

**workspace 全体の俯瞰**（このプロジェクトと派生プロジェクトの関係、研究の経過、仮説の検証状況、作業キュー）は
[../README.md](../README.md) ／ [../PROGRESS.md](../PROGRESS.md) ／ [../HYPOTHESES.md](../HYPOTHESES.md) ／
[../TODO.md](../TODO.md) を参照してください。

**数式による定義**（更新式・NARMA10・MC・IPC・相関減衰の定義、研究の問いの数式表現、
式と実装の食い違い）は [../DEFINITIONS.md](../DEFINITIONS.md) にあります。`code_reference.md` が
「コードがどう書かれているか」を、`../DEFINITIONS.md` が「式でどう定義されているか」を担当します。

## クイックスタート

```bash
cd workspace/Vicsek_reservoir_AI
bash for.sh
```

フルシミュレーションは元のパラメータ（`N=500`、入力ステップ数 `12000`）を使い、`rcut=1,2,3` の3条件を順に実行するため、実行に時間がかかる場合があります。

## 手動で実行する場合

NARMA10 データを生成します（`narma_data/<YYYYMMDD_HHMMSS>/` 配下に出力されます）。

```bash
python3 generate_narma10.py
```

Vicsek シミュレーションをコンパイルして実行します。入力パスは生成された日付 dir
配下のファイルを指定します（例）。

```bash
cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm
./vicsek_dynamic narma_data/<YYYYMMDD_HHMMSS>/narma10_input_0.0:0.5_seed666.dat data
```

解析スクリプトを直接実行してリザバー評価・集計を行います。

```bash
python analysis/reservoir_aggregate/reservoir_aggregate.py \
    --data-dir data \
    --sgm-values 0.0 0.1 0.2 0.3 0.4 0.5 \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --rcut 13.0 \
    --out analysis/reservoir_aggregate
```

評価結果は `analysis/reservoir_aggregate/<sgm>/` に保存されます。
