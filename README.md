# Vicsek Reservoir AI

Vicsek モデルのシミュレーションと、NARMA10 に対するリザバー計算の評価を行うプロジェクトです。

## ファイル構成

- `vicsek_dynamic.c`: Vicsek 粒子ダイナミクスを計算し、`data/YYYYMMDD_HHMMSS/`（`position.dat`, `params_model.json`, `README.md`）に保存します。
- `generate_narma10.py`: シミュレーションと評価で使う入力データ・教師データを生成します。
- `vicsek_rc/`: 共有ライブラリ（メトリクス・ローダ・評価・プロット）。解析スクリプトはここから import します（→ `vicsek_rc/README.md`）。
- `analysis/`: 解析スクリプトと解析タスク表（→ `analysis/README.md`, `analysis/analysis_tasks.md`）。
- `runners/`: スイープ実行スクリプト（→ `runners/README.md`）。
- `data/`: シミュレーション出力と実験カタログ（→ `data/README.md`）。
- `for.sh`: データ生成、シミュレーション、評価までの標準ワークフローをまとめて実行します。

ドキュメントはコードの近くに分散配置しています。横断的な文書はルート直下:
`prompt.md`（研究ワークフロー）, `code_reference.md`（コード技術リファレンス）, `CLAUDE.md`（プロジェクトルール）。

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
