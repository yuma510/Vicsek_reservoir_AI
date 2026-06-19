# Vicsek Reservoir — プロジェクトルール

このファイルは Claude Code が自動で読み込むルール集です。作業開始前に必ず参照してください。

---
## 0. 編集範囲の制限（最重要）

**このファイル（CLAUDE.md）があるディレクトリより上位の
ファイル・ディレクトリを編集・作成・削除してはならない。**

- 編集してよい範囲: `/mnt/d/DATA/yuma/workspace/Vicsek_reservoir_AI/` 配下のみ
- 禁止例: `/mnt/d/DATA/yuma/`、`/mnt/d/DATA/`、その他プロジェクト外の任意のパス
- 上位ディレクトリの内容を読む（参照する）ことは許可されるが、変更は一切行わない

---

## 1. ドキュメント参照ルール

作業開始前に必ず以下を読むこと:

| ファイル | 内容 |
|---|---|
| `prompt.md` | 研究目標・ワークフロー全体像（ルート） |
| `code_reference.md` | コード仕様・データフロー・CLI 引数の詳細（ルート） |
| `analysis/analysis_tasks.md` | 解析タスク一覧（状態とリンクのみ）。詳細は `analysis/tasks/task_0N_*.md` を参照 |

各ディレクトリの `README.md`（`vicsek_rc/`, `analysis/`, `data/`）も併せて参照。

**コード変更後は `code_reference.md` の該当箇所を必ず更新すること。**
仕様変更を検出した場合は実験を継続する前にその内容を要約すること。

---

## 2. データディレクトリ規則

---

## 2a. data/ ディレクトリ規則

### 個別シミュレーションディレクトリ

- ディレクトリ名は **タイムスタンプのみ**: `data/<YYYYMMDD_HHMMSS>/`
- 各ディレクトリに以下の 2 ファイルが生成される:
  - `position.dat` — シミュレーション軌道（N×(ntime/utime) 行、utime ステップごとに 1 フレーム保存）
  - `params_model.json` — パラメータ（機械可読）
- **ディレクトリの検索はディレクトリ名のパターンでなく `params_model.json` の内容で行う**
  （glob パターンは使わない）

### グループディレクトリ（実験シリーズ）

複数のシミュレーションをまとめる場合は、親ディレクトリを作りその中に個別 sim dir を並べる:

```
data/<YYYYMMDD_HHMMSS>_<シリーズ名>/   ← グループ dir（手動命名）
    <YYYYMMDD_HHMMSS>/                 ← 個別 sim dir（自動生成）
    <YYYYMMDD_HHMMSS>/
    ...
```

- グループ dir 名: `<タイムスタンプ>_<シリーズを端的に表す語>`（例: `20260524_090700_seed2_change`）
- グループ実験の目的・変化パラメータ・固定パラメータは **`data/README.md`** にまとめる

### README 更新ルール

**シミュレーションを実行したら必ず `data/README.md` を更新すること。**

- 個別実験（単発 rcut / sgm 変更など）→「個別実験一覧」テーブルに 1 行追加
- グループ実験（for ループなど）→「グループ実験一覧」テーブルに追加し、詳細セクションも作成

---

## 2b. reservoir_data/ ディレクトリ規則

`vicsek_prediction.py` が生成するリザバー評価結果の格納場所。

- ディレクトリ名: `reservoir_data/<YYYYMMDD_HHMMSS>/`（実行ごとに自動生成）
- 各ディレクトリに以下のファイルが生成される:
  - `results_reservoir.json` — NRMSE・MC・パラメータの集約（`params_model`・`params_reservoir`・`results` キー）
  - `narma10_prediction.png` — 正解と予測の重ね描き
  - `MCk.png` — 遅延 k ごとの MC_k カーブ
  - `NARMA10_prediction.dat` — 予測値の数値データ
  - `MCk_prediction/k=<k>.dat` — 各遅延 k の予測値
- `results_reservoir.json` の `data_path` フィールドに対応する `data/<dir>/` のパスが記録される（対応関係はこのフィールドで追跡）
- グループ実験のまとめ評価は `reservoir_data/<YYYYMMDD_HHMMSS>_<シリーズ名>/` とする
- グループ実験の目的・変化パラメータ・結果サマリーは **`reservoir_data/README.md`** にまとめる

### README 更新ルール

**リザバー評価を実行したら必ず `reservoir_data/README.md` を更新すること。**

- 個別評価 → 「個別評価一覧」テーブルに rcut・sgm・MC・NRMSE・対応 data dir を 1 行追加
- グループ評価 → 「グループ実験一覧」テーブルに追加し、詳細セクションも作成

---

## 2c. analysis/<解析名>/ ディレクトリ規則

各解析はサブディレクトリに分かれており、**スクリプトと出力データが同じサブ dir に置かれる**。

```
analysis/<解析名>/
    <script>.py          ← 解析スクリプト
    <YYYYMMDD_HHMMSS>/  ← 実行ごとの出力（自動生成）
        *_data.csv      ← プロット元データ（数値系列・散布点、long 形式）
        *.png
        params_used.json ← 使用パラメータ（メタデータ、JSON）
        results_*.json  ← キャッシュ（再実行時スキップ、JSON）
```

- 解析名の例: `rcut_sweep`, `sgm_sweep`, `sgm_mean_state`, `reservoir_aggregate`, `correlation_analysis`
- `--output-dir` 引数には `analysis/<解析名>` を指定する（例: `analysis/rcut_sweep`）
  スクリプトがタイムスタンプサブ dir を自動生成する
- 詳細は `analysis/README.md` を参照

### README 更新ルール

**新しい解析を実施したら必ず `analysis/README.md` を更新すること。**

- 新しい解析サブ dir を作成したとき → `analysis/README.md` のサブディレクトリ一覧テーブルに追加
- 既存解析を実行して新しい知見が得られたとき → 該当解析の概要欄に結果メモを追記

---

## 3. メタデータの管理
- メタデータから同じデータを再現できるように出力するメタデータを設定する
- **パラメータ・メタデータ**は出力データと一緒に **JSON** ファイルで出力する
  （`params_model.json` / `params_used.json` / `results_reservoir.json` 等）
- 解析のため、vicsek_dynamic.cやvicsek_prediction.pyで新しいデータを出力するときはメタデータを保持するjsonに新たに出力したデータに関するkeyを追加する

### プロット元データの保存（CSV）

- **プロットの元データ（数値系列・散布点）は CSV（long 形式）で保存する**。サマリーを JSON で保存しない
  （`*_summary.json` は廃止し `*_data.csv` 等に置き換え済み）。
- 散布図は seed ごとの個別点をそのまま 1 行にする（例: `rcut, seed, MC_test, MC_train, nrmse_test, …`）。
  集約値（mean/std）は CSV から再計算できるため保存不要。
- 時系列・カーブ系プロットも対応する CSV を出す（例: `narma10_prediction.csv`, `MCk.csv`,
  `noise_avg_data.csv`, `correlation_<tag>.csv`）。
- 書き出しは pandas（`pd.DataFrame(rows).to_csv(path, index=False)`）。
- パラメータ・結果メタデータ（`params_used.json` / `params_model.json` / `results_reservoir.json`）は
  **JSON のまま**で CSV 化しない。

### 解析プロットのパラメータ記録（`params_used.json`）

- 解析（プロット）を出力するときは、使用した**モデルパラメータ（vicsek_dynamic.c 由来）**と
  **レザバー計算パラメータ**を、各出力ディレクトリ（`analysis/<解析名>/<YYYYMMDD_HHMMSS>/`）に
  `params_used.json` として保存する。
- 掃引したパラメータ（rcut / sgm / seed / λ / train_num 等）は**固定/掃引を自動判別**し、
  固定パラメータはスカラ値、掃引パラメータは範囲（値リスト）でコンパクトに記録する。
  全 sim の `params_model.json` を丸ごとダンプしない。
- 実装は共通ユーティリティ `vicsek_rc/params_io.py`（`split_fixed_varied` / `write_params_used`）を使い、
  スクリプト側で再定義しない。
- `params_used.json` の構造:
  ```json
  {
    "model":     { "fixed": { ...スカラ... }, "swept": { "rcut": [...], "seed": [...] } },
    "reservoir": { "fixed": { "readout":1, "washout":2000, "train_num":7000,
                              "ridge_lambda":1e-11, "k_max":100 }, "swept": {} }
  }
  ```


## 4. データ保護ルール（最重要）

**解析に使用したデータ（`position.dat`、`params_model.json`、評価結果 JSON 等）を削除してはならない。**

- `data/` 配下のシミュレーションディレクトリは、そこから生成された解析結果が存在する限り削除禁止
- `reservoir_data/`、`analysis/<解析名>/` 配下の評価結果も同様に削除禁止
- ディスク節約のために `position.dat` を削除する場合は、**その `position.dat` から生成された評価結果（result JSON）がすべて保存済みであることを確認してから**行うこと
- 不完全なシミュレーションディレクトリ（行数不足）のみ削除可能（使用済みデータではないため）

### 削除可否一覧

| データ | 削除 | 条件 |
|---|---|---|
| 不完全な `position.dat`（行数不足） | **可** | 使用済みでないことを確認 |
| 完全な `position.dat` | **禁止** | 評価結果 JSON が保存済みであれば可（要確認） |
| `params_model.json` | **禁止** | 常に保持 |
| `reservoir_data/<dir>/` | **禁止** | 解析に使用済みの評価結果 |
| `analysis/<解析名>/<dir>/` | **禁止** | サマリー・プロットは研究記録 |
| テスト用一時 dir（`data/test*/` 等） | **可** | 本番実験でないことを確認 |

---

## 5. ディスク容量管理（データ保護と両立させること）

- D:\\ ドライブは 3.7TB だが実験データで逼迫しやすい
- シミュレーション実行前に空き容量を確認する: `df -h /mnt/d/`
- 各シミュレーションは約 **230MB** の `position.dat` を生成する（utime ごとサンプリング後）
- 不完全なディレクトリ（行数不足）は速やかに削除してスペースを確保する
- 大量のシミュレーションをバックグラウンド実行する場合は、進捗を監視しディスクフルになる前に止めること

---
## 7. 実験実行フロー

1 つの完全な実験サイクルは以下の順序で実行する。

### Step 1: NARMA10 入力生成（`tmp/` が空の場合のみ）

```bash
python generate_narma10.py
# → tmp/narma10_{input|target}_0.0:0.5_seed666.dat を生成
```

### Step 2: C コードのコンパイル（`vicsek_dynamic.c` を変更した場合のみ）

```bash
cc -O2 -Wall -Wextra -o vicsek_dynamic vicsek_dynamic.c -lm
```

### Step 3: シミュレーション実行

```bash
./vicsek_dynamic [input_file] [output_base] [seed_noise] [rcut] [sgm] [seed_nf]
# → data/<YYYYMMDD_HHMMSS>/{position.dat, params_model.json} を生成
```

### Step 4: リザバー評価

```bash
python vicsek_prediction.py --data-folder data --date-folder <dir> \
  --input-path tmp/narma10_input_0.0:0.5_seed666.dat \
  --target-path tmp/narma10_target_0.0:0.5_seed666.dat
# → reservoir_data/<YYYYMMDD_HHMMSS>/{results_reservoir.json, ...} を生成
```

Step 1〜4 をまとめて実行する場合は `for.sh` を参照。
