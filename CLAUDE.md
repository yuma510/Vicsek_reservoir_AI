# Vicsek Reservoir — プロジェクトルール

このファイルは Claude Code が自動で読み込むルール集です。作業開始前に必ず参照してください。

---

## 1. 編集範囲の制限

**このファイル（CLAUDE.md）があるディレクトリより上位の
ファイル・ディレクトリを編集・作成・削除してはならない。**

- 編集してよい範囲: `/mnt/d/DATA/yuma/workspace/Vicsek_reservoir_AI/` 配下のみ
- 禁止例: `/mnt/d/DATA/yuma/`、`/mnt/d/DATA/`、その他プロジェクト外の任意のパス
- 上位ディレクトリの内容を読む（参照する）ことは許可されるが、変更は一切行わない

---

## 2. データ保護ルール

**解析に使用したデータ（`position.dat`、`params_model.json`、評価結果 JSON 等）を削除してはならない。**

`data/`・`analysis/<解析名>/` 配下は、そこから生成された解析結果が存在する限り削除禁止。
削除してよいのは不完全な dir（行数不足）のみ。`position.dat` を消す場合は、そこから生成された
評価結果 JSON がすべて保存済みであることを確認してから行う。詳細は下表。

### 削除可否一覧

| データ | 削除 | 条件 |
|---|---|---|
| 不完全な `position.dat`（行数不足） | **可** | 使用済みでないことを確認 |
| 完全な `position.dat` | **禁止** | 評価結果 JSON が保存済みであれば可（要確認） |
| `params_model.json` | **禁止** | 常に保持 |
| `analysis/<解析名>/<dir>/` | **禁止** | サマリー・プロットは研究記録 |
| テスト用一時 dir（`data/test*/` 等） | **可** | 本番実験でないことを確認 |

---

## 3. ドキュメント参照ルール

作業開始前に必ず以下を読むこと:

| ファイル | 内容 |
|---|---|
| `prompt.md` | 研究目標・ワークフロー全体像（ルート） |
| `code_reference.md` | コード仕様・データフロー・CLI 引数の詳細（ルート） |
| `analysis/analysis_tasks.md` | 解析タスク一覧（状態とリンクのみ）。詳細は `analysis/tasks/task_0N_*.md` を参照 |

各ディレクトリの `README.md`（`vicsek_rc/`, `analysis/`, `data/`）も併せて参照。

**コード変更後は `code_reference.md` の該当箇所を必ず更新すること。**
仕様変更を検出した場合は実験を継続する前にその内容を要約すること。

**結果に影響する変更を行った場合は `analysis/tasks/task_0N_*.md` の該当タスクを必ず更新すること。**

対象となる変更の例:
- 解析パラメータ（λ、train_num、washout、k_max 等）の変更
- 評価手法・指標の変更（MC 計算の閾値、NRMSE の定義等）
- スイープ範囲・seed 数の変更
- バグ修正で結果が変わった場合

更新すべき内容:
- 変更した設定値と変更前の値
- 変更後の結果（NRMSE_test、MC_test、gap 等）
- 変更の理由・考察

---

## 4. データディレクトリ規則

実験データは用途別に 2 つのディレクトリへ格納する: `data/`（シミュレーション軌道）、
`analysis/<解析名>/`（解析スクリプトと出力）。
各ディレクトリの規則を以下に定める。README 更新義務は §4.3 に共通ルールとしてまとめる。

### 4.1 data/ ディレクトリ

- ディレクトリ名は **タイムスタンプのみ**: `data/<YYYYMMDD_HHMMSS>/`
- 各ディレクトリに以下の 2 ファイルが生成される:
  - `position.dat` — シミュレーション軌道（N×(ntime/utime) 行、utime ステップごとに 1 フレーム保存）
  - `params_model.json` — パラメータ（機械可読）
- **グループディレクトリは作成しない**。全シム dir を `data/` 直下に並べる
- **実験の検索は `params_model.json` の内容で行う**（`find_exp_dir()` を使用）
  - 例: `find_exp_dir("data", rcut=5, seed=10, seed_key="seed_pos")`
  - ディレクトリ名のパターンマッチは使わない
- 同じ実験セットを複数のタスクが参照してよい（`data/` は共有の軌道倉庫）

### 4.2 analysis/<解析名>/ ディレクトリ

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

### 4.3 README 更新ルール（共通）

**ディレクトリにデータを生成したら、対応する README を必ず更新すること。**

| 操作 | 更新する README | 追加内容 |
|---|---|---|
| `data/` にシムを追加（スイープ等） | `data/README.md`「実験一覧」 | 1 行追加（固定パラメータ・掃引パラメータ・シム数） |
| `analysis/` 新規サブ dir を作成 | `analysis/README.md` サブディレクトリ一覧テーブル | 1 行追加 |
| `analysis/` 既存解析を実行し新知見 | `analysis/README.md` 該当解析の概要欄 | 結果メモを追記 |

---

## 5. パラメータ・メタデータの管理

### 5.1 入力パラメータはハードコード禁止

**スクリプト内にパラメータ値を直接書いてはならない。すべて JSON ファイルから読み込むこと。**

| パラメータ種別 | JSON ファイル | 読み込み方法 |
|---|---|---|
| モデルパラメータ（N, rcut, sgm, ntime 等） | `configs/default_params.json` | vicsek_dynamic.c が読む |
| 共通リザバーパラメータ（washout, train_num, ridge_lambda 等） | `configs/default_reservoir_params.json` | `load_reservoir_defaults()` |
| 解析固有パラメータ（各解析スクリプトにしかない設定） | `analysis/<解析名>/default_params.json` | スクリプトが直接読む |

**argparse のデフォルト値は JSON から取得すること:**

```python
# ✓ 正しい — 共通パラメータ
_RC = load_reservoir_defaults()
p.add_argument("--washout", type=int, default=_RC["washout"])

# ✓ 正しい — 解析固有パラメータ
_CA = json.loads((Path(__file__).parent / "default_params.json").read_text())
p.add_argument("--frame-end", type=int, default=_CA["frame_end"])

# ✗ 禁止
p.add_argument("--washout",   type=int, default=2000)   # JSON にある値をコピー
p.add_argument("--frame-end", type=int, default=2500)   # 解析固有でも直書き禁止
```

使用したデフォルト値は `params_used.json` に必ず記録すること（記録方法は §5.3）。

### 5.2 プロット元データの保存（CSV）

- **プロットの元データ（数値系列・散布点）は CSV（long 形式）で保存する**。サマリーを JSON で保存しない
  （`*_summary.json` は廃止し `*_data.csv` 等に置き換え済み）。
- 散布図は seed ごとの個別点をそのまま 1 行にする（例: `rcut, seed, MC_test, MC_train, nrmse_test, …`）。
  集約値（mean/std）は CSV から再計算できるため保存不要。
- 時系列・カーブ系プロットも対応する CSV を出す（例: `narma10_prediction.csv`, `MCk.csv`,
  `noise_avg_data.csv`, `correlation_<tag>.csv`）。
- 書き出しは pandas（`pd.DataFrame(rows).to_csv(path, index=False)`）。
- パラメータ・結果メタデータ（`params_used.json` / `params_model.json` / `results_reservoir.json`）は
  **JSON のまま**で CSV 化しない。

### 5.3 解析プロットのパラメータ記録（`params_used.json`）

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
    "reservoir": { "fixed": { "readout":1, "washout":2000, "train_num":6000,
                              "ridge_lambda":1e-9, "k_max":100 }, "swept": {} }
  }
  ```

---

## 6. ディスク容量管理（データ保護と両立させること）

- D:\\ ドライブは 3.7TB だが実験データで逼迫しやすい
- シミュレーション実行前に空き容量を確認する: `df -h /mnt/d/`
- 各シミュレーションは約 **230MB** の `position.dat` を生成する（utime ごとサンプリング後）
- 空き容量確保のための削除は §2 の削除可否ルールに従う（不完全な dir のみ削除可）
- 大量のシミュレーションをバックグラウンド実行する場合は、進捗を監視しディスクフルになる前に止めること

---

## 7. git の管理

- コードの変更を行ったときは GitHub に push を行う
