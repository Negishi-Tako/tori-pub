# aif-with-llm — Evaluating LLM Extraction of AIF Argument Graphs

[English](#english) | [日本語](#日本語)

---

## English

Code and report for an experiment that measures **how correct the AIF / IAT argument graphs**
built by large language models from transcripts are, using a human-annotated public corpus
(AIFdb / QT30) as the gold standard.

Results and discussion are in **[`docs/report.md`](docs/report.md)** (in Japanese). The report has
two parts: Part I covers the accuracy and method of AIF extraction, and Part II covers the
accuracy and method of graph comparison (the evaluation metrics). The limitations of the experiment
(sample size, the corpus being English, how a revision of the metric reversed a conclusion, and
measurement bugs and their impact) are stated in the "Limitations" sections of each part
(§7, §15, §16).

### Scope of this repository

This repository contains only the parts needed for this experiment, extracted from a parent
project (a discussion-reproduction simulator, not public).

**Included**

| Path                          | Contents                                                                                                                                                                                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `internal/aif`                | Canonical representation of plain AIF / IAT, AIFdb client, metamodel checks, LLM annotator, baseline graphs                                                                                                                                                                                              |
| `internal/evaluation`         | Graph edit distance (`aifged.go` is the main metric of this experiment), optimal assignment (`assignment.go`, Hungarian method), CASS-style layered metrics, evidence extraction                                                                                                                         |
| `internal/llm`                | LLM client (Structured Outputs / caching / rate limiting / price table), embeddings                                                                                                                                                                                                                      |
| `internal/stats`              | Paired bootstrap CIs, Wilcoxon signed-rank test, Cliff's delta, Holm correction                                                                                                                                                                                                                          |
| `internal/prompts`            | Loading `prompts/` and hashing prompt bodies                                                                                                                                                                                                                                                             |
| `cmd/aif-eval`                | CLI for fetch → extract → evaluate → aggregate (the entry point of the experiment). Config loading and wiring also live here. `-goldcheck` counts the gold graphs' own conformance to the AIF metamodel                                                                                                  |
| `prompts/aif_*.md`            | Every prompt compared in the experiment (one file per version; the body hash is recorded in the results)                                                                                                                                                                                                 |
| `data/metrics/*.csv`          | Aggregated metrics (numbers only, no corpus text; same format as the `results.csv` written by `-report`). `n124_*.csv` are the numbers in the report (`docs/report.md`); `rerun_*.csv` are the TJ-SIF 2026 paper numbers, re-collected with `scripts/rerun_paper.sh` after fixing the utterance ordering |
| `scripts/make_report_figs.py` | Generates the report figures (matplotlib)                                                                                                                                                                                                                                                                |
| `scripts/make_paper_figs.py`  | Generates the paper figures (pipeline diagram, example generated graph). The example figure (Fig. 2) contains corpus text, so it is not included in the repository; generate it locally from the raw output (`out/`)                                                                                     |
| `scripts/rerun_paper.sh`      | Re-runs the paper's conditions (E = `v3`, K = `w1`, N = `baseline`, A = `adj` in the paper, plus a gpt-5.6-terra version of E)                                                                                                                                                                           |

**Not included**

- The parent project's retrieval, generation and CMS (Postgres / Redis / Neo4j / Qdrant / REST API /
  frontend). This experiment uses none of them.
- **The QT30 corpus text.** See "Data handling" below.

### Data handling (important)

**The QT30 corpus text is not included in this repository.**
The corpus metadata returned by the AIFdb API has an empty licence field, so the terms for
redistribution could not be confirmed. Obtain the corpus yourself from the original distributor.

```sh
make aif-fetch ARGS="-corpus qt30 -subcorpora 6"   # → data/aifdb/ (ignored by .gitignore)
```

As of 2026-10, the AIFdb subcorpus archives are broken: the nodeset JSON files are 0 bytes, and the
plain text is a single episode-level transcript rather than one file per nodeset.
In that case `-fetch` re-downloads only the broken JSON files one at a time from
`https://www.aifdb.org/json/<id>` (at 2-second intervals), and cuts each nodeset's span out of the
episode transcript to get its plain text (`internal/aif/transcript.go`). The cut-out text is not
guaranteed to match, byte for byte, the per-nodeset text the archives used to contain (so the input
to the e2e condition may differ).
Some subcorpora, such as `cutietestrun18June2020` (dev), do not include a transcript at all, so
their plain text cannot be produced. Only the e2e condition uses plain text, so those nodesets are
still fetched and kept in the sample (listed under `no_text` in `corpus.json`), and only their e2e
runs are recorded as error rows in the results.
Which inputs were used is recorded in `files` in `data/aifdb/corpus.json` (the SHA-256 of each
nodeset's graph and text, and its origin: `archive` / `nodeset-json` / `episode-excerpt`). Include
this when you report results.

For the same reason, the raw experiment output (`out/`) is not included either: the I-nodes and
L-nodes of the generated AIF graphs contain corpus text verbatim. Instead, **only aggregated
metrics without corpus text** are provided in `data/metrics/*.csv` (GED, F1, etc. per
nodeset × condition × metric version × weight).
However, the figure scripts and the report's micro averages, distance bands and threshold curves
use columns that are not in the CSVs (TP counts, distance bands, F1 per threshold, grade
distributions), so they need the raw output (`out/aif-n124-*/`).

- Corpus: <https://corpora.aifdb.org/qt30>
- Source: Hautli-Janisz et al. (2022), LREC 2022 (see the report's references for details)

### Reproduction

```sh
cp .env.example .env            # set OPENAI_API_KEY
make aif-fetch ARGS="-corpus qt30 -subcorpora 6"

# Evaluation (a prompt-set is a combination of files in prompts/; see promptSets in cmd/aif-eval)
make aif-run ARGS="-split test -sample 124 -conditions baseline,gold-loc -prompt-set v3 -out out/v3"
make aif-run ARGS="-split test -sample 124 -conditions gold-loc -prompt-set adj -out out/adj"   # baseline that uses no LLM for relations (stage 3); propositions are the stage-2 LLM output
make aif-report ARGS="-out out/v3"
```

- `-split dev` / `-split test` are fixed splits by subcorpus (= one episode).
  Prompts were designed on dev only, and reported numbers come from test.
- Both LLM responses and embeddings are cached, so **re-running gives the same numbers**
  (at zero cost).
- No external services are needed (everything runs in Go). Optimal node assignment used to be
  solved by a Python (scipy) service; the same algorithm (the same shortest augmenting path method
  and tie-breaking as scipy's `linear_sum_assignment`) has been ported to Go
  (`internal/evaluation/assignment.go`).
- By default, four GED versions (0.1 / 0.2 / 0.3 / 0.4) are computed together. The default is
  version 0.4 with the gold graph as reference. Version 0.4 is the full graph of 0.3 with TA nodes
  removed (because the generator adds TAs mechanically by construction rules; see (6) in
  `internal/evaluation/aifged.go`), so the proposition graph (proposition GED) has the same value as
  in 0.3. The older versions are kept because showing that the metric's design changed the
  conclusion is part of the results (report §11).

```sh
make aif-goldcheck   # count AIF metamodel violations in the gold graphs
make check           # fmt → vet → test
make figures         # regenerate the report figures from out/aif-n124-*/ (Python + matplotlib)
```

### License

The code is MIT licensed (`LICENSE`).
This does not apply to the corpus you fetch (see "Data handling" above).

---

## 日本語

人手アノテーション済みの公開コーパス（AIFdb / QT30）を正解として、
大規模言語モデルが書き起こしから作る **AIF / IAT 議論グラフの正しさ**を測る実験のコードと
レポートである。

結果と考察は **[`docs/report.md`](docs/report.md)** にまとめてある。レポートは 2 部構成で、
第 I 部が AIF 抽出の精度と手法、第 II 部がグラフ比較（評価指標）の精度と手法である。
実験の限界（標本の大きさ、英語コーパスであること、指標の改訂で結論が反転した経緯、
計測の不具合とその影響）は各部の「限界」節（§7・§15・§16）に明記した。

### このリポジトリの位置づけ

親プロジェクト（議論再現シミュレータ。非公開）から、本実験に必要な部分だけを切り出したもの。

**含めたもの**

| パス                          | 内容                                                                                                                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `internal/aif`                | 素の AIF / IAT の正準表現、AIFdb クライアント、メタモデル検査、LLM アノテータ、基準線グラフ                                                                                                                                         |
| `internal/evaluation`         | グラフ編集距離（`aifged.go` が本実験の主指標）、最適割当（`assignment.go`・Hungarian 法）、CASS 風の層別指標、証拠の抽出                                                                                                            |
| `internal/llm`                | LLM クライアント（Structured Outputs / キャッシュ / レート制御 / 単価表）、埋め込み                                                                                                                                                 |
| `internal/stats`              | 対応ありブートストラップ CI、Wilcoxon 符号順位検定、Cliff's delta、Holm 補正                                                                                                                                                        |
| `internal/prompts`            | `prompts/` の読み込みと本文ハッシュ                                                                                                                                                                                                 |
| `cmd/aif-eval`                | 取得 → 抽出 → 評価 → 集計の CLI（実験の入口）。設定の読み込みと配線もここ。`-goldcheck` でゴールド自体の AIF メタモデル適合を数える                                                                                                 |
| `prompts/aif_*.md`            | 実験で比較した全プロンプト（版ごとに別ファイル。本文のハッシュを結果に記録する）                                                                                                                                                    |
| `data/metrics/*.csv`          | 集計値（本文を含まない数値のみ。`-report` が書き出す `results.csv` と同じ形式）。`n124_*.csv` はレポート（`docs/report.md`）の数値、`rerun_*.csv` は発話順序の修正後に `scripts/rerun_paper.sh` で取り直した TJ-SIF 2026 論文の数値 |
| `scripts/make_report_figs.py` | レポートの図の生成（matplotlib）                                                                                                                                                                                                    |
| `scripts/make_paper_figs.py`  | 論文の図（パイプライン図・生成グラフの例）の生成。例の図（図 2）はコーパス本文を含むためリポジトリに含めず、生出力（`out/`）から手元で生成する                                                                                      |
| `scripts/rerun_paper.sh`      | 論文の条件の再実行（論文の E = `v3`、K = `w1`、N = `baseline`、A = `adj`、および E の gpt-5.6-terra 版）                                                                                                                            |

**含めなかったもの**

- 親プロジェクトの検索・生成・CMS（Postgres / Redis / Neo4j / Qdrant / REST API / フロントエンド）。
  本実験はこれらを一切使わない。
- **QT30 コーパスの本文**。下記「データの扱い」を参照。

### データの扱い（重要）

**QT30 コーパスの本文はこのリポジトリに含まれない。**
AIFdb の API が返すコーパスのメタ情報には licence フィールドが空で返るため、
再配布の条件が確認できていない。各自が一次配布元から取得すること。

```sh
make aif-fetch ARGS="-corpus qt30 -subcorpora 6"   # → data/aifdb/（.gitignore 対象）
```

AIFdb のサブコーパスのアーカイブは、2026-10 時点で中身が壊れている（ノードセットの JSON が 0 バイトで、
素テキストもノードセット単位ではなくエピソード単位の書き起こし 1 本しか入っていない）。
`-fetch` はそのとき、壊れた JSON だけを `https://www.aifdb.org/json/<id>` から 1 件ずつ取り直し
（2 秒間隔）、素テキストはエピソードの書き起こしからノードセットの区間を切り出す
（`internal/aif/transcript.go`）。切り出しは、以前アーカイブに同梱されていたノードセット単位の
テキストとバイト単位で一致する保証が無い（e2e 条件の入力が変わりうる）。
`cutietestrun18June2020`（dev）のように書き起こし自体が同梱されないサブコーパスもあり、
その素テキストは作れない。素テキストを使うのは e2e 条件だけなので、そうしたノードセットも取得・標本には残し
（`corpus.json` の `no_text` に列挙）、e2e だけを結果の error 行として記録する。
どの入力を使ったかは `data/aifdb/corpus.json` の `files`（ノードセットごとのグラフ・テキストの
SHA-256 と由来 `archive` / `nodeset-json` / `episode-excerpt`）に残るので、結果を報告するときはこれも添える。

同じ理由で、実験の生出力（`out/`）も含めない。生成された AIF グラフの I-node / L-node には
コーパス本文がそのまま入るためである。代わりに、**本文を含まない集計値のみ**を
`data/metrics/*.csv` に置いた（ノードセット × 条件 × 指標版 × 重み ごとの GED・F1 等）。
ただし図の生成スクリプトと、レポートの micro 平均・距離帯・閾値曲線は、CSV に無い列
（TP 数・距離帯・閾値別 F1・等級分布）を使うため、生出力（`out/aif-n124-*/`）が必要である。

- コーパス: <https://corpora.aifdb.org/qt30>
- 出典: Hautli-Janisz et al. (2022), LREC 2022（詳細はレポートの引用文献）

### 再現手順

```sh
cp .env.example .env            # OPENAI_API_KEY を設定する
make aif-fetch ARGS="-corpus qt30 -subcorpora 6"

# 評価（prompt-set は prompts/ の組み合わせ。cmd/aif-eval の promptSets を参照）
make aif-run ARGS="-split test -sample 124 -conditions baseline,gold-loc -prompt-set v3 -out out/v3"
make aif-run ARGS="-split test -sample 124 -conditions gold-loc -prompt-set adj -out out/adj"   # 関係（stage 3）に LLM を使わない基準線（命題は stage 2 の LLM 出力）
make aif-report ARGS="-out out/v3"
```

- `-split dev` / `-split test` はサブコーパス（= 1 エピソード）単位の固定分割。
  プロンプトの設計は dev でのみ行い、報告する数値は test から取る。
- LLM 応答と埋め込みの両方をキャッシュするので、**再実行しても同じ数値が出る**（費用も 0）。
- 外部サービスは不要（Go だけで完結する）。ノードの最適割当は以前 Python（scipy）の
  サービスで解いていたが、同じ解法（scipy の `linear_sum_assignment` と同じ最短増加路法・
  同じ同点の解き方）を Go に移植した（`internal/evaluation/assignment.go`）。
- GED は既定で 4 版（0.1 / 0.2 / 0.3 / 0.4）を同時に計算する。既定の版は 0.4 の gold 基準。
  0.4 は 0.3 の全体グラフから TA を外したもの（TA は生成側が構成規則で機械的に張るため。
  `internal/evaluation/aifged.go` の (6)）で、命題グラフ（命題 GED）は 0.3 と同じ値になる。
  古い版を残しているのは、指標の設計が結論を変えたことを示すのが結果の一部だからである
  （レポート §11）。

```sh
make aif-goldcheck   # ゴールド側の AIF メタモデル違反を数える
make check           # fmt → vet → test
make figures         # レポートの図を out/aif-n124-*/ から再生成（Python + matplotlib）
```

### ライセンス

コードは MIT（`LICENSE`）。
取得したコーパスには適用されない（上記「データの扱い」）。
