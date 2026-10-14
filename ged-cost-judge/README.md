# ged-cost-judge — LLM-as-a-judge による GED 編集コストの較正（PoC）

計画は `docs/plan.md`（第 4 版）。このフォルダには、その概念実証（PoC）の実装がある。
前実験 [`aif-with-llm/`](../aif-with-llm/) のコミット `086cdf4` から `internal/llm`（LLM クライアント）と
Hungarian 法（`internal/ged/assignment.go`）を複製した。

| パス | 内容 |
| --- | --- |
| `internal/ged` | 11 個の学習できるコストでの命題グラフの GED と、その線形化（特徴 φ） |
| `internal/fit` | 引き分けつき一対比較モデル（Rao–Kupper 型）、交互最適化（Hungarian 法 ⇄ L-BFGS）、クラスタブートストラップ、交差検証 |
| `internal/perturb` | 編集カタログと三つ組の編集計画の抽選（seed 固定） |
| `internal/simulate` | API を使わない合成データ（計画 P1） |
| `internal/synth` | LLM による元の議論の生成・変種の書き下し・判定 |
| `prompts/` | 上の 3 つのプロンプト（本文ハッシュを結果に記録する） |

## 実行

```sh
go run ./cmd/gcj -simulate -sources 250 -boot 200 -out out/sim         # 真のコストを現行値からずらした世界
go run ./cmd/gcj -simulate -null -sources 250 -boot 200 -out out/sim   # 現行値が正しい世界
go run ./cmd/gcj -pilot -sources 50 -boot 200 -lambda 1 -budget 2 -out out/pilot   # gpt-5.6-luna で通し（API を使う）
```

- `-pilot` は `OPENAI_API_KEY` を `.env`、なければ `../aif-with-llm/.env` から読む。
- LLM の応答と埋め込みは `.cache/` にキャッシュするので、同じ設定の再実行は費用 0 で同じ結果になる。
- `-budget` を超えると以降の API 呼び出しを止める。

## PoC が計画と違うところ

- 判定者は luna 1 つだけ（計画は luna・Haiku 4.5・Sonnet 5.5 の 3 つ）。人手の判定も無い。
- 別系列のモデルによる書き下しの検証と自然さの評価（G4）をしていない。代わりに「グラフを変える編集があるのにテキストが元のまま」の変種を数え、それを除いた推定も出す。
- 当てはめは (θ, θ_neg, β, τ, π) を同時に L-BFGS で解く（計画は交互）。自然さの共変量 γ は入れていない。
- 交互最適化が循環したときは、各回の θ で対応を解き直した本物の目的関数が最小の回を返す。CMA-ES による大域性の確認はしていない。
- T3 の「現行コストで差が 0 付近のものと明確なものを半々に」は、シミュレーションでだけ行っている（PoC では書き下す前に cos が分からないため）。
