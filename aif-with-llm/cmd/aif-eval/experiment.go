package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/evaluation"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// ---------- 実行 ----------

type runOptions struct {
	DataDir    string
	OutDir     string
	Sample     int
	MinLoc     int
	MaxLoc     int
	Conditions []string
	Model      string
	Threshold  float64
	Seed       int64
	// PromptSet はプロンプト一式の版（promptSets を参照）。
	PromptSet string
	// Metrics は計算する GED 実装の版。同じグラフを複数の指標で測って、
	// 「指標の設計が結論を変えていないか」を確かめるために複数指定できる。
	Metrics []string
	// Split は使う分割（splits を参照）。
	Split string
}

// 分割はサブコーパス（= QT の 1 エピソード）単位で固定する。
//
// ノードセット単位で分けるとリークになる。同一エピソードのノードセットは
// 同じ論題・同じ話者の連続した区間なので、dev で規則を作って test で測っても
// 「同じ議論の別の断片」を見ているだけになる。
//
// dev は v1 → v2 → v3 の規則を導いた 20 件を含む 3 エピソード。
// これらの規則（特に MA/CA の向き）は dev のゴールドの観察から作ったので、
// **test でプロンプトを触ってはいけない**。
const (
	splitAll  = "all"
	splitDev  = "dev"
	splitTest = "test"
)

var splits = map[string][]string{
	splitDev: {
		"cutietestrun4June2020",
		"cutietestrun18June2020",
		"cutietestrun30July2020",
	},
	splitTest: {
		"cutietestrun28May2020",
		"cutietestrun2September2020",
		"cutietestrun22October2020",
	},
}

// promptSet は各段階のプロンプトと、関係の受け方の組み合わせ。
//
// 版が上がるほど IAT の仕様に忠実になる。v1 → v2 → v3 の差は「正解に合わせた
// チューニング」ではなく、IAT が定めているが実装が伝えていなかった規則を
// 足したものである（レポート §4.3 の E1 を参照）。
type promptSet struct {
	locution   string
	illocution string
	relation   string
	// verify が空でなければ stage 4（検証パス）を掛ける。
	verify string
	// scored が true なら関係を確信度つきスキーマで受ける。
	scored bool
	// candidate が空でなければ、stage 3 を「候補対を 1 つずつ判定させる」方式で行う。
	candidate string
	// adjacent が true なら stage 3 を LLM ではなく機械的な規則で作る（基準線）。
	adjacent bool
	// graded が true なら関係を 5 段階の依存度つきスキーマで受ける。
	graded bool
	// restrictForces は stage 2 の発話行為の選択肢を、L-node に anchor できる力
	// だけに絞るか。IAT では Arguing / Restating は遷移に anchor される力なので、
	// 発話単位の選択肢に混ぜてはいけない。
	restrictForces bool
}

var promptSets = map[string]promptSet{
	// v1: 発話行為を anchor 位置で絞らない。関係の向きの規則も伝えない（初版）。
	"v1": {locution: "aif_locution_v1", illocution: "aif_illocution_v1", relation: "aif_relation_v1"},
	// v2: 発話行為を L-node に anchor できる力だけに絞る。
	"v2": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v2", restrictForces: true},
	// v3: v2 に加えて、MA / CA の向きは対話の時間順で決まること、
	// MA は質問と応答の間にも成り立つこと、言い換えの連鎖を落とさないことを伝える。
	"v3": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v3", restrictForces: true},

	// ここから先は 1 世代 1 要因。効果を分離するため、同時に 2 つ変えない。
	//
	// v4 = v3 + H1: 近接バイアスの是正。
	//   **この版のプロンプトに書いた距離分布は誤りである**（正規化前のグラフで測っていた）。
	//   実測し直すと近接バイアスは存在せず、v3 の記述の方が正しかった。
	//   本文を直すとハッシュが変わり実行結果を再現できなくなるので残すが、使わないこと。
	"v4": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v4", restrictForces: true},
	// v5 = v4 + H4: 関係ごとに確信度を申告させる。閾値は評価側で後から動かす。
	"v5": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v5", scored: true, restrictForces: true},
	// v6 = v5 + H2: 検証パス。出す係と削る係を別の呼び出しに分ける。
	"v6": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v5", verify: "aif_verify_v1", scored: true, restrictForces: true},
	// w1 = v3 + H3: 命題の過剰な正規化を抑える（関係層は v3 のまま）。
	//   cos≥0.70 の閾値で baseline（発話のコピー）に負けていたことへの対処。
	"w1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v3", restrictForces: true},

	// adj = 関係抽出の基準線。stage 3 を使わず、隣り合う命題を全部 RA でつなぐ。
	//   関係抽出にも「LLM 無しの下限」を置く。これを置いていなかったため、
	//   「どの対がつながっているか」の判断で LLM が自明な規則に負けていることに
	//   気づけていなかった（test n=124 の無向 F1: v3 0.458 対 adj 0.508。レポート §5.1）。
	"adj": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", adjacent: true, restrictForces: true},
	// p1 = 候補対分類。生成をやめ、距離 4 以内の候補を 1 対ずつ判定させる。
	//   命題層は w1（H3 採用）に合わせ、関係層の方式だけを w1 と比べられるようにする。
	"p1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", candidate: "aif_candidate_v1", restrictForces: true},

	// g1 = v3 の関係プロンプト + 5 段階の依存度。命題層は w1（H3 採用）。
	//   3 値の確信度は実測で high 49.5% / medium 49.1% / low 1.4% と実質 2 値に潰れ、
	//   動作点が「全部（AP 0.300）」と「high のみ（AP 0.395・再現率半減）」の 2 つしか
	//   取れなかった。high と medium の精度差は 2 倍近くあるので、
	//   申告そのものには情報がある。刻みを 5 段階に増やし、
	//   さらに「注釈者の確信」ではなく「dst が src にどれだけ依存しているか」を
	//   操作的な判定手順つきで訊く。関係層の土台は v3（v4 の距離規則は誤りだった）。
	"g1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v7", graded: true, restrictForces: true},
	// c1 = v3 の関係プロンプト + 3 値の確信度。命題層は w1。
	//   g1 との唯一の違いは等級の訊き方（「関係がどれだけ不可欠か」vs
	//   「注釈者は記録するか」）と段数。v5 は v4（誤った距離規則）の上に載っていたため、
	//   3 値と 5 段階を同じ土台・同じ標本で比べた条件が無かった。それを埋める。
	"c1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v8", scored: true, restrictForces: true},
}

// Result は 1 (ノードセット, 条件, 重み) の結果。JSONL で 1 行ずつ書き出す。
type Result struct {
	NodesetID string `json:"nodeset_id"`
	Condition string `json:"condition"`
	PromptSet string `json:"prompt_set"`
	// PromptVersions は使ったプロンプトの版と本文ハッシュ（再現性のため）。
	PromptVersions map[string]string         `json:"prompt_versions,omitempty"`
	Model          string                    `json:"model"`
	Comparison     *evaluation.AIFComparison `json:"comparison"`
	Usage          llm.Usage                 `json:"usage"`
	// GoldNormalize / PredViolations は前処理と厳密性の記録。
	GoldNormalize aif.NormalizeReport `json:"gold_normalize"`
	// Dropped は検証パス（stage 4）が捨てた関係の本数。
	Dropped        int             `json:"dropped,omitempty"`
	PredViolations []aif.Violation `json:"pred_violations,omitempty"`
	Err            string          `json:"error,omitempty"`
}

func runExperiment(ctx context.Context, logger *slog.Logger, cfg *config, opts runOptions) error {
	docs, err := loadDocs(opts.DataDir)
	if err != nil {
		return err
	}
	if opts.Split != splitAll {
		if _, found := splits[opts.Split]; !found {
			return fmt.Errorf("aif-eval: 未知の分割 %q（dev / test / all）", opts.Split)
		}
	}
	picked := sampleDocs(docs, opts)
	if len(picked) == 0 {
		return fmt.Errorf("aif-eval: 条件に合うノードセットが %s に無い（先に -fetch を実行）", opts.DataDir)
	}
	logger.Info("評価対象を選んだ", "split", opts.Split, "nodesets", len(picked), "候補", len(docs))

	client, err := cfg.newLLMClient()
	if err != nil {
		return err
	}
	emb, err := cfg.newEmbedder()
	if err != nil {
		return err
	}
	set, ok := promptSets[opts.PromptSet]
	if !ok {
		return fmt.Errorf("aif-eval: 未知のプロンプト版 %q", opts.PromptSet)
	}
	restrict := set.restrictForces
	ann, err := aif.NewAnnotator(aif.AnnotatorConfig{
		Client: client, Prompts: cfg.newPromptStore(), Model: opts.Model, Logger: logger,
		LocutionPrompt:         set.locution,
		IllocutionPrompt:       set.illocution,
		RelationPrompt:         set.relation,
		VerifyPrompt:           set.verify,
		ScoredRelations:        set.scored,
		CandidatePrompt:        set.candidate,
		AdjacentRelations:      set.adjacent,
		GradedRelations:        set.graded,
		RestrictLocutionForces: &restrict,
	})
	if err != nil {
		return err
	}
	promptVersions, err := ann.PromptVersions()
	if err != nil {
		return err
	}
	logger.Info("プロンプト版", "set", opts.PromptSet, "prompts", fmt.Sprint(promptVersions))

	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return fmt.Errorf("aif-eval: 出力先の作成: %w", err)
	}
	f, err := os.Create(filepath.Join(opts.OutDir, "results.jsonl"))
	if err != nil {
		return fmt.Errorf("aif-eval: 結果ファイルの作成: %w", err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)

	weights := []evaluation.AIFWeights{evaluation.DefaultAIFWeights(), evaluation.DiscountedAIFWeights()}
	var totalUsage llm.Usage
	for i, d := range picked {
		gold, rep := aif.Normalize(d.Graph)
		goldLocs := locutionsOf(gold)
		for _, cond := range opts.Conditions {
			b, err := build(ctx, ann, cond, d, goldLocs)
			if err != nil {
				logger.Error("アノテーションに失敗した", "nodeset", d.NodesetID, "condition", cond, "err", err)
				_ = enc.Encode(Result{NodesetID: d.NodesetID, Condition: cond, PromptSet: opts.PromptSet, Model: opts.Model, Err: err.Error()})
				continue
			}
			totalUsage = totalUsage.Add(b.Usage)
			label := conditionLabel(cond, opts.PromptSet, opts.Model)
			// 生成グラフは保存する。指標を後から変えても LLM を叩き直さずに
			// 比較をやり直せるようにするため（生レスポンスを捨てないのと同じ理由）。
			if err := saveGraph(opts.OutDir, d.NodesetID, label, b.Graph); err != nil {
				return err
			}
			// 確信度つきで生成した条件は、閾値ごとの変種も同じ流れで評価する。
			for _, variant := range confidenceVariants(b.Graph) {
				pred := variant.Graph
				for _, mw := range crossProduct(opts.Metrics, weights) {
					cmpOpts := evaluation.DefaultAIFCompareOptions()
					cmpOpts.Metric = mw.metric
					cmpOpts.Weights = mw.weights
					cmpOpts.MinSimilarity = opts.Threshold
					cmpOpts.Rationales = b.Rationales
					c, err := evaluation.CompareAIF(ctx, pred, gold, emb, cmpOpts)
					if err != nil {
						return err
					}
					// baseline は LLM を使わないのでプロンプト版にもモデルにも依存しない。
					// 付けると同じ数値が版の数だけ別条件として並んでしまう。
					c.Condition = label + variant.Suffix
					c.NodesetID = d.NodesetID
					res := Result{
						NodesetID: d.NodesetID, Condition: cond, PromptSet: opts.PromptSet,
						PromptVersions: promptVersions, Model: opts.Model,
						Comparison: c, Usage: b.Usage, GoldNormalize: rep,
						PredViolations: aif.Validate(pred), Dropped: b.Dropped,
					}
					if err := enc.Encode(res); err != nil {
						return fmt.Errorf("aif-eval: 結果の書き出し: %w", err)
					}
				}
			}
			logger.Info("完了", "nodeset", d.NodesetID, "condition", cond,
				"i", i+1, "n", len(picked), "cost_usd", fmt.Sprintf("%.4f", totalUsage.CostUSD))
		}
	}
	logger.Info("実行完了", "cost_usd", fmt.Sprintf("%.4f", totalUsage.CostUSD),
		"input_tokens", totalUsage.InputTokens, "output_tokens", totalUsage.OutputTokens)
	return nil
}

// metricWeights は 指標の版 × 重み の組。
type metricWeights struct {
	metric  string
	weights evaluation.AIFWeights
}

func crossProduct(metrics []string, weights []evaluation.AIFWeights) []metricWeights {
	out := make([]metricWeights, 0, len(metrics)*len(weights))
	for _, m := range metrics {
		for _, w := range weights {
			out = append(out, metricWeights{metric: m, weights: w})
		}
	}
	return out
}

// conditionLabel は集計・検定で使う条件名。
//
// プロンプト版とモデルを名前に入れるのが要点。入れないと、同じ条件を別の
// モデルで回した結果が同じ系列に混ざって平均されてしまう
// （モデル差を測る実験ではそれが結論そのものになる）。
// baseline は LLM を使わないので、版もモデルも付けない。
func conditionLabel(cond, promptSet, model string) string {
	if cond == condBaseline {
		return cond
	}
	return cond + "@" + promptSet + "/" + modelTag(model)
}

// modelTag はモデル名を表側に出す短縮形。"gpt-5.6-terra" → "5.6-terra"。
// 世代まで残すのは gpt-5-mini と gpt-5.6-mini を混同しないため。
func modelTag(model string) string {
	return strings.TrimPrefix(model, "gpt-")
}

// saveGraph は生成した AIF グラフを out/graphs/ に書き出す。
// 条件名（プロンプト版とモデルを含む）をそのままファイル名にする（"/" は "-" に置き換える）。
func saveGraph(outDir, nodesetID, label string, g aif.Graph) error {
	name := fmt.Sprintf("%s_%s.json", nodesetID, strings.ReplaceAll(label, "/", "-"))
	return writeJSON(filepath.Join(outDir, "graphs", name), g)
}

// build は条件に応じて生成側のグラフを作る。
func build(ctx context.Context, ann *aif.Annotator, cond string, d aif.Document, goldLocs []aif.LocutionInput) (built, error) {
	switch cond {
	case condBaseline:
		return built{Graph: aif.BaselineGraph(d.NodesetID, goldLocs)}, nil
	case condGoldLoc:
		a, err := ann.Annotate(ctx, d.NodesetID, goldLocs)
		if err != nil {
			return built{}, err
		}
		return built{Graph: a.Graph, Usage: a.Usage, Rationales: a.Rationales, Dropped: a.Dropped}, nil
	case condE2E:
		if strings.TrimSpace(d.Text) == "" {
			return built{}, fmt.Errorf("aif-eval: %s に素テキストが無いので e2e を実行できない（corpus.json の no_text を参照）", d.NodesetID)
		}
		a, err := ann.AnnotateTranscript(ctx, d.NodesetID, d.Text)
		if err != nil {
			return built{}, err
		}
		return built{Graph: a.Graph, Usage: a.Usage, Rationales: a.Rationales, Dropped: a.Dropped}, nil
	}
	return built{}, fmt.Errorf("aif-eval: 未知の条件 %q", cond)
}

// built は 1 条件分の生成結果。
type built struct {
	Graph      aif.Graph
	Usage      llm.Usage
	Rationales map[string]string
	// Dropped は検証パス（stage 4）が捨てた関係の本数。
	Dropped int
}

// confidenceVariants は、等級つきで生成したグラフから「閾値ごとのグラフ」を作る。
//
// 生成のやり直しは要らない（等級は S-node に載っている）ので、
// 1 回の LLM 呼び出しから P/R 曲線の各点が取れる。
// 3 値の確信度なら 2 点、5 段階の依存度なら 4 点。
// 等級を申告していないグラフでは変種を作らない（v3 以前と同じ 1 本だけ）。
func confidenceVariants(g aif.Graph) []graphVariant {
	out := []graphVariant{{"", g}}
	ladder := g.GradeLadder()
	if len(ladder) == 0 {
		return out
	}
	// 梯子の上から i+1 段までを残す閾値を順に作る（全段を残す＝無フィルタは除く）。
	for i := 0; i < len(ladder)-1; i++ {
		keep := ladder[:i+1]
		var suffix string
		if i == 0 {
			suffix = "+grade=" + keep[0]
		} else {
			suffix = "+grade≥" + keep[i]
		}
		out = append(out, graphVariant{suffix, g.FilterRelationsByConfidence(keep...)})
	}
	return out
}

// graphVariant は 1 つの閾値に対応する生成グラフ。
type graphVariant struct {
	Suffix string
	Graph  aif.Graph
}

// locutionsOf はゴールドの L-node を LocutionInput に写す。
// "話者 : 本文" 形になっているものは本文だけを取り出す。
func locutionsOf(g aif.Graph) []aif.LocutionInput {
	// L-node の並びは TA の連鎖で決まる（JSON の順序が時系列とは限らない）。
	order := g.LocutionOrder()
	out := make([]aif.LocutionInput, 0, len(order))
	for _, n := range order {
		text := n.Text
		speaker := aif.SpeakerOf(n)
		if speaker != "" && strings.HasPrefix(text, speaker) {
			text = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(text[len(speaker):]), ":"))
		}
		if text == "" {
			continue
		}
		out = append(out, aif.LocutionInput{Speaker: speaker, Text: text})
	}
	return out
}
