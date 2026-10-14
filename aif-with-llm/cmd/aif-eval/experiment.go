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

// ---------- Running ----------

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
	// PromptSet is the version of the prompt set (see promptSets).
	PromptSet string
	// Metrics are the GED implementation versions to compute. Several can be given so the same
	// graph is measured with multiple metrics, to check that the metric design is not changing the conclusion.
	Metrics []string
	// Split is the split to use (see splits).
	Split string
}

// Splits are fixed per subcorpus (= one QT episode).
//
// Splitting per nodeset would leak. Nodesets from the same episode are consecutive
// stretches with the same topic and speakers, so building rules on dev and measuring
// on test would only look at "another fragment of the same debate".
//
// dev is the 3 episodes containing the 20 items from which the v1 → v2 → v3 rules were derived.
// Those rules (especially the MA/CA direction) came from observing dev's gold, so
// **the prompts must not be touched based on test**.
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

// promptSet combines the prompts for each stage with how relations are received.
//
// Later versions follow the IAT specification more faithfully. The differences v1 → v2 → v3
// are not "tuning toward the answers"; they add rules that IAT defines but the
// implementation did not convey (see E1 in §4.3 of the report).
type promptSet struct {
	locution   string
	illocution string
	relation   string
	// If verify is non-empty, stage 4 (the verification pass) is applied.
	verify string
	// If scored is true, relations are received with a confidence schema.
	scored bool
	// If candidate is non-empty, stage 3 judges candidate pairs one at a time.
	candidate string
	// If adjacent is true, stage 3 is built by a mechanical rule instead of the LLM (baseline).
	adjacent bool
	// If graded is true, relations are received with a 5-level dependence schema.
	graded bool
	// restrictForces limits stage 2's illocutionary-force choices to forces that can be
	// anchored to an L-node. In IAT, Arguing / Restating are anchored to transitions,
	// so they must not be mixed into the per-utterance choices.
	restrictForces bool
}

var promptSets = map[string]promptSet{
	// v1: does not filter illocutions by anchor position, nor convey relation-direction rules (initial version).
	"v1": {locution: "aif_locution_v1", illocution: "aif_illocution_v1", relation: "aif_relation_v1"},
	// v2: limits illocutions to forces that can be anchored to an L-node.
	"v2": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v2", restrictForces: true},
	// v3: on top of v2, conveys that the MA / CA direction follows dialogue time order,
	// that MA also holds between a question and its answer, and that rephrase chains must not be dropped.
	"v3": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v3", restrictForces: true},

	// From here on, one factor per generation. Never change two at once, so effects stay separable.
	//
	// v4 = v3 + H1: correct proximity bias.
	//   **The distance distribution written in this version's prompt is wrong** (it was measured on unnormalized graphs).
	//   Re-measuring showed no proximity bias; v3's description was the correct one.
	//   Fixing the text would change the hash and break reproduction of results, so it is kept, but do not use it.
	"v4": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v4", restrictForces: true},
	// v5 = v4 + H4: ask for a confidence per relation. The threshold is moved later on the evaluation side.
	"v5": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v5", scored: true, restrictForces: true},
	// v6 = v5 + H2: verification pass. Proposing and pruning are separate calls.
	"v6": {locution: "aif_locution_v1", illocution: "aif_illocution_v2", relation: "aif_relation_v5", verify: "aif_verify_v1", scored: true, restrictForces: true},
	// w1 = v3 + H3: suppress over-normalization of propositions (the relation layer stays v3).
	//   Addresses losing to the baseline (copying utterances) at the cos≥0.70 threshold.
	"w1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v3", restrictForces: true},

	// adj = relation-extraction baseline. Skips stage 3 and links all adjacent propositions with RA.
	//   Relation extraction also gets a "no LLM" lower bound. Without it, we had not noticed that
	//   the LLM loses to a trivial rule at deciding which pairs are connected
	//   (test n=124 undirected F1: v3 0.458 vs adj 0.508; report §5.1).
	"adj": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", adjacent: true, restrictForces: true},
	// p1 = candidate-pair classification. Instead of generating, judge candidates within distance 4 one pair at a time.
	//   The proposition layer matches w1 (H3 adopted) so only the relation-layer method differs from w1.
	"p1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", candidate: "aif_candidate_v1", restrictForces: true},

	// g1 = v3's relation prompt + 5-level dependence. The proposition layer is w1 (H3 adopted).
	//   Measured 3-level confidence collapsed to effectively binary (high 49.5% / medium 49.1% / low 1.4%),
	//   leaving only two operating points: "all (AP 0.300)" and "high only (AP 0.395, recall halved)".
	//   Precision differs nearly 2x between high and medium, so
	//   the self-report itself carries information. Increase the levels to 5,
	//   and instead of "annotator confidence", ask "how much dst depends on src"
	//   with an operational decision procedure. The relation layer builds on v3 (v4's distance rule was wrong).
	"g1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v7", graded: true, restrictForces: true},
	// c1 = v3's relation prompt + 3-level confidence. The proposition layer is w1.
	//   The only differences from g1 are how the grade is asked ("how essential is the relation" vs
	//   "would an annotator record it") and the number of levels. v5 sat on top of v4 (the wrong distance rule),
	//   so no condition compared 3 levels and 5 levels on the same base and sample. This fills that gap.
	"c1": {locution: "aif_locution_v1", illocution: "aif_illocution_v3", relation: "aif_relation_v8", scored: true, restrictForces: true},
}

// Result is the result for one (nodeset, condition, weight). Written one line at a time as JSONL.
type Result struct {
	NodesetID string `json:"nodeset_id"`
	Condition string `json:"condition"`
	PromptSet string `json:"prompt_set"`
	// PromptVersions are the prompt versions used and their body hashes (for reproducibility).
	PromptVersions map[string]string         `json:"prompt_versions,omitempty"`
	Model          string                    `json:"model"`
	Comparison     *evaluation.AIFComparison `json:"comparison"`
	Usage          llm.Usage                 `json:"usage"`
	// GoldNormalize / PredViolations record preprocessing and strictness.
	GoldNormalize aif.NormalizeReport `json:"gold_normalize"`
	// Dropped is the number of relations discarded by the verification pass (stage 4).
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
			// Save the generated graph, so the comparison can be redone after changing metrics
			// without calling the LLM again (the same reason raw responses are kept).
			if err := saveGraph(opts.OutDir, d.NodesetID, label, b.Graph); err != nil {
				return err
			}
			// For conditions generated with confidence, per-threshold variants are evaluated the same way.
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
					// baseline does not use the LLM, so it depends on neither prompt version nor model.
					// Tagging it would list the same numbers as separate conditions, once per version.
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

// metricWeights is a (metric version × weight) pair.
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

// conditionLabel is the condition name used in aggregation and tests.
//
// The key point is including the prompt version and model in the name. Otherwise results of the
// same condition run with different models get mixed into one series and averaged
// (in an experiment measuring model differences, that would be the conclusion itself).
// baseline does not use the LLM, so it gets neither version nor model.
func conditionLabel(cond, promptSet, model string) string {
	if cond == condBaseline {
		return cond
	}
	return cond + "@" + promptSet + "/" + modelTag(model)
}

// modelTag is a short form of the model name for table rows. "gpt-5.6-terra" → "5.6-terra".
// The generation is kept so gpt-5-mini and gpt-5.6-mini are not confused.
func modelTag(model string) string {
	return strings.TrimPrefix(model, "gpt-")
}

// saveGraph writes the generated AIF graph to out/graphs/.
// The condition name (including prompt version and model) is used as the file name ("/" replaced with "-").
func saveGraph(outDir, nodesetID, label string, g aif.Graph) error {
	name := fmt.Sprintf("%s_%s.json", nodesetID, strings.ReplaceAll(label, "/", "-"))
	return writeJSON(filepath.Join(outDir, "graphs", name), g)
}

// build builds the generated graph according to the condition.
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

// built is the generation result for one condition.
type built struct {
	Graph      aif.Graph
	Usage      llm.Usage
	Rationales map[string]string
	// Dropped is the number of relations discarded by the verification pass (stage 4).
	Dropped int
}

// confidenceVariants builds per-threshold graphs from a graph generated with grades.
//
// No regeneration is needed (grades live on the S-nodes), so every point of the
// P/R curve comes from a single LLM call:
// 2 points for 3-level confidence, 4 points for 5-level dependence.
// Graphs without self-reported grades get no variants (just the one graph, as in v3 and earlier).
func confidenceVariants(g aif.Graph) []graphVariant {
	out := []graphVariant{{"", g}}
	ladder := g.GradeLadder()
	if len(ladder) == 0 {
		return out
	}
	// Build, in order, thresholds that keep the top i+1 rungs of the ladder (excluding keeping all rungs, i.e. no filter).
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

// graphVariant is the generated graph for one threshold.
type graphVariant struct {
	Suffix string
	Graph  aif.Graph
}

// locutionsOf maps the gold L-nodes to LocutionInput.
// For text of the form "speaker : text", only the text part is taken.
func locutionsOf(g aif.Graph) []aif.LocutionInput {
	// L-node order is determined by the TA chain (JSON order is not necessarily chronological).
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
