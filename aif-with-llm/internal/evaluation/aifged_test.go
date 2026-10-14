package evaluation_test

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/evaluation"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// compare compares with the default settings and the given metric version.
func compare(t *testing.T, pred, gold aif.Graph, metric string) *evaluation.AIFComparison {
	t.Helper()
	return compareWith(t, pred, gold, metric, evaluation.DefaultAIFWeights())
}

func compareWith(t *testing.T, pred, gold aif.Graph, metric string, w evaluation.AIFWeights) *evaluation.AIFComparison {
	t.Helper()
	opts := evaluation.DefaultAIFCompareOptions()
	opts.Metric, opts.Weights = metric, w
	c, err := evaluation.CompareAIF(context.Background(), pred, gold, &llm.HashEmbedder{D: 64}, opts)
	if err != nil {
		t.Fatalf("CompareAIF(%s): %v", metric, err)
	}
	return c
}

// Comparing a graph with itself must give GED 0 and all F1 1.0 in every version.
// If this breaks, every number that follows becomes meaningless.
func TestCompareAIFIdenticalGraphs(t *testing.T) {
	g := relGraph([]string{"schools should reopen", "the infection rate is falling"}, [][3]int{{1, 0, 0}})
	for _, m := range evaluation.AIFMetrics {
		c := compare(t, g, g, m)
		if c.FullGED.GED != 0 || c.ReducedGED.GED != 0 {
			t.Errorf("%s: 同一グラフの GED は 0 のはず: full=%v reduced=%v", m, c.FullGED, c.ReducedGED)
		}
		if c.FullSimilarityGold != 1 || c.ReducedSimilarityGold != 1 {
			t.Errorf("%s: 同一グラフの類似度は 1.0 のはず: %v / %v", m, c.FullSimilarityGold, c.ReducedSimilarityGold)
		}
		if c.Relations.Typed.F1 != 1 || c.Relations.Reversed != 0 || c.Forces.Accuracy != 1 || c.Propositions.PRF.F1 != 1 {
			t.Errorf("%s: 同一グラフの診断はすべて満点のはず: %+v %+v %+v", m, c.Relations, c.Forces, c.Propositions)
		}
	}
}

// How the treatment of a single relation error changes across versions.
//
//	0.1  edges are exact match or total loss (insertion 1.0 + deletion 1.0 = 2.0)
//	0.2  substitution (partial credit) if a gold edge exists on the same proposition pair, but not in the overall graph
//	0.3  partial credit applies to the overall graph too, so a relation error costs the same at any granularity
//
// Also, relation diagnostics (counting undirected hits and reversals) must not depend on the version.
func TestRelationErrorCostByMetricVersion(t *testing.T) {
	gold := twoPropGraph(aif.TypeMA, false)
	w := evaluation.DefaultAIFWeights()
	cases := []struct {
		name     string
		pred     aif.Graph
		want     float64 // edge cost in the 0.2 / 0.3 proposition graph
		reversed int
	}{
		{"種別違い（MA を RA と判定）", twoPropGraph(aif.TypeRA, false), w.EdgeTypeMismatch, 0},
		{"向き違い", twoPropGraph(aif.TypeMA, true), w.EdgeDirectionMismatch, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v1 := compare(t, tc.pred, gold, evaluation.MetricGEDv1)
			if v1.ReducedGED.EdgeCost != 2*w.EdgeCost {
				t.Errorf("0.1 は挿入 + 削除を課すはず: %v", v1.ReducedGED.EdgeCost)
			}
			v2 := compare(t, tc.pred, gold, evaluation.MetricGEDv2)
			if v2.ReducedGED.EdgeCost != tc.want || v2.ReducedGED.NodeCost != 0 {
				t.Errorf("0.2 は置換コスト %v だけを課すはず: %+v", tc.want, v2.ReducedGED)
			}
			v3 := compare(t, tc.pred, gold, evaluation.MetricGEDv3)
			if v3.ReducedGED.GED != tc.want || v3.FullGED.GED != tc.want {
				t.Errorf("0.3 は両粒度で %v のはず: reduced=%v full=%v", tc.want, v3.ReducedGED.GED, v3.FullGED.GED)
			}
			if r := v3.Relations; r.Undirected.F1 != 1 || r.Typed.F1 != 0 || r.Reversed != tc.reversed {
				t.Errorf("無向では当たり・厳密では外れ・逆向き %d 本のはず: %+v", tc.reversed, r)
			}
		})
	}
}

// A graph that differs only in the order of structural nodes (TAs).
//
// 0.1 put TAs into the Hungarian method, and TAs have substitution cost 0 with each other
// and are indistinguishable, so changing their order assigns them to different TAs
// and L→TA→L edges are counted as errors.
// 0.2 and later derive TA correspondences from L correspondences, so order does not matter.
func TestDerivedCorrespondenceIsRobustToStructuralNodeOrder(t *testing.T) {
	gold := chainGraph([]string{"alpha", "beta", "gamma"}, false)
	pred := chainGraph([]string{"alpha", "beta", "gamma"}, true)
	v1 := compare(t, pred, gold, evaluation.MetricGEDv1).FullGED
	v2 := compare(t, pred, gold, evaluation.MetricGEDv2).FullGED
	if v2.GED != 0 {
		t.Errorf("0.2 は並び順に影響されないはず: GED=%v edges=%+v", v2.GED, v2.Edges)
	}
	if v1.GED <= v2.GED {
		t.Errorf("0.1 は並び順で誤りが出るはず（この差が改良の理由）: v1=%v v2=%v", v1.GED, v2.GED)
	}
}

// Treatment of YAs anchored to transitions.
//
// On the pred side, the YA count is fixed by the harness construction rule "relations between
// non-adjacent utterances have no TA, so no YA is created". 0.2 penalizes that difference in the overall GED,
// and the value moves with weights discounting YA/TA. 0.3 removes them from the overall graph, so neither happens.
func TestRelationAnchoredYAOnlyCountsUpToV2(t *testing.T) {
	gold := twoPropGraph(aif.TypeMA, false)
	gold.Nodes = append(gold.Nodes, aif.Node{ID: "YAS", Type: aif.TypeYA, Text: "Restating", Scheme: "Restating"})
	gold.Edges = append(gold.Edges, aif.Edge{FromID: "TA0", ToID: "YAS"}, aif.Edge{FromID: "YAS", ToID: "S0"})
	pred := twoPropGraph(aif.TypeMA, false)

	uniform, discounted := evaluation.DefaultAIFWeights(), evaluation.DiscountedAIFWeights()
	v2 := compareWith(t, pred, gold, evaluation.MetricGEDv2, uniform).FullGED.GED
	v2d := compareWith(t, pred, gold, evaluation.MetricGEDv2, discounted).FullGED.GED
	if v2 == 0 || !(v2d < v2) {
		t.Errorf("0.2 は関係の YA の欠落を罰し、割引で軽くなるはず: uniform=%v discounted=%v", v2, v2d)
	}
	for _, w := range []evaluation.AIFWeights{uniform, discounted} {
		if got := compareWith(t, pred, gold, evaluation.MetricGEDv3, w).FullGED.GED; got != 0 {
			t.Errorf("0.3 (%s) では関係の YA は全体 GED に乗らないはず: %v", w.Name, got)
		}
	}
}

// Merely emitting more output must not raise similarity (the reason 0.3 made gold-based normalization the primary metric).
//
// Symmetric normalization includes pred's size in the denominator, so even at the same edit cost, "emitting more" scores higher.
func TestGoldNormalizationDoesNotRewardVerbosity(t *testing.T) {
	texts := []string{"alpha", "beta", "gamma", "delta", "epsilon"}
	gold := relGraph(texts, [][3]int{{1, 0, 0}, {2, 1, 0}, {3, 2, 0}, {4, 3, 0}})
	// Emit nothing: 4 missed (4.0). Emit 4: 2 hits, 2 missed, 2 extra (4.0).
	quiet := compare(t, relGraph(texts, nil), gold, evaluation.MetricGEDv3).ReducedGED
	loud := compare(t, relGraph(texts, [][3]int{{1, 0, 0}, {2, 1, 0}, {4, 0, 0}, {3, 0, 0}}), gold, evaluation.MetricGEDv3).ReducedGED
	if quiet.GED != loud.GED {
		t.Fatalf("この 2 つは編集コストが等しい前提のテスト: quiet=%v loud=%v", quiet.GED, loud.GED)
	}
	if math.Abs(quiet.SimilarityGold-loud.SimilarityGold) > 1e-9 {
		t.Errorf("gold 基準はコストが同じなら同じ値のはず: quiet=%.4f loud=%.4f", quiet.SimilarityGold, loud.SimilarityGold)
	}
	if !(loud.Similarity > quiet.Similarity) {
		t.Errorf("対称正規化は多く出した方を高く出すはず: quiet=%.4f loud=%.4f", quiet.Similarity, loud.Similarity)
	}
}

// Two relations on the same proposition pair must not both hit the single gold relation (recall must not exceed 1).
func TestRelationScoreConsumesEachGoldRelationOnce(t *testing.T) {
	gold := relGraph([]string{"alpha", "beta"}, [][3]int{{1, 0, 0}})
	pred := relGraph([]string{"alpha", "beta"}, [][3]int{{1, 0, 0}, {1, 0, 2}}) // RA and MA
	r := compare(t, pred, gold, evaluation.MetricGEDv3).Relations
	if r.Undirected.TP != 1 || r.Typed.TP != 1 || r.Undirected.Recall > 1 {
		t.Errorf("ゴールド 1 本に当てられるのは 1 本だけのはず: %+v", r)
	}
}

// Distance-band aggregation: missing distant relations must show up in per-band R, not in overall F1.
// Per-band precision counts both numerator and denominator by the generated-side band (counting by gold-side band can exceed 1).
func TestRelationScoreSplitsByDistance(t *testing.T) {
	texts := make([]string, 12)
	for i := range texts {
		texts[i] = "claim " + strconv.Itoa(i)
	}
	gold := relGraph(texts, [][3]int{{1, 0, 0}, {11, 0, 0}}) // one adjacent + one at distance 11
	pred := relGraph(texts, [][3]int{{1, 0, 0}})
	by := compare(t, pred, gold, evaluation.MetricGEDv3).Relations.ByDistance
	near, far := by["1"], by["10+"]
	if near != (evaluation.DistanceBucket{GoldTotal: 1, PredTotal: 1, TP: 1, TPPred: 1, TPUndirected: 1, TPUndirectedPred: 1}) {
		t.Errorf("隣接の帯は正解・生成・当たりが 1 本ずつのはず: %+v", near)
	}
	if far != (evaluation.DistanceBucket{GoldTotal: 1}) {
		t.Errorf("遠距離の帯は正解 1 本だけのはず: %+v", far)
	}
}

// relGraph is chainGraph with relations added.
// Each element of rels is {src index, dst index, type (0=RA, 1=CA, 2=MA)}.
func relGraph(texts []string, rels [][3]int) aif.Graph {
	g := chainGraph(texts, false)
	kinds := []aif.NodeType{aif.TypeRA, aif.TypeCA, aif.TypeMA}
	for i, r := range rels {
		s := "S" + strconv.Itoa(i)
		g.Nodes = append(g.Nodes, aif.Node{ID: s, Type: kinds[r[2]]})
		g.Edges = append(g.Edges,
			aif.Edge{FromID: "I" + strconv.Itoa(r[0]), ToID: s},
			aif.Edge{FromID: s, ToID: "I" + strconv.Itoa(r[1])})
	}
	out, _ := aif.Normalize(g)
	return out
}

// chainGraph builds an AIF graph where "speaker A makes n consecutive utterances" and nothing else.
// If reverseTA is true, only the order of TA nodes is reversed (same content).
func chainGraph(texts []string, reverseTA bool) aif.Graph {
	locs := make([]aif.LocutionInput, len(texts))
	for i, txt := range texts {
		locs[i] = aif.LocutionInput{Speaker: "A", Text: txt}
	}
	g := aif.BaselineGraph("chain", locs)
	if reverseTA {
		var tas, rest []aif.Node
		for _, n := range g.Nodes {
			if n.Type == aif.TypeTA {
				tas = append([]aif.Node{n}, tas...)
			} else {
				rest = append(rest, n)
			}
		}
		g.Nodes = append(rest, tas...)
	}
	return g
}

// twoPropGraph is a graph with two propositions connected by one relation.
func twoPropGraph(kind aif.NodeType, reversed bool) aif.Graph {
	rel := [3]int{1, 0, map[aif.NodeType]int{aif.TypeRA: 0, aif.TypeCA: 1, aif.TypeMA: 2}[kind]}
	if reversed {
		rel[0], rel[1] = rel[1], rel[0]
	}
	return relGraph([]string{"the schools should reopen", "the schools ought to open again"}, [][3]int{rel})
}

// Generated relations whose endpoints were not matched (over-generation) must also count in the per-band precision denominator.
// Previously they were excluded from the denominator, inflating per-band precision.
func TestDistanceBandPrecisionCountsUnmatchedPredictions(t *testing.T) {
	gold := relGraph([]string{"a", "b"}, [][3]int{{1, 0, 0}})
	// The generated side has 4 propositions. The endpoints of I3→I2 do not match the gold.
	pred := relGraph([]string{"a", "b", "c", "d"}, [][3]int{{1, 0, 0}, {3, 2, 0}})
	r := compare(t, pred, gold, evaluation.MetricGEDv3).Relations
	total := 0
	for _, b := range r.ByDistance {
		total += b.PredTotal
	}
	if total != r.PredTotal {
		t.Errorf("帯ごとの生成本数の合計 %d は生成本数 %d と一致するはず: %+v", total, r.PredTotal, r.ByDistance)
	}
}

// When two relations of different types are emitted on the same proposition pair, output order must not change the result.
// Previously the relation processed first took the gold of a different type, and the exact match was not counted.
func TestRelationMatchingIsOrderIndependent(t *testing.T) {
	gold := relGraph([]string{"a", "b"}, [][3]int{{1, 0, 1}}) // CA
	for _, order := range [][][3]int{{{1, 0, 0}, {1, 0, 1}}, {{1, 0, 1}, {1, 0, 0}}} {
		c := compare(t, relGraph([]string{"a", "b"}, order), gold, evaluation.MetricGEDv3) // RA and CA
		if c.Relations.Typed.TP != 1 || c.Relations.Undirected.TP != 1 {
			t.Errorf("順序 %v: 完全一致 1 本・過剰生成 1 本のはず: %+v", order, c.Relations)
		}
		if c.ReducedGED.GED != 1.0 {
			t.Errorf("順序 %v: 完全一致 0 + 過剰生成 1.0 のはず: %v (%+v)", order, c.ReducedGED.GED, c.ReducedGED.Edges)
		}
	}
	// Likewise for reversal: prefer the other relation's exact match over one reversed relation.
	gold = relGraph([]string{"a", "b"}, [][3]int{{1, 0, 0}})
	c := compare(t, relGraph([]string{"a", "b"}, [][3]int{{0, 1, 0}, {1, 0, 0}}), gold, evaluation.MetricGEDv3)
	if c.Relations.Typed.TP != 1 || c.Relations.Reversed != 0 {
		t.Errorf("完全一致を逆向きより優先するはず: %+v", c.Relations)
	}
}

// Deriving S-node correspondences (used in 0.2's overall graph) must not depend on output order either.
// Previously the CA emitted first took the gold RA, and 0.2's overall GED split into 3.00 / 7.75 depending on order.
func TestDeriveIsOrderIndependent(t *testing.T) {
	texts := []string{"alpha one", "beta two", "gamma three"}
	gold := relGraph(texts, [][3]int{{0, 1, 0}}) // RA
	var got []float64
	for _, order := range [][][3]int{{{0, 1, 1}, {0, 1, 0}}, {{0, 1, 0}, {0, 1, 1}}} { // CA and RA
		got = append(got, compare(t, relGraph(texts, order), gold, evaluation.MetricGEDv2).FullGED.GED)
	}
	if got[0] != got[1] {
		t.Errorf("出力の順序で 0.2 の全体 GED が変わってはいけない: %v", got)
	}
}

// 0.4's overall graph must not count TAs. If L / I / utterance YAs and relations match the gold,
// GED is 0 even if TAs are placed differently (the gold branches, the generated side is an adjacent chain). 0.3 counts the TA difference.
func TestGEDv4IgnoresTransitionStructure(t *testing.T) {
	texts := []string{"alpha one", "beta two", "gamma three"}
	pred := relGraph(texts, [][3]int{{1, 0, 0}})
	gold := relGraph(texts, [][3]int{{1, 0, 0}})
	// Add an L0→L2 TA to the gold (a branch, absent from the generated chain).
	gold.Nodes = append(gold.Nodes, aif.Node{ID: "TAx", Type: aif.TypeTA, Scheme: aif.SchemeDefaultTransition})
	gold.Edges = append(gold.Edges, aif.Edge{FromID: "L0", ToID: "TAx"}, aif.Edge{FromID: "TAx", ToID: "L2"})
	if v4 := compare(t, pred, gold, evaluation.MetricGEDv4); v4.FullGED.GED != 0 {
		t.Errorf("0.4 は TA の違いを数えないはず: %v", v4.FullGED)
	}
	if v3 := compare(t, pred, gold, evaluation.MetricGEDv3); v3.FullGED.GED != 3 {
		t.Errorf("0.3 は余分な TA 1 個と辺 2 本を数えるはず（3.0）: %v", v3.FullGED)
	}
}

// A wrongly directed edge costs the same as "emitting nothing", and 1.0 less than "emitting a relation with no gold counterpart".
// This pins down that direction errors gain nothing over "emitting nothing" (no reward for emitting without judging direction).
func TestReversedRelationCostsSameAsOmission(t *testing.T) {
	texts := []string{"alpha one", "beta two", "gamma three"}
	gold := relGraph(texts, [][3]int{{0, 1, 0}})
	cost := func(rels [][3]int) float64 {
		return compare(t, relGraph(texts, rels), gold, evaluation.MetricGEDv4).ReducedGED.GED
	}
	none, reversed, spurious := cost(nil), cost([][3]int{{1, 0, 0}}), cost([][3]int{{1, 2, 0}})
	if none != reversed || spurious != none+1 {
		t.Errorf("出さない=%v 向き違い=%v 過剰生成=%v（向き違い = 出さない、過剰生成 = 出さない + 1 のはず）", none, reversed, spurious)
	}
}
