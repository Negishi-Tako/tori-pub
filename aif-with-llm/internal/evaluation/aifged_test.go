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

// compare は既定の設定で metric の版を指定して比べる。
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

// 同一グラフ同士の比較は、どの版でも GED 0・全 F1 が 1.0 になること。
// ここが崩れると、以降の数値はすべて意味を失う。
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

// 関係 1 本の誤りの扱いが版でどう変わるか。
//
//	0.1  辺の完全一致か全損か（挿入 1.0 + 削除 1.0 = 2.0）
//	0.2  同じ命題対に正解の辺があれば置換（部分点）。ただし全体グラフには効かない
//	0.3  部分点を全体グラフにも効かせ、関係の誤りのコストを粒度によらず同じにする
//
// あわせて、関係の診断（無向で当たり・向きを逆と数える）が版によらないこと。
func TestRelationErrorCostByMetricVersion(t *testing.T) {
	gold := twoPropGraph(aif.TypeMA, false)
	w := evaluation.DefaultAIFWeights()
	cases := []struct {
		name     string
		pred     aif.Graph
		want     float64 // 0.2 / 0.3 の命題グラフでの辺のコスト
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

// 構造ノード（TA）の並び順を変えただけのグラフ。
//
// 0.1 は TA を Hungarian 法に載せており、TA どうしは置換コスト 0 で
// 互いに区別が付かないため、並び順が変わると別の TA に割り当てられて
// L→TA→L の辺が誤りとして数えられてしまう。
// 0.2 以降は TA の対応を L の対応から導出するので、並び順に影響されない。
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

// 遷移に anchor される YA の扱い。
//
// pred 側は「隣接しない発話同士の関係には TA が無いので YA も作らない」という
// harness の構成規則で YA の個数が決まる。0.2 はその差を全体 GED で罰し、
// YA/TA を割り引く重みで値が動く。0.3 は全体グラフから外すので、どちらも起きない。
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

// 出力を増やすだけで類似度が上がらないこと（0.3 で gold 基準の正規化を主指標にした理由）。
//
// 対称正規化は分母が pred の大きさを含むので、編集コストが同じでも「多く出した方」が高く出る。
func TestGoldNormalizationDoesNotRewardVerbosity(t *testing.T) {
	texts := []string{"alpha", "beta", "gamma", "delta", "epsilon"}
	gold := relGraph(texts, [][3]int{{1, 0, 0}, {2, 1, 0}, {3, 2, 0}, {4, 3, 0}})
	// 何も張らない: 4 本取りこぼし（4.0）。4 本張る: 2 本当たり・取りこぼし 2・過剰 2（4.0）。
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

// 同じ命題対に 2 本張っても、ゴールドの 1 本を二重に当てない（再現率が 1 を超えない）。
func TestRelationScoreConsumesEachGoldRelationOnce(t *testing.T) {
	gold := relGraph([]string{"alpha", "beta"}, [][3]int{{1, 0, 0}})
	pred := relGraph([]string{"alpha", "beta"}, [][3]int{{1, 0, 0}, {1, 0, 2}}) // RA と MA
	r := compare(t, pred, gold, evaluation.MetricGEDv3).Relations
	if r.Undirected.TP != 1 || r.Typed.TP != 1 || r.Undirected.Recall > 1 {
		t.Errorf("ゴールド 1 本に当てられるのは 1 本だけのはず: %+v", r)
	}
}

// 距離帯の集計: 遠い関係の取りこぼしが、全体の F1 ではなく帯ごとの R に出ること。
// 帯ごとの精度は分子・分母とも生成側の帯で数える（ゴールド側の帯で数えると 1 を超えうる）。
func TestRelationScoreSplitsByDistance(t *testing.T) {
	texts := make([]string, 12)
	for i := range texts {
		texts[i] = "claim " + strconv.Itoa(i)
	}
	gold := relGraph(texts, [][3]int{{1, 0, 0}, {11, 0, 0}}) // 隣接 1 本 + 距離 11 の 1 本
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

// relGraph は chainGraph に関係を足したグラフ。
// rels の各要素は {src の索引, dst の索引, 種別（0=RA, 1=CA, 2=MA）}。
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

// chainGraph は「話者 A が n 回続けて発話する」だけの AIF グラフを作る。
// reverseTA が true なら TA ノードの並び順だけを逆にする（内容は同じ）。
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

// twoPropGraph は命題 2 つを 1 本の関係で結んだグラフ。
func twoPropGraph(kind aif.NodeType, reversed bool) aif.Graph {
	rel := [3]int{1, 0, map[aif.NodeType]int{aif.TypeRA: 0, aif.TypeCA: 1, aif.TypeMA: 2}[kind]}
	if reversed {
		rel[0], rel[1] = rel[1], rel[0]
	}
	return relGraph([]string{"the schools should reopen", "the schools ought to open again"}, [][3]int{rel})
}

// 端点が対応付かなかった生成側の関係（過剰生成）も、帯ごとの精度の分母に入ること。
// 以前は分母から外れており、帯ごとの精度が過大になっていた。
func TestDistanceBandPrecisionCountsUnmatchedPredictions(t *testing.T) {
	gold := relGraph([]string{"a", "b"}, [][3]int{{1, 0, 0}})
	// 生成側は命題が 4 つ。I3→I2 は端点がゴールドに対応しない。
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

// 同じ命題対に種別の違う関係を 2 本出したとき、出力の順序で結果が変わらないこと。
// 以前は先に処理した関係が別種別のゴールドを取ってしまい、完全一致が数えられなかった。
func TestRelationMatchingIsOrderIndependent(t *testing.T) {
	gold := relGraph([]string{"a", "b"}, [][3]int{{1, 0, 1}}) // CA
	for _, order := range [][][3]int{{{1, 0, 0}, {1, 0, 1}}, {{1, 0, 1}, {1, 0, 0}}} {
		c := compare(t, relGraph([]string{"a", "b"}, order), gold, evaluation.MetricGEDv3) // RA と CA
		if c.Relations.Typed.TP != 1 || c.Relations.Undirected.TP != 1 {
			t.Errorf("順序 %v: 完全一致 1 本・過剰生成 1 本のはず: %+v", order, c.Relations)
		}
		if c.ReducedGED.GED != 1.0 {
			t.Errorf("順序 %v: 完全一致 0 + 過剰生成 1.0 のはず: %v (%+v)", order, c.ReducedGED.GED, c.ReducedGED.Edges)
		}
	}
	// 向きの逆転も同様: 逆向き 1 本より、もう 1 本の完全一致を優先する。
	gold = relGraph([]string{"a", "b"}, [][3]int{{1, 0, 0}})
	c := compare(t, relGraph([]string{"a", "b"}, [][3]int{{0, 1, 0}, {1, 0, 0}}), gold, evaluation.MetricGEDv3)
	if c.Relations.Typed.TP != 1 || c.Relations.Reversed != 0 {
		t.Errorf("完全一致を逆向きより優先するはず: %+v", c.Relations)
	}
}

// S-node の対応の導出（0.2 の全体グラフで使う）も、出力の順序に依存しないこと。
// 以前は先に出た CA がゴールドの RA を取り、0.2 の全体 GED が順序次第で 3.00 / 7.75 に割れていた。
func TestDeriveIsOrderIndependent(t *testing.T) {
	texts := []string{"alpha one", "beta two", "gamma three"}
	gold := relGraph(texts, [][3]int{{0, 1, 0}}) // RA
	var got []float64
	for _, order := range [][][3]int{{{0, 1, 1}, {0, 1, 0}}, {{0, 1, 0}, {0, 1, 1}}} { // CA と RA
		got = append(got, compare(t, relGraph(texts, order), gold, evaluation.MetricGEDv2).FullGED.GED)
	}
	if got[0] != got[1] {
		t.Errorf("出力の順序で 0.2 の全体 GED が変わってはいけない: %v", got)
	}
}

// 0.4 の全体グラフは TA を数えないこと。L / I / 発話の YA と関係がゴールドと同じなら、
// TA の張り方（ゴールドは分岐、生成は隣接の鎖）が違っても GED は 0。0.3 は TA の差を数える。
func TestGEDv4IgnoresTransitionStructure(t *testing.T) {
	texts := []string{"alpha one", "beta two", "gamma three"}
	pred := relGraph(texts, [][3]int{{1, 0, 0}})
	gold := relGraph(texts, [][3]int{{1, 0, 0}})
	// ゴールドに L0→L2 の TA を足す（分岐。生成側の鎖には無い）。
	gold.Nodes = append(gold.Nodes, aif.Node{ID: "TAx", Type: aif.TypeTA, Scheme: aif.SchemeDefaultTransition})
	gold.Edges = append(gold.Edges, aif.Edge{FromID: "L0", ToID: "TAx"}, aif.Edge{FromID: "TAx", ToID: "L2"})
	if v4 := compare(t, pred, gold, evaluation.MetricGEDv4); v4.FullGED.GED != 0 {
		t.Errorf("0.4 は TA の違いを数えないはず: %v", v4.FullGED)
	}
	if v3 := compare(t, pred, gold, evaluation.MetricGEDv3); v3.FullGED.GED != 3 {
		t.Errorf("0.3 は余分な TA 1 個と辺 2 本を数えるはず（3.0）: %v", v3.FullGED)
	}
}

// 向き違いの辺のコストは「出さない」と同じで、「正解に対応の無い関係を出す」より 1.0 安い。
// 向き違いに「出さない」より得を与えない（向きを判定せずに出す戦略を報わない）ことを固定する。
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
