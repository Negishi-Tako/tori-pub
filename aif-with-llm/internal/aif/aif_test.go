package aif_test

import (
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

func loadGold(t *testing.T) aif.Graph {
	t.Helper()
	b, err := os.ReadFile("../../testdata/aif/fixture.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	g, err := aif.ParseNodeset("fixture", b)
	if err != nil {
		t.Fatalf("ParseNodeset: %v", err)
	}
	return g
}

func TestParseNodesetReadsAIFdbShape(t *testing.T) {
	g := loadGold(t)
	counts := g.CountByType()
	for _, tp := range []aif.NodeType{aif.TypeI, aif.TypeL, aif.TypeYA, aif.TypeTA} {
		if counts[tp] == 0 {
			t.Errorf("fixture should contain %s nodes, got %v", tp, counts)
		}
	}
	if len(g.Edges) == 0 {
		t.Fatal("fixture should contain edges")
	}
	for _, n := range g.NodesOfType(aif.TypeYA) {
		if n.Scheme == "" {
			t.Errorf("YA node %s has no scheme", n.ID)
		}
	}
}

// Gold preprocessing must drop only "L-nodes without edges" (duplicates from the recording tool) and
// edges not in the metamodel. If propositions or relations silently vanish, evaluation breaks.
func TestNormalizeKeepsPropositionsAndRelations(t *testing.T) {
	g := loadGold(t)
	before := g.CountByType()
	norm, rep := aif.Normalize(g)
	after := norm.CountByType()

	if after[aif.TypeI] != before[aif.TypeI] {
		t.Errorf("I-node が減った: %d -> %d", before[aif.TypeI], after[aif.TypeI])
	}
	if len(norm.Relations()) != len(g.Relations()) {
		t.Errorf("関係が減った: %d -> %d", len(g.Relations()), len(norm.Relations()))
	}
	if rep.DroppedIsolatedL == 0 {
		t.Error("QT30 の複製 L-node が落ちていない（前処理が効いていない）")
	}
	if after[aif.TypeL] >= before[aif.TypeL] {
		t.Errorf("孤立 L-node が落ちていない: %d -> %d", before[aif.TypeL], after[aif.TypeL])
	}
}

func TestNormalizedGoldSatisfiesMetamodel(t *testing.T) {
	norm, _ := aif.Normalize(loadGold(t))
	if vs := aif.Validate(norm); len(vs) > 0 {
		for _, v := range vs {
			t.Logf("violation: %s", v)
		}
		t.Errorf("正規化後のゴールドに %d 件のメタモデル違反がある", len(vs))
	}
}

func TestRelationsCollapseSNodes(t *testing.T) {
	norm, _ := aif.Normalize(loadGold(t))
	rels := norm.Relations()
	if len(rels) == 0 {
		t.Fatal("fixture should contain relations")
	}
	for _, r := range rels {
		if !r.Type.IsSNode() {
			t.Errorf("relation %s has non S-node type %s", r.SNodeID, r.Type)
		}
		if len(r.SrcIDs) == 0 || r.DstID == "" {
			t.Errorf("relation %s is not fully connected: %+v", r.SNodeID, r)
		}
	}
}

// Validate must not miss broken graphs.
func TestValidateRejectsDirectIToI(t *testing.T) {
	g := aif.Graph{
		ID: "x",
		Nodes: []aif.Node{
			{ID: "I0", Type: aif.TypeI, Text: "a"},
			{ID: "I1", Type: aif.TypeI, Text: "b"},
		},
		Edges: []aif.Edge{{FromID: "I0", ToID: "I1"}},
	}
	vs := aif.Validate(g)
	if len(vs) == 0 {
		t.Fatal("I-node 同士の直結を検出していない")
	}
}

// The baseline graph must be valid AIF and have no relations.
func TestBaselineGraphIsValidAndRelationFree(t *testing.T) {
	locs := []aif.LocutionInput{
		{Speaker: "A", Text: "schools should reopen"},
		{Speaker: "B", Text: "the infection rate is still high"},
	}
	g := aif.BaselineGraph("x", locs)
	if vs := aif.Validate(g); len(vs) > 0 {
		t.Errorf("基準線グラフが AIF として妥当でない: %v", vs)
	}
	if len(g.Relations()) != 0 {
		t.Errorf("基準線グラフに関係があってはならない: %d", len(g.Relations()))
	}
	if got := g.CountByType()[aif.TypeI]; got != 2 {
		t.Errorf("I-node は発話数と同じはず: %d", got)
	}
	if got := g.CountByType()[aif.TypeTA]; got != 1 {
		t.Errorf("TA は発話数 - 1 のはず: %d", got)
	}
}

// Every edge after normalization must carry a type name (GED counts matches using it).
// Edges from folded relations must match the number of relations (one per premise).
func TestLabeledEdges(t *testing.T) {
	norm, _ := aif.Normalize(loadGold(t))
	edges := norm.LabeledEdges()
	if len(edges) != len(norm.Edges) {
		t.Errorf("辺が落ちた: %d -> %d", len(norm.Edges), len(edges))
	}
	for _, e := range edges {
		if e.Kind == "" {
			t.Errorf("辺 %s->%s の種別が決まっていない", e.FromID, e.ToID)
		}
	}
	n := 0
	for _, r := range norm.Relations() {
		n += len(r.SrcIDs)
	}
	if got := len(norm.RelationEdges()); got != n {
		t.Errorf("関係を畳んだ辺は %d 本のはず: %d", n, got)
	}
}

// L-node order must be determined by the TA chain, not by JSON order.
func TestLocutionOrderFollowsTransitions(t *testing.T) {
	g := aif.BaselineGraph("x", []aif.LocutionInput{{Speaker: "A", Text: "a"}, {Speaker: "B", Text: "b"}, {Speaker: "C", Text: "c"}})
	slices.Reverse(g.Nodes)
	var got []string
	for _, n := range g.LocutionOrder() {
		got = append(got, n.ID)
	}
	if !slices.Equal(got, []string{"L0", "L1", "L2"}) {
		t.Errorf("TA の連鎖どおりに並ぶはず: %v", got)
	}
	if pos := g.PropositionOrder(); pos["I0"] != 0 || pos["I2"] != 2 {
		t.Errorf("命題は発話の位置を引き継ぐはず: %v", pos)
	}
}

// Thresholding on 5-level dependence must drop relations rung by rung (and remain valid AIF afterward).
// Mixing in 3-level confidence must not confuse the ladders. Relations without a self-report must not be dropped.
func TestGradeLadderAndFiltering(t *testing.T) {
	id := func(prefix string, i int) string { return prefix + strconv.Itoa(i) }
	g := aif.Graph{ID: "g"}
	for i := 0; i < 5; i++ {
		g.Nodes = append(g.Nodes,
			aif.Node{ID: id("I", 2*i), Type: aif.TypeI, Text: "claim " + strconv.Itoa(i)},
			aif.Node{ID: id("I", 2*i+1), Type: aif.TypeI, Text: "reason " + strconv.Itoa(i)},
			aif.Node{ID: id("RA", i), Type: aif.TypeRA, Scheme: aif.SchemeDefaultInference, Confidence: strconv.Itoa(5 - i)},
		)
		g.Edges = append(g.Edges,
			aif.Edge{FromID: id("I", 2*i+1), ToID: id("RA", i)},
			aif.Edge{FromID: id("RA", i), ToID: id("I", 2*i)})
	}
	ladder := g.GradeLadder()
	if len(ladder) != 5 || ladder[0] != "5" {
		t.Fatalf("5 段階の梯子を返すはず: %v", ladder)
	}
	for _, tc := range []struct {
		keep []string
		want int
	}{
		{[]string{"5"}, 1},
		{[]string{"5", "4"}, 2},
		{[]string{"5", "4", "3"}, 3},
		{[]string{"5", "4", "3", "2"}, 4},
	} {
		out := g.FilterRelationsByConfidence(tc.keep...)
		if got := len(out.Relations()); got != tc.want {
			t.Errorf("keep=%v なら %d 本残るはず: %d", tc.keep, tc.want, got)
		}
		if v := aif.Validate(out); len(v) > 0 {
			t.Errorf("絞り込んだグラフも AIF として妥当であるはず: %v", v)
		}
	}
	// A 3-level vocabulary returns a 3-rung ladder.
	h := aif.Graph{ID: "h", Nodes: []aif.Node{
		{ID: "I0", Type: aif.TypeI, Text: "a"},
		{ID: "I1", Type: aif.TypeI, Text: "b"},
		{ID: "RA0", Type: aif.TypeRA, Confidence: "high"},
	}, Edges: []aif.Edge{{FromID: "I1", ToID: "RA0"}, {FromID: "RA0", ToID: "I0"}}}
	if l := h.GradeLadder(); len(l) != 3 || l[0] != "high" {
		t.Errorf("3 値の梯子を返すはず: %v", l)
	}
	h.Nodes[2].Confidence = ""
	if got := len(h.FilterRelationsByConfidence("5").Relations()); got != 1 {
		t.Errorf("確信度の申告が無ければ落とさないはず: %d 本", got)
	}
}
