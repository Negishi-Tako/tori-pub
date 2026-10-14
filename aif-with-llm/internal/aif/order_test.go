package aif_test

import (
	"slices"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// orderGraph places L-nodes in the order of ids (the JSON order) and adds TAs for each (from, to) in tas.
// If stamps is non-nil, L-nodes get timestamps.
func orderGraph(ids []string, tas [][2]string, stamps map[string]string) aif.Graph {
	g := aif.Graph{ID: "o"}
	for _, id := range ids {
		g.Nodes = append(g.Nodes, aif.Node{ID: id, Type: aif.TypeL, Text: "A : " + id, Timestamp: stamps[id]})
	}
	for i, t := range tas {
		ta := "TA" + string(rune('a'+i))
		g.Nodes = append(g.Nodes, aif.Node{ID: ta, Type: aif.TypeTA, Scheme: aif.SchemeDefaultTransition})
		g.Edges = append(g.Edges, aif.Edge{FromID: t[0], ToID: ta}, aif.Edge{FromID: ta, ToID: t[1]})
	}
	return g
}

func idsOf(nodes []aif.Node) []string {
	var out []string
	for _, n := range nodes {
		out = append(out, n.ID)
	}
	return out
}

// When TAs merge (A→C and B→C), the merge target must come after both.
// The previous implementation followed only A→C, put C right after A and pushed B to the end
// (test 17919: the moderator's question and "There are thousands of head teachers…" both have TAs to the same reply).
func TestLocutionOrderRespectsAllTransitions(t *testing.T) {
	g := orderGraph([]string{"A", "B", "C"}, [][2]string{{"A", "C"}, {"B", "C"}}, nil)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("合流先 C は A・B の両方の後に来るはず: %v", got)
	}
}

// When TAs branch (A→B and A→C, B→C), the direction of every TA must still be respected.
func TestLocutionOrderHandlesBranching(t *testing.T) {
	g := orderGraph([]string{"A", "C", "B"}, [][2]string{{"A", "C"}, {"A", "B"}, {"B", "C"}}, nil)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("A→B→C の順のはず: %v", got)
	}
}

// Utterances not ordered by TAs must be ordered by timestamp (then by JSON order if equal).
func TestLocutionOrderBreaksTiesByTimestamp(t *testing.T) {
	stamps := map[string]string{"A": "2020-05-28 19:24:31", "X": "2020-05-28 19:24:30", "B": "2020-05-28 19:24:32"}
	g := orderGraph([]string{"A", "B", "X"}, [][2]string{{"A", "B"}}, stamps)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"X", "A", "B"}) {
		t.Errorf("TA の無い X はタイムスタンプで先頭に来るはず: %v", got)
	}
}

// Even when TAs form a cycle, every L-node must be returned exactly once (no infinite loop or omission).
func TestLocutionOrderSurvivesCycles(t *testing.T) {
	g := orderGraph([]string{"A", "B", "C"}, [][2]string{{"A", "B"}, {"B", "C"}, {"C", "A"}}, nil)
	got := idsOf(g.LocutionOrder())
	slices.Sort(got)
	if !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("循環していても全 L-node を 1 回ずつ返すはず: %v", got)
	}
}

// When one I-node is anchored to multiple utterances, the earliest utterance's position must be used
// (independent of node order).
func TestPropositionOrderUsesEarliestAnchor(t *testing.T) {
	for _, reverse := range []bool{false, true} {
		g := orderGraph([]string{"L0", "L1", "L2"}, [][2]string{{"L0", "L1"}, {"L1", "L2"}}, nil)
		ya := []aif.Node{
			{ID: "YA0", Type: aif.TypeYA, Scheme: "Asserting"},
			{ID: "YA2", Type: aif.TypeYA, Scheme: "Restating"},
		}
		if reverse {
			slices.Reverse(ya)
		}
		g.Nodes = append(g.Nodes, ya...)
		g.Nodes = append(g.Nodes, aif.Node{ID: "I", Type: aif.TypeI, Text: "p"})
		g.Edges = append(g.Edges,
			aif.Edge{FromID: "L0", ToID: "YA0"}, aif.Edge{FromID: "YA0", ToID: "I"},
			aif.Edge{FromID: "L2", ToID: "YA2"}, aif.Edge{FromID: "YA2", ToID: "I"})
		if pos := g.PropositionOrder()["I"]; pos != 0 {
			t.Errorf("reverse=%v: 最も早い発話の位置 0 のはず: %d", reverse, pos)
		}
	}
}

// AnchorNodes / AnchorEdges (the overall graph of GED 0.4) must not include TAs or their edges.
func TestAnchorGraphExcludesTransitions(t *testing.T) {
	g := aif.BaselineGraph("b", []aif.LocutionInput{{Speaker: "A", Text: "x"}, {Speaker: "B", Text: "y"}})
	for _, n := range g.AnchorNodes() {
		if n.Type == aif.TypeTA {
			t.Errorf("TA が残っている: %v", n)
		}
	}
	ix := aif.NewIndex(g)
	for _, e := range g.AnchorEdges() {
		if ix.Type(e.FromID) == aif.TypeTA || ix.Type(e.ToID) == aif.TypeTA {
			t.Errorf("TA の辺が残っている: %+v", e)
		}
	}
	if got, want := len(g.AnchorNodes()), len(g.DialogueNodes())-len(g.Transitions()); got != want {
		t.Errorf("DialogueNodes から TA だけを除いた数のはず: %d vs %d", got, want)
	}
}
