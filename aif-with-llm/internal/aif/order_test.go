package aif_test

import (
	"slices"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// orderGraph は L-node を ids の順（JSON の並び）に置き、tas の (from, to) で TA を張る。
// stamps が非 nil なら L-node にタイムスタンプを付ける。
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

// TA が合流するとき（A→C と B→C）、合流先は両方の後に来ること。
// 以前の実装は A→C だけをたどって C を A の直後に置き、B を末尾に回していた
// （test の 17919: 司会の質問と "There are thousands of head teachers…" が同じ応答に TA を張っている）。
func TestLocutionOrderRespectsAllTransitions(t *testing.T) {
	g := orderGraph([]string{"A", "B", "C"}, [][2]string{{"A", "C"}, {"B", "C"}}, nil)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("合流先 C は A・B の両方の後に来るはず: %v", got)
	}
}

// TA が分岐するとき（A→B と A→C、B→C）も、すべての TA の向きを守ること。
func TestLocutionOrderHandlesBranching(t *testing.T) {
	g := orderGraph([]string{"A", "C", "B"}, [][2]string{{"A", "C"}, {"A", "B"}, {"B", "C"}}, nil)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("A→B→C の順のはず: %v", got)
	}
}

// TA で順序の決まらない発話どうしは、タイムスタンプで並べること（同じなら JSON の並び）。
func TestLocutionOrderBreaksTiesByTimestamp(t *testing.T) {
	stamps := map[string]string{"A": "2020-05-28 19:24:31", "X": "2020-05-28 19:24:30", "B": "2020-05-28 19:24:32"}
	g := orderGraph([]string{"A", "B", "X"}, [][2]string{{"A", "B"}}, stamps)
	if got := idsOf(g.LocutionOrder()); !slices.Equal(got, []string{"X", "A", "B"}) {
		t.Errorf("TA の無い X はタイムスタンプで先頭に来るはず: %v", got)
	}
}

// TA が循環していても全 L-node を 1 回ずつ返すこと（無限ループ・欠落にしない）。
func TestLocutionOrderSurvivesCycles(t *testing.T) {
	g := orderGraph([]string{"A", "B", "C"}, [][2]string{{"A", "B"}, {"B", "C"}, {"C", "A"}}, nil)
	got := idsOf(g.LocutionOrder())
	slices.Sort(got)
	if !slices.Equal(got, []string{"A", "B", "C"}) {
		t.Errorf("循環していても全 L-node を 1 回ずつ返すはず: %v", got)
	}
}

// 1 つの I-node が複数の発話に anchor されているとき、最も早い発話の位置を採ること
// （ノードの並び順に依存しない）。
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

// AnchorNodes / AnchorEdges（GED 0.4 の全体グラフ）には TA とそれに付く辺が入らないこと。
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
