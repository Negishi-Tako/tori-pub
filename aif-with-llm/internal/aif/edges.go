package aif

import "strconv"

// LabeledEdge is an edge labeled with the meaning determined by its endpoint node types.
//
// AIF edges have no type; their meaning is determined by the endpoint types. Counting edge matches
// in GED requires the meaning to match as well, so names are assigned mechanically from the endpoint types here.
// No information is added or lost.
type LabeledEdge struct {
	FromID string
	ToID   string
	Kind   string
}

// Edge kind names. Values are used only for GED matching (same value means an edge with the same meaning).
const (
	edgeAnchors       = "ANCHORS"        // L / TA -> YA
	edgeIllocutes     = "ILLOCUTES"      // YA -> I / S-node
	edgePremiseOf     = "PREMISE_OF"     // I -> RA
	edgeConcludes     = "CONCLUDES"      // RA -> I / S-node
	edgeConflictsVia  = "CONFLICTS_VIA"  // I -> CA
	edgeConflictsWith = "CONFLICTS_WITH" // CA -> I / S-node
	edgeRephrasesVia  = "REPHRASES_VIA"  // I -> MA
	edgeRephrasesTo   = "REPHRASES_TO"   // MA -> I / S-node
	edgeNext          = "NEXT"           // L -> TA
	edgeLeadsTo       = "LEADS_TO"       // TA -> L
)

// relationEdgeKind is the kind name of a relation (S-node) folded into a single edge between propositions.
var relationEdgeKind = map[NodeType]string{
	TypeRA: "support",
	TypeCA: "attack",
	TypeMA: "rephrase",
}

func edgeKind(from, to NodeType) (string, bool) {
	switch {
	case from == TypeL && to == TypeYA, from == TypeTA && to == TypeYA:
		return edgeAnchors, true
	case from == TypeYA:
		return edgeIllocutes, true
	case from == TypeI && to == TypeRA:
		return edgePremiseOf, true
	case from == TypeRA:
		return edgeConcludes, true
	case from == TypeI && to == TypeCA:
		return edgeConflictsVia, true
	case from == TypeCA:
		return edgeConflictsWith, true
	case from == TypeI && to == TypeMA:
		return edgeRephrasesVia, true
	case from == TypeMA:
		return edgeRephrasesTo, true
	case from == TypeL && to == TypeTA:
		return edgeNext, true
	case from == TypeTA && to == TypeL:
		return edgeLeadsTo, true
	}
	return "", false
}

// LabeledEdges returns all edges with kind names, keeping every node.
func (g Graph) LabeledEdges() []LabeledEdge {
	return g.labeledEdges(nil)
}

// labeledEdges returns, with kind names, only the edges between nodes in keep. A nil keep means all edges.
func (g Graph) labeledEdges(keep map[string]bool) []LabeledEdge {
	ix := NewIndex(g)
	out := make([]LabeledEdge, 0, len(g.Edges))
	for _, e := range g.Edges {
		if keep != nil && (!keep[e.FromID] || !keep[e.ToID]) {
			continue
		}
		if k, ok := edgeKind(ix.Type(e.FromID), ix.Type(e.ToID)); ok {
			out = append(out, LabeledEdge{FromID: e.FromID, ToID: e.ToID, Kind: k})
		}
	}
	return out
}

// RelationEdges folds relations (RA/CA/MA) into edges between propositions.
// An RA with multiple premises yields one edge per premise. These are the edges of the proposition graph (I-nodes and relations only).
func (g Graph) RelationEdges() []LabeledEdge {
	var out []LabeledEdge
	for _, r := range g.Relations() {
		for _, src := range r.SrcIDs {
			out = append(out, LabeledEdge{FromID: src, ToID: r.DstID, Kind: relationEdgeKind[r.Type]})
		}
	}
	return out
}

// dialogueKeep is the set of nodes kept in DialogueNodes / DialogueEdges (withTA=true) and
// AnchorNodes / AnchorEdges (withTA=false).
//
// It keeps L / I (and TA if withTA) and only **YAs anchored to an utterance that illocute a proposition**.
// The following are dropped, each to exclude from GED "things that are not about LLM quality":
//
//	S-nodes (RA/CA/MA)      … relations are folded into one edge, so they are not counted as nodes.
//	                          Counting them as both nodes and edges would penalize the same error twice.
//	YAs anchored to TAs     … on the pred side their count is fixed by the harness construction rule
//	                          "relations between non-adjacent utterances have no TA, so no YA is created"
//	                          (annotate.go), so differences from the gold do not reflect LLM errors.
//	                          Illocution accuracy is measured separately by evaluation.ForceScore.
//	TA (withTA=false)       … dropped for the same reason. Pred-side TAs are fixed by the construction rule
//	                          "one between each pair of adjacent utterances", while gold TAs branch and also link
//	                          non-adjacent utterances. Even for the baseline given the gold segmentation, over 124 test items
//	                          445 of 1,732 gold TAs and 492 of 1,779 generated TAs go unmatched, and those TAs and
//	                          their edges alone make up about 18% of the 0.3 overall GED denominator (irreducible whatever the LLM answers).
func (g Graph) dialogueKeep(withTA bool) map[string]bool {
	keep := make(map[string]bool, len(g.Nodes))
	for _, n := range g.Nodes {
		switch n.Type {
		case TypeL, TypeI:
			keep[n.ID] = true
		case TypeTA:
			keep[n.ID] = withTA
		}
	}
	for _, il := range g.Illocutions() {
		if il.AnchorType == TypeL && il.TargetType == TypeI {
			keep[il.YAID] = true
		}
	}
	return keep
}

func (g Graph) nodesIn(keep map[string]bool) []Node {
	out := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if keep[n.ID] {
			out = append(out, n)
		}
	}
	return out
}

// DialogueNodes returns the dialogue-layer nodes (L / TA / I / utterance YAs) in input order (the overall graph of GED 0.3).
func (g Graph) DialogueNodes() []Node { return g.nodesIn(g.dialogueKeep(true)) }

// DialogueEdges is the dialogue-layer edges plus relations folded into single edges between propositions (RelationEdges).
//
// The difference from LabeledEdges is that one relation becomes "one edge" rather than
// "1 S-node + 2 edges (+ 1 YA + 2 edges)". Otherwise a granularity mismatch remains: one type
// confusion costs 0.75 at the proposition level but 4.75 overall (the edges on both sides miss).
// Folding makes the cost of a relation error the same regardless of granularity.
func (g Graph) DialogueEdges() []LabeledEdge {
	return append(g.labeledEdges(g.dialogueKeep(true)), g.RelationEdges()...)
}

// AnchorNodes is DialogueNodes without TAs (L / I / utterance YAs; the overall graph of GED 0.4).
func (g Graph) AnchorNodes() []Node { return g.nodesIn(g.dialogueKeep(false)) }

// AnchorEdges is the edges among AnchorNodes (L→YA→I) plus the folded relation edges.
func (g Graph) AnchorEdges() []LabeledEdge {
	return append(g.labeledEdges(g.dialogueKeep(false)), g.RelationEdges()...)
}

// BaselineGraph builds the "no LLM" lower-bound AIF graph.
//
// It treats each given utterance as a proposition, anchors all of them with Asserting,
// and adds no relations (RA/CA/MA). Normalized GED gets smaller as the denominator grows, so
// without showing what value a "do nothing" system gets, we cannot tell whether the LLM's value
// is good or bad. It serves as that baseline.
func BaselineGraph(id string, locs []LocutionInput) Graph {
	g := Graph{ID: id}
	var prev string
	for i, l := range locs {
		lid, yid, iid := idxID("L", i), idxID("YA", i), idxID("I", i)
		g.Nodes = append(g.Nodes,
			Node{ID: lid, Type: TypeL, Text: l.Speaker + " : " + l.Text, Speaker: l.Speaker},
			Node{ID: yid, Type: TypeYA, Text: "Asserting", Scheme: "Asserting"},
			Node{ID: iid, Type: TypeI, Text: l.Text},
		)
		g.Edges = append(g.Edges, Edge{FromID: lid, ToID: yid}, Edge{FromID: yid, ToID: iid})
		if prev != "" {
			ta := idxID("TA", i-1)
			g.Nodes = append(g.Nodes, Node{ID: ta, Type: TypeTA, Scheme: SchemeDefaultTransition})
			g.Edges = append(g.Edges, Edge{FromID: prev, ToID: ta}, Edge{FromID: ta, ToID: lid})
		}
		prev = lid
	}
	out, _ := Normalize(g)
	return out
}

func idxID(prefix string, i int) string {
	return prefix + strconv.Itoa(i)
}
