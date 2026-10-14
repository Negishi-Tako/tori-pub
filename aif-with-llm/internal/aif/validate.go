package aif

import (
	"fmt"
	"sort"
	"strings"
)

// Violation is a single AIF metamodel violation.
type Violation struct {
	NodeID string `json:"node_id,omitempty"`
	Rule   string `json:"rule"`
	Detail string `json:"detail"`
}

func (v Violation) String() string {
	if v.NodeID == "" {
		return fmt.Sprintf("%s: %s", v.Rule, v.Detail)
	}
	return fmt.Sprintf("%s (node %s): %s", v.Rule, v.NodeID, v.Detail)
}

// Validate checks the metamodel constraints of the AIF core ontology + IAT.
//
// Since we claim "strict AIF", generated graphs are required to pass this check.
// The constraints are as follows (Chesñevar et al. 2006 / Budzynska & Reed 2011 /
// the AIFdb schema):
//
//	R1  I-nodes are never connected directly (relations always go through an S-node)
//	R2  an RA has one or more premise I-nodes and exactly one conclusion I-node
//	R3  a CA has exactly one attacker and exactly one target
//	R4  an MA has exactly one source and exactly one target of the rephrase
//	R5  a YA is anchored to exactly one L or TA and illocutes exactly one I or S-node
//	R6  a TA is a transition from L to L
//	R7  an L-node connects directly only to L / TA / YA
//	R8  a YA's scheme is in the IAT set of illocutionary forces
//	R9  node types are one of the 7 types, and both ends of every edge exist
//	R10 S-nodes do not form self-loops (premise and conclusion are not the same I-node)
func Validate(g Graph) []Violation {
	ix := NewIndex(g)
	var vs []Violation
	add := func(id, rule, detail string) { vs = append(vs, Violation{NodeID: id, Rule: rule, Detail: detail}) }

	seen := map[string]bool{}
	for _, n := range g.Nodes {
		if !n.Type.Valid() {
			add(n.ID, "R9", fmt.Sprintf("未知のノード種別 %q", n.Type))
		}
		if seen[n.ID] {
			add(n.ID, "R9", "ノード ID が重複している")
		}
		seen[n.ID] = true
	}
	for _, e := range g.Edges {
		if _, ok := ix.Node(e.FromID); !ok {
			add(e.FromID, "R9", "辺の始点ノードが存在しない")
		}
		if _, ok := ix.Node(e.ToID); !ok {
			add(e.ToID, "R9", "辺の終点ノードが存在しない")
		}
	}

	for _, n := range g.Nodes {
		switch n.Type {
		case TypeI:
			for _, id := range ix.Out(n.ID) {
				if ix.Type(id) == TypeI {
					add(n.ID, "R1", "I-node から I-node へ直接辺を張っている")
				}
			}
		case TypeRA, TypeCA, TypeMA:
			var prem, conc int
			for _, id := range ix.In(n.ID) {
				if ix.Type(id) == TypeI {
					prem++
				}
			}
			var dst string
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t == TypeI || t.IsSNode() {
					conc++
					dst = id
				}
			}
			rule := map[NodeType]string{TypeRA: "R2", TypeCA: "R3", TypeMA: "R4"}[n.Type]
			switch {
			case prem == 0:
				add(n.ID, rule, string(n.Type)+" に前提側の I-node が無い")
			case n.Type != TypeRA && prem != 1:
				add(n.ID, rule, fmt.Sprintf("%s の入力は 1 本でなければならない（%d 本）", n.Type, prem))
			}
			if conc != 1 {
				add(n.ID, rule, fmt.Sprintf("%s の出力はちょうど 1 本でなければならない（%d 本）", n.Type, conc))
			}
			if dst != "" {
				for _, id := range ix.In(n.ID) {
					if id == dst {
						add(n.ID, "R10", "前提と結論が同じノードを指している")
					}
				}
			}
		case TypeYA:
			var anchors, targets int
			for _, id := range ix.In(n.ID) {
				if t := ix.Type(id); t == TypeL || t == TypeTA {
					anchors++
				}
			}
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t == TypeI || t.IsSNode() {
					targets++
				}
			}
			if anchors != 1 {
				add(n.ID, "R5", fmt.Sprintf("YA の anchor は L か TA ちょうど 1 つ（%d 個）", anchors))
			}
			if targets != 1 {
				add(n.ID, "R5", fmt.Sprintf("YA が illocute する対象はちょうど 1 つ（%d 個）", targets))
			}
			if !ValidForce(n.Scheme) {
				add(n.ID, "R8", fmt.Sprintf("未知の発話行為 %q", n.Scheme))
			}
		case TypeTA:
			var from, to int
			for _, id := range ix.In(n.ID) {
				if ix.Type(id) == TypeL {
					from++
				}
			}
			for _, id := range ix.Out(n.ID) {
				if ix.Type(id) == TypeL {
					to++
				}
			}
			if from != 1 || to != 1 {
				add(n.ID, "R6", fmt.Sprintf("TA は L から L への遷移（in=%d out=%d）", from, to))
			}
		case TypeL:
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t != TypeYA && t != TypeTA {
					add(n.ID, "R7", fmt.Sprintf("L-node から %s へ直接辺を張っている", t))
				}
			}
		}
	}
	return vs
}

// NormalizeReport is a breakdown of what Normalize dropped.
// It is returned rather than swallowed so the report can state what preprocessing discarded.
type NormalizeReport struct {
	DroppedEdges     int      `json:"dropped_edges"`
	DroppedIsolatedL int      `json:"dropped_isolated_l"`
	DroppedOther     int      `json:"dropped_other"`
	DroppedNodeIDs   []string `json:"dropped_node_ids,omitempty"`
	Violations       int      `json:"violations"`
}

// Normalize puts gold / generated graphs into a comparable form.
//
// Dropped:
//   - edges whose endpoints do not exist
//   - edges of shapes the AIF metamodel does not have (L→I directly, etc.; annotation noise)
//   - isolated nodes (degree 0). QT30 contains many edgeless L-nodes from the recording tool,
//     "speaker name: a copy of the previous utterance", and keeping them would let the
//     L-node count alone move GED.
//
// Not dropped: everything else. Nothing is thinned out based on content quality.
func Normalize(g Graph) (Graph, NormalizeReport) {
	ix := NewIndex(g)
	var rep NormalizeReport

	edges := make([]Edge, 0, len(g.Edges))
	deg := map[string]int{}
	seenEdge := map[[2]string]bool{}
	for _, e := range g.Edges {
		ft, tt := ix.Type(e.FromID), ix.Type(e.ToID)
		if ft == "" || tt == "" || !allowedEdge(ft, tt) || e.FromID == e.ToID || seenEdge[[2]string{e.FromID, e.ToID}] {
			rep.DroppedEdges++
			continue
		}
		seenEdge[[2]string{e.FromID, e.ToID}] = true
		edges = append(edges, e)
		deg[e.FromID]++
		deg[e.ToID]++
	}

	nodes := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if deg[n.ID] == 0 {
			if n.Type == TypeL {
				rep.DroppedIsolatedL++
			} else {
				rep.DroppedOther++
			}
			rep.DroppedNodeIDs = append(rep.DroppedNodeIDs, n.ID)
			continue
		}
		n.Text = strings.TrimSpace(n.Text)
		nodes = append(nodes, n)
	}
	sort.Strings(rep.DroppedNodeIDs)

	out := Graph{ID: g.ID, Nodes: nodes, Edges: edges}
	rep.Violations = len(Validate(out))
	return out, rep
}

// allowedEdge reports whether (source type, target type) is a pair the AIF metamodel allows.
func allowedEdge(from, to NodeType) bool {
	switch from {
	case TypeI:
		return to.IsSNode()
	case TypeRA, TypeCA, TypeMA:
		return to == TypeI || to.IsSNode()
	case TypeL:
		return to == TypeYA || to == TypeTA
	case TypeYA:
		return to == TypeI || to.IsSNode()
	case TypeTA:
		return to == TypeL || to == TypeYA
	}
	return false
}
