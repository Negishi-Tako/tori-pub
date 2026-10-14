// Package aif handles a strict canonical representation of AIF (Argument Interchange Format).
//
// The representation follows only the AIF core ontology (Chesñevar et al. 2006) and the IAT
// (Inference Anchoring Theory: Budzynska & Reed 2011) metamodel, with no custom extensions,
// because comparing against AIFdb's gold annotations requires "plain AIF".
//
// Dependency direction: aif depends only on llm / prompts.
// Evaluation (GED) lives in internal/evaluation (one-way: evaluation -> aif).
package aif

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// NodeType is an AIF node type. It is closed over 7 types: the AIF core ontology's I-node and
// S-nodes (RA/CA/MA), plus IAT's L-node, YA-node and TA-node.
// Do not add more (the moment you do, it is no longer "plain AIF").
type NodeType string

const (
	TypeI  NodeType = "I"  // Information node (proposition)
	TypeL  NodeType = "L"  // Locution (utterance; IAT)
	TypeRA NodeType = "RA" // Rule Application (inference)
	TypeCA NodeType = "CA" // Conflict Application (conflict)
	TypeMA NodeType = "MA" // Preference/Rephrase Application (rephrase)
	TypeYA NodeType = "YA" // Illocutionary Anchoring (illocution; IAT)
	TypeTA NodeType = "TA" // Transition Application (utterance transition; IAT)
)

// IsSNode reports whether t is RA/CA/MA (an inference, conflict or rephrase S-node).
func (t NodeType) IsSNode() bool { return t == TypeRA || t == TypeCA || t == TypeMA }

// Valid reports whether t is one of the 7 AIF types.
func (t NodeType) Valid() bool {
	switch t {
	case TypeI, TypeL, TypeRA, TypeCA, TypeMA, TypeYA, TypeTA:
		return true
	}
	return false
}

// Default schemes. These take the defaults used by AIFdb annotations as-is.
const (
	SchemeDefaultInference  = "Default Inference"
	SchemeDefaultConflict   = "Default Conflict"
	SchemeDefaultRephrase   = "Default Rephrase"
	SchemeDefaultTransition = "Default Transition"
	SchemeDefaultIllocuting = "Default Illocuting"
)

// IllocutionaryForces is the set of illocutionary forces allowed on YA-nodes.
// It follows the IAT standard set (Budzynska & Reed 2011 / the QT30 annotation
// guidelines).
var IllocutionaryForces = []string{
	"Asserting",
	"Arguing",
	"Restating",
	"Analysing",
	"Agreeing",
	"Disagreeing",
	"PureQuestioning",
	"AssertiveQuestioning",
	"RhetoricalQuestioning",
	"Challenging",
	SchemeDefaultIllocuting,
}

// LocutionForces are the illocutionary forces usable for YAs anchored to an L-node (i.e. forces
// that put forward a single utterance). Arguing / Restating / Analysing carry "relations between
// utterances" and are anchored to transitions (TA), so they are not included here.
//
// Counting QT30's gold (the 6 subcorpora fetched by this project, over 5,000 YAs),
// L→I anchors are Asserting 90.7% / PureQuestioning 5.9% / AssertiveQuestioning 1.1% /
// Agreeing 0.8% / RhetoricalQuestioning 0.7%, with Arguing only 0.1%.
// Meanwhile TA→RA anchors are Arguing 97.6%, TA→MA Restating 86.1%,
// and TA→CA Disagreeing 91.9%. This division is how IAT is applied in practice, and
// annotators (human or LLM) must not be shown both sets at the same time.
var LocutionForces = []string{
	"Asserting",
	"PureQuestioning",
	"AssertiveQuestioning",
	"RhetoricalQuestioning",
	"Challenging",
	"Agreeing",
	"Disagreeing",
	SchemeDefaultIllocuting,
}

// TransitionForces are the illocutionary forces usable for YAs anchored to a TA (i.e. forces that illocute an S-node).
var TransitionForces = []string{
	"Arguing",
	"Restating",
	"Disagreeing",
	"Challenging",
	"Analysing",
	SchemeDefaultIllocuting,
}

// ValidForce reports whether f is a valid IAT illocutionary force.
func ValidForce(f string) bool { return slices.Contains(IllocutionaryForces, f) }

// Node is a single AIF node.
type Node struct {
	ID   string   `json:"nodeID"`
	Type NodeType `json:"type"`
	Text string   `json:"text"`
	// Scheme is the scheme name of an S-node / YA / TA ("Default Inference", etc.).
	// Empty for I-nodes and L-nodes.
	Scheme string `json:"scheme,omitempty"`
	// Speaker is the speaker of an L-node. AIFdb stores it as locutions[].personID,
	// but matching uses the display name, so it is normalized and kept here.
	Speaker   string `json:"speaker,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
	// Confidence is the grade self-reported by the LLM, meaningful only for S-nodes.
	// It holds either a 3-level confidence ("high" / "medium" / "low")
	// or a 5-level dependence ("5"–"1") (GradeLadder tells which).
	// Empty means no self-report (i.e. not subject to filtering).
	// Attaching it at generation time lets the threshold be moved later to draw a P/R curve,
	// i.e. without calling the LLM again for every threshold.
	Confidence string `json:"confidence,omitempty"`
}

// Edge is a single AIF edge. AIF edges have no type (meaning is determined by the endpoint node types).
type Edge struct {
	FromID string `json:"fromID"`
	ToID   string `json:"toID"`
}

// Graph is the AIF graph of one nodeset.
type Graph struct {
	// ID is the nodeset identifier (AIFdb's nodesetID, or the generator-side identifier).
	ID    string `json:"id"`
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// Relation is an S-node (RA/CA/MA) folded into a single relation.
// Used when computing GED per proposition and when writing evidence.
type Relation struct {
	SNodeID string   `json:"snode_id"`
	Type    NodeType `json:"type"`
	// SrcIDs are the premise-side I-nodes (RA may have several).
	SrcIDs []string `json:"src_ids"`
	// DstID is the conclusion / conflicted / rephrased I-node.
	DstID string `json:"dst_id"`
	// Force is the scheme of the YA that illocutes the S-node (empty if none).
	Force string `json:"force,omitempty"`
}

// Index looks up nodes and adjacency by node ID.
// Linear search in every Graph method would be O(n^2), so it is built once up front.
type Index struct {
	byID map[string]Node
	out  map[string][]string
	in   map[string][]string
}

// NewIndex builds the index for g.
func NewIndex(g Graph) *Index {
	ix := &Index{
		byID: make(map[string]Node, len(g.Nodes)),
		out:  make(map[string][]string, len(g.Nodes)),
		in:   make(map[string][]string, len(g.Nodes)),
	}
	for _, n := range g.Nodes {
		ix.byID[n.ID] = n
	}
	for _, e := range g.Edges {
		ix.out[e.FromID] = append(ix.out[e.FromID], e.ToID)
		ix.in[e.ToID] = append(ix.in[e.ToID], e.FromID)
	}
	return ix
}

func (ix *Index) Node(id string) (Node, bool) { n, ok := ix.byID[id]; return n, ok }
func (ix *Index) Out(id string) []string      { return ix.out[id] }
func (ix *Index) In(id string) []string       { return ix.in[id] }

// Type returns the node type of id, or the empty string if it does not exist.
func (ix *Index) Type(id string) NodeType {
	if n, ok := ix.byID[id]; ok {
		return n.Type
	}
	return ""
}

// NodesOfType returns the nodes of type t in input order.
func (g Graph) NodesOfType(t NodeType) []Node {
	out := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if n.Type == t {
			out = append(out, n)
		}
	}
	return out
}

// CountByType returns the number of nodes per type.
func (g Graph) CountByType() map[NodeType]int {
	m := make(map[NodeType]int, 7)
	for _, n := range g.Nodes {
		m[n.Type]++
	}
	return m
}

// Relations folds RA/CA/MA into relations. Incoming S-node edges from I-nodes are the
// premises, and outgoing edges to I-nodes are the conclusion.
// In AIF a CA can attack an RA (undercut), but this does not occur in QT30, so only
// I-node-to-I-node relations are handled and the rest are dropped (Validate reports how many).
func (g Graph) Relations() []Relation {
	ix := NewIndex(g)
	out := make([]Relation, 0, 16)
	for _, n := range g.Nodes {
		if !n.Type.IsSNode() {
			continue
		}
		var srcs []string
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeI {
				srcs = append(srcs, id)
			}
		}
		var dst string
		for _, id := range ix.Out(n.ID) {
			if ix.Type(id) == TypeI {
				dst = id
				break
			}
		}
		if len(srcs) == 0 || dst == "" {
			continue
		}
		sort.Strings(srcs)
		r := Relation{SNodeID: n.ID, Type: n.Type, SrcIDs: srcs, DstID: dst}
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeYA {
				ya, _ := ix.Node(id)
				r.Force = ya.Scheme
				break
			}
		}
		out = append(out, r)
	}
	return out
}

// Illocution pairs an L-node (or TA) with the content linked to it via a YA.
type Illocution struct {
	YAID string `json:"ya_id"`
	// AnchorID is the L-node or TA the YA is anchored to.
	AnchorID   string   `json:"anchor_id"`
	AnchorType NodeType `json:"anchor_type"`
	// TargetID is the I-node or S-node the YA illocutes.
	TargetID   string   `json:"target_id"`
	TargetType NodeType `json:"target_type"`
	Force      string   `json:"force"`
}

// Illocutions folds YAs into (anchor, target) pairs.
func (g Graph) Illocutions() []Illocution {
	ix := NewIndex(g)
	out := make([]Illocution, 0, 16)
	for _, n := range g.Nodes {
		if n.Type != TypeYA {
			continue
		}
		il := Illocution{YAID: n.ID, Force: n.Scheme}
		for _, id := range ix.In(n.ID) {
			if t := ix.Type(id); t == TypeL || t == TypeTA {
				il.AnchorID, il.AnchorType = id, t
				break
			}
		}
		for _, id := range ix.Out(n.ID) {
			if t := ix.Type(id); t == TypeI || t.IsSNode() {
				il.TargetID, il.TargetType = id, t
				break
			}
		}
		if il.TargetID == "" {
			continue
		}
		out = append(out, il)
	}
	return out
}

// Transition is a TA as a (from L, to L) pair.
type Transition struct {
	TAID   string `json:"ta_id"`
	FromID string `json:"from_id"`
	ToID   string `json:"to_id"`
}

// Transitions returns the L→TA→L transitions.
func (g Graph) Transitions() []Transition {
	ix := NewIndex(g)
	out := make([]Transition, 0, 16)
	for _, n := range g.Nodes {
		if n.Type != TypeTA {
			continue
		}
		var from, to string
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeL {
				from = id
				break
			}
		}
		for _, id := range ix.Out(n.ID) {
			if ix.Type(id) == TypeL {
				to = id
				break
			}
		}
		if from == "" || to == "" {
			continue
		}
		out = append(out, Transition{TAID: n.ID, FromID: from, ToID: to})
	}
	return out
}

// LocutionOrder returns the L-nodes in chronological order.
//
// Each TA (L→TA→L) is treated as an ordering constraint "earlier utterance → later utterance",
// and an order satisfying all constraints (a topological order) is returned. Utterances not ordered
// by the constraints are ordered by timestamp, then by input order. When TAs form a cycle,
// the earliest remaining utterance is emitted to break it (no infinite loop).
//
// QT30's gold TAs are not a single chain. In the 124 test nodesets, 194 L-nodes have
// 2 or more outgoing TAs, and 412 TAs connect utterances that are not chronologically adjacent.
// The previous implementation "followed only the first TA from each L and appended unreached chains
// at the end", pushing utterances to the end at every branch. Checked against positions in the episode
// transcripts (1,642 adjacent pairs whose positions could be uniquely identified), the old implementation
// reversed 81 pairs and this one reverses 28. Only 3 of 1,489 TAs point backward in the transcript, so using
// TAs as ordering constraints is sound. Timestamps are annotation creation times, and utterances added later
// lag behind, so they are only used as a fallback.
func (g Graph) LocutionOrder() []Node {
	locs := g.NodesOfType(TypeL)
	idx := make(map[string]int, len(locs))
	for i, n := range locs {
		idx[n.ID] = i
	}
	succ := map[string][]string{}
	indeg := map[string]int{}
	for _, t := range g.Transitions() {
		if _, ok := idx[t.FromID]; !ok {
			continue
		}
		if _, ok := idx[t.ToID]; !ok {
			continue
		}
		succ[t.FromID] = append(succ[t.FromID], t.ToID)
		indeg[t.ToID]++
	}
	earlier := func(a, b int) bool {
		if ta, tb := locs[a].Timestamp, locs[b].Timestamp; ta != tb {
			return ta < tb
		}
		return a < b
	}
	done := make([]bool, len(locs))
	out := make([]Node, 0, len(locs))
	for len(out) < len(locs) {
		best, bestFree := -1, false
		for i, n := range locs {
			if done[i] {
				continue
			}
			free := indeg[n.ID] == 0
			// Prefer utterances whose constraints are resolved, and pick the earliest among them.
			// Only when none are resolved (a cycle) pick from the unresolved utterances.
			if best == -1 || (free && !bestFree) || (free == bestFree && earlier(i, best)) {
				best, bestFree = i, free
			}
		}
		done[best] = true
		out = append(out, locs[best])
		for _, s := range succ[locs[best].ID] {
			indeg[s]--
		}
	}
	return out
}

// PropositionOrder maps each I-node to the index of the utterance it came from.
//
// Used to measure how many utterances apart the propositions linked by a relation are.
// Overall F1 is dominated by relations between adjacent utterances, so counts are split by distance band.
//
// When an I-node is anchored to multiple utterances (restatements or quotes; 17 in the test gold),
// the earliest utterance's position is used. Previously it was the position of the last utterance seen
// in Illocutions order, so the result depended on node order.
func (g Graph) PropositionOrder() map[string]int {
	pos := map[string]int{}
	for i, n := range g.LocutionOrder() {
		pos[n.ID] = i
	}
	out := make(map[string]int, len(pos))
	for _, il := range g.Illocutions() {
		if il.AnchorType != TypeL || il.TargetType != TypeI {
			continue
		}
		p, ok := pos[il.AnchorID]
		if !ok {
			continue
		}
		if q, seen := out[il.TargetID]; !seen || p < q {
			out[il.TargetID] = p
		}
	}
	return out
}

// DistanceBand is the band of inter-utterance distance. It is the cut used to read
// "whether distant relations are captured", which GED does not show.
//
// Band "1" also includes distance 0 (two propositions from the same utterance). The generator creates
// only one proposition per utterance, so distance 0 is never generated, and the gold has only 1 of 1,090 in test.
// Adding a band would change the table and figure columns, so it is folded into band "1" and documented here.
func DistanceBand(d int) string {
	switch {
	case d <= 1:
		return "1"
	case d <= 4:
		return "2-4"
	case d <= 9:
		return "5-9"
	default:
		return "10+"
	}
}

// DistanceBands are the values DistanceBand can return (in table column order).
var DistanceBands = []string{"1", "2-4", "5-9", "10+"}

// Grade ladders, ordered strictest first. Keeping only index 0 is the strictest threshold.
//
// The 3-level (confidence) ladder collapsed to effectively binary in practice because low was almost
// never used, so a 5-level (dependence) ladder was added. Which vocabulary is used is detected from the graph.
var gradeLadders = [][]string{
	{"5", "4", "3", "2", "1"},
	{"high", "medium", "low"},
}

// GradeLadder returns the grade ladder used by g's S-nodes.
// It returns nil for graphs without self-reported grades (or with an unknown vocabulary).
func (g Graph) GradeLadder() []string {
	seen := map[string]bool{}
	for _, n := range g.Nodes {
		if n.Type.IsSNode() && n.Confidence != "" {
			seen[n.Confidence] = true
		}
	}
	if len(seen) == 0 {
		return nil
	}
	for _, ladder := range gradeLadders {
		known := 0
		for _, lv := range ladder {
			if seen[lv] {
				known++
			}
		}
		if known == len(seen) {
			return ladder
		}
	}
	return nil
}

// FilterRelationsByConfidence drops relations whose self-reported grade is not in keep.
//
// S-nodes with empty confidence have "no self-report" and are always kept (compatibility with v3 and earlier prompts).
// YAs illocuting the relation are dropped too (keeping them would leave dangling YAs, violating AIF).
func (g Graph) FilterRelationsByConfidence(keep ...string) Graph {
	ok := map[string]bool{}
	for _, k := range keep {
		ok[k] = true
	}
	drop := map[string]bool{}
	for _, n := range g.Nodes {
		if n.Type.IsSNode() && n.Confidence != "" && !ok[n.Confidence] {
			drop[n.ID] = true
		}
	}
	if len(drop) == 0 {
		return g
	}
	for _, il := range g.Illocutions() {
		if drop[il.TargetID] {
			drop[il.YAID] = true
		}
	}
	out := Graph{ID: g.ID}
	for _, n := range g.Nodes {
		if !drop[n.ID] {
			out.Nodes = append(out.Nodes, n)
		}
	}
	for _, e := range g.Edges {
		if !drop[e.FromID] && !drop[e.ToID] {
			out.Edges = append(out.Edges, e)
		}
	}
	return out
}

// SpeakerOf returns the speaker of an L-node. If the speaker is empty, it is taken from the "name : text" form of the text.
func SpeakerOf(n Node) string {
	if n.Speaker != "" {
		return n.Speaker
	}
	if i := strings.Index(n.Text, " : "); i > 0 && i < 60 {
		return strings.TrimSpace(n.Text[:i])
	}
	return ""
}

// String returns a one-line human-readable representation (for logs and evidence output).
func (n Node) String() string {
	if n.Scheme != "" && n.Text == "" {
		return fmt.Sprintf("%s(%s)", n.Type, n.Scheme)
	}
	t := n.Text
	if len([]rune(t)) > 80 {
		t = string([]rune(t)[:80]) + "…"
	}
	return fmt.Sprintf("%s[%s]", n.Type, t)
}
