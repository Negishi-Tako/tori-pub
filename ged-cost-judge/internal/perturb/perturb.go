// Package perturb holds the edit catalog (plan §4.2) and draws edit plans for triplets (§4.3).
// The kinds and amounts of edits are decided here by random draws (fixed seed). The generating LLM does not know "which is closer".
package perturb

import (
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
)

// Kind is a kind of edit.
type Kind string

const (
	PPara  Kind = "P-para"  // paraphrase a proposition (meaning preserved)
	PShift Kind = "P-shift" // change a proposition's content (role preserved)
	PNeg   Kind = "P-neg"   // flip polarity or quantification
	PDel   Kind = "P-del"   // missing isolated proposition
	PAdd   Kind = "P-add"   // extra isolated proposition
	PDelR  Kind = "P-del+R" // proposition missing along with its relations
	RDel   Kind = "R-del"   // missing relation only
	RAdd   Kind = "R-add"   // extra relation only
	RType  Kind = "R-type"  // relabel the type
	RRev   Kind = "R-rev"   // reverse the direction
	XSpk   Kind = "X-spk"   // swap speakers (proposition graph unchanged)
	XSurf  Kind = "X-surf"  // change wording (proposition graph unchanged)
)

// Edit is a single edit.
type Edit struct {
	Kind Kind `json:"kind"`
	// Prop is the target proposition (P-para / P-shift / P-neg / P-del / P-del+R), or the new proposition's ID for P-add.
	Prop string `json:"prop,omitempty"`
	// Edge is the target edge (R-del / R-type / R-rev), or the edge to add for R-add.
	Edge *ged.Edge `json:"edge,omitempty"`
	// NewKind is the new type for R-type.
	NewKind string `json:"new_kind,omitempty"`
	// Strength is the strength for P-para (weak / medium / strong) and P-shift (small / medium / large).
	Strength string `json:"strength,omitempty"`
}

// Variant is a variant with edits applied.
type Variant struct {
	Graph ged.Graph `json:"graph"`
	Edits []Edit    `json:"edits"`
	// Origin maps variant propositions → source propositions (absent for added propositions).
	Origin map[string]string `json:"origin"`
	// Changed holds propositions whose content was changed, with their edits.
	Changed map[string]Edit `json:"changed"`
	// Neg is the number of polarity flips.
	Neg int `json:"neg"`
}

type state struct {
	g        ged.Graph
	touchedP map[string]bool
	touchedE map[ged.Edge]bool
	added    int
}

func (s *state) degree(id string) int {
	n := 0
	for _, e := range s.g.Edges {
		if e.From == id || e.To == id {
			n++
		}
	}
	return n
}

func (s *state) hasEdge(a, b string) bool {
	for _, e := range s.g.Edges {
		if (e.From == a && e.To == b) || (e.From == b && e.To == a) {
			return true
		}
	}
	return false
}

func (s *state) freeProps(ok func(id string) bool) []string {
	var out []string
	for _, p := range s.g.Props {
		if !s.touchedP[p.ID] && ok(p.ID) {
			out = append(out, p.ID)
		}
	}
	return out
}

func (s *state) freeEdges(ok func(e ged.Edge) bool) []ged.Edge {
	var out []ged.Edge
	for _, e := range s.g.Edges {
		if !s.touchedE[e] && ok(e) {
			out = append(out, e)
		}
	}
	return out
}

func pick[T any](rng *rand.Rand, xs []T) (T, bool) {
	var zero T
	if len(xs) == 0 {
		return zero, false
	}
	return xs[rng.IntN(len(xs))], true
}

// draw picks and applies one edit of kind on the current graph. Returns false if none can be applied.
func (s *state) draw(rng *rand.Rand, kind Kind) (Edit, bool) {
	all := func(string) bool { return true }
	switch kind {
	case PPara, PShift, PNeg:
		id, ok := pick(rng, s.freeProps(func(id string) bool { return id[0] != 'n' }))
		if !ok {
			return Edit{}, false
		}
		e := Edit{Kind: kind, Prop: id}
		switch kind {
		case PPara:
			e.Strength = []string{"weak", "medium", "strong"}[rng.IntN(3)]
		case PShift:
			e.Strength = []string{"small", "medium", "large"}[rng.IntN(3)]
		}
		s.touchedP[id] = true
		return e, true
	case PDel, PDelR:
		want := func(id string) bool { return s.degree(id) == 0 }
		if kind == PDelR {
			want = func(id string) bool { d := s.degree(id); return d >= 1 && d <= 2 }
		}
		id, ok := pick(rng, s.freeProps(want))
		if !ok {
			return Edit{}, false
		}
		s.removeProp(id)
		return Edit{Kind: kind, Prop: id}, true
	case PAdd:
		s.added++
		id := fmt.Sprintf("n%d", s.added)
		s.g.Props = append(s.g.Props, ged.Prop{ID: id})
		s.touchedP[id] = true
		return Edit{Kind: kind, Prop: id}, true
	case RDel:
		e, ok := pick(rng, s.freeEdges(func(ged.Edge) bool { return true }))
		if !ok {
			return Edit{}, false
		}
		s.g.Edges = slices.DeleteFunc(s.g.Edges, func(x ged.Edge) bool { return x == e })
		return Edit{Kind: kind, Edge: &e}, true
	case RAdd:
		ids := s.freeProps(all)
		var cands [][2]string
		for _, a := range ids {
			for _, b := range ids {
				if a != b && !s.hasEdge(a, b) {
					cands = append(cands, [2]string{a, b})
				}
			}
		}
		ab, ok := pick(rng, cands)
		if !ok {
			return Edit{}, false
		}
		e := ged.Edge{From: ab[0], To: ab[1], Kind: ged.Kinds[rng.IntN(3)]}
		s.g.Edges = append(s.g.Edges, e)
		s.touchedE[e] = true
		return Edit{Kind: kind, Edge: &e}, true
	case RType:
		e, ok := pick(rng, s.freeEdges(func(ged.Edge) bool { return true }))
		if !ok {
			return Edit{}, false
		}
		others := slices.DeleteFunc(slices.Clone(ged.Kinds), func(k string) bool { return k == e.Kind })
		nk := others[rng.IntN(len(others))]
		s.replaceEdge(e, ged.Edge{From: e.From, To: e.To, Kind: nk})
		return Edit{Kind: kind, Edge: &e, NewKind: nk}, true
	case RRev:
		e, ok := pick(rng, s.freeEdges(func(ged.Edge) bool { return true }))
		if !ok {
			return Edit{}, false
		}
		s.replaceEdge(e, ged.Edge{From: e.To, To: e.From, Kind: e.Kind})
		return Edit{Kind: kind, Edge: &e}, true
	case XSpk, XSurf:
		return Edit{Kind: kind}, true
	}
	return Edit{}, false
}

func (s *state) removeProp(id string) {
	s.g.Props = slices.DeleteFunc(s.g.Props, func(p ged.Prop) bool { return p.ID == id })
	s.g.Edges = slices.DeleteFunc(s.g.Edges, func(e ged.Edge) bool { return e.From == id || e.To == id })
	s.touchedP[id] = true
}

func (s *state) replaceEdge(old, nw ged.Edge) {
	for i, e := range s.g.Edges {
		if e == old {
			s.g.Edges[i] = nw
			break
		}
	}
	s.touchedE[nw] = true
}

// Build applies kinds to S in order and returns the variant at each step (steps[i] has the first i+1 applied).
// Edits that cannot be applied are skipped.
func Build(rng *rand.Rand, src ged.Graph, kinds []Kind) []Variant {
	s := &state{g: src.Clone(), touchedP: map[string]bool{}, touchedE: map[ged.Edge]bool{}}
	var edits []Edit
	var steps []Variant
	for _, k := range kinds {
		e, ok := s.draw(rng, k)
		if !ok {
			continue
		}
		edits = append(edits, e)
		steps = append(steps, snapshot(src, s.g, edits))
	}
	if len(steps) == 0 {
		steps = append(steps, snapshot(src, s.g, nil))
	}
	return steps
}

func snapshot(src, g ged.Graph, edits []Edit) Variant {
	v := Variant{Graph: g.Clone(), Edits: slices.Clone(edits), Origin: map[string]string{}, Changed: map[string]Edit{}}
	orig := map[string]bool{}
	for _, p := range src.Props {
		orig[p.ID] = true
	}
	for _, p := range g.Props {
		if orig[p.ID] {
			v.Origin[p.ID] = p.ID
		}
	}
	for _, e := range edits {
		switch e.Kind {
		case PPara, PShift, PNeg:
			v.Changed[e.Prop] = e
			if e.Kind == PNeg {
				v.Neg++
			}
		}
	}
	return v
}

// TripletType is the kind of triplet (plan §4.3).
type TripletType string

const (
	T1Monotone  TripletType = "T1" // A's edits ⊂ B's edits
	T2SameKind  TripletType = "T2" // same kind, differing only in amount or strength
	T3Contrast  TripletType = "T3" // edits of different kinds
	T4Invariant TripletType = "T4" // only edits that leave the proposition graph unchanged
)

// DrawType picks a triplet type with proportions 10 / 10 / 70 / 10%.
func DrawType(rng *rand.Rand) TripletType {
	switch r := rng.Float64(); {
	case r < 0.1:
		return T1Monotone
	case r < 0.2:
		return T2SameKind
	case r < 0.9:
		return T3Contrast
	default:
		return T4Invariant
	}
}

// graphKinds are the edit kinds and weights used in T3.
var graphKinds = []struct {
	k Kind
	w float64
}{
	{PPara, 0.12}, {PShift, 0.12}, {PNeg, 0.05}, {PDel, 0.05}, {PAdd, 0.08}, {PDelR, 0.12},
	{RDel, 0.11}, {RAdd, 0.08}, {RType, 0.17}, {RRev, 0.08}, {XSpk, 0.01}, {XSurf, 0.01},
}

func drawKind(rng *rand.Rand) Kind {
	r := rng.Float64()
	for _, x := range graphKinds {
		if r < x.w {
			return x.k
		}
		r -= x.w
	}
	return RType
}

// Plan draws the sequences of edit kinds for one triplet (for A and for B).
// In T1, A's sequence is the prefix of B's (calling Build with the same seed gives A ⊂ B).
func Plan(rng *rand.Rand, typ TripletType) (a, b []Kind) {
	switch typ {
	case T1Monotone:
		n := 1 + rng.IntN(2)
		b = make([]Kind, n+1+rng.IntN(2))
		for i := range b {
			b[i] = drawKind(rng)
		}
		return b[:n], b
	case T2SameKind:
		k := []Kind{PPara, PShift, PDelR, RDel, RType}[rng.IntN(5)]
		return []Kind{k}, []Kind{k, k}
	case T4Invariant:
		return []Kind{XSurf}, []Kind{XSpk}
	default:
		na, nb := 1+rng.IntN(3), 1+rng.IntN(3)
		for range na {
			a = append(a, drawKind(rng))
		}
		for range nb {
			b = append(b, drawKind(rng))
		}
		return a, b
	}
}
