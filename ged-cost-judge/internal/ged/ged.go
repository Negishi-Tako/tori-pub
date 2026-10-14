// Package ged computes the GED of proposition graphs (I-nodes and edges folded from RA/CA/MA)
// with 11 learnable costs (plan §6.2).
//
// It follows the same procedure as the previous experiment's aif-ged/0.4 proposition GED (match propositions with
// the Hungarian method, then match edges from that matching as exact → type mismatch → direction mismatch), generalizing 3 points:
//
//   - separate costs for missing (only on the reference side) and extra (only on the variant side)
//   - separate type-mismatch costs per type pair (RA↔CA / RA↔MA / CA↔MA)
//   - substitution cost as a piecewise-linear function of cos (knots 1, 0.9, 0.7, 0.5, 0.3)
//
// With the proposition matching fixed, GED is linear in θ (GED = θ·φ + number of extra edges).
// Compute returns those features φ.
package ged

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Relation types.
const (
	RA = "RA" // support
	CA = "CA" // attack
	MA = "MA" // rephrase
)

// Kinds lists the relation types.
var Kinds = []string{RA, CA, MA}

// Prop is a proposition (I-node).
type Prop struct {
	ID   string `json:"id"`
	Text string `json:"text,omitempty"`
}

// Edge is a relation from premise From to conclusion To.
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Kind string `json:"kind"`
}

// Graph is a proposition graph.
type Graph struct {
	Props []Prop `json:"props"`
	Edges []Edge `json:"edges"`
}

// Clone returns a deep copy.
func (g Graph) Clone() Graph {
	return Graph{Props: slices.Clone(g.Props), Edges: slices.Clone(g.Edges)}
}

// Parameter indices (the # in plan §6.2 minus 1).
const (
	PNodeMissing = iota // missing I-node (proposition only on the reference side; an insertion in GED)
	PNodeExtra          // extra I-node (proposition only on the variant side; a deletion in GED)
	PSub09              // substitution function increment, cos 1 → 0.9
	PSub07              // likewise 0.9 → 0.7
	PSub05              // likewise 0.7 → 0.5
	PSub03              // likewise 0.5 → 0.3 (constant for cos < 0.3)
	PEdgeMissing        // missing edge
	PTypeRACA           // type mismatch RA↔CA
	PTypeRAMA           // type mismatch RA↔MA
	PTypeCAMA           // type mismatch CA↔MA
	PDirection          // direction mismatch
	NumParams
)

// ParamNames are the names shown in tables.
var ParamNames = [NumParams]string{
	"I 欠落", "I 余分", "置換 cos0.9", "置換 cos0.7", "置換 cos0.5", "置換 cos0.3",
	"辺の欠落", "種別 RA↔CA", "種別 RA↔MA", "種別 CA↔MA", "向き違い",
}

// EdgeExtraCost is the cost of an extra edge. β and θ scales cannot be separated, so it is fixed at 1 (the scale reference).
const EdgeExtraCost = 1.0

// Knots are the knots (cos) of the substitution function.
var Knots = [5]float64{1, 0.9, 0.7, 0.5, 0.3}

// Theta is the 11 costs.
type Theta [NumParams]float64

// Current is the current cost table (aif-ged/0.4 proposition graph). Substitution is piecewise-linear 1 − cos.
func Current() Theta {
	return Theta{1, 1, 0.1, 0.2, 0.2, 0.2, 1, 0.75, 0.75, 0.75, 1}
}

// SubBasis is the basis h_k(cos) of the substitution function. f(cos) = Σ θ_k h_k(cos).
func SubBasis(cos float64) [4]float64 {
	var h [4]float64
	for k := range 4 {
		hi, lo := Knots[k], Knots[k+1]
		h[k] = math.Min(1, math.Max(0, (hi-cos)/(hi-lo)))
	}
	return h
}

// Sub is the substitution cost for cos.
func (t Theta) Sub(cos float64) float64 {
	h := SubBasis(cos)
	return t[PSub09]*h[0] + t[PSub07]*h[1] + t[PSub05]*h[2] + t[PSub03]*h[3]
}

// Features is the linearization with the matching fixed. GED = Phi·θ + EdgeExtra·EdgeExtraCost.
type Features struct {
	Phi       [NumParams]float64
	EdgeExtra float64
	// Assign maps variant proposition i → reference proposition index (-1 if unmatched). Used for convergence checks.
	Assign []int
}

// Cost is the GED at θ.
func (f Features) Cost(t Theta) float64 {
	c := f.EdgeExtra * EdgeExtraCost
	for i := range NumParams {
		c += f.Phi[i] * t[i]
	}
	return c
}

// Signature identifies the matching.
func (f Features) Signature() string {
	var b strings.Builder
	for _, j := range f.Assign {
		b.WriteString(strconv.Itoa(j))
		b.WriteByte(',')
	}
	return b.String()
}

// Sim returns the cos between reference proposition refID and variant proposition varID.
type Sim func(refID, varID string) float64

func typePair(a, b string) int {
	if a > b {
		a, b = b, a
	}
	switch a + b {
	case CA + RA:
		return PTypeRACA
	case MA + RA:
		return PTypeRAMA
	case CA + MA:
		return PTypeCAMA
	}
	return -1
}

// Compute optimally matches propositions at θ and returns the resulting features. ref is the reference (source argument S); v is the variant.
func Compute(ref, v Graph, sim Sim, t Theta) (Features, error) {
	n, m := len(v.Props), len(ref.Props)
	sub := make([][]float64, n)
	del := make([]float64, n)
	cos := make([][]float64, n)
	for i, a := range v.Props {
		sub[i] = make([]float64, m)
		cos[i] = make([]float64, m)
		del[i] = t[PNodeExtra]
		for j, b := range ref.Props {
			c := math.Min(1, math.Max(0, sim(b.ID, a.ID)))
			cos[i][j] = c
			sub[i][j] = t.Sub(c)
		}
	}
	ins := make([]float64, m)
	for j := range ins {
		ins[j] = t[PNodeMissing]
	}
	match, err := bipartiteMatch(sub, del, ins)
	if err != nil {
		return Features{}, fmt.Errorf("ged: 命題の対応付け: %w", err)
	}

	var f Features
	f.Assign = make([]int, n)
	usedRef := make([]bool, m)
	pairs := make(map[string]string, len(match))
	for i := range n {
		j, ok := match[i]
		if !ok {
			f.Assign[i] = -1
			f.Phi[PNodeExtra]++
			continue
		}
		f.Assign[i] = j
		usedRef[j] = true
		pairs[v.Props[i].ID] = ref.Props[j].ID
		h := SubBasis(cos[i][j])
		for k := range 4 {
			f.Phi[PSub09+k] += h[k]
		}
	}
	for j := range m {
		if !usedRef[j] {
			f.Phi[PNodeMissing]++
		}
	}

	// Edges: match every edge per pass in the order exact → type mismatch → direction mismatch.
	refUsed := make([]bool, len(ref.Edges))
	type mapped struct {
		from, to, kind string
		done           bool
	}
	var ms []*mapped
	for _, e := range v.Edges {
		fr, ok1 := pairs[e.From]
		to, ok2 := pairs[e.To]
		if !ok1 || !ok2 {
			f.EdgeExtra++
			continue
		}
		ms = append(ms, &mapped{from: fr, to: to, kind: e.Kind})
	}
	find := func(from, to string, ok func(Edge) bool) int {
		for k, e := range ref.Edges {
			if !refUsed[k] && e.From == from && e.To == to && ok(e) {
				return k
			}
		}
		return -1
	}
	for _, x := range ms {
		if k := find(x.from, x.to, func(e Edge) bool { return e.Kind == x.kind }); k >= 0 {
			refUsed[k], x.done = true, true
		}
	}
	for _, x := range ms {
		if x.done {
			continue
		}
		if k := find(x.from, x.to, func(Edge) bool { return true }); k >= 0 {
			refUsed[k], x.done = true, true
			if p := typePair(ref.Edges[k].Kind, x.kind); p >= 0 {
				f.Phi[p]++
			}
		}
	}
	for _, x := range ms {
		if x.done {
			continue
		}
		if k := find(x.to, x.from, func(Edge) bool { return true }); k >= 0 {
			refUsed[k], x.done = true, true
			f.Phi[PDirection]++
		}
	}
	for _, x := range ms {
		if !x.done {
			f.EdgeExtra++
		}
	}
	for k := range ref.Edges {
		if !refUsed[k] {
			f.Phi[PEdgeMissing]++
		}
	}
	return f, nil
}
