// Package perturb は編集カタログ（計画 §4.2）と、三つ組の編集計画の抽選（§4.3）を持つ。
// 編集の種類と量はここで乱数（seed 固定）で決める。生成側の LLM は「どちらが近いか」を知らない。
package perturb

import (
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
)

// Kind は編集の種類。
type Kind string

const (
	PPara  Kind = "P-para"  // 命題の言い換え（意味は保つ）
	PShift Kind = "P-shift" // 命題の内容を変える（役割は保つ）
	PNeg   Kind = "P-neg"   // 極性・量化の反転
	PDel   Kind = "P-del"   // 孤立した命題の欠落
	PAdd   Kind = "P-add"   // 孤立した命題の余分
	PDelR  Kind = "P-del+R" // 関係ごと命題が欠落
	RDel   Kind = "R-del"   // 関係だけの欠落
	RAdd   Kind = "R-add"   // 関係だけの余分
	RType  Kind = "R-type"  // 種別の付け替え
	RRev   Kind = "R-rev"   // 向きの反転
	XSpk   Kind = "X-spk"   // 話者の入れ替え（命題グラフは不変）
	XSurf  Kind = "X-surf"  // 言い回しの変更（命題グラフは不変）
)

// Edit は 1 つの編集。
type Edit struct {
	Kind Kind `json:"kind"`
	// Prop は対象の命題（P-para / P-shift / P-neg / P-del / P-del+R）、P-add なら新しい命題の ID。
	Prop string `json:"prop,omitempty"`
	// Edge は対象の辺（R-del / R-type / R-rev）、R-add なら足す辺。
	Edge *ged.Edge `json:"edge,omitempty"`
	// NewKind は R-type の付け替え先。
	NewKind string `json:"new_kind,omitempty"`
	// Strength は P-para（weak / medium / strong）と P-shift（small / medium / large）の強さ。
	Strength string `json:"strength,omitempty"`
}

// Variant は編集を適用した変種。
type Variant struct {
	Graph ged.Graph `json:"graph"`
	Edits []Edit    `json:"edits"`
	// Origin は変種の命題 → 元の命題（足した命題には無い）。
	Origin map[string]string `json:"origin"`
	// Changed は内容を変えた命題とその編集。
	Changed map[string]Edit `json:"changed"`
	// Neg は極性反転の個数。
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

// draw は今のグラフで kind の編集を 1 つ選んで適用する。適用できなければ false。
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

// Build は S に kinds を順に適用し、各段階の変種を返す（steps[i] は i+1 個目までを適用したもの）。
// 適用できない編集は飛ばす。
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

// TripletType は三つ組の種類（計画 §4.3）。
type TripletType string

const (
	T1Monotone  TripletType = "T1" // A の編集 ⊂ B の編集
	T2SameKind  TripletType = "T2" // 同じ種類で量か強さだけが違う
	T3Contrast  TripletType = "T3" // 種類の違う編集どうし
	T4Invariant TripletType = "T4" // 命題グラフを変えない編集だけ
)

// DrawType は割合（10 / 10 / 70 / 10%）で三つ組の種類を選ぶ。
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

// graphKinds は T3 で使う編集の種類と重み。
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

// Plan は三つ組 1 つ分の編集の種類の列（A 用と B 用）を抽選する。
// T1 では B の列の先頭が A の列になる（Build を同じ seed で呼ぶと A ⊂ B になる）。
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
