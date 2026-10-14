package ged

import (
	"math"
	"testing"
)

// スライドの例: S = 主張 c を p1 が支持し p2 が攻撃する。
func uniform() Graph {
	return Graph{
		Props: []Prop{{ID: "c"}, {ID: "p1"}, {ID: "p2"}},
		Edges: []Edge{{From: "p1", To: "c", Kind: RA}, {From: "p2", To: "c", Kind: CA}},
	}
}

func identity(a, b string) float64 {
	if a == b {
		return 1
	}
	return 0.2
}

func TestExampleFromSlides(t *testing.T) {
	s := uniform()
	th := Current()

	// A: 理由 p1 と支持の矢印を消す → 主張の欠落 1.0 + 辺の欠落 1.0。
	a := Graph{Props: []Prop{{ID: "c"}, {ID: "p2"}}, Edges: []Edge{{From: "p2", To: "c", Kind: CA}}}
	fa, err := Compute(s, a, identity, th)
	if err != nil {
		t.Fatal(err)
	}
	if got := fa.Cost(th); math.Abs(got-2.0) > 1e-9 {
		t.Fatalf("A: GED = %v, want 2.0 (phi=%v)", got, fa.Phi)
	}

	// B: 反論 p2 を支持に変える → 種別違い RA↔CA 0.75（消して足す 2.0 より安い）。
	b := uniform()
	b.Edges[1].Kind = RA
	fb, err := Compute(s, b, identity, th)
	if err != nil {
		t.Fatal(err)
	}
	if got := fb.Cost(th); math.Abs(got-0.75) > 1e-9 || fb.Phi[PTypeRACA] != 1 {
		t.Fatalf("B: GED = %v, phi=%v", got, fb.Phi)
	}

	// 向きの反転は向き違い 1 回。
	r := uniform()
	r.Edges[0] = Edge{From: "c", To: "p1", Kind: RA}
	fr, _ := Compute(s, r, identity, th)
	if fr.Phi[PDirection] != 1 || fr.Cost(th) != 1 {
		t.Fatalf("reverse: phi=%v", fr.Phi)
	}

	// 関係のない辺を足すと余分 1 本。
	x := uniform()
	x.Edges = append(x.Edges, Edge{From: "p1", To: "p2", Kind: MA})
	fx, _ := Compute(s, x, identity, th)
	if fx.EdgeExtra != 1 || fx.Cost(th) != 1 {
		t.Fatalf("extra edge: %+v", fx)
	}
}

func TestSubstitutionIsOneMinusCosAboveKnot(t *testing.T) {
	th := Current()
	for _, c := range []float64{1, 0.95, 0.9, 0.8, 0.6, 0.45, 0.3} {
		if got, want := th.Sub(c), 1-c; math.Abs(got-want) > 1e-9 {
			t.Errorf("Sub(%v) = %v, want %v", c, got, want)
		}
	}
	if got := th.Sub(0.1); math.Abs(got-0.7) > 1e-9 {
		t.Errorf("Sub(0.1) = %v, want 0.7 (cos < 0.3 は一定)", got)
	}
}

// 線形化が一致すること: 同じ対応なら Cost(θ) は Compute(θ) の GED と同じ。
func TestFeaturesAreLinear(t *testing.T) {
	s := uniform()
	v := uniform()
	v.Edges[0].Kind = MA
	sim := func(a, b string) float64 {
		if a == b {
			if a == "p2" {
				return 0.62
			}
			return 1
		}
		return 0.1
	}
	th := Theta{1.3, 0.7, 0.15, 0.3, 0.2, 0.1, 1.2, 1.5, 0.6, 0.9, 1.2}
	f, _ := Compute(s, v, sim, th)
	want := th.Sub(0.62) + th[PTypeRAMA]
	if got := f.Cost(th); math.Abs(got-want) > 1e-9 {
		t.Fatalf("Cost = %v, want %v", got, want)
	}
}
