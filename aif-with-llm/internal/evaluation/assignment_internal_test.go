package evaluation

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// On small matrices, the result must match brute force and return the optimum.
// Integer matrices with many ties (where structural nodes have substitution cost 0) are included.
func TestLinearSumAssignmentIsOptimal(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 300 {
		n := 1 + rng.IntN(6)
		cost := make([][]float64, n)
		for i := range cost {
			cost[i] = make([]float64, n)
			for j := range cost[i] {
				if trial%2 == 0 {
					cost[i][j] = float64(rng.IntN(3)) // full of ties
				} else {
					cost[i][j] = rng.Float64()
				}
			}
		}
		got, err := linearSumAssignment(cost)
		if err != nil {
			t.Fatal(err)
		}
		if !isPermutation(got) {
			t.Fatalf("割当が置換になっていない: %v", got)
		}
		if g, w := total(cost, got), bruteForce(cost); math.Abs(g-w) > 1e-9 {
			t.Fatalf("最適でない: got %v want %v (cost=%v)", g, w, cost)
		}
	}
}

// A constant matrix returns the identity permutation (the same tie-breaking as scipy).
func TestLinearSumAssignmentTieBreakMatchesScipy(t *testing.T) {
	cost := [][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
	got, err := linearSumAssignment(cost)
	if err != nil {
		t.Fatal(err)
	}
	for i, j := range got {
		if i != j {
			t.Fatalf("定数行列の解は単位置換のはず: %v", got)
		}
	}
}

func TestBipartiteMatchPrefersSubstitutionOnlyWhenCheaper(t *testing.T) {
	// For 0 → 0, substitution at 0.2 is cheaper than deletion + insertion at 2.0. 1 has no substitution target, so it is deleted.
	m, err := bipartiteMatch([][]float64{{0.2}, {1e6}}, []float64{1, 1}, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 1 || m[0] != 0 {
		t.Fatalf("got %v", m)
	}
	if m, err := bipartiteMatch(nil, nil, nil); err != nil || len(m) != 0 {
		t.Fatalf("空入力: %v %v", m, err)
	}
}

func isPermutation(p []int) bool {
	seen := map[int]bool{}
	for _, j := range p {
		if j < 0 || j >= len(p) || seen[j] {
			return false
		}
		seen[j] = true
	}
	return true
}

func total(cost [][]float64, p []int) float64 {
	s := 0.0
	for i, j := range p {
		s += cost[i][j]
	}
	return s
}

func bruteForce(cost [][]float64) float64 {
	n := len(cost)
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	best := math.Inf(1)
	var rec func(k int)
	rec = func(k int) {
		if k == n {
			best = math.Min(best, total(cost, p))
			return
		}
		for i := k; i < n; i++ {
			p[k], p[i] = p[i], p[k]
			rec(k + 1)
			p[k], p[i] = p[i], p[k]
		}
	}
	rec(0)
	return best
}

// Per-threshold proposition F1 must count "the maximum number of one-to-one pairs formable from pairs at or above the threshold".
// Cutting the cos-sum-maximizing assignment (0.99, 0.50) at the threshold gives 1 pair, but (0.56, 0.56) gives 2.
func TestAlignmentCountsMaximumPairsAboveThreshold(t *testing.T) {
	sim := simTable{{"p0", "g0"}: 0.99, {"p1", "g1"}: 0.50, {"p0", "g1"}: 0.56, {"p1", "g0"}: 0.56}
	hungarian := map[string]string{"p0": "g0", "p1": "g1"} // the sum-maximizing assignment
	a, err := alignmentOf(hungarian, sim, []string{"p0", "p1"}, []string{"g0", "g1"}, 0.55)
	if err != nil {
		t.Fatal(err)
	}
	if a.PRF.TP != 2 || a.AboveThreshold != 1 {
		t.Errorf("TP は閾値以上で作れる最大の 2 組、AboveThreshold は割当のうち閾値以上の 1 組のはず: %+v", a)
	}
	if a.F1ByThreshold["0.55"] != 1 || a.F1ByThreshold["0.60"] != 0.5 {
		t.Errorf("閾値曲線も同じ数え方のはず: %v", a.F1ByThreshold)
	}
}

// Illocutions must be compared only on YAs anchored to utterances (L-nodes). Gold TA→YA→I must not be mixed in.
func TestForceScoreIgnoresTransitionAnchoredYA(t *testing.T) {
	mk := func(withTA bool) aif.Graph {
		g := aif.Graph{ID: "f", Nodes: []aif.Node{
			{ID: "L0", Type: aif.TypeL, Text: "A : x"}, {ID: "L1", Type: aif.TypeL, Text: "B : y"},
			{ID: "T", Type: aif.TypeTA}, {ID: "Y", Type: aif.TypeYA, Scheme: "Asserting"}, {ID: "I", Type: aif.TypeI, Text: "x"},
		}, Edges: []aif.Edge{{FromID: "L0", ToID: "T"}, {FromID: "T", ToID: "L1"}, {FromID: "L0", ToID: "Y"}, {FromID: "Y", ToID: "I"}}}
		if withTA {
			g.Nodes = append(g.Nodes, aif.Node{ID: "YT", Type: aif.TypeYA, Scheme: "Arguing"})
			g.Edges = append(g.Edges, aif.Edge{FromID: "T", ToID: "YT"}, aif.Edge{FromID: "YT", ToID: "I"})
		}
		return g
	}
	if s := forceScore(mk(false), mk(true), map[string]string{"I": "I"}); s.Compared != 1 || s.Exact != 1 {
		t.Errorf("L→YA→I の Asserting どうしを比べて一致のはず: %+v", s)
	}
}
