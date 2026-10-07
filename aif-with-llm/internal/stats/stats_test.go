package stats_test

import (
	"math"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/stats"
)

func TestDescribe(t *testing.T) {
	s := stats.Describe([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if s.N != 8 || s.Mean != 5 {
		t.Errorf("mean = %f, n = %d; want 5, 8", s.Mean, s.N)
	}
	// 標本標準偏差（n-1）で 2.138…
	if math.Abs(s.SD-2.1381) > 1e-3 {
		t.Errorf("sd = %f, want ~2.138", s.SD)
	}
	if s.Min != 2 || s.Max != 9 {
		t.Errorf("min/max = %f/%f", s.Min, s.Max)
	}
	if got := stats.Describe(nil); got.N != 0 {
		t.Errorf("empty describe = %+v", got)
	}
}

func TestCliffsDelta(t *testing.T) {
	// 完全に上回る場合は 1、下回る場合は -1。
	if d := stats.CliffsDelta([]float64{4, 5, 6}, []float64{1, 2, 3}); d != 1 {
		t.Errorf("delta = %f, want 1", d)
	}
	if d := stats.CliffsDelta([]float64{1, 2, 3}, []float64{4, 5, 6}); d != -1 {
		t.Errorf("delta = %f, want -1", d)
	}
	// 同じ分布なら 0 付近。
	if d := stats.CliffsDelta([]float64{1, 2, 3}, []float64{1, 2, 3}); d != 0 {
		t.Errorf("delta = %f, want 0", d)
	}
	if got := stats.DeltaMagnitude(0.05); got != "negligible" {
		t.Errorf("magnitude(0.05) = %s", got)
	}
	if got := stats.DeltaMagnitude(0.8); got != "large" {
		t.Errorf("magnitude(0.8) = %s", got)
	}
}

func TestWilcoxonSignedRank(t *testing.T) {
	// 全ての対で同じ向きに差がある → 有意になるはず。
	diffs := make([]float64, 20)
	for i := range diffs {
		diffs[i] = 0.5 + float64(i)*0.01
	}
	if p := stats.WilcoxonSignedRank(diffs); p > 0.001 {
		t.Errorf("p = %f, want a very small value for a consistent difference", p)
	}
	// 差が無ければ有意にならない。
	zero := make([]float64, 20)
	if p := stats.WilcoxonSignedRank(zero); p != 1 {
		t.Errorf("p = %f, want 1 when every difference is zero", p)
	}
	// 向きがばらついていれば有意にならない。
	mixed := []float64{1, -1, 2, -2, 3, -3, 1.5, -1.5}
	if p := stats.WilcoxonSignedRank(mixed); p < 0.5 {
		t.Errorf("p = %f, want a large value for symmetric differences", p)
	}
}

func TestBootstrapCIIsDeterministic(t *testing.T) {
	diffs := []float64{0.1, 0.2, 0.15, 0.3, 0.05, 0.25}
	lo1, hi1 := stats.BootstrapCI(diffs, 2000, 0.05, 42)
	lo2, hi2 := stats.BootstrapCI(diffs, 2000, 0.05, 42)
	if lo1 != lo2 || hi1 != hi2 {
		t.Fatal("bootstrap CI must be reproducible for the same seed")
	}
	mean := stats.Describe(diffs).Mean
	if mean < lo1 || mean > hi1 {
		t.Errorf("CI [%f, %f] should contain the mean %f", lo1, hi1, mean)
	}
}

func TestHolmAdjust(t *testing.T) {
	p := []float64{0.01, 0.02, 0.03, 0.04}
	adj := stats.HolmAdjust(p)
	// 最小の p は n 倍される。
	if math.Abs(adj[0]-0.04) > 1e-9 {
		t.Errorf("adjusted[0] = %f, want 0.04", adj[0])
	}
	// 補正後は単調非減少。
	for i := 1; i < len(adj); i++ {
		if adj[i] < adj[i-1] {
			t.Errorf("holm output is not monotonic: %v", adj)
			break
		}
	}
	// 1 を超えない。
	for _, v := range stats.HolmAdjust([]float64{0.5, 0.6, 0.9}) {
		if v > 1 {
			t.Errorf("adjusted p exceeds 1: %f", v)
		}
	}
	if got := stats.HolmAdjust(nil); len(got) != 0 {
		t.Errorf("empty input should give empty output")
	}
}

func TestComparePairedRejectsMismatchedLength(t *testing.T) {
	if _, err := stats.ComparePaired([]float64{1, 2}, []float64{1}, 100, 1); err == nil {
		t.Fatal("mismatched lengths must be rejected")
	}
}
