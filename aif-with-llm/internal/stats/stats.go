// Package stats provides statistics for comparisons between conditions.
//
// Principles:
//   - Never make claims from p-values alone. Always report effect sizes and confidence intervals alongside
//   - Comparisons between conditions use paired tests (pairing the same nodesets)
//   - Apply multiple-comparison correction (Holm)
package stats

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// Summary is the summary statistics of one series.
type Summary struct {
	N    int     `json:"n"`
	Mean float64 `json:"mean"`
	SD   float64 `json:"sd"`
	Min  float64 `json:"min"`
	Max  float64 `json:"max"`
}

func Describe(xs []float64) Summary {
	s := Summary{N: len(xs)}
	if len(xs) == 0 {
		return s
	}
	s.Min, s.Max = xs[0], xs[0]
	var sum float64
	for _, x := range xs {
		sum += x
		s.Min = math.Min(s.Min, x)
		s.Max = math.Max(s.Max, x)
	}
	s.Mean = sum / float64(len(xs))
	if len(xs) > 1 {
		var ss float64
		for _, x := range xs {
			d := x - s.Mean
			ss += d * d
		}
		s.SD = math.Sqrt(ss / float64(len(xs)-1))
	}
	return s
}

// PairedComparison is the result of comparing two paired conditions.
type PairedComparison struct {
	N int `json:"n"`
	// MeanDiff is the mean of a - b.
	MeanDiff float64 `json:"mean_diff"`
	// CILow / CIHigh are the bootstrap confidence interval of the difference.
	CILow  float64 `json:"ci_low"`
	CIHigh float64 `json:"ci_high"`
	// PValue is from the Wilcoxon signed-rank test (two-sided, normal approximation).
	PValue float64 `json:"p_value"`
	// CliffsDelta is the effect size (-1 to 1).
	CliffsDelta float64 `json:"cliffs_delta"`
	// Magnitude is the conventional interpretation of Cliff's delta.
	Magnitude string `json:"magnitude"`
}

// ComparePaired compares two paired series.
//
// a and b must correspond in the same order (the same nodesets).
func ComparePaired(a, b []float64, bootstrapIters int, seed uint64) (PairedComparison, error) {
	if len(a) != len(b) {
		return PairedComparison{}, fmt.Errorf("stats: paired series must have the same length (%d vs %d)", len(a), len(b))
	}
	out := PairedComparison{N: len(a)}
	if len(a) == 0 {
		return out, nil
	}
	diffs := make([]float64, len(a))
	for i := range a {
		diffs[i] = a[i] - b[i]
	}
	out.MeanDiff = Describe(diffs).Mean
	out.CILow, out.CIHigh = BootstrapCI(diffs, bootstrapIters, 0.05, seed)
	out.PValue = WilcoxonSignedRank(diffs)
	out.CliffsDelta = CliffsDelta(a, b)
	out.Magnitude = DeltaMagnitude(out.CliffsDelta)
	return out, nil
}

// BootstrapCI computes a confidence interval for the mean difference by bootstrapping.
//
// seed is passed to guarantee that the same data yields the same interval
// (reported numbers must be reproducible later).
func BootstrapCI(diffs []float64, iters int, alpha float64, seed uint64) (low, high float64) {
	if len(diffs) == 0 {
		return 0, 0
	}
	if iters <= 0 {
		iters = 10000
	}
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.05
	}
	rng := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15))
	means := make([]float64, iters)
	for i := range iters {
		var sum float64
		for range diffs {
			sum += diffs[rng.IntN(len(diffs))]
		}
		means[i] = sum / float64(len(diffs))
	}
	sort.Float64s(means)
	lowIdx := int(alpha / 2 * float64(iters))
	highIdx := int((1-alpha/2)*float64(iters)) - 1
	if highIdx >= iters {
		highIdx = iters - 1
	}
	if lowIdx < 0 {
		lowIdx = 0
	}
	return means[lowIdx], means[highIdx]
}

// WilcoxonSignedRank returns the two-sided p-value of the signed-rank test (normal approximation with continuity correction).
//
// Because it is a normal approximation, accuracy drops when there are few effective pairs (nonzero differences).
// Treat results with fewer than 10 pairs as indicative only (this experiment has n=20–124).
func WilcoxonSignedRank(diffs []float64) float64 {
	type entry struct {
		abs  float64
		sign float64
	}
	var entries []entry
	for _, d := range diffs {
		if d == 0 {
			continue // pairs with zero difference are excluded from the test (standard practice)
		}
		sign := 1.0
		if d < 0 {
			sign = -1
		}
		entries = append(entries, entry{abs: math.Abs(d), sign: sign})
	}
	n := len(entries)
	if n == 0 {
		return 1
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].abs < entries[j].abs })

	// Ties get the average rank.
	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i
		for j+1 < n && entries[j+1].abs == entries[i].abs {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			ranks[k] = avg
		}
		i = j + 1
	}
	var wPlus, wMinus float64
	for i, e := range entries {
		if e.sign > 0 {
			wPlus += ranks[i]
		} else {
			wMinus += ranks[i]
		}
	}
	w := math.Min(wPlus, wMinus)
	nf := float64(n)
	mean := nf * (nf + 1) / 4
	variance := nf * (nf + 1) * (2*nf + 1) / 24
	if variance <= 0 {
		return 1
	}
	z := (math.Abs(w-mean) - 0.5) / math.Sqrt(variance) // continuity correction
	if z < 0 {
		z = 0
	}
	return 2 * (1 - normalCDF(z))
}

func normalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

// CliffsDelta is an effect size: how strongly a tends to be larger than b (-1 to 1).
//
// It is rank-based, so it assumes no distribution shape. The mean difference alone can
// overstate "a difference buried in the variance", so this is reported alongside.
func CliffsDelta(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var greater, less int
	for _, x := range a {
		for _, y := range b {
			switch {
			case x > y:
				greater++
			case x < y:
				less++
			}
		}
	}
	return float64(greater-less) / float64(len(a)*len(b))
}

// DeltaMagnitude applies the conventional cutoffs for Cliff's delta (Romano et al. 2006).
func DeltaMagnitude(delta float64) string {
	d := math.Abs(delta)
	switch {
	case d < 0.147:
		return "negligible"
	case d < 0.33:
		return "small"
	case d < 0.474:
		return "medium"
	default:
		return "large"
	}
}

// HolmAdjust adjusts p-values with the Holm-Bonferroni method.
//
// Adjusted values are returned in the same order as the input. The more metrics there are, the stricter
// each threshold becomes, which guards against "adding metrics to fish for hits".
func HolmAdjust(pvalues []float64) []float64 {
	n := len(pvalues)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return pvalues[idx[i]] < pvalues[idx[j]] })

	prev := 0.0
	for rank, i := range idx {
		adj := float64(n-rank) * pvalues[i]
		// Preserve monotonicity (a lower-ranked p-value must not fall below a higher-ranked one).
		adj = math.Max(adj, prev)
		adj = math.Min(adj, 1)
		out[i] = adj
		prev = adj
	}
	return out
}
