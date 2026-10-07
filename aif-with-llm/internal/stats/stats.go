// Package stats は条件間比較の統計処理。
//
// 方針:
//   - p 値だけで主張しない。効果量と信頼区間を必ず併記する
//   - 条件間比較は対応のある検定（同じノードセット同士を組にする）
//   - 多重比較補正（Holm）を掛ける
package stats

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// Summary は 1 系列の要約統計。
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

// PairedComparison は対応のある 2 条件の比較結果。
type PairedComparison struct {
	N int `json:"n"`
	// MeanDiff は a - b の平均。
	MeanDiff float64 `json:"mean_diff"`
	// CILow / CIHigh は bootstrap による差の信頼区間。
	CILow  float64 `json:"ci_low"`
	CIHigh float64 `json:"ci_high"`
	// PValue は Wilcoxon 符号順位検定（両側・正規近似）。
	PValue float64 `json:"p_value"`
	// CliffsDelta は効果量（-1〜1）。
	CliffsDelta float64 `json:"cliffs_delta"`
	// Magnitude は Cliff's delta の慣例的な解釈。
	Magnitude string `json:"magnitude"`
}

// ComparePaired は対応のある 2 系列を比べる。
//
// a と b は同じ順序で対応していること（同じノードセット）。
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

// BootstrapCI は差の平均の信頼区間をブートストラップで求める。
//
// seed を渡すのは、同じデータから同じ区間が出ることを保証するため
// （報告した数値を後から再現できないと困る）。
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

// WilcoxonSignedRank は符号順位検定の両側 p 値（正規近似・連続性補正あり）。
//
// 正規近似なので、有効な対（差が 0 でないもの）が少ないと精度が落ちる。
// 対が 10 未満のときは参考値として扱うこと（本実験は n=20〜124）。
func WilcoxonSignedRank(diffs []float64) float64 {
	type entry struct {
		abs  float64
		sign float64
	}
	var entries []entry
	for _, d := range diffs {
		if d == 0 {
			continue // 差が 0 の対は検定から外す（標準的な扱い）
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

	// 同順位は平均順位を割り当てる。
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
	z := (math.Abs(w-mean) - 0.5) / math.Sqrt(variance) // 連続性補正
	if z < 0 {
		z = 0
	}
	return 2 * (1 - normalCDF(z))
}

func normalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

// CliffsDelta は効果量。a が b より大きい傾向をどれだけ持つか（-1〜1）。
//
// 順位ベースなので分布の形を仮定しない。平均差だけでは
// 「ばらつきの中に埋もれた差」を過大評価しかねないので併記する。
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

// DeltaMagnitude は Cliff's delta の慣例的な区切り（Romano et al. 2006）。
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

// HolmAdjust は Holm-Bonferroni 法で p 値を補正する。
//
// 入力と同じ順序で補正後の値を返す。指標を増やすほど個々の閾値が厳しくなるので、
// 「指標を増やして当たりを探す」ことへの歯止めになる。
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
		// 単調性を保つ（順位が下の p 値が上を下回らないようにする）。
		adj = math.Max(adj, prev)
		adj = math.Min(adj, 1)
		out[i] = adj
		prev = adj
	}
	return out
}
