package fit

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
)

// Majority は三つ組ごとの判定の多数（同数で割れたら ok = false）。
func Majority(n int, js []Judgment) (maj []Answer, ok []bool) {
	counts := make([][3]int, n)
	for _, k := range js {
		counts[k.Triplet][k.Answer]++
	}
	maj = make([]Answer, n)
	ok = make([]bool, n)
	for t, c := range counts {
		best, top, tied := 0, -1, false
		for a, v := range c {
			switch {
			case v > top:
				best, top, tied = a, v, false
			case v == top:
				tied = true
			}
		}
		maj[t], ok[t] = Answer(best), top > 0 && !tied
	}
	return maj, ok
}

// Agreement は Δ の符号が判定の多数（A / B）と一致する割合（計画 §7.3）。
// 多数が「同程度」の三つ組と多数が決まらない三つ組は除く。Δ = 0 は 0.5 と数える。
// only が nil でなければ only[t] が true の三つ組だけを数える。
func Agreement(delta []float64, js []Judgment, only []bool) (rate float64, n int) {
	maj, ok := Majority(len(delta), js)
	hit := 0.0
	for t, d := range delta {
		if !ok[t] || maj[t] == AnswerTie || (only != nil && !only[t]) {
			continue
		}
		n++
		switch {
		case math.Abs(d) < 1e-9:
			hit += 0.5
		case (d > 0) == (maj[t] == AnswerA):
			hit++
		}
	}
	if n == 0 {
		return math.NaN(), 0
	}
	return hit / float64(n), n
}

// subset は元の議論の番号の列（重複可）から三つ組と判定を組み直す。
func subset(trips []Triplet, js []Judgment, sources []int) ([]Triplet, []Judgment) {
	bySource := map[int][]int{}
	for i, t := range trips {
		bySource[t.Source] = append(bySource[t.Source], i)
	}
	byTriplet := make([][]Judgment, len(trips))
	for _, k := range js {
		byTriplet[k.Triplet] = append(byTriplet[k.Triplet], k)
	}
	var ot []Triplet
	var oj []Judgment
	for c, s := range sources {
		for _, i := range bySource[s] {
			t := trips[i]
			t.Source = c // 重複して引いた議論は別のクラスタとして扱う
			idx := len(ot)
			ot = append(ot, t)
			for _, k := range byTriplet[i] {
				k.Triplet = idx
				oj = append(oj, k)
			}
		}
	}
	return ot, oj
}

func sourceIDs(trips []Triplet) []int {
	seen := map[int]bool{}
	var out []int
	for _, t := range trips {
		if !seen[t.Source] {
			seen[t.Source] = true
			out = append(out, t.Source)
		}
	}
	slices.Sort(out)
	return out
}

// Bootstrap は元の議論単位のクラスタブートストラップ（計画 §6.4）。
// 各回の推定値 [θ..., θ_neg] を返す。
func Bootstrap(trips []Triplet, js []Judgment, opts Options, est Params, reps int, seed uint64) ([][]float64, error) {
	ids := sourceIDs(trips)
	out := make([][]float64, reps)
	errs := make([]error, reps)
	var wg sync.WaitGroup
	sem := make(chan struct{}, max(1, runtime.NumCPU()/2))
	for r := range reps {
		rng := rand.New(rand.NewPCG(seed, uint64(r)))
		draw := make([]int, len(ids))
		for i := range draw {
			draw[i] = ids[rng.IntN(len(ids))]
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(r int, draw []int) {
			defer func() { <-sem; wg.Done() }()
			bt, bj := subset(trips, js, draw)
			res, err := Fit(bt, bj, opts, &est)
			if err != nil {
				errs[r] = err
				return
			}
			v := append(res.Params.Theta[:], res.Params.ThetaNeg)
			out[r] = slices.Clone(v)
		}(r, draw)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// CrossValidate は元の議論単位の k 分割交差検証。学習に使っていない三つ組での Δ を返す。
func CrossValidate(trips []Triplet, js []Judgment, opts Options, k int, seed uint64) ([]float64, error) {
	ids := sourceIDs(trips)
	rng := rand.New(rand.NewPCG(seed, 7))
	rng.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	fold := map[int]int{}
	for i, s := range ids {
		fold[s] = i % k
	}
	out := make([]float64, len(trips))
	for f := range k {
		var train []int
		for _, s := range ids {
			if fold[s] != f {
				train = append(train, s)
			}
		}
		tt, tj := subset(trips, js, train)
		res, err := Fit(tt, tj, opts, nil)
		if err != nil {
			return nil, err
		}
		var test []Triplet
		var idx []int
		for i, t := range trips {
			if fold[t.Source] == f {
				test = append(test, t)
				idx = append(idx, i)
			}
		}
		d, err := Delta(test, res.Params.Theta)
		if err != nil {
			return nil, err
		}
		for i, v := range d {
			out[idx[i]] = v
		}
	}
	return out, nil
}

// Quantile は昇順に並べた値の分位点（線形補間）。
func Quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := min(lo+1, len(sorted)-1)
	return sorted[lo] + (pos-float64(lo))*(sorted[hi]-sorted[lo])
}

// SD は標本標準偏差。
func SD(v []float64) float64 {
	if len(v) < 2 {
		return math.NaN()
	}
	m := 0.0
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	s := 0.0
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(v)-1))
}
