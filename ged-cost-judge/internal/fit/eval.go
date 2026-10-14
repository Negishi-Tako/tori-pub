package fit

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
)

// Majority is the majority judgment per triplet (ok = false on a tie).
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

// Agreement is the fraction where the sign of Δ agrees with the majority judgment (A / B) (plan §7.3).
// Triplets whose majority is "about the same" or has no majority are excluded. Δ = 0 counts as 0.5.
// If only is non-nil, only triplets with only[t] true are counted.
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

// subset rebuilds triplets and judgments from a list of source argument indices (duplicates allowed).
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
			t.Source = c // an argument drawn more than once is treated as a separate cluster
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

// Bootstrap is a cluster bootstrap over source arguments (plan §6.4).
// It returns the estimates [θ..., θ_neg] of each round.
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

// CrossValidate is k-fold cross-validation over source arguments. It returns Δ on triplets not used for training.
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

// Quantile is the quantile of values sorted in ascending order (linear interpolation).
func Quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := min(lo+1, len(sorted)-1)
	return sorted[lo] + (pos-float64(lo))*(sorted[hi]-sorted[lo])
}

// SD is the sample standard deviation.
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
