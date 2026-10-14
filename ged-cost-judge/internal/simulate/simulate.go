// Package simulate synthesizes triplets and judgments without using the API (plan §6.5 / P1).
//
// The judges' "true costs" θ* are set off from the current values, and 3-way judgments are produced from GED differences at those costs.
// The goal is to check whether estimation recovers θ* (identifiability and parameter recovery).
package simulate

import (
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/fit"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
)

// Judge is a synthetic judge.
type Judge struct {
	Beta, Tau, Pi float64
}

// Config configures the simulation.
type Config struct {
	Sources  int // number of source arguments (4 triplets each)
	PerSrc   int
	Seed     uint64
	True     ged.Theta
	TrueNeg  float64
	Judges   []Judge
	Sigma    float64 // standard deviation of the per-triplet offset ε_t shared by all judges
	Random   float64 // probability of answering at random
	MinProps int
	MaxProps int
}

// Default is the main scenario of plan §6.5 (β ≈ 2, σ = 1, 10% random).
func Default() Config {
	tr := ged.Current()
	tr[ged.PNodeMissing] = 1.3
	tr[ged.PNodeExtra] = 0.7
	tr[ged.PEdgeMissing] = 1.2
	tr[ged.PTypeRACA] = 1.5
	tr[ged.PDirection] = 1.2
	return Config{
		Sources: 250, PerSrc: 4, Seed: 1,
		True: tr, TrueNeg: 1.5,
		Judges: []Judge{{2.0, 0.6, 0.3}, {1.5, 0.9, -0.2}, {2.5, 0.4, 0.0}},
		Sigma:  1, Random: 0.1, MinProps: 6, MaxProps: 18,
	}
}

// Meta is auxiliary information about a triplet.
type Meta struct {
	Type   perturb.TripletType
	EditsA []perturb.Edit
	EditsB []perturb.Edit
	// TrueDelta is Δ at θ* (including the polarity-flip weight).
	TrueDelta float64
}

type world struct {
	rng  *rand.Rand
	base map[[2]string]float64
}

func (w *world) baseCos(a, b string) float64 {
	if a > b {
		a, b = b, a
	}
	k := [2]string{a, b}
	if c, ok := w.base[k]; ok {
		return c
	}
	c := 0.1 + 0.35*w.rng.Float64()
	w.base[k] = c
	return c
}

func (w *world) source(src int, cfg Config) ged.Graph {
	n := cfg.MinProps + w.rng.IntN(cfg.MaxProps-cfg.MinProps+1)
	var g ged.Graph
	for i := range n {
		g.Props = append(g.Props, ged.Prop{ID: fmt.Sprintf("s%d.p%d", src, i)})
	}
	for i := 1; i < n; i++ {
		if w.rng.Float64() < 0.15 {
			continue // propositions without relations
		}
		parent := w.rng.IntN(i)
		kind := ged.RA
		switch r := w.rng.Float64(); {
		case r < 0.25:
			kind = ged.CA
		case r < 0.4:
			kind = ged.MA
		}
		g.Edges = append(g.Edges, ged.Edge{From: g.Props[i].ID, To: g.Props[parent].ID, Kind: kind})
	}
	return g
}

func (w *world) cosFor(e perturb.Edit) float64 {
	u := w.rng.Float64()
	switch e.Kind {
	case perturb.PPara:
		switch e.Strength {
		case "weak":
			return 0.9 + 0.07*u
		case "medium":
			return 0.83 + 0.07*u
		default:
			return 0.75 + 0.08*u
		}
	case perturb.PShift:
		switch e.Strength {
		case "small":
			return 0.6 + 0.15*u
		case "medium":
			return 0.45 + 0.15*u
		default:
			return 0.25 + 0.2*u
		}
	case perturb.PNeg:
		return 0.85 + 0.1*u
	}
	return 1
}

// sim builds a table of cos between variant and source propositions and returns a function that looks it up.
// Estimation processes triplets in parallel, so the table is fully populated up front (the function only reads).
func (w *world) sim(src ged.Graph, v perturb.Variant, tag string) ged.Sim {
	own := map[string]float64{}
	for _, id := range slices.Sorted(maps.Keys(v.Changed)) { // fix the order of random draws
		own[id] = w.cosFor(v.Changed[id])
	}
	table := map[[2]string]float64{}
	for _, r := range src.Props {
		for _, p := range v.Graph.Props {
			var c float64
			switch o, ok := v.Origin[p.ID]; {
			case !ok:
				c = w.baseCos(r.ID, tag+"/"+p.ID) // an added proposition is unrelated to every proposition
			case o == r.ID:
				c = 1
				if x, ok := own[p.ID]; ok {
					c = x
				}
			default:
				c = w.baseCos(r.ID, o)
			}
			table[[2]string{r.ID, p.ID}] = c
		}
	}
	return func(ref, vid string) float64 { return table[[2]string{ref, vid}] }
}

// Generate synthesizes triplets and judgments.
func Generate(cfg Config) ([]fit.Triplet, []fit.Judgment, []Meta, error) {
	w := &world{rng: rand.New(rand.NewPCG(cfg.Seed, 0x5eed)), base: map[[2]string]float64{}}
	th0 := ged.Current()
	var trips []fit.Triplet
	var metas []Meta
	for s := range cfg.Sources {
		src := w.source(s, cfg)
		for k := range cfg.PerSrc {
			typ := perturb.DrawType(w.rng)
			var va, vb perturb.Variant
			var sa, sb ged.Sim
			for attempt := 0; ; attempt++ {
				ka, kb := perturb.Plan(w.rng, typ)
				if typ == perturb.T1Monotone {
					steps := perturb.Build(w.rng, src, kb)
					va, vb = steps[min(len(ka), len(steps))-1], steps[len(steps)-1]
				} else {
					va = last(perturb.Build(w.rng, src, ka))
					vb = last(perturb.Build(w.rng, src, kb))
				}
				tag := fmt.Sprintf("t%d.%d", s, k)
				sa, sb = w.sim(src, va, tag+"a"), w.sim(src, vb, tag+"b")
				if typ != perturb.T3Contrast || attempt >= 50 {
					break
				}
				// For T3, half have a difference near 0 under current costs and half a clear difference (plan §4.3).
				d, err := delta(src, va, vb, sa, sb, th0)
				if err != nil {
					return nil, nil, nil, err
				}
				if wantNear := k%2 == 0; (math.Abs(d) < 0.3) == wantNear {
					break
				}
			}
			trips = append(trips, fit.Triplet{
				Source: s, Ref: src, A: va.Graph, B: vb.Graph, SimA: sa, SimB: sb,
				NegDiff: float64(vb.Neg - va.Neg),
			})
			metas = append(metas, Meta{Type: typ, EditsA: va.Edits, EditsB: vb.Edits})
		}
	}

	trueDelta, err := fit.Delta(trips, cfg.True)
	if err != nil {
		return nil, nil, nil, err
	}
	var js []fit.Judgment
	for t := range trips {
		d := trueDelta[t] + cfg.TrueNeg*trips[t].NegDiff
		metas[t].TrueDelta = d
		eps := w.rng.NormFloat64() * cfg.Sigma
		for j, jd := range cfg.Judges {
			for _, o := range []float64{1, -1} {
				var y fit.Answer
				if w.rng.Float64() < cfg.Random {
					y = fit.Answer(w.rng.IntN(3))
				} else {
					z := jd.Beta*d + eps + jd.Pi*o
					pa := 1 / (1 + math.Exp(-(z - jd.Tau)))
					pb := 1 / (1 + math.Exp(-(-z - jd.Tau)))
					switch u := w.rng.Float64(); {
					case u < pa:
						y = fit.AnswerA
					case u < pa+pb:
						y = fit.AnswerB
					default:
						y = fit.AnswerTie
					}
				}
				js = append(js, fit.Judgment{Triplet: t, Judge: j, Order: o, Answer: y})
			}
		}
	}
	return trips, js, metas, nil
}

func last(vs []perturb.Variant) perturb.Variant { return vs[len(vs)-1] }

func delta(src ged.Graph, va, vb perturb.Variant, sa, sb ged.Sim, th ged.Theta) (float64, error) {
	fa, err := ged.Compute(src, va.Graph, sa, th)
	if err != nil {
		return 0, err
	}
	fb, err := ged.Compute(src, vb.Graph, sb, th)
	if err != nil {
		return 0, err
	}
	return fb.Cost(th) - fa.Cost(th), nil
}
