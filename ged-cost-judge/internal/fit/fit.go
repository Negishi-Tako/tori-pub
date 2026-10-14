// Package fit estimates GED costs from 3-way judgments on triplets (plan §6).
//
// Model (Rao–Kupper-style paired comparison with ties; the same form as an ordered logit):
//
//	Δ_t(θ) = GED_θ(S, B) − GED_θ(S, A)                     (positive means A is closer)
//	z_k    = β_j·(Δ_t(θ) + θ_neg·Δneg_t) + π_j·o_k
//	P(A) = σ(z_k − τ_j),  P(B) = σ(−z_k − τ_j),  P(same) = the rest
//
// GED is a minimum over proposition matchings, so it has points where it is not differentiable in θ.
// With the matching fixed it is linear in θ, so "solve the matching → fix it and fit with L-BFGS" is
// repeated until the matching stops changing (plan §6.3).
//
// Differences from the plan (PoC): the fit solves (θ, θ_neg, β, τ, π) jointly with L-BFGS rather than alternating.
// Constraints θ ≥ 0, τ > 0, β > 0 are imposed via softplus / exp reparameterization. The naturalness covariate γ is not included.
package fit

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/optimize"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
)

// Answer is a 3-way answer.
type Answer int

const (
	AnswerA Answer = iota
	AnswerB
	AnswerTie
)

func (a Answer) String() string {
	return [...]string{"A", "B", "同程度"}[a]
}

// Triplet is a triplet (S, A, B).
type Triplet struct {
	// Source is the source argument index. The unit for cluster bootstrap and cross-validation.
	Source int
	Ref    ged.Graph
	A, B   ged.Graph
	SimA   ged.Sim
	SimB   ged.Sim
	// NegDiff is the difference in the number of polarity flips (B − A). GED cannot observe it.
	NegDiff float64
}

// Judgment is a single judgment.
type Judgment struct {
	Triplet int
	Judge   int
	// Order is +1 if A was shown first, −1 if second.
	Order  float64
	Answer Answer
}

// Params are the parameters to estimate.
type Params struct {
	Theta    ged.Theta
	ThetaNeg float64
	Beta     []float64
	Tau      []float64
	Pi       []float64
}

// Options configures estimation.
type Options struct {
	Judges int
	// Theta0 is the center of L2 regularization (the current values).
	Theta0 ged.Theta
	// Lambda is the coefficient of λ‖θ − θ₀‖².
	Lambda float64
	// If FitNeg is false, θ_neg is fixed at 0.
	FitNeg bool
	// If FixTheta is true, θ is fixed at Theta0 and only the auxiliary parameters are fitted
	// (to give fixed rules a chance at calibration too and compare log loss; plan §7.3).
	FixTheta bool
	MaxOuter int
	// AuxRidge is the L2 coefficient on log β, log τ and π (0 means 1e-4).
	// With a single judge and few judgments, perfect separation makes β diverge, so at small scale it acts as a weak prior.
	AuxRidge float64
	// Fixed holds θ indices and values to fix (for drawing profile likelihoods).
	Fixed map[int]float64
}

// Result is the estimation result.
type Result struct {
	Params Params
	// NegLogLik is the negative log-likelihood excluding penalties.
	NegLogLik float64
	Outer     int
	Converged bool
	// Cycled indicates the matching returned to an earlier state (the alternating optimization cycled).
	// In that case, the round with the smallest true objective, re-solving the matching at each round's θ, is returned.
	Cycled bool
}

// diff is Δ_t = dphi·θ + c with the matching fixed.
type diff struct {
	dphi [ged.NumParams]float64
	c    float64
	sig  string
}

// Features re-solves the matching of every triplet at θ and returns the linearization of Δ.
func features(trips []Triplet, th ged.Theta) ([]diff, error) {
	out := make([]diff, len(trips))
	errs := make([]error, len(trips))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for i := range trips {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			t := trips[i]
			fa, err := ged.Compute(t.Ref, t.A, t.SimA, th)
			if err != nil {
				errs[i] = err
				return
			}
			fb, err := ged.Compute(t.Ref, t.B, t.SimB, th)
			if err != nil {
				errs[i] = err
				return
			}
			var d diff
			for p := range ged.NumParams {
				d.dphi[p] = fb.Phi[p] - fa.Phi[p]
			}
			d.c = (fb.EdgeExtra - fa.EdgeExtra) * ged.EdgeExtraCost
			d.sig = fa.Signature() + "|" + fb.Signature()
			out[i] = d
		}(i)
	}
	wg.Wait()
	return out, errors.Join(errs...)
}

// Signatures identifies the proposition matching (A and B) per triplet at θ.
func Signatures(trips []Triplet, th ged.Theta) ([]string, error) {
	ds, err := features(trips, th)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(ds))
	for i, d := range ds {
		out[i] = d.sig
	}
	return out, nil
}

// Delta returns Δ_t at θ (the difference in real GED).
func Delta(trips []Triplet, th ged.Theta) ([]float64, error) {
	ds, err := features(trips, th)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(ds))
	for i, d := range ds {
		out[i] = d.value(th)
	}
	return out, nil
}

func (d diff) value(th ged.Theta) float64 {
	v := d.c
	for p := range ged.NumParams {
		v += d.dphi[p] * th[p]
	}
	return v
}

// Variable vector layout: u[11] (θ = softplus(u)), neg, b[J] (β = exp), s[J] (τ = exp), p[J] (π).
type layout struct{ j int }

func (l layout) size() int      { return ged.NumParams + 1 + 3*l.j }
func (l layout) neg() int       { return ged.NumParams }
func (l layout) beta(j int) int { return ged.NumParams + 1 + j }
func (l layout) tau(j int) int  { return ged.NumParams + 1 + l.j + j }
func (l layout) pi(j int) int   { return ged.NumParams + 1 + 2*l.j + j }

func softplus(u float64) float64 {
	if u > 30 {
		return u
	}
	return math.Log1p(math.Exp(u))
}

func invSoftplus(t float64) float64 {
	t = math.Max(t, 1e-4)
	if t > 30 {
		return t
	}
	return math.Log(math.Expm1(t))
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// logSigmoid computes log σ(x) stably.
func logSigmoid(x float64) float64 {
	if x < 0 {
		return x - math.Log1p(math.Exp(x))
	}
	return -math.Log1p(math.Exp(-x))
}

func (l layout) pack(p Params) []float64 {
	x := make([]float64, l.size())
	for i := range ged.NumParams {
		x[i] = invSoftplus(p.Theta[i])
	}
	x[l.neg()] = p.ThetaNeg
	for j := range l.j {
		x[l.beta(j)] = math.Log(p.Beta[j])
		x[l.tau(j)] = math.Log(p.Tau[j])
		x[l.pi(j)] = p.Pi[j]
	}
	return x
}

func (l layout) unpack(x []float64) Params {
	p := Params{Beta: make([]float64, l.j), Tau: make([]float64, l.j), Pi: make([]float64, l.j)}
	for i := range ged.NumParams {
		p.Theta[i] = softplus(x[i])
	}
	p.ThetaNeg = x[l.neg()]
	for j := range l.j {
		p.Beta[j] = math.Exp(x[l.beta(j)])
		p.Tau[j] = math.Exp(x[l.tau(j)])
		p.Pi[j] = x[l.pi(j)]
	}
	return p
}

// answerLogLik returns the log-likelihood of one judgment and its derivatives with respect to z and τ.
func answerLogLik(y Answer, z, tau float64) (ll, dz, dtau float64) {
	const floor = 1e-12
	switch y {
	case AnswerA:
		a := sigmoid(z - tau)
		return logSigmoid(z - tau), 1 - a, -(1 - a)
	case AnswerB:
		b := sigmoid(-z - tau)
		return logSigmoid(-z - tau), -(1 - b), -(1 - b)
	default:
		a, b := sigmoid(z-tau), sigmoid(-z-tau)
		pt := math.Max(1-a-b, floor)
		return math.Log(pt), (-a*(1-a) + b*(1-b)) / pt, (a*(1-a) + b*(1-b)) / pt
	}
}

// objective is the penalized negative log-likelihood and its gradient.
type objective struct {
	l      layout
	ds     []diff
	neg    []float64
	js     []Judgment
	opts   Options
	ridge  float64 // a very weak penalty on auxiliary parameters for numerical stability
	nllRaw float64
}

func (o *objective) eval(x, grad []float64) float64 {
	l := o.l
	p := l.unpack(x)
	if grad != nil {
		clear(grad)
	}
	var gTheta [ged.NumParams]float64
	var gNeg float64
	gBeta := make([]float64, l.j)
	gTau := make([]float64, l.j)
	gPi := make([]float64, l.j)

	nll := 0.0
	for _, k := range o.js {
		d := o.ds[k.Triplet]
		D := d.value(p.Theta) + p.ThetaNeg*o.neg[k.Triplet]
		j := k.Judge
		z := p.Beta[j]*D + p.Pi[j]*k.Order
		ll, dz, dtau := answerLogLik(k.Answer, z, p.Tau[j])
		nll -= ll
		if grad == nil {
			continue
		}
		for i := range ged.NumParams {
			gTheta[i] -= dz * p.Beta[j] * d.dphi[i]
		}
		gNeg -= dz * p.Beta[j] * o.neg[k.Triplet]
		gBeta[j] -= dz * D
		gTau[j] -= dtau
		gPi[j] -= dz * k.Order
	}
	o.nllRaw = nll

	f := nll
	for i := range ged.NumParams {
		dev := p.Theta[i] - o.opts.Theta0[i]
		f += o.opts.Lambda * dev * dev
		gTheta[i] += 2 * o.opts.Lambda * dev
	}
	f += o.ridge * p.ThetaNeg * p.ThetaNeg
	gNeg += 2 * o.ridge * p.ThetaNeg
	for j := range l.j {
		lb, lt := math.Log(p.Beta[j]), math.Log(p.Tau[j])
		f += o.ridge * (lb*lb + lt*lt + p.Pi[j]*p.Pi[j])
		if grad != nil {
			grad[l.beta(j)] = gBeta[j]*p.Beta[j] + 2*o.ridge*lb
			grad[l.tau(j)] = gTau[j]*p.Tau[j] + 2*o.ridge*lt
			grad[l.pi(j)] = gPi[j] + 2*o.ridge*p.Pi[j]
		}
	}
	if grad != nil {
		for i := range ged.NumParams {
			if _, fixed := o.opts.Fixed[i]; !o.opts.FixTheta && !fixed {
				grad[i] = gTheta[i] * sigmoid(x[i])
			}
		}
		if o.opts.FitNeg {
			grad[l.neg()] = gNeg
		}
	}
	return f
}

// Init returns the initial values (θ = Theta0, β = 1, τ = 0.5, π = 0).
func Init(opts Options) Params {
	p := Params{Theta: opts.Theta0, Beta: make([]float64, opts.Judges), Tau: make([]float64, opts.Judges), Pi: make([]float64, opts.Judges)}
	for j := range opts.Judges {
		p.Beta[j], p.Tau[j] = 1, 0.5
	}
	return p
}

// Fit estimates by alternating optimization. If init is nil, it starts from Init(opts).
func Fit(trips []Triplet, js []Judgment, opts Options, init *Params) (Result, error) {
	if opts.Judges <= 0 {
		return Result{}, errors.New("fit: Judges must be positive")
	}
	if opts.MaxOuter <= 0 {
		opts.MaxOuter = 20
	}
	p := Init(opts)
	if init != nil {
		p = clone(*init)
	}
	if opts.FixTheta {
		p.Theta = opts.Theta0
	}
	if !opts.FitNeg {
		p.ThetaNeg = 0
	}
	for i, v := range opts.Fixed {
		p.Theta[i] = v
	}
	neg := make([]float64, len(trips))
	for i, t := range trips {
		neg[i] = t.NegDiff
	}
	l := layout{j: opts.Judges}

	ds, err := features(trips, p.Theta)
	if err != nil {
		return Result{}, err
	}
	ridge := opts.AuxRidge
	if ridge <= 0 {
		ridge = 1e-4
	}
	seen := map[string]bool{sigOf(ds): true}
	best, bestF := Result{}, math.Inf(1)
	for outer := 1; outer <= opts.MaxOuter; outer++ {
		obj := &objective{l: l, ds: ds, neg: neg, js: js, opts: opts, ridge: ridge}
		prob := optimize.Problem{
			Func: func(x []float64) float64 { return obj.eval(x, nil) },
			Grad: func(g, x []float64) { obj.eval(x, g) },
		}
		r, err := optimize.Minimize(prob, l.pack(p), &optimize.Settings{
			GradientThreshold: 1e-6,
			MajorIterations:   5000,
		}, &optimize.LBFGS{})
		if err != nil && r == nil {
			return best, fmt.Errorf("fit: L-BFGS: %w", err)
		}
		p = l.unpack(r.X)

		// Re-solve the matching at this θ and measure the objective under real GED (GED is a minimum over matchings).
		next, err := features(trips, p.Theta)
		if err != nil {
			return best, err
		}
		trueObj := &objective{l: l, ds: next, neg: neg, js: js, opts: opts, ridge: ridge}
		f := trueObj.eval(r.X, nil)
		cur := Result{Params: p, NegLogLik: trueObj.nllRaw, Outer: outer}
		if f < bestF {
			best, bestF = cur, f
		}
		sig := sigOf(next)
		if sig == sigOf(ds) {
			cur.Converged = true
			return cur, nil
		}
		if seen[sig] {
			best.Cycled, best.Outer = true, outer
			return best, nil
		}
		seen[sig] = true
		ds = next
	}
	best.Outer = opts.MaxOuter
	return best, nil
}

func sigOf(ds []diff) string {
	h := sha256.New()
	for _, d := range ds {
		h.Write([]byte(d.sig))
		h.Write([]byte{0})
	}
	return string(h.Sum(nil))
}

func clone(p Params) Params {
	q := p
	q.Beta = append([]float64(nil), p.Beta...)
	q.Tau = append([]float64(nil), p.Tau...)
	q.Pi = append([]float64(nil), p.Pi...)
	return q
}

// NegLogLik is the negative log-likelihood at the given parameters (no penalty; matching solved at p.Theta).
func NegLogLik(trips []Triplet, js []Judgment, p Params) (float64, error) {
	ds, err := features(trips, p.Theta)
	if err != nil {
		return 0, err
	}
	nll := 0.0
	for _, k := range js {
		D := ds[k.Triplet].value(p.Theta) + p.ThetaNeg*trips[k.Triplet].NegDiff
		z := p.Beta[k.Judge]*D + p.Pi[k.Judge]*k.Order
		ll, _, _ := answerLogLik(k.Answer, z, p.Tau[k.Judge])
		nll -= ll
	}
	return nll, nil
}
