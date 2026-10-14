// Package fit は三つ組の 3 択判定から GED のコストを推定する（計画 §6）。
//
// モデル（Rao–Kupper 型の引き分けつき一対比較。順序ロジットと同じ形）:
//
//	Δ_t(θ) = GED_θ(S, B) − GED_θ(S, A)                     （正なら A が近い）
//	z_k    = β_j·(Δ_t(θ) + θ_neg·Δneg_t) + π_j·o_k
//	P(A) = σ(z_k − τ_j),  P(B) = σ(−z_k − τ_j),  P(同程度) = 残り
//
// GED は命題の対応についての最小値なので θ について微分できない点を持つ。
// 対応を固定すれば θ について線形なので、「対応を解く → 対応を固定して L-BFGS で当てはめる」を
// 対応が変わらなくなるまで繰り返す（計画 §6.3）。
//
// 計画との違い（PoC）: 当てはめは (θ, θ_neg, β, τ, π) を交互にではなく同時に L-BFGS で解く。
// 制約 θ ≥ 0・τ > 0・β > 0 は softplus / exp の変数変換で課す。自然さの共変量 γ は入れていない。
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

// Answer は 3 択の回答。
type Answer int

const (
	AnswerA Answer = iota
	AnswerB
	AnswerTie
)

func (a Answer) String() string {
	return [...]string{"A", "B", "同程度"}[a]
}

// Triplet は三つ組 (S, A, B)。
type Triplet struct {
	// Source は元の議論の番号。クラスタブートストラップと交差検証の単位。
	Source int
	Ref    ged.Graph
	A, B   ged.Graph
	SimA   ged.Sim
	SimB   ged.Sim
	// NegDiff は極性反転の個数の差（B − A）。GED はこれを観測できない。
	NegDiff float64
}

// Judgment は 1 回の判定。
type Judgment struct {
	Triplet int
	Judge   int
	// Order は A を先に見せたら +1、後なら −1。
	Order  float64
	Answer Answer
}

// Params は推定するパラメータ。
type Params struct {
	Theta    ged.Theta
	ThetaNeg float64
	Beta     []float64
	Tau      []float64
	Pi       []float64
}

// Options は推定の設定。
type Options struct {
	Judges int
	// Theta0 は L2 正則化の中心（現行値）。
	Theta0 ged.Theta
	// Lambda は λ‖θ − θ₀‖² の係数。
	Lambda float64
	// FitNeg が false なら θ_neg を 0 に固定する。
	FitNeg bool
	// FixTheta が true なら θ を Theta0 に固定し、補助パラメータだけを当てはめる
	// （固定の規則にも較正の機会を与えて対数損失を比べるため。計画 §7.3）。
	FixTheta bool
	MaxOuter int
	// AuxRidge は log β・log τ・π にかける L2 の係数（0 なら 1e-4）。
	// 判定者が 1 人で判定が少ないと、完全に分離できて β が発散するので、小規模では弱い事前分布として使う。
	AuxRidge float64
	// Fixed は値を固定する θ の添字と値（プロファイル尤度を描くため）。
	Fixed map[int]float64
}

// Result は推定の結果。
type Result struct {
	Params Params
	// NegLogLik は罰則を除いた負の対数尤度。
	NegLogLik float64
	Outer     int
	Converged bool
	// Cycled は対応が以前の状態に戻った（交互最適化が循環した）ことを表す。
	// そのときは、各回の θ で対応を解き直した本物の目的関数が最小の回を返す。
	Cycled bool
}

// diff は対応を固定したときの Δ_t = dphi·θ + c。
type diff struct {
	dphi [ged.NumParams]float64
	c    float64
	sig  string
}

// Features は θ で全三つ組の対応を解き直し、Δ の線形化を返す。
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

// Signatures は θ での三つ組ごとの命題の対応（A と B）の識別子。
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

// Delta は θ での Δ_t（本物の GED の差）を返す。
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

// 変数ベクトルの並び: u[11]（θ = softplus(u)）, neg, b[J]（β = exp）, s[J]（τ = exp）, p[J]（π）。
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

// logSigmoid は log σ(x) を安定に計算する。
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

// answerLogLik は 1 判定の対数尤度と、z・τ についての微分を返す。
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

// objective は罰則つきの負の対数尤度と勾配。
type objective struct {
	l      layout
	ds     []diff
	neg    []float64
	js     []Judgment
	opts   Options
	ridge  float64 // 補助パラメータの数値安定用のごく弱い罰則
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

// Init は初期値（θ = Theta0、β = 1、τ = 0.5、π = 0）。
func Init(opts Options) Params {
	p := Params{Theta: opts.Theta0, Beta: make([]float64, opts.Judges), Tau: make([]float64, opts.Judges), Pi: make([]float64, opts.Judges)}
	for j := range opts.Judges {
		p.Beta[j], p.Tau[j] = 1, 0.5
	}
	return p
}

// Fit は交互最適化で推定する。init が nil なら Init(opts) から始める。
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

		// この θ で対応を解き直し、本物の GED での目的関数を測る（GED は対応についての最小値）。
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

// NegLogLik は与えたパラメータでの負の対数尤度（罰則なし、対応は p.Theta で解く）。
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
