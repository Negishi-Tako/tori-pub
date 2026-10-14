package main

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/synth"
)

// 書き下し（G3）と検証（G4）を三つ組の枠ごとに通す。
//
//   - 変種を書き下したら、別のリクエストで luna に検証させる（計画では生成と別系列のモデルだが、PoC では luna だけ）
//   - 不合格なら理由を添えて 1 回だけ作り直し、それでも不合格なら編集計画を引き直す（枠ごとに最大 3 回）
//   - 「グラフを変える編集があるのに文章が元のまま」は検証の前に機械的に落とす
//   - T3 は、現行の値段で差が 0 付近の組と明確な組を半々にする（計画 §4.3）。
//     書き下す前に cos が分からないので、編集の強さから見込んだ cos で差を予測して選ぶ

const nearThreshold = 0.3

type variantCheck struct {
	Tries        int                 `json:"tries"`
	Passed       bool                `json:"passed"`
	Naturalness  int                 `json:"naturalness,omitempty"`
	Problems     [][]string          `json:"problems,omitempty"`
	Verification *synth.Verification `json:"verification,omitempty"`
}

type verifyStats struct {
	mu        sync.Mutex
	byKind    map[perturb.Kind]*[4]int // 試行した変種, 1 回目で合格, 作り直しで合格, 不合格
	reasons   map[string]int
	natural   [6]int
	slots     int
	accepted  int
	discarded int
	replans   int
}

func (s *verifyStats) record(v perturb.Variant, passedAt int, problems [][]string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := map[perturb.Kind]bool{}
	for _, e := range v.Edits {
		if e.Kind == perturb.XSpk || seen[e.Kind] {
			continue
		}
		seen[e.Kind] = true
		c := s.byKind[e.Kind]
		if c == nil {
			c = &[4]int{}
			s.byKind[e.Kind] = c
		}
		c[0]++
		switch passedAt {
		case 0:
			c[1]++
		case 1:
			c[2]++
		default:
			c[3]++
		}
	}
	for _, ps := range problems {
		for _, p := range ps {
			s.reasons[reasonOf(p)]++
		}
	}
}

func reasonOf(p string) string {
	switch {
	case strings.Contains(p, "not clearly visible"), strings.Contains(p, "not checked"):
		return "編集が文章に出ていない"
	case strings.Contains(p, "left unchanged"):
		return "文章が元のまま（機械検査）"
	case strings.Contains(p, "no edit asked for"):
		return "指示していない変更がある"
	case strings.Contains(p, "unnaturally"):
		return "不自然（自然さ 2 以下）"
	}
	return "出力を使えない（形式の誤りなど）"
}

type slotSpec struct {
	source, k int
	typ       perturb.TripletType
}

type verifyPipeline struct {
	ctx     context.Context
	g       *synth.Generator
	mt      *meter
	sources []synth.Discussion
	seed    uint64
	stats   *verifyStats
}

// buildVerified は全枠の三つ組を作る。失敗した枠は Err つきで返す。
func buildVerified(ctx context.Context, g *synth.Generator, mt *meter, sources []synth.Discussion, srcErr []error, rng *rand.Rand, seed uint64) ([]*pilotTriplet, *verifyStats) {
	pp := &verifyPipeline{ctx: ctx, g: g, mt: mt, sources: sources, seed: seed,
		stats: &verifyStats{byKind: map[perturb.Kind]*[4]int{}, reasons: map[string]int{}}}
	var slots []slotSpec
	for i := range sources {
		if srcErr[i] != nil {
			continue
		}
		for k := range 4 {
			slots = append(slots, slotSpec{source: i, k: k, typ: perturb.DrawType(rng)})
		}
	}
	trips := make([]*pilotTriplet, len(slots))
	parallel(len(slots), 8, func(i int) error {
		trips[i] = pp.slot(slots[i])
		return nil
	})
	pp.stats.slots = len(slots)
	for _, t := range trips {
		pp.stats.replans += t.Replans
		if t.Err == "" {
			pp.stats.accepted++
			pp.stats.natural[t.CheckA.Naturalness]++
			pp.stats.natural[t.CheckB.Naturalness]++
		} else {
			pp.stats.discarded++
		}
	}
	return trips, pp.stats
}

func (pp *verifyPipeline) slot(s slotSpec) *pilotTriplet {
	rng := rand.New(rand.NewPCG(pp.seed^0xc0ffee, uint64(s.source*4+s.k)))
	src := pp.sources[s.source]
	prefix := fmt.Sprintf("s%d.", s.source)
	t := &pilotTriplet{Source: s.source, Type: s.typ}
	var near *bool
	if s.typ == perturb.T3Contrast {
		v := s.k%2 == 0
		near = &v
		t.NearTarget = near
	}
	for plan := range 3 {
		if err := pp.mt.check(); err != nil {
			t.Err = err.Error()
			return t
		}
		t.Replans = plan
		a, b, pred := drawPair(rng, src.Graph, s.typ, near)
		t.A, t.B, t.EditsA, t.EditsB, t.PredDelta = a, b, a.Edits, b.Edits, pred
		da, sa, ca, ok := pp.realizeVerified(src, a, prefix)
		t.CheckA = ca
		if !ok {
			continue
		}
		db, sb, cb, ok := pp.realizeVerified(src, b, prefix)
		t.CheckB = cb
		if !ok {
			continue
		}
		t.TextA, t.TextB, t.StatA, t.StatB, t.Err = da, db, sa, sb, ""
		return t
	}
	t.Err = "discarded: 3 回の編集計画とも検証に通らなかった"
	return t
}

// drawPair は編集計画を引く。near が非 nil なら、予測した差が目標（0 付近か明確か）に合うまで引き直す。
func drawPair(rng *rand.Rand, src ged.Graph, typ perturb.TripletType, near *bool) (a, b perturb.Variant, pred float64) {
	for range 40 {
		ka, kb := perturb.Plan(rng, typ)
		if typ == perturb.T1Monotone {
			steps := perturb.Build(rng, src, kb)
			a, b = steps[min(len(ka), len(steps))-1], steps[len(steps)-1]
		} else {
			a, b = lastOf(perturb.Build(rng, src, ka)), lastOf(perturb.Build(rng, src, kb))
		}
		if near == nil {
			return a, b, math.NaN()
		}
		pred = predictedDelta(src, a, b)
		if (math.Abs(pred) < nearThreshold) == *near {
			return a, b, pred
		}
	}
	return a, b, pred
}

// predictedDelta は編集の強さから見込んだ cos で、現行の値段での差 Δ を予測する。
func predictedDelta(src ged.Graph, a, b perturb.Variant) float64 {
	th := ged.Current()
	fa, errA := ged.Compute(src, a.Graph, predictedSim(a), th)
	fb, errB := ged.Compute(src, b.Graph, predictedSim(b), th)
	if errA != nil || errB != nil {
		return math.NaN()
	}
	return fb.Cost(th) - fa.Cost(th)
}

func predictedSim(v perturb.Variant) ged.Sim {
	para := map[string]float64{"weak": 0.935, "medium": 0.865, "strong": 0.79}
	shift := map[string]float64{"small": 0.675, "medium": 0.525, "large": 0.35}
	own := map[string]float64{}
	for id, e := range v.Changed {
		switch e.Kind {
		case perturb.PPara:
			own[id] = para[e.Strength]
		case perturb.PShift:
			own[id] = shift[e.Strength]
		case perturb.PNeg:
			own[id] = 0.9
		}
	}
	return func(ref, vid string) float64 {
		o, ok := v.Origin[vid]
		switch {
		case !ok || o != ref:
			return 0.3 // 別の命題どうしの cos の見込み
		case own[vid] > 0:
			return own[vid]
		}
		return 1
	}
}

// realizeVerified は書き下して検証し、不合格なら理由を添えて 1 回だけ作り直す。
func (pp *verifyPipeline) realizeVerified(src synth.Discussion, v perturb.Variant, prefix string) (synth.Discussion, synth.RealizeStats, variantCheck, bool) {
	var rec variantCheck
	feedback := ""
	n := len(synth.Instructions(v))
	for r := range 2 {
		if pp.mt.check() != nil {
			break
		}
		rec.Tries++
		d, st, resp, err := pp.g.Realize(pp.ctx, src, v, prefix, r, feedback)
		pp.mt.add("realize", resp)
		if err != nil {
			rec.Problems = append(rec.Problems, []string{"error: " + err.Error()})
			feedback = "the output could not be used"
			continue
		}
		if n == 0 {
			rec.Passed = true
			pp.stats.record(v, r, nil)
			return d, st, rec, true
		}
		if changesGraph(v.Edits) && slices.Equal(d.Utterances, src.Utterances) {
			rec.Problems = append(rec.Problems, []string{"the text was left unchanged"})
			feedback = "the text was left unchanged; every edit must be visible in the text"
			continue
		}
		ver, resp, err := pp.g.Verify(pp.ctx, src, d, v)
		pp.mt.add("verify", resp)
		if err != nil {
			rec.Problems = append(rec.Problems, []string{"error: " + err.Error()})
			continue
		}
		probs := ver.Problems(n)
		if len(probs) == 0 {
			rec.Passed, rec.Naturalness, rec.Verification = true, ver.Natural(), &ver
			pp.stats.record(v, r, rec.Problems)
			return d, st, rec, true
		}
		rec.Problems = append(rec.Problems, probs)
		feedback = strings.Join(probs, "; ")
	}
	pp.stats.record(v, -1, rec.Problems)
	return synth.Discussion{}, synth.RealizeStats{}, rec, false
}

// writeVerifyReport は検証と割り付けの集計を書く。
func writeVerifyReport(b *bytes.Buffer, s *verifyStats, kept []*pilotTriplet, d0 []float64) {
	fmt.Fprintln(b, "\n## 書き下しの検証（G4 の代わり。luna が別のリクエストで検証）")
	fmt.Fprintf(b, "\n- 三つ組の枠 %d、採用 %d、破棄 %d（3 回の編集計画とも不合格）。編集計画の引き直し 計 %d 回\n", s.slots, s.accepted, s.discarded, s.replans)
	fmt.Fprintln(b, "- 合格の基準: 指示した編集がすべて文章に出ている・指示外の変更が無い・自然さ 3 以上。不合格は理由を添えて 1 回だけ作り直す")
	fmt.Fprintln(b, "\n| 編集 | 書き下した変種 | 1 回目で合格 | 作り直しで合格 | 不合格 |")
	fmt.Fprintln(b, "| --- | --: | --: | --: | --: |")
	for _, k := range slices.Sorted(func(yield func(perturb.Kind) bool) {
		for k := range s.byKind {
			if !yield(k) {
				return
			}
		}
	}) {
		c := s.byKind[k]
		fmt.Fprintf(b, "| %s | %d | %d（%.0f%%） | %d | %d（%.0f%%） |\n", k, c[0], c[1], pct(c[1], c[0]), c[2], c[3], pct(c[3], c[0]))
	}
	fmt.Fprintln(b, "\n不合格の理由（試行ごと、重複あり）:")
	fmt.Fprintln(b, "\n| 理由 | 回数 |")
	fmt.Fprintln(b, "| --- | --: |")
	for _, r := range slices.Sorted(func(yield func(string) bool) {
		for r := range s.reasons {
			if !yield(r) {
				return
			}
		}
	}) {
		fmt.Fprintf(b, "| %s | %d |\n", r, s.reasons[r])
	}
	fmt.Fprintf(b, "\n採用した変種の自然さ（検証者の評価）: 5 = %d、4 = %d、3 = %d（話者の入れ替えだけの変種は評価なし: %d）\n",
		s.natural[5], s.natural[4], s.natural[3], s.natural[0])

	// T3 の割り付けが実際の差でどうなったか（予測は見込みの cos、実際は埋め込みの cos）。
	var nearN, nearHit, clearN, clearHit int
	for i, t := range kept {
		if t.NearTarget == nil {
			continue
		}
		isNear := math.Abs(d0[i]) < nearThreshold
		if *t.NearTarget {
			nearN++
			if isNear {
				nearHit++
			}
		} else {
			clearN++
			if !isNear {
				clearHit++
			}
		}
	}
	fmt.Fprintln(b, "\n## T3 の割り付け（現行の値段で差が 0 付近 / 明確）")
	fmt.Fprintf(b, "\n- 「0 付近（|Δ| < %.1f）」を狙った組: %d、実際の埋め込みで 0 付近だったもの %s\n", nearThreshold, nearN, ratio(nearHit, nearN))
	fmt.Fprintf(b, "- 「明確」を狙った組: %d、実際に明確だったもの %s\n", clearN, ratio(clearHit, clearN))
}
