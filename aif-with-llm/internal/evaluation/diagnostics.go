package evaluation

import (
	"maps"
	"slices"
	"sort"
	"strconv"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// alignmentOf turns a proposition (or utterance) matching into P/R/F1, counting pairs at or above threshold as matched.
//
// TP is "the maximum number of one-to-one pairs formable using only pairs with cos at or above the threshold", and FP / FN
// are the remainders on the generated and gold sides. The maximum matching is re-solved for each threshold.
//
// Previously the Hungarian assignment (the single assignment maximizing the sum of cos) was cut at the threshold.
// That assignment ignores the threshold, so it does not pick another assignment even if that would form more pairs above it
// (e.g. given assignments with cos (0.99, 0.50) and (0.56, 0.56), it picks the former with the larger sum,
// and TP at threshold 0.55 becomes 1 instead of 2). For the §12.2 curves, which compare rankings while moving the threshold,
// counting without the best matching per threshold distorts the curve's shape for reasons of assignment.
//
// matching (the Hungarian assignment) is used only for the mean cos and AboveThreshold.
// Relation matching and GED are still counted on top of this assignment.
func alignmentOf(matching map[string]string, sim simTable, predIDs, goldIDs []string, threshold float64) (PropositionAlignment, error) {
	nPred, nGold := len(predIDs), len(goldIDs)
	// Fix the summation order (so map iteration order does not wobble the last floating-point digits).
	var sum float64
	above := 0
	for _, p := range slices.Sorted(maps.Keys(matching)) {
		c := sim.get(p, matching[p])
		sum += c
		if c >= threshold {
			above++
		}
	}
	prfAt := func(t float64) (PRF, error) {
		n, err := maxMatchedAbove(predIDs, goldIDs, sim, t)
		if err != nil {
			return PRF{}, err
		}
		return newPRF(n, nPred-n, nGold-n), nil
	}
	prf, err := prfAt(threshold)
	if err != nil {
		return PropositionAlignment{}, err
	}
	out := PropositionAlignment{Threshold: threshold, Aligned: len(matching), AboveThreshold: above, PRF: prf}
	if len(matching) > 0 {
		out.MeanSimilarity = sum / float64(len(matching))
	}
	if nGold > 0 {
		out.CountRatio = float64(nPred) / float64(nGold)
	}
	out.F1ByThreshold = make(map[string]float64, len(PropositionThresholds))
	for _, t := range PropositionThresholds {
		p, err := prfAt(t)
		if err != nil {
			return PropositionAlignment{}, err
		}
		out.F1ByThreshold[strconv.FormatFloat(t, 'f', 2, 64)] = p.F1
	}
	return out, nil
}

// maxMatchedAbove returns the maximum number of one-to-one (generated, gold) pairs using only pairs with cos at or above t.
//
// It solves a minimum-cost assignment with substitution cost 0 for pairs at or above the threshold and, for the rest, higher
// than "deletion + insertion" (2). Each added pair lowers the cost by 2, so the minimum-cost solution has the most pairs.
func maxMatchedAbove(predIDs, goldIDs []string, sim simTable, t float64) (int, error) {
	if len(predIDs) == 0 || len(goldIDs) == 0 {
		return 0, nil
	}
	sub := make([][]float64, len(predIDs))
	for i, p := range predIDs {
		sub[i] = make([]float64, len(goldIDs))
		for j, g := range goldIDs {
			if sim.get(p, g) < t {
				sub[i][j] = 3
			}
		}
	}
	del := slices.Repeat([]float64{1}, len(predIDs))
	ins := slices.Repeat([]float64{1}, len(goldIDs))
	m, err := bipartiteMatch(sub, del, ins)
	if err != nil {
		return 0, err
	}
	n := 0
	for i, j := range m {
		if sub[i][j] == 0 {
			n++
		}
	}
	return n, nil
}

// goldRelInst is one gold relation (an RA with multiple premises counts once per premise).
//
// The key point is the used flag, so each one is consumed only once. If pred draws two relations
// (RA and MA, etc.) on the same proposition pair, matching both to the same gold relation would
// make TP exceed the number of gold relations and inflate recall.
type goldRelInst struct {
	typ      string
	src, dst string
	used     bool
}

// goldRelIndex looks up gold relations by proposition pair.
type goldRelIndex struct {
	insts  []goldRelInst
	byPair map[[2]string][]int
}

func newGoldRelIndex(rels []aif.Relation) *goldRelIndex {
	ix := &goldRelIndex{byPair: map[[2]string][]int{}}
	for _, r := range rels {
		for _, s := range r.SrcIDs {
			ix.insts = append(ix.insts, goldRelInst{typ: string(r.Type), src: s, dst: r.DstID})
			k := [2]string{s, r.DstID}
			ix.byPair[k] = append(ix.byPair[k], len(ix.insts)-1)
		}
	}
	return ix
}

// take takes and consumes one unconsumed gold relation connecting the proposition pair pair.
// If exact, only relations of type typ are taken. Otherwise any type is taken, preferring typ.
func (ix *goldRelIndex) take(pair [2]string, typ string, exact bool) (goldRelInst, bool) {
	cand := ix.byPair[pair]
	for _, i := range cand {
		if !ix.insts[i].used && ix.insts[i].typ == typ {
			ix.insts[i].used = true
			return ix.insts[i], true
		}
	}
	if exact {
		return goldRelInst{}, false
	}
	for _, i := range cand {
		if !ix.insts[i].used {
			ix.insts[i].used = true
			return ix.insts[i], true
		}
	}
	return goldRelInst{}, false
}

// relOutcome is the matching result for one generated relation.
type relOutcome int

const (
	relSpurious relOutcome = iota // no gold counterpart (over-generation)
	relCorrect                    // type and direction both match
	relMistyped                   // direction matches, type differs
	relReversed                   // direction reversed
)

// newDistanceBuckets returns a table with every distance band initialized to 0.
// All bands are created up front to distinguish "none in that band" from "not counted".
func newDistanceBuckets() map[string]DistanceBucket {
	out := make(map[string]DistanceBucket, len(aif.DistanceBands))
	for _, b := range aif.DistanceBands {
		out[b] = DistanceBucket{}
	}
	return out
}

func relationScore(pred, gold aif.Graph, matching map[string]string, sim simTables, opts AIFCompareOptions) (RelationScore, PairSimilarityStats, AIFEvidence) {
	predIx, goldIx := aif.NewIndex(pred), aif.NewIndex(gold)
	textOf := func(ix *aif.Index, id string) string {
		if n, ok := ix.Node(id); ok {
			return n.Text
		}
		return ""
	}

	// Index the gold relations. Each is consumed only once.
	goldRelIx := newGoldRelIndex(gold.Relations())

	score := RelationScore{
		TypeConfusion: map[string]map[string]int{},
		ByDistance:    newDistanceBuckets(),
	}
	var ev AIFEvidence
	goldTotal := len(goldRelIx.insts)
	score.GoldTotal = goldTotal

	// Inter-utterance distance. Proposition → utterance position is computed independently per graph.
	predPos, goldPos := pred.PropositionOrder(), gold.PropositionOrder()
	bandOf := func(pos map[string]int, a, b string) (string, bool) {
		pa, ok1 := pos[a]
		pb, ok2 := pos[b]
		if !ok1 || !ok2 {
			return "", false
		}
		d := pa - pb
		if d < 0 {
			d = -d
		}
		return aif.DistanceBand(d), true
	}
	for _, inst := range goldRelIx.insts {
		if band, ok := bandOf(goldPos, inst.src, inst.dst); ok {
			bucket := score.ByDistance[band]
			bucket.GoldTotal++
			score.ByDistance[band] = bucket
		}
	}

	// ---- List generated relations one by one (per premise) ----
	type predInst struct {
		typ      string
		ms, md   string // matched gold propositions (meaningful only when both ends matched)
		mapped   bool
		band     string
		bandOK   bool
		outcome  relOutcome
		goldType string
		ev       RelationEvidence
	}
	var insts []*predInst
	for _, r := range pred.Relations() {
		for _, s := range r.SrcIDs {
			ms, mok := matching[s]
			md, dok := matching[r.DstID]
			p := &predInst{typ: string(r.Type), ms: ms, md: md, mapped: mok && dok}
			p.ev = RelationEvidence{
				Side:               "pred",
				Type:               p.typ,
				SrcText:            textOf(predIx, s),
				DstText:            textOf(predIx, r.DstID),
				Rationale:          opts.Rationales[r.SNodeID],
				PairSimilarity:     sim.predSelf.get(s, r.DstID),
				SrcMatchSimilarity: sim.cross.get(s, ms),
			}
			if mok {
				p.ev.SrcMatch = textOf(goldIx, ms)
			}
			if dok {
				p.ev.DstMatch = textOf(goldIx, md)
			}
			// The denominator of per-band precision is counted by the generated-side band, whether or not the endpoints matched.
			// Excluding unmatched relations (over-generation) from the denominator would inflate per-band precision.
			if p.band, p.bandOK = bandOf(predPos, s, r.DstID); p.bandOK {
				bucket := score.ByDistance[p.band]
				bucket.PredTotal++
				score.ByDistance[p.band] = bucket
			}
			insts = append(insts, p)
		}
	}

	// ---- Match against the gold (3 passes) ----
	//
	// If an earlier relation takes a gold relation of a different type, a later relation can no longer match exactly
	// (e.g. when RA and CA are emitted on the same proposition pair). So the result does not depend on processing order,
	// every relation is matched per pass, in the order "exact", "direction matches but type differs", "direction reversed".
	// This maximizes strict TP, direction-only TP and undirected TP lexicographically, in that order.
	passes := []struct {
		outcome relOutcome
		take    func(p *predInst) (goldRelInst, bool)
	}{
		{relCorrect, func(p *predInst) (goldRelInst, bool) { return goldRelIx.take([2]string{p.ms, p.md}, p.typ, true) }},
		{relMistyped, func(p *predInst) (goldRelInst, bool) { return goldRelIx.take([2]string{p.ms, p.md}, p.typ, false) }},
		{relReversed, func(p *predInst) (goldRelInst, bool) { return goldRelIx.take([2]string{p.md, p.ms}, p.typ, false) }},
	}
	for _, pass := range passes {
		for _, p := range insts {
			if !p.mapped || p.outcome != relSpurious {
				continue
			}
			if g, ok := pass.take(p); ok {
				p.outcome, p.goldType = pass.outcome, g.typ
			}
		}
	}

	// ---- Tally (in generated-side order) ----
	var typedTP, untypedTP, undirectedTP int
	for _, p := range insts {
		e := p.ev
		if p.outcome == relSpurious {
			ev.Spurious = append(ev.Spurious, e)
			continue
		}
		undirectedTP++
		typed := p.outcome == relCorrect
		switch p.outcome {
		case relCorrect:
			ev.Correct = append(ev.Correct, e)
		case relMistyped:
			e.GoldType = p.goldType
			ev.Mistyped = append(ev.Mistyped, e)
		case relReversed:
			score.Reversed++
			e.GoldType = p.goldType
			ev.Reversed = append(ev.Reversed, e)
		}
		if p.outcome != relReversed {
			untypedTP++
			score.TypeConfusion[p.typ] = incr(score.TypeConfusion[p.typ], p.goldType)
		}
		if typed {
			typedTP++
		}
		// Hits count in both the gold-side band (for recall) and the generated-side band (for precision).
		// Counting only one would misalign the bands of numerator and denominator.
		if p.bandOK {
			bucket := score.ByDistance[p.band]
			bucket.TPUndirectedPred++
			if typed {
				bucket.TPPred++
			}
			score.ByDistance[p.band] = bucket
		}
		if band, ok := bandOf(goldPos, p.ms, p.md); ok {
			bucket := score.ByDistance[band]
			bucket.TPUndirected++
			if typed {
				bucket.TP++
			}
			score.ByDistance[band] = bucket
		}
	}
	predTotal := len(insts)
	score.PredTotal = predTotal

	// Missed gold relations (those not consumed).
	reverse := map[string]string{}
	for p, g := range matching {
		reverse[g] = p
	}
	for _, inst := range goldRelIx.insts {
		if inst.used {
			continue
		}
		e := RelationEvidence{
			Side:           "gold",
			Type:           inst.typ,
			SrcText:        textOf(goldIx, inst.src),
			DstText:        textOf(goldIx, inst.dst),
			PairSimilarity: sim.goldSelf.get(inst.src, inst.dst),
		}
		if p, ok := reverse[inst.src]; ok {
			e.SrcMatch = textOf(predIx, p)
		}
		if p, ok := reverse[inst.dst]; ok {
			e.DstMatch = textOf(predIx, p)
		}
		ev.Missed = append(ev.Missed, e)
	}

	score.Typed = newPRF(typedTP, predTotal-typedTP, goldTotal-typedTP)
	score.Untyped = newPRF(untypedTP, predTotal-untypedTP, goldTotal-untypedTP)
	score.Undirected = newPRF(undirectedTP, predTotal-undirectedTP, goldTotal-undirectedTP)

	// Take the mean cos between propositions connected by relations before truncating the evidence.
	// Taking it after truncation would average only the top few.
	mean := func(in []RelationEvidence) (float64, int) {
		if len(in) == 0 {
			return 0, 0
		}
		var sum float64
		for _, e := range in {
			sum += e.PairSimilarity
		}
		return sum / float64(len(in)), len(in)
	}
	var ps PairSimilarityStats
	ps.CorrectMean, ps.CorrectN = mean(ev.Correct)
	ps.SpuriousMean, ps.SpuriousN = mean(ev.Spurious)
	ps.MissedMean, ps.MissedN = mean(ev.Missed)

	for _, list := range []*[]RelationEvidence{&ev.Correct, &ev.Reversed, &ev.Mistyped, &ev.Spurious, &ev.Missed} {
		*list = (*list)[:min(len(*list), opts.EvidenceLimit)]
	}
	return score, ps, ev
}

func incr(m map[string]int, k string) map[string]int {
	if m == nil {
		m = map[string]int{}
	}
	m[k]++
	return m
}

// forceScore compares, for matched I-nodes, the schemes of the YAs that illocute them.
//
// Only YAs **anchored to an utterance (L-node)** are compared. The generator only creates L→YA→I, so mixing in
// gold TA→YA→I (4 in test) would compare transition forces (Arguing, etc.) against utterance forces.
// When an I-node has multiple utterance YAs (restatements or quotes), the YA of the earliest utterance is used
// (the same rule as PropositionOrder). Previously it was the last one seen in Illocutions order.
func forceScore(pred, gold aif.Graph, matching map[string]string) ForceScore {
	forceOf := func(g aif.Graph) map[string]string {
		pos := map[string]int{}
		for i, n := range g.LocutionOrder() {
			pos[n.ID] = i
		}
		out := map[string]string{}
		at := map[string]int{}
		for _, il := range g.Illocutions() {
			if il.TargetType != aif.TypeI || il.AnchorType != aif.TypeL {
				continue
			}
			p := pos[il.AnchorID]
			if q, seen := at[il.TargetID]; !seen || p < q {
				out[il.TargetID], at[il.TargetID] = il.Force, p
			}
		}
		return out
	}
	pf, gf := forceOf(pred), forceOf(gold)
	out := ForceScore{Confusion: map[string]map[string]int{}}
	for p, g := range matching {
		a, aok := pf[p]
		b, bok := gf[g]
		if !aok || !bok {
			continue
		}
		out.Compared++
		out.Confusion[a] = incr(out.Confusion[a], b)
		if a == b {
			out.Exact++
		}
		if forceFamily(a) == forceFamily(b) {
			out.SameFamily++
		}
	}
	if out.Compared > 0 {
		out.Accuracy = float64(out.Exact) / float64(out.Compared)
	}
	return out
}

func pairEvidence(pred, gold aif.Graph, matching map[string]string, sim simTable, limit int) (best, worst []PropositionPair) {
	predIx, goldIx := aif.NewIndex(pred), aif.NewIndex(gold)
	var pairs []PropositionPair
	for p, g := range matching {
		pn, ok := predIx.Node(p)
		if !ok || pn.Type != aif.TypeI {
			continue
		}
		gn, ok := goldIx.Node(g)
		if !ok {
			continue
		}
		pairs = append(pairs, PropositionPair{
			PredID: p, PredText: pn.Text, GoldID: g, GoldText: gn.Text,
			Similarity: sim.get(p, g),
		})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Similarity != pairs[j].Similarity {
			return pairs[i].Similarity > pairs[j].Similarity
		}
		return pairs[i].PredID < pairs[j].PredID
	})
	if len(pairs) <= limit {
		return pairs, nil
	}
	best = pairs[:limit]
	// Keep the top and bottom from overlapping (with few items, the same pair would appear in both).
	start := len(pairs) - limit
	if start < limit {
		start = limit
	}
	tail := pairs[start:]
	worst = make([]PropositionPair, len(tail))
	for i := range tail {
		worst[i] = tail[len(tail)-1-i]
	}
	return best, worst
}
