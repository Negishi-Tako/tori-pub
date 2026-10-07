package evaluation

import (
	"maps"
	"slices"
	"sort"
	"strconv"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// alignmentOf は命題（または発話）の対応を、閾値 threshold で「対応付いた」と数えた P/R/F1 にする。
//
// TP は「cos が閾値以上の対だけを使って作れる 1 対 1 の対応の最大数」、FP / FN は
// それぞれ生成側・ゴールド側の残り。閾値ごとに最大の対応を解き直す。
//
// 以前は Hungarian 法の割当（cos の総和を最大にする 1 つの割当）を閾値で切っていた。
// その割当は閾値を見ないので、閾値以上の対をより多く作れる別の割当があっても選ばない
// （例: cos が (0.99, 0.50) の割当と (0.56, 0.56) の割当があると総和の大きい前者を選び、
// 閾値 0.55 での TP が 2 ではなく 1 になる）。閾値を上げ下げして順位を比べる §12.2 の曲線では、
// 閾値ごとに最良の対応で数えないと、曲線の形が割当の都合で歪む。
//
// matching（Hungarian 法の割当）は平均 cos と AboveThreshold にだけ使う。
// 関係の照合と GED は引き続きこの割当の上で数える。
func alignmentOf(matching map[string]string, sim simTable, predIDs, goldIDs []string, threshold float64) (PropositionAlignment, error) {
	nPred, nGold := len(predIDs), len(goldIDs)
	// 加算順を固定する（map の反復順で浮動小数の最終桁が揺れないように）。
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

// maxMatchedAbove は cos が t 以上の (生成, ゴールド) 対だけを使った 1 対 1 の対応の最大数を返す。
//
// 閾値以上の対の置換コストを 0、それ以外を「削除 + 挿入」（2）より高くして最小コスト割当を解く。
// 対を 1 つ増やすごとにコストが 2 下がるので、最小コスト解は対の数が最大の解になる。
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

// goldRelInst はゴールドの関係 1 本（前提が複数ある RA は前提ごとに 1 本と数える）。
//
// used を持たせて 1 本を 1 回しか消費させないのが要点。pred が同じ命題対に
// 2 本（RA と MA 等）張ったとき、両方を同じゴールドの関係に当ててしまうと
// TP が正解の本数を超え、再現率が過大になる。
type goldRelInst struct {
	typ      string
	src, dst string
	used     bool
}

// goldRelIndex はゴールドの関係を命題対から引くための索引。
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

// take は命題対 pair を結ぶ未消費のゴールド関係を 1 本取って消費する。
// exact なら種別が typ のものだけを取る。そうでなければ種別を問わず、typ と同じものを優先する。
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

// relOutcome は生成側の関係 1 本の照合結果。
type relOutcome int

const (
	relSpurious relOutcome = iota // 正解に対応が無い（過剰生成）
	relCorrect                    // 種別・向きとも一致
	relMistyped                   // 向きは一致・種別が違う
	relReversed                   // 向きが逆
)

// newDistanceBuckets は全距離帯を 0 で初期化した表を返す。
// 「その帯に 1 本も無かった」と「集計していない」を区別するため、先に全帯を作る。
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

	// ゴールド側の関係を索引化する。1 本を 1 回だけ消費する。
	goldRelIx := newGoldRelIndex(gold.Relations())

	score := RelationScore{
		TypeConfusion: map[string]map[string]int{},
		ByDistance:    newDistanceBuckets(),
	}
	var ev AIFEvidence
	goldTotal := len(goldRelIx.insts)
	score.GoldTotal = goldTotal

	// 発話間距離。命題 → 発話位置はグラフごとに独立に求める。
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

	// ---- 生成側の関係を 1 本ずつ（前提ごとに）並べる ----
	type predInst struct {
		typ      string
		ms, md   string // 対応先のゴールド命題（両端とも対応したときだけ意味を持つ）
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
			// 帯ごとの精度の分母は、端点が対応したかどうかに関係なく生成側の帯で数える。
			// 対応しなかった関係（過剰生成）を分母から外すと、帯ごとの精度が過大になる。
			if p.band, p.bandOK = bandOf(predPos, s, r.DstID); p.bandOK {
				bucket := score.ByDistance[p.band]
				bucket.PredTotal++
				score.ByDistance[p.band] = bucket
			}
			insts = append(insts, p)
		}
	}

	// ---- ゴールドとの照合（3 パス） ----
	//
	// 先に処理した関係が別種別のゴールドを取ってしまうと、後の関係が完全一致できなくなる
	// （同じ命題対に RA と CA を出したときなど）。処理順で結果が変わらないよう、
	// 「完全一致」「向きは合うが種別違い」「向きが逆」の順に、パスごとに全件を照合する。
	// これで厳密 TP・向きのみ TP・無向 TP の順に辞書式に最大になる。
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

	// ---- 集計（生成側の順に） ----
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
		// 当たりはゴールド側の帯（再現率用）と生成側の帯（精度用）の
		// 両方に数える。片方だけだと分子と分母の帯が揃わない。
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

	// 取りこぼしたゴールドの関係（消費されなかったもの）。
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

	// 証拠を打ち切る前に、関係で結ばれた命題どうしの平均 cos を取る。
	// 打ち切り後に取ると上位数件だけの平均になってしまう。
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

// forceScore は対応付いた I-node について、それを illocute する YA のスキームを比べる。
//
// 比べるのは**発話（L-node）に anchor された** YA だけ。生成側は L→YA→I しか作らないので、
// ゴールドの TA→YA→I（test で 4 本）を混ぜると、遷移の力（Arguing 等）と発話の力を比べてしまう。
// 1 つの I-node に発話の YA が複数ある場合（言い直し・引用）は、最も早い発話の YA を採る
// （PropositionOrder と同じ規則）。以前は Illocutions の並びで最後に見たものになっていた。
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
	// 上位と下位が重ならないようにする（件数が少ないと同じ組が両方に出てしまう）。
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
