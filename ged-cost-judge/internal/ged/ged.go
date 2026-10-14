// Package ged は命題グラフ（I-node と RA/CA/MA を畳んだ辺）の GED を、
// 学習できる 11 個のコスト（計画 §6.2）で計算する。
//
// 前実験の aif-ged/0.4 の命題 GED と同じ手順（Hungarian 法で命題を対応付け、
// 辺はその対応から完全一致 → 種別違い → 向き違いの順に照合）を取り、次の 3 点を一般化した。
//
//   - 欠落（参照側にだけある）と余分（変種側にだけある）のコストを分ける
//   - 種別違いのコストを種別の組（RA↔CA / RA↔MA / CA↔MA）ごとに分ける
//   - 置換コストを cos の折れ線（節点 1, 0.9, 0.7, 0.5, 0.3）にする
//
// 命題の対応を固定すると GED は θ について線形になる（GED = θ·φ + 辺の余分の本数）。
// Compute はその特徴 φ を返す。
package ged

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// 関係の種別。
const (
	RA = "RA" // 支持
	CA = "CA" // 攻撃
	MA = "MA" // 言い換え
)

// Kinds は関係の種別の一覧。
var Kinds = []string{RA, CA, MA}

// Prop は命題（I-node）。
type Prop struct {
	ID   string `json:"id"`
	Text string `json:"text,omitempty"`
}

// Edge は前提 From から結論 To への関係。
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Kind string `json:"kind"`
}

// Graph は命題グラフ。
type Graph struct {
	Props []Prop `json:"props"`
	Edges []Edge `json:"edges"`
}

// Clone は深いコピーを返す。
func (g Graph) Clone() Graph {
	return Graph{Props: slices.Clone(g.Props), Edges: slices.Clone(g.Edges)}
}

// パラメータの添字（計画 §6.2 の # から 1 を引いたもの）。
const (
	PNodeMissing = iota // I-node の欠落（参照側にだけある命題。GED では挿入）
	PNodeExtra          // I-node の余分（変種側にだけある命題。GED では削除）
	PSub09              // 置換関数の増分 cos 1 → 0.9
	PSub07              // 同 0.9 → 0.7
	PSub05              // 同 0.7 → 0.5
	PSub03              // 同 0.5 → 0.3（cos < 0.3 は一定）
	PEdgeMissing        // 辺の欠落
	PTypeRACA           // 種別違い RA↔CA
	PTypeRAMA           // 種別違い RA↔MA
	PTypeCAMA           // 種別違い CA↔MA
	PDirection          // 向き違い
	NumParams
)

// ParamNames は表に出す名前。
var ParamNames = [NumParams]string{
	"I 欠落", "I 余分", "置換 cos0.9", "置換 cos0.7", "置換 cos0.5", "置換 cos0.3",
	"辺の欠落", "種別 RA↔CA", "種別 RA↔MA", "種別 CA↔MA", "向き違い",
}

// EdgeExtraCost は辺の余分のコスト。β と θ の尺度を分けられないので 1 に固定する（尺度の基準）。
const EdgeExtraCost = 1.0

// Knots は置換関数の節点（cos）。
var Knots = [5]float64{1, 0.9, 0.7, 0.5, 0.3}

// Theta は 11 個のコスト。
type Theta [NumParams]float64

// Current は現行のコスト表（aif-ged/0.4 の命題グラフ）。置換は 1 − cos の折れ線。
func Current() Theta {
	return Theta{1, 1, 0.1, 0.2, 0.2, 0.2, 1, 0.75, 0.75, 0.75, 1}
}

// SubBasis は置換関数の基底 h_k(cos)。f(cos) = Σ θ_k h_k(cos)。
func SubBasis(cos float64) [4]float64 {
	var h [4]float64
	for k := range 4 {
		hi, lo := Knots[k], Knots[k+1]
		h[k] = math.Min(1, math.Max(0, (hi-cos)/(hi-lo)))
	}
	return h
}

// Sub は cos に対する置換コスト。
func (t Theta) Sub(cos float64) float64 {
	h := SubBasis(cos)
	return t[PSub09]*h[0] + t[PSub07]*h[1] + t[PSub05]*h[2] + t[PSub03]*h[3]
}

// Features は対応を固定したときの線形化。GED = Phi·θ + EdgeExtra·EdgeExtraCost。
type Features struct {
	Phi       [NumParams]float64
	EdgeExtra float64
	// Assign は変種側の命題 i → 参照側の命題の添字（対応しなければ -1）。収束判定に使う。
	Assign []int
}

// Cost は θ での GED。
func (f Features) Cost(t Theta) float64 {
	c := f.EdgeExtra * EdgeExtraCost
	for i := range NumParams {
		c += f.Phi[i] * t[i]
	}
	return c
}

// Signature は対応の識別子。
func (f Features) Signature() string {
	var b strings.Builder
	for _, j := range f.Assign {
		b.WriteString(strconv.Itoa(j))
		b.WriteByte(',')
	}
	return b.String()
}

// Sim は参照側の命題 refID と変種側の命題 varID の cos を返す。
type Sim func(refID, varID string) float64

func typePair(a, b string) int {
	if a > b {
		a, b = b, a
	}
	switch a + b {
	case CA + RA:
		return PTypeRACA
	case MA + RA:
		return PTypeRAMA
	case CA + MA:
		return PTypeCAMA
	}
	return -1
}

// Compute は θ で命題を最適に対応付け、そのときの特徴を返す。ref は参照（元の議論 S）、v は変種。
func Compute(ref, v Graph, sim Sim, t Theta) (Features, error) {
	n, m := len(v.Props), len(ref.Props)
	sub := make([][]float64, n)
	del := make([]float64, n)
	cos := make([][]float64, n)
	for i, a := range v.Props {
		sub[i] = make([]float64, m)
		cos[i] = make([]float64, m)
		del[i] = t[PNodeExtra]
		for j, b := range ref.Props {
			c := math.Min(1, math.Max(0, sim(b.ID, a.ID)))
			cos[i][j] = c
			sub[i][j] = t.Sub(c)
		}
	}
	ins := make([]float64, m)
	for j := range ins {
		ins[j] = t[PNodeMissing]
	}
	match, err := bipartiteMatch(sub, del, ins)
	if err != nil {
		return Features{}, fmt.Errorf("ged: 命題の対応付け: %w", err)
	}

	var f Features
	f.Assign = make([]int, n)
	usedRef := make([]bool, m)
	pairs := make(map[string]string, len(match))
	for i := range n {
		j, ok := match[i]
		if !ok {
			f.Assign[i] = -1
			f.Phi[PNodeExtra]++
			continue
		}
		f.Assign[i] = j
		usedRef[j] = true
		pairs[v.Props[i].ID] = ref.Props[j].ID
		h := SubBasis(cos[i][j])
		for k := range 4 {
			f.Phi[PSub09+k] += h[k]
		}
	}
	for j := range m {
		if !usedRef[j] {
			f.Phi[PNodeMissing]++
		}
	}

	// 辺: 完全一致 → 種別違い → 向き違いの順に、パスごとに全件を照合する。
	refUsed := make([]bool, len(ref.Edges))
	type mapped struct {
		from, to, kind string
		done           bool
	}
	var ms []*mapped
	for _, e := range v.Edges {
		fr, ok1 := pairs[e.From]
		to, ok2 := pairs[e.To]
		if !ok1 || !ok2 {
			f.EdgeExtra++
			continue
		}
		ms = append(ms, &mapped{from: fr, to: to, kind: e.Kind})
	}
	find := func(from, to string, ok func(Edge) bool) int {
		for k, e := range ref.Edges {
			if !refUsed[k] && e.From == from && e.To == to && ok(e) {
				return k
			}
		}
		return -1
	}
	for _, x := range ms {
		if k := find(x.from, x.to, func(e Edge) bool { return e.Kind == x.kind }); k >= 0 {
			refUsed[k], x.done = true, true
		}
	}
	for _, x := range ms {
		if x.done {
			continue
		}
		if k := find(x.from, x.to, func(Edge) bool { return true }); k >= 0 {
			refUsed[k], x.done = true, true
			if p := typePair(ref.Edges[k].Kind, x.kind); p >= 0 {
				f.Phi[p]++
			}
		}
	}
	for _, x := range ms {
		if x.done {
			continue
		}
		if k := find(x.to, x.from, func(Edge) bool { return true }); k >= 0 {
			refUsed[k], x.done = true, true
			f.Phi[PDirection]++
		}
	}
	for _, x := range ms {
		if !x.done {
			f.EdgeExtra++
		}
	}
	for k := range ref.Edges {
		if !refUsed[k] {
			f.Phi[PEdgeMissing]++
		}
	}
	return f, nil
}
