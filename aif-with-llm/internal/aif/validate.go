package aif

import (
	"fmt"
	"sort"
	"strings"
)

// Violation は AIF メタモデル違反 1 件。
type Violation struct {
	NodeID string `json:"node_id,omitempty"`
	Rule   string `json:"rule"`
	Detail string `json:"detail"`
}

func (v Violation) String() string {
	if v.NodeID == "" {
		return fmt.Sprintf("%s: %s", v.Rule, v.Detail)
	}
	return fmt.Sprintf("%s (node %s): %s", v.Rule, v.NodeID, v.Detail)
}

// Validate は AIF 中核オントロジー + IAT のメタモデル制約を検査する。
//
// 「厳密な AIF」を主張する以上、生成側のグラフはこの検査を通ることを条件にする。
// 参照するのは以下の制約（Chesñevar et al. 2006 / Budzynska & Reed 2011 /
// AIFdb のスキーマ）:
//
//	R1  I-node 同士を直接つながない（関係は必ず S-node を経由する）
//	R2  RA は前提 I-node を 1 つ以上、結論 I-node をちょうど 1 つ持つ
//	R3  CA は攻撃元をちょうど 1 つ、攻撃先をちょうど 1 つ持つ
//	R4  MA は言い換え元・先をちょうど 1 つずつ持つ
//	R5  YA は L か TA にちょうど 1 つ anchor され、I か S-node をちょうど 1 つ illocute する
//	R6  TA は L から L への遷移である
//	R7  L-node は L / TA / YA 以外へ直接つながない
//	R8  YA のスキームは IAT の発話行為集合に含まれる
//	R9  ノード種別は 7 種のいずれか、辺の両端が存在する
//	R10 S-node は自己ループを作らない（前提と結論が同一 I-node でない）
func Validate(g Graph) []Violation {
	ix := NewIndex(g)
	var vs []Violation
	add := func(id, rule, detail string) { vs = append(vs, Violation{NodeID: id, Rule: rule, Detail: detail}) }

	seen := map[string]bool{}
	for _, n := range g.Nodes {
		if !n.Type.Valid() {
			add(n.ID, "R9", fmt.Sprintf("未知のノード種別 %q", n.Type))
		}
		if seen[n.ID] {
			add(n.ID, "R9", "ノード ID が重複している")
		}
		seen[n.ID] = true
	}
	for _, e := range g.Edges {
		if _, ok := ix.Node(e.FromID); !ok {
			add(e.FromID, "R9", "辺の始点ノードが存在しない")
		}
		if _, ok := ix.Node(e.ToID); !ok {
			add(e.ToID, "R9", "辺の終点ノードが存在しない")
		}
	}

	for _, n := range g.Nodes {
		switch n.Type {
		case TypeI:
			for _, id := range ix.Out(n.ID) {
				if ix.Type(id) == TypeI {
					add(n.ID, "R1", "I-node から I-node へ直接辺を張っている")
				}
			}
		case TypeRA, TypeCA, TypeMA:
			var prem, conc int
			for _, id := range ix.In(n.ID) {
				if ix.Type(id) == TypeI {
					prem++
				}
			}
			var dst string
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t == TypeI || t.IsSNode() {
					conc++
					dst = id
				}
			}
			rule := map[NodeType]string{TypeRA: "R2", TypeCA: "R3", TypeMA: "R4"}[n.Type]
			switch {
			case prem == 0:
				add(n.ID, rule, string(n.Type)+" に前提側の I-node が無い")
			case n.Type != TypeRA && prem != 1:
				add(n.ID, rule, fmt.Sprintf("%s の入力は 1 本でなければならない（%d 本）", n.Type, prem))
			}
			if conc != 1 {
				add(n.ID, rule, fmt.Sprintf("%s の出力はちょうど 1 本でなければならない（%d 本）", n.Type, conc))
			}
			if dst != "" {
				for _, id := range ix.In(n.ID) {
					if id == dst {
						add(n.ID, "R10", "前提と結論が同じノードを指している")
					}
				}
			}
		case TypeYA:
			var anchors, targets int
			for _, id := range ix.In(n.ID) {
				if t := ix.Type(id); t == TypeL || t == TypeTA {
					anchors++
				}
			}
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t == TypeI || t.IsSNode() {
					targets++
				}
			}
			if anchors != 1 {
				add(n.ID, "R5", fmt.Sprintf("YA の anchor は L か TA ちょうど 1 つ（%d 個）", anchors))
			}
			if targets != 1 {
				add(n.ID, "R5", fmt.Sprintf("YA が illocute する対象はちょうど 1 つ（%d 個）", targets))
			}
			if !ValidForce(n.Scheme) {
				add(n.ID, "R8", fmt.Sprintf("未知の発話行為 %q", n.Scheme))
			}
		case TypeTA:
			var from, to int
			for _, id := range ix.In(n.ID) {
				if ix.Type(id) == TypeL {
					from++
				}
			}
			for _, id := range ix.Out(n.ID) {
				if ix.Type(id) == TypeL {
					to++
				}
			}
			if from != 1 || to != 1 {
				add(n.ID, "R6", fmt.Sprintf("TA は L から L への遷移（in=%d out=%d）", from, to))
			}
		case TypeL:
			for _, id := range ix.Out(n.ID) {
				if t := ix.Type(id); t != TypeYA && t != TypeTA {
					add(n.ID, "R7", fmt.Sprintf("L-node から %s へ直接辺を張っている", t))
				}
			}
		}
	}
	return vs
}

// NormalizeReport は Normalize が落としたものの内訳。
// 「前処理で何を捨てたか」を報告に書けるよう、握り潰さず返す。
type NormalizeReport struct {
	DroppedEdges     int      `json:"dropped_edges"`
	DroppedIsolatedL int      `json:"dropped_isolated_l"`
	DroppedOther     int      `json:"dropped_other"`
	DroppedNodeIDs   []string `json:"dropped_node_ids,omitempty"`
	Violations       int      `json:"violations"`
}

// Normalize はゴールド／生成グラフを比較可能な形に整える。
//
// 落とすもの:
//   - 端点が存在しない辺
//   - AIF のメタモデルに無い形の辺（L→I の直結など。アノテーション由来のノイズ）
//   - 孤立ノード（次数 0）。QT30 には収録ツール由来の「話者名: 直前の発話の複製」
//     という辺を持たない L-node が大量に混ざっており、これを残すと L-node の
//     個数だけで GED が動いてしまう。
//
// 落とさないもの: 上記以外はすべて。内容の良し悪しでの間引きはしない。
func Normalize(g Graph) (Graph, NormalizeReport) {
	ix := NewIndex(g)
	var rep NormalizeReport

	edges := make([]Edge, 0, len(g.Edges))
	deg := map[string]int{}
	seenEdge := map[[2]string]bool{}
	for _, e := range g.Edges {
		ft, tt := ix.Type(e.FromID), ix.Type(e.ToID)
		if ft == "" || tt == "" || !allowedEdge(ft, tt) || e.FromID == e.ToID || seenEdge[[2]string{e.FromID, e.ToID}] {
			rep.DroppedEdges++
			continue
		}
		seenEdge[[2]string{e.FromID, e.ToID}] = true
		edges = append(edges, e)
		deg[e.FromID]++
		deg[e.ToID]++
	}

	nodes := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if deg[n.ID] == 0 {
			if n.Type == TypeL {
				rep.DroppedIsolatedL++
			} else {
				rep.DroppedOther++
			}
			rep.DroppedNodeIDs = append(rep.DroppedNodeIDs, n.ID)
			continue
		}
		n.Text = strings.TrimSpace(n.Text)
		nodes = append(nodes, n)
	}
	sort.Strings(rep.DroppedNodeIDs)

	out := Graph{ID: g.ID, Nodes: nodes, Edges: edges}
	rep.Violations = len(Validate(out))
	return out, rep
}

// allowedEdge は AIF のメタモデルが許す (始点種別, 終点種別) の組か。
func allowedEdge(from, to NodeType) bool {
	switch from {
	case TypeI:
		return to.IsSNode()
	case TypeRA, TypeCA, TypeMA:
		return to == TypeI || to.IsSNode()
	case TypeL:
		return to == TypeYA || to == TypeTA
	case TypeYA:
		return to == TypeI || to.IsSNode()
	case TypeTA:
		return to == TypeL || to == TypeYA
	}
	return false
}
