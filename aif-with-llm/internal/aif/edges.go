package aif

import "strconv"

// LabeledEdge は辺に「両端のノード種別から決まる意味」を名前として付けたもの。
//
// AIF の辺は種別を持たず、両端の種別で意味が決まる。GED で辺の一致を数えるには
// 意味まで揃っている必要があるので、ここで両端の種別から機械的に名前を付ける。
// 情報は増えも減りもしない。
type LabeledEdge struct {
	FromID string
	ToID   string
	Kind   string
}

// 辺の種別名。値は GED の一致判定にだけ使う（同じ値なら同じ意味の辺）。
const (
	edgeAnchors       = "ANCHORS"        // L / TA -> YA
	edgeIllocutes     = "ILLOCUTES"      // YA -> I / S-node
	edgePremiseOf     = "PREMISE_OF"     // I -> RA
	edgeConcludes     = "CONCLUDES"      // RA -> I / S-node
	edgeConflictsVia  = "CONFLICTS_VIA"  // I -> CA
	edgeConflictsWith = "CONFLICTS_WITH" // CA -> I / S-node
	edgeRephrasesVia  = "REPHRASES_VIA"  // I -> MA
	edgeRephrasesTo   = "REPHRASES_TO"   // MA -> I / S-node
	edgeNext          = "NEXT"           // L -> TA
	edgeLeadsTo       = "LEADS_TO"       // TA -> L
)

// relationEdgeKind は関係（S-node）を命題間の辺 1 本に畳んだときの種別名。
var relationEdgeKind = map[NodeType]string{
	TypeRA: "support",
	TypeCA: "attack",
	TypeMA: "rephrase",
}

func edgeKind(from, to NodeType) (string, bool) {
	switch {
	case from == TypeL && to == TypeYA, from == TypeTA && to == TypeYA:
		return edgeAnchors, true
	case from == TypeYA:
		return edgeIllocutes, true
	case from == TypeI && to == TypeRA:
		return edgePremiseOf, true
	case from == TypeRA:
		return edgeConcludes, true
	case from == TypeI && to == TypeCA:
		return edgeConflictsVia, true
	case from == TypeCA:
		return edgeConflictsWith, true
	case from == TypeI && to == TypeMA:
		return edgeRephrasesVia, true
	case from == TypeMA:
		return edgeRephrasesTo, true
	case from == TypeL && to == TypeTA:
		return edgeNext, true
	case from == TypeTA && to == TypeL:
		return edgeLeadsTo, true
	}
	return "", false
}

// LabeledEdges は全ノードを残したまま、すべての辺に種別名を付けて返す。
func (g Graph) LabeledEdges() []LabeledEdge {
	return g.labeledEdges(nil)
}

// labeledEdges は keep に含まれるノード同士の辺だけを、種別名を付けて返す。keep が nil なら全辺。
func (g Graph) labeledEdges(keep map[string]bool) []LabeledEdge {
	ix := NewIndex(g)
	out := make([]LabeledEdge, 0, len(g.Edges))
	for _, e := range g.Edges {
		if keep != nil && (!keep[e.FromID] || !keep[e.ToID]) {
			continue
		}
		if k, ok := edgeKind(ix.Type(e.FromID), ix.Type(e.ToID)); ok {
			out = append(out, LabeledEdge{FromID: e.FromID, ToID: e.ToID, Kind: k})
		}
	}
	return out
}

// RelationEdges は関係（RA/CA/MA）を命題間の辺に畳んだもの。
// 前提が複数ある RA は前提ごとに 1 本とする。命題グラフ（I-node と関係だけ）の辺になる。
func (g Graph) RelationEdges() []LabeledEdge {
	var out []LabeledEdge
	for _, r := range g.Relations() {
		for _, src := range r.SrcIDs {
			out = append(out, LabeledEdge{FromID: src, ToID: r.DstID, Kind: relationEdgeKind[r.Type]})
		}
	}
	return out
}

// dialogueKeep は DialogueNodes / DialogueEdges（withTA=true）と
// AnchorNodes / AnchorEdges（withTA=false）に残すノードの集合。
//
// 残すのは L / I（と withTA なら TA）と、**発話に anchor され命題を illocute する YA** だけ。
// 落とすのは次の種類で、いずれも「LLM の質ではないもの」を GED から外すため。
//
//	S-node（RA/CA/MA）  … 関係は辺 1 本に畳むので、ノードとしては数えない。
//	                      ノードでも辺でも数えると同じ誤りを二重に罰する。
//	遷移に anchor される YA … pred 側は「隣接しない発話同士の関係には TA が無いので
//	                      YA も作らない」（annotate.go）という harness の構成規則で
//	                      個数が決まっており、ゴールドとの差が LLM の誤りを表さない。
//	                      発話行為の当たりは evaluation.ForceScore で別に測る。
//	TA（withTA=false）    … 同じ理由で外す。pred 側の TA は「並べた発話の隣どうしに 1 本ずつ」
//	                      という構成規則で決まるが、ゴールドの TA は分岐し、隣接しない発話も結ぶ。
//	                      発話分割をゴールドどおりに与えた基準線でも、test 124 件で
//	                      ゴールドの TA 1,732 個のうち 445 個、生成側 1,779 個のうち 492 個が対応せず、
//	                      その TA と辺だけで 0.3 の全体 GED の分母の約 18% を占める（LLM がどう答えても消えない）。
func (g Graph) dialogueKeep(withTA bool) map[string]bool {
	keep := make(map[string]bool, len(g.Nodes))
	for _, n := range g.Nodes {
		switch n.Type {
		case TypeL, TypeI:
			keep[n.ID] = true
		case TypeTA:
			keep[n.ID] = withTA
		}
	}
	for _, il := range g.Illocutions() {
		if il.AnchorType == TypeL && il.TargetType == TypeI {
			keep[il.YAID] = true
		}
	}
	return keep
}

func (g Graph) nodesIn(keep map[string]bool) []Node {
	out := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if keep[n.ID] {
			out = append(out, n)
		}
	}
	return out
}

// DialogueNodes は対話層（L / TA / I / 発話の YA）のノードを入力順で返す（GED 0.3 の全体グラフ）。
func (g Graph) DialogueNodes() []Node { return g.nodesIn(g.dialogueKeep(true)) }

// DialogueEdges は対話層の辺に、関係を命題間の辺 1 本に畳んだもの（RelationEdges）を足したもの。
//
// LabeledEdges との違いは、関係 1 本が「S-node 1 個 + 辺 2 本（+ YA 1 個 + 辺 2 本）」
// ではなく「辺 1 本」になること。そのままでは種別の取り違え 1 件が
// 命題単位では 0.75 なのに全体では 4.75 になる（辺が両側とも外れる）という
// 粒度間の不整合が残る。畳んでおけば、関係の誤りのコストは粒度によらず同じ値になる。
func (g Graph) DialogueEdges() []LabeledEdge {
	return append(g.labeledEdges(g.dialogueKeep(true)), g.RelationEdges()...)
}

// AnchorNodes は DialogueNodes から TA を除いたもの（L / I / 発話の YA。GED 0.4 の全体グラフ）。
func (g Graph) AnchorNodes() []Node { return g.nodesIn(g.dialogueKeep(false)) }

// AnchorEdges は AnchorNodes どうしの辺（L→YA→I）に、関係を畳んだ辺を足したもの。
func (g Graph) AnchorEdges() []LabeledEdge {
	return append(g.labeledEdges(g.dialogueKeep(false)), g.RelationEdges()...)
}

// BaselineGraph は「LLM を使わない下限」の AIF グラフを作る。
//
// 与えられた発話をそのまま命題とみなし、すべて Asserting で anchor し、
// 関係（RA/CA/MA）は一切張らない。正規化 GED は分母が大きいほど小さく出るため、
// 「何もしない系」がどの程度の値を取るのかを示さないと、LLM の値が良いのか
// 悪いのか判断できない。その基準線として使う。
func BaselineGraph(id string, locs []LocutionInput) Graph {
	g := Graph{ID: id}
	var prev string
	for i, l := range locs {
		lid, yid, iid := idxID("L", i), idxID("YA", i), idxID("I", i)
		g.Nodes = append(g.Nodes,
			Node{ID: lid, Type: TypeL, Text: l.Speaker + " : " + l.Text, Speaker: l.Speaker},
			Node{ID: yid, Type: TypeYA, Text: "Asserting", Scheme: "Asserting"},
			Node{ID: iid, Type: TypeI, Text: l.Text},
		)
		g.Edges = append(g.Edges, Edge{FromID: lid, ToID: yid}, Edge{FromID: yid, ToID: iid})
		if prev != "" {
			ta := idxID("TA", i-1)
			g.Nodes = append(g.Nodes, Node{ID: ta, Type: TypeTA, Scheme: SchemeDefaultTransition})
			g.Edges = append(g.Edges, Edge{FromID: prev, ToID: ta}, Edge{FromID: ta, ToID: lid})
		}
		prev = lid
	}
	out, _ := Normalize(g)
	return out
}

func idxID(prefix string, i int) string {
	return prefix + strconv.Itoa(i)
}
