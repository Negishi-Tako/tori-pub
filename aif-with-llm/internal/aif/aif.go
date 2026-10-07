// Package aif は AIF（Argument Interchange Format）の厳密な正準表現を扱う。
//
// AIF 中核オントロジー（Chesñevar et al. 2006）と IAT（Inference Anchoring Theory:
// Budzynska & Reed 2011）のメタモデルにだけ従った表現を持ち、独自の拡張は一切入れない。
// AIFdb のゴールドアノテーションと突き合わせるには「素の AIF」でなければならないため。
//
// 依存の向き: aif は llm / prompts にしか依存しない。
// 評価（GED）は internal/evaluation 側に置く（evaluation -> aif の一方向）。
package aif

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// NodeType は AIF のノード種別。AIF 中核オントロジーの I-node と S-node
// （RA/CA/MA）に、IAT の L-node・YA-node・TA-node を加えた 7 種で閉じる。
// ここを増やさないこと（増やした時点で「素の AIF」ではなくなる）。
type NodeType string

const (
	TypeI  NodeType = "I"  // Information node（命題）
	TypeL  NodeType = "L"  // Locution（発話。IAT）
	TypeRA NodeType = "RA" // Rule Application（推論）
	TypeCA NodeType = "CA" // Conflict Application（対立）
	TypeMA NodeType = "MA" // Preference/Rephrase Application（言い換え）
	TypeYA NodeType = "YA" // Illocutionary Anchoring（発話行為。IAT）
	TypeTA NodeType = "TA" // Transition Application（発話遷移。IAT）
)

// IsSNode は RA/CA/MA（推論・対立・言い換えの S-node）か。
func (t NodeType) IsSNode() bool { return t == TypeRA || t == TypeCA || t == TypeMA }

// Valid は AIF の 7 種のいずれかか。
func (t NodeType) Valid() bool {
	switch t {
	case TypeI, TypeL, TypeRA, TypeCA, TypeMA, TypeYA, TypeTA:
		return true
	}
	return false
}

// 既定スキーム。AIFdb のアノテーションが使う既定値をそのまま採る。
const (
	SchemeDefaultInference  = "Default Inference"
	SchemeDefaultConflict   = "Default Conflict"
	SchemeDefaultRephrase   = "Default Rephrase"
	SchemeDefaultTransition = "Default Transition"
	SchemeDefaultIllocuting = "Default Illocuting"
)

// IllocutionaryForces は YA-node に許す発話行為（illocutionary force）の集合。
// IAT の標準セット（Budzynska & Reed 2011 / QT30 のアノテーションガイドライン）に
// 合わせる。
var IllocutionaryForces = []string{
	"Asserting",
	"Arguing",
	"Restating",
	"Analysing",
	"Agreeing",
	"Disagreeing",
	"PureQuestioning",
	"AssertiveQuestioning",
	"RhetoricalQuestioning",
	"Challenging",
	SchemeDefaultIllocuting,
}

// LocutionForces は L-node に anchor される YA（＝発話 1 つを put forward する力）に
// 使える発話行為。Arguing / Restating / Analysing は「発話どうしの関係」を担う力で、
// 遷移（TA）に anchor されるものなのでここには入らない。
//
// QT30 のゴールド（本 PJ が取得した 6 サブコーパス・YA 5,000 件超）を数えると、
// L→I の anchor は Asserting 90.7% / PureQuestioning 5.9% / AssertiveQuestioning 1.1% /
// Agreeing 0.8% / RhetoricalQuestioning 0.7% で、Arguing は 0.1% しかない。
// 一方 TA→RA の anchor は Arguing 97.6%、TA→MA は Restating 86.1%、
// TA→CA は Disagreeing 91.9%。この住み分けが IAT の運用であり、
// アノテータ（人でも LLM でも）に両方の集合を同時に見せてはいけない。
var LocutionForces = []string{
	"Asserting",
	"PureQuestioning",
	"AssertiveQuestioning",
	"RhetoricalQuestioning",
	"Challenging",
	"Agreeing",
	"Disagreeing",
	SchemeDefaultIllocuting,
}

// TransitionForces は TA に anchor される YA（＝S-node を illocute する力）に使える発話行為。
var TransitionForces = []string{
	"Arguing",
	"Restating",
	"Disagreeing",
	"Challenging",
	"Analysing",
	SchemeDefaultIllocuting,
}

// ValidForce は f が IAT の発話行為として妥当か。
func ValidForce(f string) bool { return slices.Contains(IllocutionaryForces, f) }

// Node は AIF の 1 ノード。
type Node struct {
	ID   string   `json:"nodeID"`
	Type NodeType `json:"type"`
	Text string   `json:"text"`
	// Scheme は S-node / YA / TA のスキーム名（"Default Inference" 等）。
	// I-node と L-node では空。
	Scheme string `json:"scheme,omitempty"`
	// Speaker は L-node の話者。AIFdb では locutions[].personID で持つが、
	// 突き合わせに使うのは表示名なのでここに正規化して持たせる。
	Speaker   string `json:"speaker,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
	// Confidence は S-node に対してだけ意味を持つ、LLM が申告した等級。
	// 3 値の確信度（"high" / "medium" / "low"）か、
	// 5 段階の依存度（"5"〜"1"）のどちらかが入る（どちらかは GradeLadder で判別する）。
	// 空なら申告が無い（＝絞り込みの対象外）。
	// 生成時に付けておけば閾値を後から動かして P/R 曲線を引ける
	// ＝閾値を変えるたびに LLM を叩き直さずに済む。
	Confidence string `json:"confidence,omitempty"`
}

// Edge は AIF の 1 辺。AIF の辺は種別を持たない（両端のノード種別で意味が決まる）。
type Edge struct {
	FromID string `json:"fromID"`
	ToID   string `json:"toID"`
}

// Graph は 1 ノードセット分の AIF グラフ。
type Graph struct {
	// ID はノードセット識別子（AIFdb の nodesetID、または生成側の識別子）。
	ID    string `json:"id"`
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// Relation は S-node（RA/CA/MA）を 1 本の関係として畳んだもの。
// GED を命題単位で取るときと、証拠の書き出しで使う。
type Relation struct {
	SNodeID string   `json:"snode_id"`
	Type    NodeType `json:"type"`
	// SrcIDs は前提側の I-node（RA は複数ありうる）。
	SrcIDs []string `json:"src_ids"`
	// DstID は結論側 / 対立先 / 言い換え先の I-node。
	DstID string `json:"dst_id"`
	// Force は S-node を illocute する YA のスキーム（無ければ空）。
	Force string `json:"force,omitempty"`
}

// Index はノード ID からノード・隣接を引くための索引。
// Graph のメソッドで毎回線形探索すると O(n^2) になるため、まとめて作る。
type Index struct {
	byID map[string]Node
	out  map[string][]string
	in   map[string][]string
}

// NewIndex は g の索引を作る。
func NewIndex(g Graph) *Index {
	ix := &Index{
		byID: make(map[string]Node, len(g.Nodes)),
		out:  make(map[string][]string, len(g.Nodes)),
		in:   make(map[string][]string, len(g.Nodes)),
	}
	for _, n := range g.Nodes {
		ix.byID[n.ID] = n
	}
	for _, e := range g.Edges {
		ix.out[e.FromID] = append(ix.out[e.FromID], e.ToID)
		ix.in[e.ToID] = append(ix.in[e.ToID], e.FromID)
	}
	return ix
}

func (ix *Index) Node(id string) (Node, bool) { n, ok := ix.byID[id]; return n, ok }
func (ix *Index) Out(id string) []string      { return ix.out[id] }
func (ix *Index) In(id string) []string       { return ix.in[id] }

// Type は id のノード種別。存在しなければ空文字。
func (ix *Index) Type(id string) NodeType {
	if n, ok := ix.byID[id]; ok {
		return n.Type
	}
	return ""
}

// NodesOfType は種別 t のノードを入力順で返す。
func (g Graph) NodesOfType(t NodeType) []Node {
	out := make([]Node, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if n.Type == t {
			out = append(out, n)
		}
	}
	return out
}

// CountByType はノード種別ごとの個数。
func (g Graph) CountByType() map[NodeType]int {
	m := make(map[NodeType]int, 7)
	for _, n := range g.Nodes {
		m[n.Type]++
	}
	return m
}

// Relations は RA/CA/MA を関係に畳んで返す。S-node の入辺のうち I-node から来る
// ものを前提、出辺のうち I-node へ向かうものを結論とする。
// AIF では CA が RA を攻撃することもあるが（undercut）、QT30 には現れないため
// I-node 同士に限って扱い、それ以外は落とす（落とした件数は Validate で分かる）。
func (g Graph) Relations() []Relation {
	ix := NewIndex(g)
	out := make([]Relation, 0, 16)
	for _, n := range g.Nodes {
		if !n.Type.IsSNode() {
			continue
		}
		var srcs []string
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeI {
				srcs = append(srcs, id)
			}
		}
		var dst string
		for _, id := range ix.Out(n.ID) {
			if ix.Type(id) == TypeI {
				dst = id
				break
			}
		}
		if len(srcs) == 0 || dst == "" {
			continue
		}
		sort.Strings(srcs)
		r := Relation{SNodeID: n.ID, Type: n.Type, SrcIDs: srcs, DstID: dst}
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeYA {
				ya, _ := ix.Node(id)
				r.Force = ya.Scheme
				break
			}
		}
		out = append(out, r)
	}
	return out
}

// Illocution は L-node（または TA）と、そこに YA で紐づく内容の組。
type Illocution struct {
	YAID string `json:"ya_id"`
	// AnchorID は YA が anchor される L-node か TA。
	AnchorID   string   `json:"anchor_id"`
	AnchorType NodeType `json:"anchor_type"`
	// TargetID は YA が illocute する I-node か S-node。
	TargetID   string   `json:"target_id"`
	TargetType NodeType `json:"target_type"`
	Force      string   `json:"force"`
}

// Illocutions は YA を (anchor, target) の組に畳んで返す。
func (g Graph) Illocutions() []Illocution {
	ix := NewIndex(g)
	out := make([]Illocution, 0, 16)
	for _, n := range g.Nodes {
		if n.Type != TypeYA {
			continue
		}
		il := Illocution{YAID: n.ID, Force: n.Scheme}
		for _, id := range ix.In(n.ID) {
			if t := ix.Type(id); t == TypeL || t == TypeTA {
				il.AnchorID, il.AnchorType = id, t
				break
			}
		}
		for _, id := range ix.Out(n.ID) {
			if t := ix.Type(id); t == TypeI || t.IsSNode() {
				il.TargetID, il.TargetType = id, t
				break
			}
		}
		if il.TargetID == "" {
			continue
		}
		out = append(out, il)
	}
	return out
}

// Transitions は TA を (from L, to L) の組で返す。
type Transition struct {
	TAID   string `json:"ta_id"`
	FromID string `json:"from_id"`
	ToID   string `json:"to_id"`
}

// Transitions は L→TA→L の遷移を返す。
func (g Graph) Transitions() []Transition {
	ix := NewIndex(g)
	out := make([]Transition, 0, 16)
	for _, n := range g.Nodes {
		if n.Type != TypeTA {
			continue
		}
		var from, to string
		for _, id := range ix.In(n.ID) {
			if ix.Type(id) == TypeL {
				from = id
				break
			}
		}
		for _, id := range ix.Out(n.ID) {
			if ix.Type(id) == TypeL {
				to = id
				break
			}
		}
		if from == "" || to == "" {
			continue
		}
		out = append(out, Transition{TAID: n.ID, FromID: from, ToID: to})
	}
	return out
}

// LocutionOrder は L-node を時系列に並べて返す。
//
// TA（L→TA→L）を「前の発話 → 後の発話」という順序の制約とみなし、
// 制約をすべて満たす並び（トポロジカル順）を返す。制約で決まらない発話どうしは
// タイムスタンプ、それも同じなら入力順で並べる。TA が循環しているときは、
// 残りのうち最も早い発話から出して循環を切る（無限ループにしない）。
//
// QT30 のゴールドの TA は 1 本の鎖ではない。test 124 ノードセットでは、出ていく TA を
// 2 本以上持つ L-node が 194 個あり、時系列で隣接しない発話を結ぶ TA が 412 本ある。
// 以前の実装は「各 L から最初の TA だけをたどり、たどれなかった鎖を末尾に足す」もので、
// 分岐のたびに発話を末尾へ追いやっていた。エピソードの書き起こしの位置と突き合わせると
// （位置を一意に特定できた 1,642 の隣接対）、旧実装は 81 対、この実装は 28 対で順序が逆になる。
// 書き起こしで後ろ向きの TA は 1,489 本中 3 本だけなので、TA を順序の制約として使うのは妥当である。
// タイムスタンプはアノテーションの作成時刻で、後から足した発話は遅れるので、補助にしか使わない。
func (g Graph) LocutionOrder() []Node {
	locs := g.NodesOfType(TypeL)
	idx := make(map[string]int, len(locs))
	for i, n := range locs {
		idx[n.ID] = i
	}
	succ := map[string][]string{}
	indeg := map[string]int{}
	for _, t := range g.Transitions() {
		if _, ok := idx[t.FromID]; !ok {
			continue
		}
		if _, ok := idx[t.ToID]; !ok {
			continue
		}
		succ[t.FromID] = append(succ[t.FromID], t.ToID)
		indeg[t.ToID]++
	}
	earlier := func(a, b int) bool {
		if ta, tb := locs[a].Timestamp, locs[b].Timestamp; ta != tb {
			return ta < tb
		}
		return a < b
	}
	done := make([]bool, len(locs))
	out := make([]Node, 0, len(locs))
	for len(out) < len(locs) {
		best, bestFree := -1, false
		for i, n := range locs {
			if done[i] {
				continue
			}
			free := indeg[n.ID] == 0
			// 制約の解けた発話を優先し、その中で最も早いものを選ぶ。
			// 解けた発話が無い（循環）ときだけ、解けていない発話から選ぶ。
			if best == -1 || (free && !bestFree) || (free == bestFree && earlier(i, best)) {
				best, bestFree = i, free
			}
		}
		done[best] = true
		out = append(out, locs[best])
		for _, s := range succ[locs[best].ID] {
			indeg[s]--
		}
	}
	return out
}

// PropositionOrder は I-node を「何番目の発話から出たか」に対応づける。
//
// 関係が何発話離れた命題どうしを結んでいるかを測るために使う。
// 全体の F1 は隣接発話どうしの関係に支配されるので、距離帯ごとに分けて数える。
//
// 1 つの I-node が複数の発話に anchor されている場合（言い直し・引用。test のゴールドで 17 個）は、
// 最も早い発話の位置を採る。以前は Illocutions の並びで最後に見た発話の位置になっており、
// 結果がノードの並び順に依存していた。
func (g Graph) PropositionOrder() map[string]int {
	pos := map[string]int{}
	for i, n := range g.LocutionOrder() {
		pos[n.ID] = i
	}
	out := make(map[string]int, len(pos))
	for _, il := range g.Illocutions() {
		if il.AnchorType != TypeL || il.TargetType != TypeI {
			continue
		}
		p, ok := pos[il.AnchorID]
		if !ok {
			continue
		}
		if q, seen := out[il.TargetID]; !seen || p < q {
			out[il.TargetID] = p
		}
	}
	return out
}

// DistanceBand は発話間距離の帯。GED では見えない「遠い関係を拾えているか」を
// 読むための区切り。
//
// 帯 "1" は距離 0（同じ発話から出た 2 つの命題どうし）も含む。生成側は 1 発話から
// 命題を 1 つしか作らないので距離 0 は生成されず、ゴールドでも test 1,090 本中 1 本しかない。
// 帯を増やすと表と図の列が変わるため、帯 "1" に含めたうえでここに明記する。
func DistanceBand(d int) string {
	switch {
	case d <= 1:
		return "1"
	case d <= 4:
		return "2-4"
	case d <= 9:
		return "5-9"
	default:
		return "10+"
	}
}

// DistanceBands は DistanceBand が返しうる値（表の列順）。
var DistanceBands = []string{"1", "2-4", "5-9", "10+"}

// 等級の梯子。厳しい順に並べる。0 番目だけを残すのが最も厳しい閾値。
//
// 3 値（確信度）は実測で low がほぼ使われず実質 2 値に潰れたため、
// 5 段階（依存度）を足した。どちらの語彙が使われているかはグラフを見て判別する。
var gradeLadders = [][]string{
	{"5", "4", "3", "2", "1"},
	{"high", "medium", "low"},
}

// GradeLadder は g の S-node が使っている等級の梯子を返す。
// 等級の申告が無い（または未知の語彙の）グラフでは nil を返す。
func (g Graph) GradeLadder() []string {
	seen := map[string]bool{}
	for _, n := range g.Nodes {
		if n.Type.IsSNode() && n.Confidence != "" {
			seen[n.Confidence] = true
		}
	}
	if len(seen) == 0 {
		return nil
	}
	for _, ladder := range gradeLadders {
		known := 0
		for _, lv := range ladder {
			if seen[lv] {
				known++
			}
		}
		if known == len(seen) {
			return ladder
		}
	}
	return nil
}

// FilterRelationsByConfidence は、申告された等級が keep に含まれない関係を落とす。
//
// 確信度が空の S-node は「申告が無い」ので常に残す（v3 以前のプロンプトとの互換）。
// 関係を illocute する YA も一緒に落とす（残すと宙に浮いた YA になり AIF 違反になる）。
func (g Graph) FilterRelationsByConfidence(keep ...string) Graph {
	ok := map[string]bool{}
	for _, k := range keep {
		ok[k] = true
	}
	drop := map[string]bool{}
	for _, n := range g.Nodes {
		if n.Type.IsSNode() && n.Confidence != "" && !ok[n.Confidence] {
			drop[n.ID] = true
		}
	}
	if len(drop) == 0 {
		return g
	}
	for _, il := range g.Illocutions() {
		if drop[il.TargetID] {
			drop[il.YAID] = true
		}
	}
	out := Graph{ID: g.ID}
	for _, n := range g.Nodes {
		if !drop[n.ID] {
			out.Nodes = append(out.Nodes, n)
		}
	}
	for _, e := range g.Edges {
		if !drop[e.FromID] && !drop[e.ToID] {
			out.Edges = append(out.Edges, e)
		}
	}
	return out
}

// SpeakerOf は L-node の話者を返す。話者が空なら本文の "名前 : 本文" 形から拾う。
func SpeakerOf(n Node) string {
	if n.Speaker != "" {
		return n.Speaker
	}
	if i := strings.Index(n.Text, " : "); i > 0 && i < 60 {
		return strings.TrimSpace(n.Text[:i])
	}
	return ""
}

// String は人が読む用の 1 行表現（ログ・証拠書き出し用）。
func (n Node) String() string {
	if n.Scheme != "" && n.Text == "" {
		return fmt.Sprintf("%s(%s)", n.Type, n.Scheme)
	}
	t := n.Text
	if len([]rune(t)) > 80 {
		t = string([]rune(t)[:80]) + "…"
	}
	return fmt.Sprintf("%s[%s]", n.Type, t)
}
