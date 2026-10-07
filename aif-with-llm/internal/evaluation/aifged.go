package evaluation

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// AIF グラフ間 GED の実装バージョン。手順を変えたら新しい版を足す。
//
// 0.1 → 0.2 で直したのは次の 2 点。どちらも「測りたいもの」ではなく
// 「測り方」の欠陥だった（レポート §11 を参照）。
//
//	(1) 対応付けの縮退
//	    0.1 は全ノードを Hungarian 法に載せていた。ところが TA は互いに
//	    置換コスト 0、同じスキームの YA も 0 なので、これらは全部が等価になり、
//	    最適解が大量に縮退して任意の割当が返る。両端の L が正しく対応していても
//	    間の TA が別物に割り当てられ、L→TA→L の辺が誤りとして数えられていた
//	    （ゴールドの発話をそのまま使う baseline ですら辺の精度 0.386）。
//	    0.2 は内容を持つノード（L と I）だけを対応付け、TA / YA / S-node は
//	    その対応から導出する。AIF ではこれらは構造ノードで、L と I の対応が
//	    決まれば一意に決まるため、独立に探索する対象ではない。
//
//	(2) 辺の「完全一致か全損か」
//	    0.1 は辺が 1 つでもずれれば挿入 1.0 + 削除 1.0 = 2.0 を課していた。
//	    このため「辺を 1 本足す」ことの期待コストは P(1-2p)（p は精度）となり、
//	    精度が 0.5 を超えないと必ず損をする。0.5 という分岐点は
//	    「挿入コスト = 削除コスト = 1」と置いたことの副産物でしかない。
//	    0.2 は同じ端点対に正解の辺があるなら置換として扱い、
//	    種別違いは EdgeTypeMismatch、向き違いは EdgeDirectionMismatch を課す。
//	    全損になるのは、正解に対応が存在しない純粋な過剰生成だけ。
//
// 0.2 → 0.3 で直したのは次の 3 点。いずれも 0.2 に残っていた「測り方」の欠陥で、
// 1 つ目は条件の順位を反転させるほどの大きさだった（レポート §12.1）。
//
//	(3) 正規化の分母が pred のサイズに依存する
//	    0.2 の分母は「pred を全部消して gold を全部作る」コストなので、
//	    関係を多く出すほど分母が膨らんでコストが希釈される。
//	    実測では、分母を gold 側だけに固定すると
//	    「LLM が baseline を上回る」「命題 GED が v1<v2<v3 と単調になる」の
//	    両方が消えた（符号が分母の定義で決まっていた）。
//	    0.3 は gold 基準の正規化（SimilarityGold）を主指標にする。
//	    参照側の大きさで割るのは WER / TER と同じ思想で、
//	    「出力を増やすと得をする」経路が無い。0〜1 には収まらず、
//	    基準線より悪いグラフは負の値になるが、それは表現できた方が正しい。
//
//	(4) 全体グラフに辺の部分点が入っていなかった
//	    0.2 は labelled を命題グラフにだけ渡していたため、同じ 1 本の誤りが
//	    命題単位では 0.75 / 1.0、全体では 4.75 / 4.00 になっていた
//	    （関係 1 本が S-node と辺 2 本に展開され、辺が両側とも外れる）。
//	    0.3 は全体グラフでも関係を辺 1 本に畳み（aif.Graph.DialogueEdges）、
//	    部分点を効かせる。関係の誤りのコストは粒度によらず同じ値になる。
//
//	(5) 遷移に anchor される YA が LLM の質を表していなかった
//	    pred 側は「隣接しない発話同士の関係には TA が無いので YA も作らない」
//	    という harness の構成規則で個数が決まる（gold 508 個 vs pred 422 個）。
//	    0.3 は全体グラフから外し、発話行為は ForceScore で測る。
//
// 0.3 → 0.4 で直したのは次の 1 点。命題グラフ（ReducedGED）は 0.3 と同じで、全体グラフだけが変わる。
//
//	(6) 全体グラフに TA が残っていた
//	    (5) と同じ理由（pred 側の個数が harness の構成規則で決まる）が TA にも当てはまる。
//	    pred の TA は並べた発話の隣どうしに機械的に張るが、ゴールドの TA は分岐し、
//	    隣接しない発話も結ぶ。発話分割をゴールドどおりに与えた基準線でも
//	    TA の 1/4 前後が対応せず、全体 GED の分母の約 18% が LLM の質と無関係なコストになっていた。
//	    0.4 は全体グラフを L / I / 発話の YA と、関係を畳んだ辺だけにする（aif.Graph.AnchorEdges）。
//
// 0.2 以降の辺の照合と構造ノードの導出は、出力の順序に依存しない（完全一致 → 種別違い →
// 向き違いの順にパスごとに照合する）。辺の照合は公開時のコードレビューで、
// S-node の導出（derive）はその後の監査で直した（レポート §15.2(f)）。
// どちらも、同じ命題対に種別の違う関係を複数出した場合にだけ値が変わる。
const (
	MetricGEDv1 = "aif-ged/0.1"
	MetricGEDv2 = "aif-ged/0.2"
	MetricGEDv3 = "aif-ged/0.3"
	MetricGEDv4 = "aif-ged/0.4"
)

// AIFComparisonVersion は既定の実装バージョン。
const AIFComparisonVersion = MetricGEDv4

// AIFMetrics は既定で計算する GED の版。古い版を残すのは、
// 「指標の設計が結論を変えた」ことを示すのがこの実験の結果の一部だから。
var AIFMetrics = []string{MetricGEDv1, MetricGEDv2, MetricGEDv3, MetricGEDv4}

// AIFWeights は AIF グラフ間の編集距離（GED）の重み。
//
// 重みの決め方（レポートに書く根拠）:
//   - 削除/挿入コストはノード種別ごとに置く。既定はすべて 1.0 の一様重みにする。
//     一様が「特別な仮定を置かない」基準であり、まずここを主指標にする。
//   - 置換コストは種別が同じ場合のみ許す。I-node / L-node は埋め込みの
//     コサイン類似度から 1 - cos で決める（Bergmann らの議論グラフ検索で使われる
//     「S-node は種別一致、I-node は意味的類似度」という設計に合わせる）。
//   - RA/CA/MA の取り違えは「関係はあるが向き・種別が違う」状態なので、
//     削除 + 挿入（= 2.0）より安く、しかし完全一致より高い中間の値にする。
//   - YA の発話行為は、同系統（主張系 / 質問系 / 同意系）の取り違えを
//     系統違いより軽くする。Asserting と Arguing の取り違えと、
//     Asserting と PureQuestioning の取り違えを同じ重さで数えるのは乱暴なため。
//   - TA 同士は内容を持たないので置換コスト 0。
//
// 感度分析として YA/TA の削除コストを下げた重み（DiscountedAIFWeights）も併せて
// 計算する。YA/TA は L-node と I-node の構造からほぼ機械的に決まるため、
// 一様重みだとそこが二重に効いてしまうという批判に答えるため。
type AIFWeights struct {
	// Name は重み設定の識別子（レポートの表側に出る）。
	Name string `json:"name"`
	// NodeDelete はノード種別ごとの削除・挿入コスト。未設定の種別は 1.0。
	NodeDelete map[aif.NodeType]float64 `json:"node_delete"`
	// EdgeCost は辺 1 本あたりの挿入・削除コスト。
	EdgeCost float64 `json:"edge_cost"`
	// SNodeTypeMismatch は RA/CA/MA を取り違えたときの置換コスト。
	SNodeTypeMismatch float64 `json:"snode_type_mismatch"`
	// ForceFamilyMatch は発話行為が別ラベルだが同系統のときの置換コスト。
	ForceFamilyMatch float64 `json:"force_family_match"`
	// ForceMismatch は発話行為が別系統のときの置換コスト。
	ForceMismatch float64 `json:"force_mismatch"`
	// Forbidden は種別違いの置換に与える禁止コスト。
	Forbidden float64 `json:"forbidden"`
	// EdgeTypeMismatch は、同じ命題対を結ぶ正解の辺はあるが種別が違うときの
	// 置換コスト（命題単位のグラフ用）。S-node の取り違えと同じ扱いにする。
	EdgeTypeMismatch float64 `json:"edge_type_mismatch"`
	// EdgeDirectionMismatch は、向きだけが逆の正解の辺があるときの置換コスト。
	// 「関係の存在は当てている」ぶん、挿入 + 削除（2.0）より安くする。
	//
	// 既定値 1.0 は、正解の辺を取りこぼしたときの削除コスト（EdgeCost = 1.0）と同じである。
	// つまり向きの逆な関係を出すことは、何も出さないことと同じコストになり、
	// 「出さない」に対する得は 0、「正解に対応の無い関係を出す」（挿入 1.0 + 削除 1.0）に対する得は 1.0。
	// これを 1.0 未満にすると、向きを誤った関係を出す方が出さないより必ず得になり、
	// 向きを判定せずに両向きの片方を出す戦略が報われてしまうので、1.0 を下限として置いている。
	// 向きの誤りを「関係の存在は当てた」として数えたいときは、RelationScore.Undirected を使う。
	EdgeDirectionMismatch float64 `json:"edge_direction_mismatch"`
}

// DefaultAIFWeights は一様重み（主指標）。
func DefaultAIFWeights() AIFWeights {
	return AIFWeights{
		Name:              "uniform",
		NodeDelete:        map[aif.NodeType]float64{},
		EdgeCost:          1.0,
		SNodeTypeMismatch: 0.75,
		ForceFamilyMatch:  0.5,
		ForceMismatch:     1.0,
		Forbidden:         1e6,

		EdgeTypeMismatch:      0.75,
		EdgeDirectionMismatch: 1.0,
	}
}

// DiscountedAIFWeights は YA/TA の削除コストを半分にした感度分析用の重み。
func DiscountedAIFWeights() AIFWeights {
	w := DefaultAIFWeights()
	w.Name = "ya-ta-discounted"
	w.NodeDelete = map[aif.NodeType]float64{aif.TypeYA: 0.5, aif.TypeTA: 0.5}
	return w
}

func (w AIFWeights) deleteCost(t aif.NodeType) float64 {
	if c, ok := w.NodeDelete[t]; ok {
		return c
	}
	return 1.0
}

// forceFamily は発話行為の系統。系統違いの取り違えを重く数えるために使う。
func forceFamily(force string) string {
	switch force {
	case "Asserting", "Arguing", "Restating", "Analysing", aif.SchemeDefaultIllocuting:
		return "assertive"
	case "PureQuestioning", "AssertiveQuestioning", "RhetoricalQuestioning", "Challenging":
		return "interrogative"
	case "Agreeing", "Disagreeing":
		return "responsive"
	}
	return "other"
}

// AIFComparison は 1 ノードセット分の比較結果。
type AIFComparison struct {
	NodesetID string `json:"nodeset_id"`
	// Condition は生成条件（"llm-e2e" / "llm-gold-loc" / "baseline" 等）。
	Condition string `json:"condition"`
	Weights   string `json:"weights"`

	CountsPred map[aif.NodeType]int `json:"counts_pred"`
	CountsGold map[aif.NodeType]int `json:"counts_gold"`

	// Metric は使った GED 実装の版（MetricGEDv1〜MetricGEDv4）。
	Metric string `json:"metric"`

	// FullGED は対話層まで含む AIF グラフ全体の GED（何を含むかは版で違う。CompareAIF を参照）。
	FullGED AIFDistance `json:"full_ged"`
	// ReducedGED は I-node と RA/CA/MA だけに畳んだ GED（レポートの「命題 GED」）。
	ReducedGED AIFDistance `json:"reduced_ged"`
	// FullSimilarity / ReducedSimilarity は 1 - 正規化 GED（対称分母）。
	// 0.2 までの主指標だが、pred のサイズに分母が依存するので条件の比較には使わない。
	FullSimilarity    float64 `json:"full_similarity"`
	ReducedSimilarity float64 `json:"reduced_similarity"`
	// FullSimilarityGold / ReducedSimilarityGold は gold 基準の正規化（0.3 の主指標）。
	FullSimilarityGold    float64 `json:"full_similarity_gold"`
	ReducedSimilarityGold float64 `json:"reduced_similarity_gold"`

	// Propositions は I-node の対応付けの診断。
	Propositions PropositionAlignment `json:"propositions"`
	// Locutions は L-node の対応付けの診断（stage 1 = 発話分割の質）。
	//
	// `gold-loc` / `baseline` はゴールドの L-node をそのまま使うので 1.0 に近くなり、
	// 意味を持つのは `e2e`（分割を LLM に任せた条件）だけ。
	// 「発話分割が上流のボトルネック」かどうかは、0.2 までは
	// `e2e` と `gold-loc` の差からの間接的な推定しか無かった。
	// 命題層と同じ通貨（対応付けの cos と件数比）で直接読めるようにする。
	Locutions PropositionAlignment `json:"locutions"`
	// Relations は RA/CA/MA の一致（GED の内訳として最も解釈しやすい部分）。
	Relations RelationScore `json:"relations"`
	// RelationPairSimilarity は関係で結ばれた命題どうしの平均 cos を、
	// 当たった関係 / 過剰生成 / 取りこぼし別に集計したもの。
	RelationPairSimilarity PairSimilarityStats `json:"relation_pair_similarity"`
	// Forces は対応付いた I-node 上での発話行為の一致。
	Forces ForceScore `json:"forces"`

	Evidence AIFEvidence `json:"evidence"`
}

// AIFDistance は 1 粒度分の編集距離。内訳を必ず持たせる。
// 「値が動いたのはノードのせいか辺のせいか」が分からない指標は、
// 改善したかどうかの判断に使えないため。
type AIFDistance struct {
	GED float64 `json:"ged"`
	// NormalizedGED は「A を全部消して B を全部作り直すコスト」で割った値（0〜1）。
	NormalizedGED float64 `json:"normalized_ged"`
	// Similarity は 1 - NormalizedGED。
	Similarity float64 `json:"similarity"`
	NodeCost   float64 `json:"node_cost"`
	EdgeCost   float64 `json:"edge_cost"`
	// Denominator は対称正規化の分母（取りうる最大コスト）。pred の大きさに依存する。
	Denominator float64 `json:"denominator"`
	// GoldCost は「gold を何も無いところから作るコスト」。pred に依存しないので、
	// 条件間で共通の分母になる（0.3 の主指標）。
	GoldCost float64 `json:"gold_cost"`
	// NormalizedGEDGold は GED / GoldCost、SimilarityGold は 1 - それ。
	// pred が gold より大きく外れていれば 1 を超え、類似度は負になりうる。
	NormalizedGEDGold float64 `json:"normalized_ged_gold"`
	SimilarityGold    float64 `json:"similarity_gold"`
	// Matched は対応が付いたノード対の数、Derived はそのうち構造から導出したもの。
	Matched int `json:"matched"`
	Derived int `json:"derived"`
	// Edges は辺の内訳。
	Edges EdgeBreakdown `json:"edges"`
}

// EdgeBreakdown は辺の編集内訳。
type EdgeBreakdown struct {
	// Exact は種別・向きとも一致した辺（コスト 0）。
	Exact int `json:"exact"`
	// TypeSubstituted は同じ命題対を結ぶが種別が違った辺。
	TypeSubstituted int `json:"type_substituted"`
	// DirectionSubstituted は向きだけが逆だった辺。
	DirectionSubstituted int `json:"direction_substituted"`
	// Inserted は正解に対応が無い辺（純粋な過剰生成）、Deleted は取りこぼした正解の辺。
	Inserted int `json:"inserted"`
	Deleted  int `json:"deleted"`
}

// PropositionAlignment は I-node の対応付け結果。
type PropositionAlignment struct {
	// Threshold は「対応付いた」と数えるコサイン類似度の下限。
	Threshold float64 `json:"threshold"`
	// Aligned は Hungarian 法が対応付けたペア数（閾値は見ない）。
	Aligned int `json:"aligned"`
	// AboveThreshold は Hungarian 法の割当（関係の照合・GED と共通）のうち、閾値以上のペア数。
	// PRF.TP（閾値以上の対だけで作れる 1 対 1 の対応の最大数）以下になる。
	AboveThreshold int     `json:"above_threshold"`
	MeanSimilarity float64 `json:"mean_similarity"`
	PRF            PRF     `json:"prf"`
	// CountRatio は 生成側 I-node 数 / ゴールド I-node 数。
	CountRatio float64 `json:"count_ratio"`
	// F1ByThreshold は閾値を変えたときの命題 F1。
	//
	// 既定の 0.55 は、同一ノードセット内で*実際に関係がある*命題対の平均 cos
	// （実測 0.44〜0.48）のすぐ上でしかない。1 点の値だけで
	// 「命題層は実用水準」と言えないので、曲線として併記する。
	F1ByThreshold map[string]float64 `json:"f1_by_threshold,omitempty"`
}

// PropositionThresholds は F1ByThreshold を出す閾値。
var PropositionThresholds = []float64{0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9}

// RelationScore は関係の一致。
//
// Reversed（向きだけが逆）を独立に数えるのが要点。GED でも P/R でも、
// 向きの誤りは「関係が 1 本足りず 1 本余分」として二重に罰せられ、
// 「関係の存在自体は当てられている」ことが見えなくなるため。
type RelationScore struct {
	// Typed は種別（RA/CA/MA）まで一致した場合のみ TP に数えたもの。
	Typed PRF `json:"typed"`
	// Untyped は種別を無視し、向きだけ合っていれば TP に数えたもの。
	Untyped PRF `json:"untyped"`
	// Undirected は向きも無視したもの。
	Undirected PRF `json:"undirected"`
	// Reversed は向きだけが逆だった関係の数。
	Reversed int `json:"reversed"`
	// TypeConfusion は 生成側の種別 → ゴールドの種別 の混同行列
	// （向き一致で対応付いた関係のみ）。
	TypeConfusion map[string]map[string]int `json:"type_confusion"`
	// PredTotal / GoldTotal は関係の総数。
	PredTotal int `json:"pred_total"`
	GoldTotal int `json:"gold_total"`
	// ByDistance は発話間距離の帯ごとの一致。
	//
	// ゴールドの関係の約 7 割は隣接発話どうしなので、全体の F1 は隣接の山に支配され、
	// 「遠い関係を拾えていない」という失敗が見えない。帯ごとに分けて数える。
	ByDistance map[string]DistanceBucket `json:"by_distance"`
}

// DistanceBucket は 1 つの距離帯の集計。
//
// 当たりを 2 通りに数えるのが要点。生成側の距離と ゴールド側の距離は
// 命題の対応付けのずれで食い違うことがあり、片方の数え方だけだと
// 分子と分母の帯が揃わない。実際、当初は TP をゴールド帯・分母を生成帯で
// 数えていたため、隣接を機械的につなぐ条件で帯ごとの精度が 1 を超えた
// （生成はすべて距離 1 なのに、当たった相手のゴールド距離が 2-4 だった）。
//
//	帯ごとの精度   = TPPred / PredTotal       （どちらも生成側の帯）
//	帯ごとの再現率 = TPGold / GoldTotal       （どちらもゴールド側の帯）
type DistanceBucket struct {
	GoldTotal int `json:"gold_total"`
	PredTotal int `json:"pred_total"`
	// TP は種別・向きとも当たった関係を**ゴールド側**の距離帯に数えたもの
	// （再現率の分子）。後方互換のため名前はそのまま。
	TP int `json:"tp"`
	// TPPred は同じ当たりを**生成側**の距離帯に数えたもの（精度の分子）。
	TPPred int `json:"tp_pred"`
	// TPUndirected / TPUndirectedPred は種別と向きを無視した当たり。
	TPUndirected     int `json:"tp_undirected"`
	TPUndirectedPred int `json:"tp_undirected_pred"`
}

// PairSimilarityStats は関係で結ばれた命題どうしの cos の平均。
//
// 「過剰生成された関係は、単に話題が近いだけの命題どうしを結んでいるのではないか」
// という問いに答えるための集計。当たった関係と過剰生成で平均 cos が変わらなければ、
// 埋め込みの近さでは関係の有無を説明できない＝埋め込みに頼る対応付けの限界を示す。
type PairSimilarityStats struct {
	CorrectMean  float64 `json:"correct_mean"`
	CorrectN     int     `json:"correct_n"`
	SpuriousMean float64 `json:"spurious_mean"`
	SpuriousN    int     `json:"spurious_n"`
	MissedMean   float64 `json:"missed_mean"`
	MissedN      int     `json:"missed_n"`
}

// ForceScore は発話行為（YA）の一致。
type ForceScore struct {
	Compared   int                       `json:"compared"`
	Exact      int                       `json:"exact"`
	SameFamily int                       `json:"same_family"`
	Accuracy   float64                   `json:"accuracy"`
	Confusion  map[string]map[string]int `json:"confusion"`
}

// AIFEvidence は GED の数値を裏付ける具体例。レポートの定性分析はここを引く。
type AIFEvidence struct {
	// BestPairs / WorstPairs は対応付いた I-node のうち類似度が高い順・低い順。
	BestPairs  []PropositionPair `json:"best_pairs"`
	WorstPairs []PropositionPair `json:"worst_pairs"`
	// Correct は種別・向きとも当たった関係。
	Correct []RelationEvidence `json:"correct"`
	// Reversed は向きが逆だった関係。
	Reversed []RelationEvidence `json:"reversed"`
	// Mistyped は向きは合っているが種別が違った関係。
	Mistyped []RelationEvidence `json:"mistyped"`
	// Spurious はゴールドに対応が無い関係（過剰生成）。
	Spurious []RelationEvidence `json:"spurious"`
	// Missed はゴールドにあって生成できなかった関係。
	Missed []RelationEvidence `json:"missed"`
}

// PropositionPair は対応付いた I-node の組。
type PropositionPair struct {
	PredID     string  `json:"pred_id"`
	PredText   string  `json:"pred_text"`
	GoldID     string  `json:"gold_id"`
	GoldText   string  `json:"gold_text"`
	Similarity float64 `json:"similarity"`
}

// RelationEvidence は関係 1 本の具体例。
type RelationEvidence struct {
	// Side は "pred" か "gold"。
	Side     string `json:"side"`
	Type     string `json:"type"`
	GoldType string `json:"gold_type,omitempty"`
	SrcText  string `json:"src_text"`
	DstText  string `json:"dst_text"`
	// SrcMatch / DstMatch は対応付いた相手側の命題文（あれば）。
	SrcMatch string `json:"src_match,omitempty"`
	DstMatch string `json:"dst_match,omitempty"`
	// Rationale は LLM が返した根拠（pred 側のみ）。
	Rationale string `json:"rationale,omitempty"`
	// PairSimilarity は関係で結ばれた 2 つの命題どうしの埋め込み類似度
	// （同じグラフ内での cos）。「意味が近いだけで関係を張ってしまった」のか
	// を見分けるために持つ。RQ3（埋め込みの限界）の主な証拠。
	PairSimilarity float64 `json:"pair_similarity"`
	// SrcMatchSimilarity は pred の src が対応付いたゴールド命題との cos。
	// 命題の対応は取れているのに関係だけ外している、という切り分けに使う。
	SrcMatchSimilarity float64 `json:"src_match_similarity,omitempty"`
}

// AIFCompareOptions は比較の設定。
type AIFCompareOptions struct {
	// Metric は GED 実装の版（MetricGEDv1〜v4）。空なら既定（AIFComparisonVersion）。
	Metric  string
	Weights AIFWeights
	// MinSimilarity は I-node が「対応付いた」と数えるコサイン類似度の下限。
	MinSimilarity float64
	// EvidenceLimit は証拠として書き出す件数の上限（種類ごと）。
	EvidenceLimit int
	// Rationales は S-node ID → LLM の根拠。生成側の Annotation から渡す。
	Rationales map[string]string
}

// DefaultAIFCompareOptions は既定の設定。
func DefaultAIFCompareOptions() AIFCompareOptions {
	return AIFCompareOptions{
		Metric:        AIFComparisonVersion,
		Weights:       DefaultAIFWeights(),
		MinSimilarity: 0.55,
		EvidenceLimit: 8,
	}
}

// CompareAIF は LLM が作った AIF グラフ pred を、ゴールド gold と突き合わせる。
//
// 手順（Riesen & Bunke 2009 の二部グラフ近似の枠組み。ただし割当のコストは
// ノードのラベルだけで決め、Riesen & Bunke のように周りの辺の照合コストを足さない）:
//  1. I / L のテキストを埋め込み、置換コスト 1 - cos を作る
//  2. 全ノードの最適割当を Hungarian 法で 1 回だけ解く（assign）
//
// 辺を割当のコストに入れないのは意図的である。入れると命題の対応付けが関係の予測に
// 引きずられ、「関係が合うように命題を対応させる」ことになって、その対応の上で数える
// 関係の P/R が過大になる（評価したいものを割当が先取りする）。命題の対応は内容だけで決め、
// 関係はその固定した対応の上で数える。ここで得る GED は、辺まで含めた最適値の上界である。
//  3. 0.1 はその割当をそのまま、0.2 以降は L / I の割当だけを使い、
//     構造ノード（S / TA / YA）の対応は構造から導出する（derive）
//  4. その対応のもとでノードと辺の編集コストを数える（distance）
//  5. 指標の版によらない層別の診断（命題・発話・関係・発話行為）を付ける
func CompareAIF(ctx context.Context, pred, gold aif.Graph, emb llm.Embedder, opts AIFCompareOptions) (*AIFComparison, error) {
	if emb == nil {
		return nil, fmt.Errorf("evaluation: embedder is required to compare AIF graphs")
	}
	if opts.EvidenceLimit <= 0 {
		opts.EvidenceLimit = 8
	}
	if opts.Weights.Forbidden == 0 {
		opts.Weights = DefaultAIFWeights()
	}
	if opts.Metric == "" {
		opts.Metric = AIFComparisonVersion
	}
	if !slices.Contains(AIFMetrics, opts.Metric) {
		return nil, fmt.Errorf("evaluation: 未知の GED 版 %q", opts.Metric)
	}

	// ---- 埋め込み（I-node と L-node のテキストだけ。S/YA/TA は文を持たない） ----
	sim, err := textSimilarity(ctx, pred, gold, emb)
	if err != nil {
		return nil, err
	}

	// ---- ノードの最適割当（1 回だけ） ----
	//
	// 種別違いの置換は禁止コストにしてあるので、コスト行列は種別ごとの
	// ブロック対角になる。つまり全ノードで解いても、L だけ・I だけで解いても
	// L↔L / I↔I の割当は同じになる。だから解くのは 1 回で足りる。
	rawMatch, err := assign(pred, gold, sim.cross, opts.Weights)
	if err != nil {
		return nil, err
	}
	contentMatch := restrictToTypes(rawMatch, pred, gold, aif.TypeL, aif.TypeI)
	propMatch := restrictToTypes(rawMatch, pred, gold, aif.TypeI)

	// ---- 全体（厳密 AIF）の対応 ----
	var fullCorr correspondence
	if opts.Metric == MetricGEDv1 {
		fullCorr = correspondence{pairs: rawMatch}
	} else {
		fullCorr = derive(pred, gold, contentMatch)
	}

	out := &AIFComparison{
		NodesetID:  gold.ID,
		Metric:     opts.Metric,
		Weights:    opts.Weights.Name,
		CountsPred: pred.CountByType(),
		CountsGold: gold.CountByType(),
	}

	// 辺の部分点（種別違い / 向き違いの置換）は 0.2 以降で効かせる。
	labelled := opts.Metric != MetricGEDv1
	// 全体グラフの取り方は版で違う。0.3 は関係を辺 1 本に畳み、
	// 遷移に anchor される YA を外す（aif.Graph.DialogueEdges の説明を参照）。
	// 0.4 はさらに TA を外す（aif.Graph.AnchorEdges の説明を参照）。
	predFull, goldFull := pred.Nodes, gold.Nodes
	predFullEdges, goldFullEdges := pred.LabeledEdges(), gold.LabeledEdges()
	labelledFull := false
	switch opts.Metric {
	case MetricGEDv3:
		predFull, goldFull = pred.DialogueNodes(), gold.DialogueNodes()
		predFullEdges, goldFullEdges = pred.DialogueEdges(), gold.DialogueEdges()
		labelledFull = true
	case MetricGEDv4:
		predFull, goldFull = pred.AnchorNodes(), gold.AnchorNodes()
		predFullEdges, goldFullEdges = pred.AnchorEdges(), gold.AnchorEdges()
		labelledFull = true
	}
	out.FullGED = distance(predFull, goldFull, predFullEdges, goldFullEdges,
		fullCorr, sim.cross, opts.Weights, labelledFull)
	predI, goldI := pred.NodesOfType(aif.TypeI), gold.NodesOfType(aif.TypeI)
	out.ReducedGED = distance(predI, goldI, pred.RelationEdges(), gold.RelationEdges(),
		correspondence{pairs: propMatch}, sim.cross, opts.Weights, labelled)
	out.FullSimilarity = out.FullGED.Similarity
	out.ReducedSimilarity = out.ReducedGED.Similarity
	out.FullSimilarityGold = out.FullGED.SimilarityGold
	out.ReducedSimilarityGold = out.ReducedGED.SimilarityGold

	// ---- 診断（指標の版によらず同じ。命題の対応だけを使う） ----
	if out.Propositions, err = alignmentOf(propMatch, sim.cross, nodeIDs(predI), nodeIDs(goldI), opts.MinSimilarity); err != nil {
		return nil, err
	}
	locMatch := restrictToTypes(rawMatch, pred, gold, aif.TypeL)
	predL, goldL := pred.NodesOfType(aif.TypeL), gold.NodesOfType(aif.TypeL)
	if out.Locutions, err = alignmentOf(locMatch, sim.cross, nodeIDs(predL), nodeIDs(goldL), opts.MinSimilarity); err != nil {
		return nil, err
	}
	out.Relations, out.RelationPairSimilarity, out.Evidence = relationScore(pred, gold, propMatch, sim, opts)
	out.Forces = forceScore(pred, gold, propMatch)
	out.Evidence.BestPairs, out.Evidence.WorstPairs = pairEvidence(pred, gold, propMatch, sim.cross, opts.EvidenceLimit)
	return out, nil
}

func nodeIDs(nodes []aif.Node) []string {
	out := make([]string, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	return out
}

// assign は生成側の全ノードをゴールド側の全ノードに最適割当し、生成側 ID → ゴールド側 ID を返す。
func assign(pred, gold aif.Graph, sim simTable, w AIFWeights) (map[string]string, error) {
	del := make([]float64, len(pred.Nodes))
	cost := make([][]float64, len(pred.Nodes))
	for i, a := range pred.Nodes {
		del[i] = w.deleteCost(a.Type)
		cost[i] = make([]float64, len(gold.Nodes))
		for j, b := range gold.Nodes {
			cost[i][j] = substitutionCost(a, b, sim, w)
		}
	}
	ins := make([]float64, len(gold.Nodes))
	for j, b := range gold.Nodes {
		ins[j] = w.deleteCost(b.Type)
	}
	m, err := bipartiteMatch(cost, del, ins)
	if err != nil {
		return nil, fmt.Errorf("evaluation: AIF ノードの対応付け: %w", err)
	}
	out := make(map[string]string, len(m))
	for i, j := range m {
		out[pred.Nodes[i].ID] = gold.Nodes[j].ID
	}
	return out, nil
}

// correspondence は生成側ノード ID → ゴールド側ノード ID の対応。
type correspondence struct {
	pairs map[string]string
	// derived は構造から導出した対応の数（探索ではなく決定で決まったもの）。
	derived int
}

// restrictToTypes は割当のうち、指定した種別のノードだけを残す。
func restrictToTypes(m map[string]string, pred, gold aif.Graph, types ...aif.NodeType) map[string]string {
	keep := map[aif.NodeType]bool{}
	for _, t := range types {
		keep[t] = true
	}
	pi, gi := aif.NewIndex(pred), aif.NewIndex(gold)
	out := make(map[string]string, len(m))
	for a, b := range m {
		if keep[pi.Type(a)] && keep[gi.Type(b)] {
			out[a] = b
		}
	}
	return out
}

// derive は L と I の対応から、構造ノード（S-node / TA / YA）の対応を決める。
//
// AIF では S-node は前提と結論の I-node で、TA は前後の L-node で、
// YA は anchor と illocute 先で、それぞれ一意に決まる。探索の対象ではないので、
// Hungarian 法には載せずにここで決める（載せると内容を持たない TA どうしが
// 全部等価になり、最適解が縮退して任意の割当が返ってしまう）。
func derive(pred, gold aif.Graph, content map[string]string) correspondence {
	out := correspondence{pairs: make(map[string]string, len(content)*2)}
	for a, b := range content {
		out.pairs[a] = b
	}

	// S-node: (前提の集合, 結論) が一致すれば対応。向きだけ逆のものも
	// 対応とみなす（種別違い・向き違いは置換コストで払わせる）。
	//
	// 対応は安い順に 4 パスで決める（同じ向き・同じ種別 → 同じ向き・別種別 →
	// 逆向き・同じ種別 → 逆向き・別種別）。以前は生成側の出力順に貪欲に決めていたため、
	// 同じ命題対に CA と RA を出し、ゴールドが RA のとき、先に出た CA がゴールドの RA を取って
	// 0.2 の全体 GED が順序次第で 3.00 / 7.75 に割れていた。
	// また鍵が同じゴールドの S-node が複数あると map の上書きで片方が対応しなかったので、
	// 鍵ごとに候補を並べて持つ。
	key := func(srcs []string, dst string, m map[string]string) (string, bool) {
		mapped := make([]string, 0, len(srcs))
		for _, s := range srcs {
			v, ok := m[s]
			if !ok {
				return "", false
			}
			mapped = append(mapped, v)
		}
		d, ok := m[dst]
		if !ok {
			return "", false
		}
		sort.Strings(mapped)
		return strings.Join(mapped, "\x1f") + "\x1e" + d, true
	}
	type goldS struct {
		id  string
		typ aif.NodeType
	}
	goldKey := map[string][]goldS{}
	goldRev := map[string][]goldS{}
	for _, r := range gold.Relations() {
		self := map[string]string{r.DstID: r.DstID}
		for _, s := range r.SrcIDs {
			self[s] = s
		}
		if k, ok := key(r.SrcIDs, r.DstID, self); ok {
			goldKey[k] = append(goldKey[k], goldS{r.SNodeID, r.Type})
		}
		// 向きが逆の探索用（前提 1 本のものだけ意味がある）
		if len(r.SrcIDs) == 1 {
			if k, ok := key([]string{r.DstID}, r.SrcIDs[0], self); ok {
				goldRev[k] = append(goldRev[k], goldS{r.SNodeID, r.Type})
			}
		}
	}
	type predS struct {
		id  string
		typ aif.NodeType
		key string
	}
	var predRels []predS
	for _, r := range pred.Relations() {
		if k, ok := key(r.SrcIDs, r.DstID, content); ok {
			predRels = append(predRels, predS{r.SNodeID, r.Type, k})
		}
	}
	usedGoldS := map[string]bool{}
	for _, pass := range []struct {
		index    map[string][]goldS
		sameType bool
	}{{goldKey, true}, {goldKey, false}, {goldRev, true}, {goldRev, false}} {
		for _, p := range predRels {
			if _, done := out.pairs[p.id]; done {
				continue
			}
			for _, g := range pass.index[p.key] {
				if usedGoldS[g.id] || (pass.sameType && g.typ != p.typ) {
					continue
				}
				usedGoldS[g.id] = true
				out.pairs[p.id] = g.id
				out.derived++
				break
			}
		}
	}

	// TA: 前後の L-node が対応していれば対応。端点の組が同じ TA が複数あっても取りこぼさない。
	goldTA := map[[2]string][]string{}
	for _, t := range gold.Transitions() {
		k := [2]string{t.FromID, t.ToID}
		goldTA[k] = append(goldTA[k], t.TAID)
	}
	usedTA := map[string]bool{}
	for _, t := range pred.Transitions() {
		f, ok1 := content[t.FromID]
		d, ok2 := content[t.ToID]
		if !ok1 || !ok2 {
			continue
		}
		for _, g := range goldTA[[2]string{f, d}] {
			if usedTA[g] {
				continue
			}
			usedTA[g] = true
			out.pairs[t.TAID] = g
			out.derived++
			break
		}
	}

	// YA: (anchor, illocute 先) がともに対応していれば対応。
	goldYA := map[[2]string][]string{}
	for _, il := range gold.Illocutions() {
		k := [2]string{il.AnchorID, il.TargetID}
		goldYA[k] = append(goldYA[k], il.YAID)
	}
	usedYA := map[string]bool{}
	for _, il := range pred.Illocutions() {
		a, ok1 := out.pairs[il.AnchorID]
		t, ok2 := out.pairs[il.TargetID]
		if !ok1 || !ok2 {
			continue
		}
		for _, g := range goldYA[[2]string{a, t}] {
			if usedYA[g] {
				continue
			}
			usedYA[g] = true
			out.pairs[il.YAID] = g
			out.derived++
			break
		}
	}
	return out
}

// distance は 1 粒度分の編集コストを集計する。
//
// labelled が true なら、同じ命題対を結ぶ正解の辺があるときに
// 「削除 + 挿入」ではなく置換として扱う（種別違い / 向き違いの部分点）。
func distance(predNodes, goldNodes []aif.Node, predEdges, goldEdges []aif.LabeledEdge,
	corr correspondence, sim simTable, w AIFWeights, labelled bool,
) AIFDistance {
	out := AIFDistance{Derived: corr.derived}

	// ---- ノード ----
	goldByID := make(map[string]aif.Node, len(goldNodes))
	for _, n := range goldNodes {
		goldByID[n.ID] = n
	}
	usedGold := map[string]bool{}
	for _, a := range predNodes {
		b, ok := corr.pairs[a.ID]
		if !ok {
			out.NodeCost += w.deleteCost(a.Type)
			continue
		}
		gn, ok := goldByID[b]
		if !ok {
			out.NodeCost += w.deleteCost(a.Type)
			continue
		}
		usedGold[b] = true
		out.Matched++
		out.NodeCost += substitutionCost(a, gn, sim, w)
	}
	for _, b := range goldNodes {
		if !usedGold[b.ID] {
			out.NodeCost += w.deleteCost(b.Type)
		}
	}

	// ---- 辺 ----
	//
	// 先に処理した辺が別種別の正解を取ってしまうと、後の辺が完全一致できなくなる。
	// 処理順で結果が変わらないよう、「完全一致」「種別違い」「向き違い」の順に、
	// パスごとに全件を照合する（安いコストの対応から先に確定させる）。
	type ekey struct{ from, to, kind string }
	remaining := map[ekey]int{}          // まだ対応していない正解の辺（種別ごと）
	remainingPair := map[[2]string]int{} // 同じく、端点対ごと（種別を問わない）
	for _, e := range goldEdges {
		remaining[ekey{e.FromID, e.ToID, e.Kind}]++
		remainingPair[[2]string{e.FromID, e.ToID}]++
	}
	type mappedEdge struct {
		from, to, kind string
		done           bool
	}
	var mapped []*mappedEdge
	for _, e := range predEdges {
		f, ok1 := corr.pairs[e.FromID]
		t, ok2 := corr.pairs[e.ToID]
		if !ok1 || !ok2 {
			out.Edges.Inserted++
			out.EdgeCost += w.EdgeCost
			continue
		}
		mapped = append(mapped, &mappedEdge{from: f, to: t, kind: e.Kind})
	}
	for _, m := range mapped {
		if k := (ekey{m.from, m.to, m.kind}); remaining[k] > 0 {
			remaining[k]--
			remainingPair[[2]string{m.from, m.to}]--
			m.done = true
			out.Edges.Exact++
		}
	}
	if labelled {
		// 完全一致の後は、どの種別の正解を取ってもコストは同じなので端点対の本数だけ見る。
		for _, sub := range []struct {
			reversed bool
			count    *int
			cost     float64
		}{
			{false, &out.Edges.TypeSubstituted, w.EdgeTypeMismatch},
			{true, &out.Edges.DirectionSubstituted, w.EdgeDirectionMismatch},
		} {
			for _, m := range mapped {
				pair := [2]string{m.from, m.to}
				if sub.reversed {
					pair = [2]string{m.to, m.from}
				}
				if m.done || remainingPair[pair] == 0 {
					continue
				}
				remainingPair[pair]--
				m.done = true
				*sub.count++
				out.EdgeCost += sub.cost
			}
		}
	}
	for _, m := range mapped {
		if !m.done {
			out.Edges.Inserted++
			out.EdgeCost += w.EdgeCost
		}
	}
	for _, n := range remainingPair {
		out.Edges.Deleted += n
	}
	out.EdgeCost += float64(out.Edges.Deleted) * w.EdgeCost

	// ---- 正規化 ----
	//
	// 2 通り出す。対称分母（0.2 までの主指標）は pred の大きさに依存するので、
	// 「関係を多く出すほど得をする」経路がある。gold 基準はそれが無く、
	// 条件をまたいで同じ分母になるので条件比較に使える。
	for _, n := range predNodes {
		out.Denominator += w.deleteCost(n.Type)
	}
	for _, n := range goldNodes {
		out.GoldCost += w.deleteCost(n.Type)
	}
	out.GoldCost += float64(len(goldEdges)) * w.EdgeCost
	out.Denominator += out.GoldCost + float64(len(predEdges))*w.EdgeCost
	out.GED = out.NodeCost + out.EdgeCost
	if out.Denominator > 0 {
		out.NormalizedGED = out.GED / out.Denominator
	}
	out.Similarity = 1 - out.NormalizedGED
	if out.GoldCost > 0 {
		out.NormalizedGEDGold = out.GED / out.GoldCost
	}
	out.SimilarityGold = 1 - out.NormalizedGEDGold
	return out
}

// substitutionCost はノード a を b に置き換えるコスト。
func substitutionCost(a, b aif.Node, sim simTable, w AIFWeights) float64 {
	switch {
	case a.Type == b.Type && (a.Type == aif.TypeI || a.Type == aif.TypeL):
		// 意味的な近さ。負のコサインは 0 に丸め（「無関係」より悪くはしない）、
		// 浮動小数の誤差で cos が 1 をわずかに超えても負のコストにしない。
		return math.Max(0, 1-math.Max(0, sim.get(a.ID, b.ID)))
	case a.Type == b.Type && a.Type == aif.TypeTA:
		return 0
	case a.Type == b.Type && a.Type == aif.TypeYA:
		switch {
		case a.Scheme == b.Scheme:
			return 0
		case forceFamily(a.Scheme) == forceFamily(b.Scheme):
			return w.ForceFamilyMatch
		default:
			return w.ForceMismatch
		}
	case a.Type == b.Type:
		return 0 // 同種の S-node
	case a.Type.IsSNode() && b.Type.IsSNode():
		return w.SNodeTypeMismatch
	default:
		return w.Forbidden
	}
}

// Add は 2 つの集計を足し合わせる（ノードセットをまたいだ集計用）。
func (b DistanceBucket) Add(o DistanceBucket) DistanceBucket {
	return DistanceBucket{
		GoldTotal:        b.GoldTotal + o.GoldTotal,
		PredTotal:        b.PredTotal + o.PredTotal,
		TP:               b.TP + o.TP,
		TPPred:           b.TPPred + o.TPPred,
		TPUndirected:     b.TPUndirected + o.TPUndirected,
		TPUndirectedPred: b.TPUndirectedPred + o.TPUndirectedPred,
	}
}
