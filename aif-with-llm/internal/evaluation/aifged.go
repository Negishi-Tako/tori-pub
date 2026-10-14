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

// Implementation versions of GED between AIF graphs. When the procedure changes, add a new version.
//
// 0.1 → 0.2 fixed the following 2 points. Both were defects in "how we measure",
// not in "what we want to measure" (see §11 of the report).
//
//	(1) Degenerate matching
//	    0.1 put every node into the Hungarian method. But TAs have substitution cost 0
//	    with each other, as do YAs with the same scheme, so all of them were equivalent,
//	    the optimum was massively degenerate, and an arbitrary assignment came back. Even when
//	    the L-nodes at both ends were matched correctly, the TA between them could be assigned to
//	    something else, and L→TA→L edges were counted as errors
//	    (even the baseline, which uses the gold utterances as-is, had edge precision 0.386).
//	    0.2 matches only content-bearing nodes (L and I) and derives TA / YA / S-node
//	    correspondences from that. In AIF these are structural nodes, uniquely determined once
//	    the L and I correspondence is fixed, so they are not something to search independently.
//
//	(2) Edges were "exact match or total loss"
//	    0.1 charged insertion 1.0 + deletion 1.0 = 2.0 whenever an edge was off at all.
//	    So the expected cost of "adding one edge" was P(1-2p) (p is precision),
//	    a guaranteed loss unless precision exceeded 0.5. That 0.5 break-even point was
//	    merely a by-product of setting "insertion cost = deletion cost = 1".
//	    0.2 treats an edge as a substitution if a gold edge exists on the same endpoint pair,
//	    charging EdgeTypeMismatch for a type difference and EdgeDirectionMismatch for a direction difference.
//	    Only pure over-generation with no gold counterpart is a total loss.
//
// 0.2 → 0.3 fixed the following 3 points. All were remaining "how we measure" defects in 0.2,
// and the first was large enough to reverse the ranking of conditions (report §12.1).
//
//	(3) The normalization denominator depended on pred's size
//	    0.2's denominator is the cost of "deleting all of pred and building all of gold",
//	    so emitting more relations inflated the denominator and diluted the cost.
//	    Measured: fixing the denominator to the gold side alone made both
//	    "the LLM beats the baseline" and "proposition GED is monotonic v1<v2<v3"
//	    disappear (the sign was determined by how the denominator was defined).
//	    0.3 uses gold-based normalization (SimilarityGold) as the primary metric.
//	    Dividing by the reference size is the same idea as WER / TER, and
//	    leaves no path where "emitting more output pays off". It does not stay within 0–1:
//	    graphs worse than the baseline get negative values, and it is better to be able to express that.
//
//	(4) The overall graph did not give partial credit for edges
//	    0.2 passed labelled only to the proposition graph, so the same single error
//	    cost 0.75 / 1.0 at the proposition level but 4.75 / 4.00 overall
//	    (one relation expands into an S-node and 2 edges, and the edges on both sides miss).
//	    0.3 folds relations into one edge in the overall graph too (aif.Graph.DialogueEdges),
//	    so partial credit applies. A relation error costs the same regardless of granularity.
//
//	(5) YAs anchored to transitions did not reflect LLM quality
//	    On the pred side their count is fixed by the harness construction rule "relations between
//	    non-adjacent utterances have no TA, so no YA is created" (gold 508 vs pred 422).
//	    0.3 removes them from the overall graph and measures illocutions with ForceScore.
//
// 0.3 → 0.4 fixed the following 1 point. The proposition graph (ReducedGED) is the same as 0.3; only the overall graph changes.
//
//	(6) TAs remained in the overall graph
//	    The same reason as (5) (the pred-side count is fixed by the harness construction rule) applies to TAs.
//	    Pred TAs are placed mechanically between adjacent utterances, while gold TAs branch and
//	    also link non-adjacent utterances. Even for the baseline given the gold segmentation,
//	    roughly 1/4 of TAs went unmatched, and about 18% of the overall GED denominator was cost unrelated to LLM quality.
//	    0.4 restricts the overall graph to L / I / utterance YAs plus folded relation edges (aif.Graph.AnchorEdges).
//
// From 0.2 on, edge matching and structural-node derivation do not depend on output order (matching is
// done per pass in the order exact → type mismatch → direction mismatch). Edge matching was fixed in the
// code review at publication, and S-node derivation (derive) in a later audit (report §15.2(f)).
// Both change values only when multiple relations of different types are emitted on the same proposition pair.
const (
	MetricGEDv1 = "aif-ged/0.1"
	MetricGEDv2 = "aif-ged/0.2"
	MetricGEDv3 = "aif-ged/0.3"
	MetricGEDv4 = "aif-ged/0.4"
)

// AIFComparisonVersion is the default implementation version.
const AIFComparisonVersion = MetricGEDv4

// AIFMetrics are the GED versions computed by default. Old versions are kept because
// showing that "the metric design changed the conclusion" is part of this experiment's results.
var AIFMetrics = []string{MetricGEDv1, MetricGEDv2, MetricGEDv3, MetricGEDv4}

// AIFWeights are the weights of the edit distance (GED) between AIF graphs.
//
// How the weights are chosen (the rationale stated in the report):
//   - Deletion/insertion costs are set per node type. The default is a uniform weight of 1.0 for all.
//     Uniform is the "no special assumptions" reference, so it is the primary metric first.
//   - Substitution is allowed only within the same type. For I-nodes / L-nodes the cost is 1 - cos
//     of the embedding cosine similarity (matching the design used in Bergmann et al.'s argument graph
//     retrieval: "S-nodes by type match, I-nodes by semantic similarity").
//   - Confusing RA/CA/MA is a state of "there is a relation but its direction or type is wrong", so it gets
//     an intermediate value: cheaper than deletion + insertion (= 2.0) but more than an exact match.
//   - For YA illocutions, confusions within the same family (assertive / questioning / agreeing) are
//     lighter than confusions across families. Weighing Asserting vs Arguing the same as
//     Asserting vs PureQuestioning would be crude.
//   - TAs have no content, so substitution between TAs costs 0.
//
// As a sensitivity analysis, weights with reduced YA/TA deletion cost (DiscountedAIFWeights) are
// computed too. YA/TA are determined almost mechanically by the structure of L-nodes and I-nodes,
// which answers the criticism that uniform weights count them twice.
type AIFWeights struct {
	// Name identifies the weight setting (shown in report table rows).
	Name string `json:"name"`
	// NodeDelete is the deletion/insertion cost per node type. Unset types cost 1.0.
	NodeDelete map[aif.NodeType]float64 `json:"node_delete"`
	// EdgeCost is the insertion/deletion cost per edge.
	EdgeCost float64 `json:"edge_cost"`
	// SNodeTypeMismatch is the substitution cost for confusing RA/CA/MA.
	SNodeTypeMismatch float64 `json:"snode_type_mismatch"`
	// ForceFamilyMatch is the substitution cost when illocutions have different labels but the same family.
	ForceFamilyMatch float64 `json:"force_family_match"`
	// ForceMismatch is the substitution cost when illocutions are from different families.
	ForceMismatch float64 `json:"force_mismatch"`
	// Forbidden is the prohibitive cost given to substitutions across types.
	Forbidden float64 `json:"forbidden"`
	// EdgeTypeMismatch is the substitution cost when a gold edge connects the same proposition pair
	// but with a different type (for proposition-level graphs). Treated the same as confusing S-nodes.
	EdgeTypeMismatch float64 `json:"edge_type_mismatch"`
	// EdgeDirectionMismatch is the substitution cost when a gold edge exists with only the direction reversed.
	// It is cheaper than insertion + deletion (2.0), since "the existence of the relation is right".
	//
	// The default 1.0 equals the deletion cost of missing a gold edge (EdgeCost = 1.0).
	// That is, emitting a reversed relation costs the same as emitting nothing:
	// the gain over "emit nothing" is 0, and over "emit a relation with no gold counterpart" (insertion 1.0 + deletion 1.0) it is 1.0.
	// Below 1.0, emitting a wrongly directed relation would always beat emitting nothing,
	// rewarding a strategy of emitting one of the two directions without judging direction, so 1.0 is the floor.
	// To count direction errors as "got the existence of the relation right", use RelationScore.Undirected.
	EdgeDirectionMismatch float64 `json:"edge_direction_mismatch"`
}

// DefaultAIFWeights are the uniform weights (primary metric).
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

// DiscountedAIFWeights are sensitivity-analysis weights with YA/TA deletion cost halved.
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

// forceFamily is the family of an illocution. Used to weigh cross-family confusions more heavily.
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

// AIFComparison is the comparison result for one nodeset.
type AIFComparison struct {
	NodesetID string `json:"nodeset_id"`
	// Condition is the generation condition ("llm-e2e" / "llm-gold-loc" / "baseline", etc.).
	Condition string `json:"condition"`
	Weights   string `json:"weights"`

	CountsPred map[aif.NodeType]int `json:"counts_pred"`
	CountsGold map[aif.NodeType]int `json:"counts_gold"`

	// Metric is the GED implementation version used (MetricGEDv1–MetricGEDv4).
	Metric string `json:"metric"`

	// FullGED is the GED of the whole AIF graph including the dialogue layer (what it includes depends on the version; see CompareAIF).
	FullGED AIFDistance `json:"full_ged"`
	// ReducedGED is the GED folded down to I-nodes and RA/CA/MA only (the report's "proposition GED").
	ReducedGED AIFDistance `json:"reduced_ged"`
	// FullSimilarity / ReducedSimilarity are 1 - normalized GED (symmetric denominator).
	// The primary metric up to 0.2, but the denominator depends on pred's size, so it is not used to compare conditions.
	FullSimilarity    float64 `json:"full_similarity"`
	ReducedSimilarity float64 `json:"reduced_similarity"`
	// FullSimilarityGold / ReducedSimilarityGold use gold-based normalization (the primary metric in 0.3).
	FullSimilarityGold    float64 `json:"full_similarity_gold"`
	ReducedSimilarityGold float64 `json:"reduced_similarity_gold"`

	// Propositions is the diagnostic of I-node matching.
	Propositions PropositionAlignment `json:"propositions"`
	// Locutions is the diagnostic of L-node matching (stage 1 = segmentation quality).
	//
	// `gold-loc` / `baseline` use the gold L-nodes as-is, so this is close to 1.0;
	// it is meaningful only for `e2e` (the condition that leaves segmentation to the LLM).
	// Up to 0.2, whether "segmentation is the upstream bottleneck" could only be estimated
	// indirectly from the difference between `e2e` and `gold-loc`.
	// This makes it directly readable in the same currency as the proposition layer (matching cos and count ratio).
	Locutions PropositionAlignment `json:"locutions"`
	// Relations is the RA/CA/MA agreement (the most interpretable part of the GED breakdown).
	Relations RelationScore `json:"relations"`
	// RelationPairSimilarity is the mean cos between propositions connected by relations,
	// aggregated separately for hit relations / over-generation / misses.
	RelationPairSimilarity PairSimilarityStats `json:"relation_pair_similarity"`
	// Forces is the illocution agreement on matched I-nodes.
	Forces ForceScore `json:"forces"`

	Evidence AIFEvidence `json:"evidence"`
}

// AIFDistance is the edit distance at one granularity. It always carries a breakdown,
// because a metric that cannot tell "whether the value moved due to nodes or edges"
// cannot be used to judge whether things improved.
type AIFDistance struct {
	GED float64 `json:"ged"`
	// NormalizedGED is divided by "the cost of deleting all of A and rebuilding all of B" (0–1).
	NormalizedGED float64 `json:"normalized_ged"`
	// Similarity is 1 - NormalizedGED.
	Similarity float64 `json:"similarity"`
	NodeCost   float64 `json:"node_cost"`
	EdgeCost   float64 `json:"edge_cost"`
	// Denominator is the denominator of symmetric normalization (the maximum possible cost). It depends on pred's size.
	Denominator float64 `json:"denominator"`
	// GoldCost is "the cost of building gold from nothing". It does not depend on pred,
	// so it is a common denominator across conditions (the primary metric in 0.3).
	GoldCost float64 `json:"gold_cost"`
	// NormalizedGEDGold is GED / GoldCost, and SimilarityGold is 1 minus that.
	// If pred is far off from gold, it can exceed 1 and the similarity can be negative.
	NormalizedGEDGold float64 `json:"normalized_ged_gold"`
	SimilarityGold    float64 `json:"similarity_gold"`
	// Matched is the number of matched node pairs; Derived is how many of those were derived from structure.
	Matched int `json:"matched"`
	Derived int `json:"derived"`
	// Edges is the edge breakdown.
	Edges EdgeBreakdown `json:"edges"`
}

// EdgeBreakdown is the breakdown of edge edits.
type EdgeBreakdown struct {
	// Exact counts edges matching in both type and direction (cost 0).
	Exact int `json:"exact"`
	// TypeSubstituted counts edges connecting the same proposition pair but with a different type.
	TypeSubstituted int `json:"type_substituted"`
	// DirectionSubstituted counts edges with only the direction reversed.
	DirectionSubstituted int `json:"direction_substituted"`
	// Inserted counts edges with no gold counterpart (pure over-generation); Deleted counts missed gold edges.
	Inserted int `json:"inserted"`
	Deleted  int `json:"deleted"`
}

// PropositionAlignment is the result of I-node matching.
type PropositionAlignment struct {
	// Threshold is the minimum cosine similarity counted as "matched".
	Threshold float64 `json:"threshold"`
	// Aligned is the number of pairs matched by the Hungarian method (ignoring the threshold).
	Aligned int `json:"aligned"`
	// AboveThreshold is the number of pairs at or above the threshold in the Hungarian assignment (shared with relation matching and GED).
	// It is at most PRF.TP (the maximum number of one-to-one pairs formable from pairs at or above the threshold).
	AboveThreshold int     `json:"above_threshold"`
	MeanSimilarity float64 `json:"mean_similarity"`
	PRF            PRF     `json:"prf"`
	// CountRatio is the number of generated I-nodes / the number of gold I-nodes.
	CountRatio float64 `json:"count_ratio"`
	// F1ByThreshold is the proposition F1 as the threshold varies.
	//
	// The default 0.55 is only just above the mean cos of proposition pairs that *actually hold a relation*
	// within the same nodeset (measured 0.44–0.48). A single value cannot show that
	// "the proposition layer is at a practical level", so the curve is reported alongside.
	F1ByThreshold map[string]float64 `json:"f1_by_threshold,omitempty"`
}

// PropositionThresholds are the thresholds at which F1ByThreshold is reported.
var PropositionThresholds = []float64{0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9}

// RelationScore is the relation agreement.
//
// The key point is counting Reversed (only the direction reversed) separately. In both GED and P/R,
// a direction error is penalized twice as "one relation missing and one extra",
// hiding the fact that "the existence of the relation itself was right".
type RelationScore struct {
	// Typed counts as TP only when the type (RA/CA/MA) also matches.
	Typed PRF `json:"typed"`
	// Untyped ignores the type and counts as TP if only the direction matches.
	Untyped PRF `json:"untyped"`
	// Undirected also ignores the direction.
	Undirected PRF `json:"undirected"`
	// Reversed is the number of relations with only the direction reversed.
	Reversed int `json:"reversed"`
	// TypeConfusion is the confusion matrix of generated type → gold type
	// (only relations matched with the same direction).
	TypeConfusion map[string]map[string]int `json:"type_confusion"`
	// PredTotal / GoldTotal are the total numbers of relations.
	PredTotal int `json:"pred_total"`
	GoldTotal int `json:"gold_total"`
	// ByDistance is the agreement per inter-utterance distance band.
	//
	// About 70% of gold relations are between adjacent utterances, so overall F1 is dominated by the adjacent peak
	// and the failure "distant relations are not captured" is invisible. Counts are split by band.
	ByDistance map[string]DistanceBucket `json:"by_distance"`
}

// DistanceBucket aggregates one distance band.
//
// The key point is counting hits in two ways. The generated-side distance and the gold-side distance
// can disagree due to misaligned proposition matching, and counting only one way
// misaligns the bands of numerator and denominator. In fact, TP was initially counted by gold band
// and the denominator by generated band, so per-band precision exceeded 1 for the condition that mechanically
// links adjacent propositions (all generated relations had distance 1, but the gold distance of the hits was 2-4).
//
//	per-band precision = TPPred / PredTotal       (both generated-side bands)
//	per-band recall    = TPGold / GoldTotal       (both gold-side bands)
type DistanceBucket struct {
	GoldTotal int `json:"gold_total"`
	PredTotal int `json:"pred_total"`
	// TP counts relations hit in both type and direction by the **gold-side** distance band
	// (the recall numerator). The name is kept for backward compatibility.
	TP int `json:"tp"`
	// TPPred counts the same hits by the **generated-side** distance band (the precision numerator).
	TPPred int `json:"tp_pred"`
	// TPUndirected / TPUndirectedPred are hits ignoring type and direction.
	TPUndirected     int `json:"tp_undirected"`
	TPUndirectedPred int `json:"tp_undirected_pred"`
}

// PairSimilarityStats is the mean cos between propositions connected by relations.
//
// An aggregate answering "do over-generated relations just connect propositions that are merely on similar topics?"
// If the mean cos does not differ between hit relations and over-generation, embedding closeness cannot
// explain whether a relation exists, i.e. it shows the limits of embedding-based matching.
type PairSimilarityStats struct {
	CorrectMean  float64 `json:"correct_mean"`
	CorrectN     int     `json:"correct_n"`
	SpuriousMean float64 `json:"spurious_mean"`
	SpuriousN    int     `json:"spurious_n"`
	MissedMean   float64 `json:"missed_mean"`
	MissedN      int     `json:"missed_n"`
}

// ForceScore is the illocution (YA) agreement.
type ForceScore struct {
	Compared   int                       `json:"compared"`
	Exact      int                       `json:"exact"`
	SameFamily int                       `json:"same_family"`
	Accuracy   float64                   `json:"accuracy"`
	Confusion  map[string]map[string]int `json:"confusion"`
}

// AIFEvidence is concrete examples backing the GED numbers. The report's qualitative analysis draws from here.
type AIFEvidence struct {
	// BestPairs / WorstPairs are matched I-nodes in descending / ascending order of similarity.
	BestPairs  []PropositionPair `json:"best_pairs"`
	WorstPairs []PropositionPair `json:"worst_pairs"`
	// Correct are relations hit in both type and direction.
	Correct []RelationEvidence `json:"correct"`
	// Reversed are relations with the direction reversed.
	Reversed []RelationEvidence `json:"reversed"`
	// Mistyped are relations with the right direction but the wrong type.
	Mistyped []RelationEvidence `json:"mistyped"`
	// Spurious are relations with no gold counterpart (over-generation).
	Spurious []RelationEvidence `json:"spurious"`
	// Missed are gold relations that were not generated.
	Missed []RelationEvidence `json:"missed"`
}

// PropositionPair is a pair of matched I-nodes.
type PropositionPair struct {
	PredID     string  `json:"pred_id"`
	PredText   string  `json:"pred_text"`
	GoldID     string  `json:"gold_id"`
	GoldText   string  `json:"gold_text"`
	Similarity float64 `json:"similarity"`
}

// RelationEvidence is a concrete example of one relation.
type RelationEvidence struct {
	// Side is "pred" or "gold".
	Side     string `json:"side"`
	Type     string `json:"type"`
	GoldType string `json:"gold_type,omitempty"`
	SrcText  string `json:"src_text"`
	DstText  string `json:"dst_text"`
	// SrcMatch / DstMatch are the matched proposition texts on the other side (if any).
	SrcMatch string `json:"src_match,omitempty"`
	DstMatch string `json:"dst_match,omitempty"`
	// Rationale is the rationale returned by the LLM (pred side only).
	Rationale string `json:"rationale,omitempty"`
	// PairSimilarity is the embedding similarity between the two propositions connected by the relation
	// (cos within the same graph). Kept to tell whether a relation was drawn "just because the meanings are close".
	// The main evidence for RQ3 (limits of embeddings).
	PairSimilarity float64 `json:"pair_similarity"`
	// SrcMatchSimilarity is the cos between pred's src and the gold proposition it was matched to.
	// Used to isolate cases where the propositions are matched but only the relation is missed.
	SrcMatchSimilarity float64 `json:"src_match_similarity,omitempty"`
}

// AIFCompareOptions configures comparison.
type AIFCompareOptions struct {
	// Metric is the GED implementation version (MetricGEDv1–v4). Empty means the default (AIFComparisonVersion).
	Metric  string
	Weights AIFWeights
	// MinSimilarity is the minimum cosine similarity for an I-node to count as "matched".
	MinSimilarity float64
	// EvidenceLimit caps the number of evidence items written (per kind).
	EvidenceLimit int
	// Rationales maps S-node ID → LLM rationale. Passed from the generated-side Annotation.
	Rationales map[string]string
}

// DefaultAIFCompareOptions returns the default settings.
func DefaultAIFCompareOptions() AIFCompareOptions {
	return AIFCompareOptions{
		Metric:        AIFComparisonVersion,
		Weights:       DefaultAIFWeights(),
		MinSimilarity: 0.55,
		EvidenceLimit: 8,
	}
}

// CompareAIF compares the LLM-built AIF graph pred against the gold graph gold.
//
// Procedure (within the bipartite approximation framework of Riesen & Bunke 2009, except that assignment
// cost is determined by node labels alone, without adding the matching cost of surrounding edges as Riesen & Bunke do):
//  1. Embed I / L text and build substitution costs 1 - cos
//  2. Solve the optimal assignment of all nodes with the Hungarian method, once (assign)
//
// Leaving edges out of the assignment cost is intentional. Including them would let relation predictions
// pull proposition matching along, "matching propositions so that relations agree", inflating the relation
// P/R counted on that matching (the assignment would pre-empt what we want to evaluate). Proposition matching is
// decided by content alone, and relations are counted on that fixed matching. The GED obtained here is an upper bound on the optimum including edges.
//  3. 0.1 uses that assignment as-is; 0.2 and later use only the L / I assignment and
//     derive the structural node (S / TA / YA) correspondences from structure (derive)
//  4. Count node and edge edit costs under that correspondence (distance)
//  5. Attach layered diagnostics independent of the metric version (propositions, utterances, relations, illocutions)
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

	// ---- Embeddings (only I-node and L-node text; S/YA/TA have no sentences) ----
	sim, err := textSimilarity(ctx, pred, gold, emb)
	if err != nil {
		return nil, err
	}

	// ---- Optimal node assignment (once) ----
	//
	// Substitutions across types have a prohibitive cost, so the cost matrix is
	// block-diagonal by type. That is, solving over all nodes or over L only / I only
	// gives the same L↔L / I↔I assignment, so solving once is enough.
	rawMatch, err := assign(pred, gold, sim.cross, opts.Weights)
	if err != nil {
		return nil, err
	}
	contentMatch := restrictToTypes(rawMatch, pred, gold, aif.TypeL, aif.TypeI)
	propMatch := restrictToTypes(rawMatch, pred, gold, aif.TypeI)

	// ---- Overall (strict AIF) correspondence ----
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

	// Partial credit for edges (type / direction mismatch substitutions) applies from 0.2 on.
	labelled := opts.Metric != MetricGEDv1
	// How the overall graph is taken depends on the version. 0.3 folds relations into one edge
	// and removes YAs anchored to transitions (see aif.Graph.DialogueEdges).
	// 0.4 also removes TAs (see aif.Graph.AnchorEdges).
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

	// ---- Diagnostics (same regardless of metric version; uses only the proposition matching) ----
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

// assign optimally assigns all generated nodes to all gold nodes and returns generated ID → gold ID.
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

// correspondence maps generated node IDs → gold node IDs.
type correspondence struct {
	pairs map[string]string
	// derived is the number of correspondences derived from structure (decided rather than searched).
	derived int
}

// restrictToTypes keeps only the assigned nodes of the given types.
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

// derive determines structural node (S-node / TA / YA) correspondences from the L and I correspondence.
//
// In AIF, an S-node is uniquely determined by its premise and conclusion I-nodes, a TA by the preceding and
// following L-nodes, and a YA by its anchor and illocution target. They are not search targets, so they are
// decided here rather than put into the Hungarian method (putting them in would make all content-less TAs
// equivalent, degenerating the optimum and returning an arbitrary assignment).
func derive(pred, gold aif.Graph, content map[string]string) correspondence {
	out := correspondence{pairs: make(map[string]string, len(content)*2)}
	for a, b := range content {
		out.pairs[a] = b
	}

	// S-node: matched if (set of premises, conclusion) agree. Ones with only the direction reversed
	// also count as matched (type / direction mismatches are paid via substitution cost).
	//
	// Matches are decided cheapest first in 4 passes (same direction & same type → same direction & other type →
	// reversed & same type → reversed & other type). Previously they were decided greedily in generated output order,
	// so when CA and RA were emitted on the same proposition pair and the gold was RA, the earlier CA took the gold RA
	// and 0.2's overall GED split into 3.00 / 7.75 depending on order.
	// Also, when several gold S-nodes shared a key, the map overwrite left one unmatched,
	// so candidates are kept as a list per key.
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
		// for searching the reversed direction (meaningful only with a single premise)
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

	// TA: matched if the preceding and following L-nodes are matched. Multiple TAs with the same endpoint pair are not missed.
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

	// YA: matched if both (anchor, illocution target) are matched.
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

// distance aggregates the edit cost at one granularity.
//
// If labelled is true, when a gold edge connects the same proposition pair it is treated
// as a substitution rather than "deletion + insertion" (partial credit for type / direction mismatches).
func distance(predNodes, goldNodes []aif.Node, predEdges, goldEdges []aif.LabeledEdge,
	corr correspondence, sim simTable, w AIFWeights, labelled bool,
) AIFDistance {
	out := AIFDistance{Derived: corr.derived}

	// ---- Nodes ----
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

	// ---- Edges ----
	//
	// If an earlier edge takes a gold edge of a different type, a later edge can no longer match exactly.
	// So the result does not depend on processing order, every edge is matched per pass in the order
	// "exact", "type mismatch", "direction mismatch" (cheapest correspondences are fixed first).
	type ekey struct{ from, to, kind string }
	remaining := map[ekey]int{}          // gold edges not yet matched (per type)
	remainingPair := map[[2]string]int{} // likewise, per endpoint pair (any type)
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
		// After exact matches, taking a gold edge of any type costs the same, so only the per-endpoint-pair count matters.
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

	// ---- Normalization ----
	//
	// Two are reported. The symmetric denominator (the primary metric up to 0.2) depends on pred's size, so
	// there is a path where "emitting more relations pays off". The gold-based one has no such path and
	// has the same denominator across conditions, so it can be used to compare conditions.
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

// substitutionCost is the cost of substituting node a with b.
func substitutionCost(a, b aif.Node, sim simTable, w AIFWeights) float64 {
	switch {
	case a.Type == b.Type && (a.Type == aif.TypeI || a.Type == aif.TypeL):
		// Semantic closeness. Negative cosine is clamped to 0 (no worse than "unrelated"),
		// and cos slightly above 1 due to floating-point error does not produce a negative cost.
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
		return 0 // S-nodes of the same type
	case a.Type.IsSNode() && b.Type.IsSNode():
		return w.SNodeTypeMismatch
	default:
		return w.Forbidden
	}
}

// Add sums two aggregates (for aggregating across nodesets).
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
