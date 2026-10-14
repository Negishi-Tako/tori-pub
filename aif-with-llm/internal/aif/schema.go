package aif

// Types used for LLM Structured Outputs.
// They follow the constraints of OpenAI's strict mode (in strict mode every field is required,
// so there are no optional fields; absence is represented by the empty string).

// LocutionResult is the output of L-node segmentation (stage 1).
type LocutionResult struct {
	Locutions []LocutionItem `json:"locutions" jsonschema_description:"Ordered locutions (AIF L-nodes) extracted from the transcript"`
}

type LocutionItem struct {
	Speaker string `json:"speaker" jsonschema_description:"Speaker name exactly as written in the transcript"`
	Text    string `json:"text" jsonschema_description:"The locution text, taken verbatim from the transcript"`
}

// IllocutionResult is the output of YA + I-node (stage 2).
type IllocutionResult struct {
	Items []IllocutionItem `json:"items" jsonschema_description:"One entry per input locution, in the same order"`
}

type IllocutionItem struct {
	Index       int    `json:"index" jsonschema_description:"0-based index of the locution this entry describes"`
	Force       string `json:"force" jsonschema:"enum=Asserting,enum=Arguing,enum=Restating,enum=Analysing,enum=Agreeing,enum=Disagreeing,enum=PureQuestioning,enum=AssertiveQuestioning,enum=RhetoricalQuestioning,enum=Challenging,enum=Default Illocuting" jsonschema_description:"Illocutionary force of the locution (the YA-node scheme)"`
	Proposition string `json:"proposition" jsonschema_description:"Propositional content as a standalone declarative sentence (the I-node). Empty string if the locution has no propositional content"`
}

// IllocutionResultAnchored is the stage 2 variant that limits illocutions to YAs anchored
// to L-nodes (aif_illocution_v2). Removing Arguing / Restating / Analysing from the choices
// prevents forces that should be anchored to transitions from being attached to utterances.
type IllocutionResultAnchored struct {
	Items []IllocutionItemAnchored `json:"items" jsonschema_description:"One entry per input locution, in the same order"`
}

type IllocutionItemAnchored struct {
	Index       int    `json:"index" jsonschema_description:"0-based index of the locution this entry describes"`
	Force       string `json:"force" jsonschema:"enum=Asserting,enum=PureQuestioning,enum=AssertiveQuestioning,enum=RhetoricalQuestioning,enum=Challenging,enum=Agreeing,enum=Disagreeing,enum=Default Illocuting" jsonschema_description:"Illocutionary force with which the speaker puts this locution forward (the YA-node scheme anchored in the L-node)"`
	Proposition string `json:"proposition" jsonschema_description:"Propositional content as a standalone declarative sentence (the I-node). Empty string if the locution has no propositional content"`
}

// RelationResult is the output of RA/CA/MA (stage 3).
type RelationResult struct {
	Relations []RelationItem `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItem struct {
	Src       int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst       int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type      string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force     string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
}

// RelationResultScored is the stage 3 output with confidence added (aif_relation_v5 and later).
//
// It is a separate type so the v3-and-earlier schema does not change by a single byte.
// In Structured Outputs strict mode every field is required, so adding a field to the
// existing RelationItem would change past conditions too.
//
// Confidence is 3-level rather than numeric because the decimals an LLM self-reports
// cluster around 0.8 / 0.9 and do not work as thresholds. With 3 levels, a P/R curve
// can be drawn from 3 points: "high only", "high+medium" and "all".
type RelationResultScored struct {
	Relations []RelationItemScored `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItemScored struct {
	Src        int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst        int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type       string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force      string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale  string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
	Confidence string `json:"confidence" jsonschema:"enum=high,enum=medium,enum=low" jsonschema_description:"How certain you are that this relation holds AND that its type and direction are right. high = an annotator following the guidelines would certainly record it; low = you would not be surprised if it were not annotated"`
}

// VerifyResult is the output of the verification pass (stage 4).
//
// It reviews generated relations one at a time and decides whether to keep or discard each.
// Measurements show bulk generation cannot curb its own over-generation
// (over-generation exceeds the number of hits), so a separate call acts as the pruner.
type VerifyResult struct {
	Items []VerifyItem `json:"items" jsonschema_description:"One entry per candidate relation, in the same order as the input"`
}

type VerifyItem struct {
	Relation int    `json:"relation" jsonschema_description:"0-based index of the candidate relation this entry judges"`
	Keep     bool   `json:"keep" jsonschema_description:"true to keep the relation, false to drop it"`
	Reason   string `json:"reason" jsonschema_description:"One short sentence for the decision"`
}

// CandidateResult is the output of candidate-pair classification (an alternative stage 3).
//
// Instead of generating "a list of relations", the LLM judges "does this pair hold a relation"
// for each candidate. With generation, the choice of which pairs to link lost even to the
// trivial rule "link all adjacent propositions" (test n=124 undirected F1: 0.458 vs 0.508),
// so selection is no longer left to free generation.
type CandidateResult struct {
	Items []CandidateItem `json:"items" jsonschema_description:"One verdict per candidate pair, in the same order as the input"`
}

type CandidateItem struct {
	Candidate  int    `json:"candidate" jsonschema_description:"0-based index of the candidate pair this verdict is for"`
	Relation   string `json:"relation" jsonschema:"enum=none,enum=RA,enum=CA,enum=MA" jsonschema_description:"The AIF S-node type, or none if the two propositions stand in no argumentative relation"`
	Source     string `json:"source" jsonschema:"enum=earlier,enum=later" jsonschema_description:"Which of the two propositions is the source of the relation (premise for RA, attacker for CA, reformulation for MA). Ignored when relation is none"`
	Confidence string `json:"confidence" jsonschema:"enum=high,enum=medium,enum=low" jsonschema_description:"How certain you are of this verdict, including the type and the direction"`
	Rationale  string `json:"rationale" jsonschema_description:"One short sentence for the verdict. Empty string when relation is none"`
}

// RelationResultGraded is the stage 3 output with a 5-level "dependence" grade
// (aif_relation_v7 and later).
//
// The 3-level confidence (RelationResultScored) **collapsed to effectively binary** in practice:
// high 49.5% / medium 49.1% / low 1.4%. Only two operating points were available, "all" and
// "high only", and the latter halves recall.
// Meanwhile precision differs nearly 2x between high (0.395) and medium+low (0.207), so
// the self-report itself carries information. The coarse granularity was the problem.
//
// Two things are changed:
//   - 3 levels → 5 levels for more operating points.
//   - Instead of "would an annotator record it" (= annotator confidence), ask
//     "how much does dst depend on src" (= a property of the relation itself).
//     The former invites drifting to the middle; the latter admits an operational decision procedure.
type RelationResultGraded struct {
	Relations []RelationItemGraded `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItemGraded struct {
	Src       int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst       int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type      string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force     string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
	// Dependency is the 5-level dependence; 5 is strongest. 1 means "not a relation", so it is not output.
	Dependency string `json:"dependency" jsonschema:"enum=5,enum=4,enum=3,enum=2" jsonschema_description:"How strongly the target depends on the source, on the 5-point scale defined in the instructions. 5 is strongest. Do not emit relations you would grade 1"`
	// Test is the result of the decision procedure applied to determine the dependence. It records the grade's basis
	// (the grade alone cannot later be checked for why it was given).
	Test string `json:"test" jsonschema_description:"Which of the tests in the instructions you applied, and its outcome, in at most ten words"`
}
