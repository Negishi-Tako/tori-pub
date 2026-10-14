package aif

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/prompts"
)

// Annotator is an LLM annotator that builds a "plain AIF" graph from transcript text.
//
// It is split into 3 stages, following the IAT annotation procedure:
//
//	stage 1  transcript  → L-nodes (speaker + utterance unit)
//	stage 2  L-node      → YA-node (illocution) + I-node (propositional content)
//	stage 3  I-node      → RA / CA / MA (+ YAs anchored to transitions)
//
// TA-nodes are placed mechanically between adjacent L-nodes (in IAT the sequence of utterances
// itself is the transition, so it is not something for the LLM to judge).
type Annotator struct {
	cfg    AnnotatorConfig
	logger *slog.Logger
}

// AnnotatorConfig configures an Annotator.
type AnnotatorConfig struct {
	Client  llm.Client
	Prompts *prompts.Store
	Model   string
	// Prompt version for each stage. Empty means the default (aif_*_v1).
	LocutionPrompt   string
	IllocutionPrompt string
	RelationPrompt   string
	// If VerifyPrompt is non-empty, a verification pass (stage 4) runs after stage 3
	// and decides, one relation at a time, whether to keep or discard each generated relation.
	// Bulk generation cannot curb its own over-generation, so a separate pruner is used.
	VerifyPrompt string
	// If ScoredRelations is true, stage 3 is received with a confidence schema
	// (aif_relation_v5 and later). The threshold is moved downstream (on the evaluation side).
	ScoredRelations bool
	// If GradedRelations is true, stage 3 is received with a 5-level dependence schema
	// (aif_relation_v7 and later). Finer levels were added because 3-level confidence collapsed to effectively binary.
	GradedRelations bool
	// If CandidatePrompt is non-empty, stage 3 judges candidate pairs one at a time
	// instead of having the LLM generate a list of relations.
	//
	// With generation, the choice of which pairs to link lost to the trivial rule
	// "link all adjacent propositions" (test n=124 undirected F1: 0.458 vs 0.508). Rather than leave
	// selection to free generation, we fix the candidate set ourselves and let the LLM only judge.
	CandidatePrompt string
	// CandidateMaxDistance is the maximum proposition-rank distance included in candidates.
	// Measured on dev, k=4 covers 90.5% of gold relations with 46 pairs per nodeset,
	// of which 17.6% actually hold a relation. 0 means the default of 4.
	CandidateMaxDistance int
	// If AdjacentRelations is true, stage 3 is built by the mechanical rule
	// "link all adjacent propositions" instead of the LLM (baseline).
	// Relation extraction must always have one "no LLM" lower bound too.
	AdjacentRelations bool
	// If RestrictLocutionForces is true, stage 2's choices are limited to illocutionary forces
	// that can be anchored to an L-node (aif.LocutionForces). This matches IAT practice,
	// and the default is true. If false, all 11 forces are offered (v1-equivalent, for comparison).
	RestrictLocutionForces *bool
	// Inputs exceeding MaxLocutions are refused. Silently truncating more than fits
	// in one prompt would quietly corrupt the graph under evaluation.
	MaxLocutions int
	Logger       *slog.Logger
}

const (
	defaultLocutionPrompt   = "aif_locution_v1"
	defaultIllocutionPrompt = "aif_illocution_v2"
	defaultRelationPrompt   = "aif_relation_v2"
	defaultMaxLocutions     = 120
	// Maximum proposition-rank distance included in candidates. Covers 90.5% of gold on dev.
	defaultCandidateMaxDistance = 4
)

// NewAnnotator creates an annotator.
func NewAnnotator(cfg AnnotatorConfig) (*Annotator, error) {
	if cfg.Client == nil {
		return nil, fmt.Errorf("aif: llm client is nil")
	}
	if cfg.Prompts == nil {
		return nil, fmt.Errorf("aif: prompt store is nil")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("aif: model is empty")
	}
	if cfg.LocutionPrompt == "" {
		cfg.LocutionPrompt = defaultLocutionPrompt
	}
	if cfg.IllocutionPrompt == "" {
		cfg.IllocutionPrompt = defaultIllocutionPrompt
	}
	if cfg.RelationPrompt == "" && cfg.CandidatePrompt == "" && !cfg.AdjacentRelations {
		cfg.RelationPrompt = defaultRelationPrompt
	}
	if cfg.CandidateMaxDistance <= 0 {
		cfg.CandidateMaxDistance = defaultCandidateMaxDistance
	}
	if cfg.MaxLocutions <= 0 {
		cfg.MaxLocutions = defaultMaxLocutions
	}
	if cfg.RestrictLocutionForces == nil {
		t := true
		cfg.RestrictLocutionForces = &t
	}
	lg := cfg.Logger
	if lg == nil {
		lg = slog.Default()
	}
	return &Annotator{cfg: cfg, logger: lg}, nil
}

// PromptVersions returns the prompt versions in use and their body hashes.
//
// For reproducibility of extraction results, this lets results record which version produced a graph.
func (a *Annotator) PromptVersions() (map[string]string, error) {
	out := map[string]string{}
	for stage, name := range map[string]string{
		"locution":   a.cfg.LocutionPrompt,
		"illocution": a.cfg.IllocutionPrompt,
		"relation":   a.cfg.RelationPrompt,
		"verify":     a.cfg.VerifyPrompt,
		"candidate":  a.cfg.CandidatePrompt,
	} {
		if name == "" {
			continue
		}
		p, err := a.cfg.Prompts.Get(name)
		if err != nil {
			return nil, err
		}
		sha := p.SHA256
		if len(sha) > 12 {
			sha = sha[:12]
		}
		out[stage] = p.Version + "@" + sha
	}
	return out, nil
}

// LocutionInput is the utterance unit passed to stage 2 onward.
// Also used by conditions that pass the gold L-nodes as-is (conditions that give the segmentation).
type LocutionInput struct {
	Speaker string
	Text    string
}

// Annotation is the annotation result for one nodeset.
type Annotation struct {
	Graph Graph `json:"graph"`
	// Raw holds the raw LLM response of each stage. It is kept so that changing evaluation
	// metrics later does not require regeneration.
	Raw map[string]json.RawMessage `json:"raw,omitempty"`
	// Usage is the total tokens and cost over the 3 stages.
	Usage llm.Usage `json:"usage"`
	// Violations are AIF metamodel violations in the generated graph (0 is normal).
	Violations []Violation `json:"violations,omitempty"`
	// Rationales are the per-relation rationales. Used in qualitative analysis.
	Rationales map[string]string `json:"rationales,omitempty"`
	// Dropped is the number of relations discarded by the verification pass (stage 4). 0 can mean either
	// "nothing was discarded" or "no verification pass was run" (tell by whether Raw["verify"] exists).
	Dropped int `json:"dropped,omitempty"`
}

// Segment is stage 1. It cuts the transcript into L-nodes.
func (a *Annotator) Segment(ctx context.Context, id, transcript string) ([]LocutionInput, json.RawMessage, llm.Usage, error) {
	data := map[string]any{"Transcript": strings.TrimSpace(transcript)}
	out, resp, err := call[LocutionResult](ctx, a, "locution", a.cfg.LocutionPrompt, data,
		"aif_locutions", "Locution segmentation for IAT annotation")
	if err != nil {
		return nil, nil, llm.Usage{}, err
	}
	locs := make([]LocutionInput, 0, len(out.Locutions))
	for _, l := range out.Locutions {
		text := strings.TrimSpace(l.Text)
		if text == "" {
			continue
		}
		locs = append(locs, LocutionInput{Speaker: strings.TrimSpace(l.Speaker), Text: text})
	}
	if len(locs) == 0 {
		return nil, rawOf(resp), usageOf(resp), fmt.Errorf("aif: %s: 発話単位が 1 つも返らなかった", id)
	}
	return locs, rawOf(resp), usageOf(resp), nil
}

// Annotate applies stages 2 and 3 to the given L-nodes and assembles an AIF graph.
// To start from a transcript, call Segment first (AnnotateTranscript does both).
func (a *Annotator) Annotate(ctx context.Context, id string, locs []LocutionInput) (*Annotation, error) {
	if len(locs) == 0 {
		return nil, fmt.Errorf("aif: %s: 発話単位が空", id)
	}
	if len(locs) > a.cfg.MaxLocutions {
		return nil, fmt.Errorf("aif: %s: 発話単位が %d 個で上限 %d を超えた（分割して実行すること）",
			id, len(locs), a.cfg.MaxLocutions)
	}
	ann := &Annotation{Raw: map[string]json.RawMessage{}, Rationales: map[string]string{}}

	// ---- stage 2: illocutions and propositional content ----
	type promptLoc struct {
		Index   int
		Speaker string
		Text    string
	}
	pl := make([]promptLoc, 0, len(locs))
	for i, l := range locs {
		pl = append(pl, promptLoc{Index: i, Speaker: l.Speaker, Text: l.Text})
	}
	forces := IllocutionaryForces
	if *a.cfg.RestrictLocutionForces {
		forces = LocutionForces
	}
	data := map[string]any{"Locutions": pl, "Forces": forces}
	var items []IllocutionItem
	var resp *llm.Response
	if *a.cfg.RestrictLocutionForces {
		out, r, err := call[IllocutionResultAnchored](ctx, a, "illocution", a.cfg.IllocutionPrompt,
			data, "aif_illocutions_anchored", "Illocutionary force and propositional content per locution")
		if err != nil {
			return nil, err
		}
		resp = r
		for _, it := range out.Items {
			items = append(items, IllocutionItem(it))
		}
	} else {
		out, r, err := call[IllocutionResult](ctx, a, "illocution", a.cfg.IllocutionPrompt,
			data, "aif_illocutions", "Illocutionary force and propositional content per locution")
		if err != nil {
			return nil, err
		}
		resp = r
		items = out.Items
	}
	ann.Raw["illocution"] = rawOf(resp)
	ann.Usage = ann.Usage.Add(usageOf(resp))

	// ---- Assemble the graph (L / TA / YA / I) ----
	g := Graph{ID: id}
	locID := make([]string, len(locs))
	for i, l := range locs {
		locID[i] = fmt.Sprintf("L%d", i)
		g.Nodes = append(g.Nodes, Node{ID: locID[i], Type: TypeL, Text: l.Speaker + " : " + l.Text, Speaker: l.Speaker})
	}
	// TAs are placed between adjacent Ls. In IAT the sequence of utterances itself is the transition;
	// letting the LLM decide this would turn the evaluation into a "guess the transitions" task.
	taBetween := map[[2]int]string{}
	for i := 0; i+1 < len(locs); i++ {
		ta := fmt.Sprintf("TA%d", i)
		g.Nodes = append(g.Nodes, Node{ID: ta, Type: TypeTA, Scheme: SchemeDefaultTransition})
		g.Edges = append(g.Edges, Edge{FromID: locID[i], ToID: ta}, Edge{FromID: ta, ToID: locID[i+1]})
		taBetween[[2]int{i, i + 1}] = ta
	}

	propID := make([]string, len(locs))   // locution index -> I-node ID (empty if none)
	propBody := make([]string, len(locs)) // locution index -> proposition text
	seen := map[int]bool{}
	for _, it := range items {
		if it.Index < 0 || it.Index >= len(locs) {
			a.logger.Warn("illocution refers to an unknown locution", "nodeset", id, "index", it.Index)
			continue
		}
		if seen[it.Index] {
			continue
		}
		seen[it.Index] = true
		prop := strings.TrimSpace(it.Proposition)
		force := strings.TrimSpace(it.Force)
		if !ValidForce(force) {
			a.logger.Warn("unknown illocutionary force; using the default", "nodeset", id, "force", force)
			force = SchemeDefaultIllocuting
		}
		if prop == "" {
			// Utterances without propositional content (procedural remarks, etc.) get neither an I-node nor a YA,
			// because AIF does not allow "a YA without content".
			continue
		}
		iid := fmt.Sprintf("I%d", it.Index)
		yid := fmt.Sprintf("YA%d", it.Index)
		propID[it.Index] = iid
		propBody[it.Index] = prop
		g.Nodes = append(g.Nodes,
			Node{ID: iid, Type: TypeI, Text: prop},
			Node{ID: yid, Type: TypeYA, Text: force, Scheme: force},
		)
		g.Edges = append(g.Edges, Edge{FromID: locID[it.Index], ToID: yid}, Edge{FromID: yid, ToID: iid})
	}

	// ---- stage 3: relations between propositions ----
	// Force is the illocution assigned by stage 2. The stage 3 prompt (v3 and later) uses it
	// to judge question–answer relations (MA in IAT).
	type promptProp struct {
		Index   int
		Speaker string
		Text    string
		Force   string
	}
	var props []promptProp
	forceOf := map[int]string{}
	for _, it := range items {
		if it.Index >= 0 && it.Index < len(locs) {
			forceOf[it.Index] = it.Force
		}
	}
	propIdxOf := map[int]int{}   // number in the prompt -> locution index
	propText := map[int]string{} // number in the prompt -> proposition text (used in stage 4)
	for i, pid := range propID {
		if pid == "" {
			continue
		}
		n := len(props)
		propIdxOf[n] = i
		props = append(props, promptProp{Index: n, Speaker: locs[i].Speaker, Text: propBody[i], Force: forceOf[i]})
		propText[n] = propBody[i]
	}
	if len(props) >= 2 {
		// Whatever the method, from here on everything is handled as []RelationItemScored.
		var rels []RelationItemScored
		var resp2 *llm.Response
		switch {
		case a.cfg.AdjacentRelations:
			// Baseline: link all adjacent propositions. No LLM.
			// The type is the most frequent one, RA; the direction is "later proposition → earlier proposition" (73% measured on dev).
			for i := 0; i+1 < len(props); i++ {
				rels = append(rels, RelationItemScored{
					Src: i + 1, Dst: i, Type: string(TypeRA),
					Force: SchemeDefaultInference, Rationale: "adjacent (baseline)",
				})
			}
		case a.cfg.CandidatePrompt != "":
			out, r, err := a.classifyCandidates(ctx, id, props, propText)
			if err != nil {
				return nil, err
			}
			resp2, rels = r, out
		case a.cfg.GradedRelations:
			out, r, err := call[RelationResultGraded](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations_graded", "Argumentative relations between propositions, with a dependency grade")
			if err != nil {
				return nil, err
			}
			resp2 = r
			for _, it := range out.Relations {
				rels = append(rels, RelationItemScored{
					Src: it.Src, Dst: it.Dst, Type: it.Type, Force: it.Force,
					Rationale:  strings.TrimSpace(it.Rationale + " [" + it.Test + "]"),
					Confidence: strings.TrimSpace(it.Dependency),
				})
			}
		case a.cfg.ScoredRelations:
			out, r, err := call[RelationResultScored](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations_scored", "Argumentative relations between propositions, with confidence")
			if err != nil {
				return nil, err
			}
			resp2, rels = r, out.Relations
		default:
			out, r, err := call[RelationResult](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations", "Argumentative relations between propositions")
			if err != nil {
				return nil, err
			}
			resp2 = r
			for _, it := range out.Relations {
				rels = append(rels, RelationItemScored{
					Src: it.Src, Dst: it.Dst, Type: it.Type,
					Force: it.Force, Rationale: it.Rationale,
				})
			}
		}
		if resp2 != nil {
			ann.Raw["relation"] = rawOf(resp2)
			ann.Usage = ann.Usage.Add(usageOf(resp2))
		}

		// ---- stage 4: verification pass (optional) ----
		// Have each candidate reviewed one at a time to decide what to discard. The number dropped is recorded
		// (without knowing how many were pruned, we cannot tell whether it helped).
		if a.cfg.VerifyPrompt != "" && len(rels) > 0 {
			kept, resp3, err := a.verify(ctx, id, props, propText, rels)
			if err != nil {
				return nil, err
			}
			ann.Raw["verify"] = rawOf(resp3)
			ann.Usage = ann.Usage.Add(usageOf(resp3))
			ann.Dropped = len(rels) - len(kept)
			rels = kept
		}

		seenRel := map[[3]string]bool{}
		for k, r := range rels {
			si, sok := propIdxOf[r.Src]
			di, dok := propIdxOf[r.Dst]
			if !sok || !dok || si == di {
				a.logger.Warn("relation refers to an unknown proposition", "nodeset", id, "src", r.Src, "dst", r.Dst)
				continue
			}
			t := NodeType(strings.TrimSpace(r.Type))
			if !t.IsSNode() {
				a.logger.Warn("unknown relation type", "nodeset", id, "type", r.Type)
				continue
			}
			key := [3]string{propID[si], propID[di], string(t)}
			if seenRel[key] {
				continue
			}
			seenRel[key] = true
			sid := fmt.Sprintf("%s%d", t, k)
			g.Nodes = append(g.Nodes, Node{
				ID: sid, Type: t, Scheme: defaultScheme(t),
				Confidence: strings.TrimSpace(r.Confidence),
			})
			g.Edges = append(g.Edges, Edge{FromID: propID[si], ToID: sid}, Edge{FromID: sid, ToID: propID[di]})
			ann.Rationales[sid] = strings.TrimSpace(r.Rationale)

			// A YA carrying a relation is anchored to the TA between the two utterances (IAT).
			// Relations between non-adjacent utterances have no TA, so no YA is created either.
			// YAs carrying relations are anchored to transitions, so they do not take per-utterance forces (Asserting, etc.).
			force := strings.TrimSpace(r.Force)
			if !slices.Contains(TransitionForces, force) {
				force = SchemeDefaultIllocuting
			}
			ta, ok := taBetween[[2]int{si, di}]
			if !ok {
				ta, ok = taBetween[[2]int{di, si}]
			}
			if ok {
				yid := fmt.Sprintf("YA_%s", sid)
				g.Nodes = append(g.Nodes, Node{ID: yid, Type: TypeYA, Text: force, Scheme: force})
				g.Edges = append(g.Edges, Edge{FromID: ta, ToID: yid}, Edge{FromID: yid, ToID: sid})
			}
		}
	}

	norm, _ := Normalize(g)
	ann.Graph = norm
	ann.Violations = Validate(norm)
	if len(ann.Violations) > 0 {
		a.logger.Warn("生成した AIF グラフにメタモデル違反がある", "nodeset", id, "violations", len(ann.Violations))
	}
	return ann, nil
}

// AnnotateTranscript runs all 3 stages starting from a transcript (the end-to-end condition).
func (a *Annotator) AnnotateTranscript(ctx context.Context, id, transcript string) (*Annotation, error) {
	locs, raw, usage, err := a.Segment(ctx, id, transcript)
	if err != nil {
		return nil, err
	}
	ann, err := a.Annotate(ctx, id, locs)
	if err != nil {
		return nil, err
	}
	ann.Raw["locution"] = raw
	ann.Usage = ann.Usage.Add(usage)
	return ann, nil
}

// call is a single-stage LLM call. It reads the prompt, expands the template, and
// receives the result via Structured Outputs.
func call[T any](ctx context.Context, a *Annotator, stage, promptName string, data any, schemaName, schemaDesc string) (T, *llm.Response, error) {
	var zero T
	p, err := a.cfg.Prompts.Get(promptName)
	if err != nil {
		return zero, nil, err
	}
	rendered, err := p.Render(data)
	if err != nil {
		return zero, nil, err
	}
	req := llm.Request{
		Model: a.cfg.Model,
		Tag:   "aif_" + stage,
		Messages: []llm.Message{
			llm.System("You are an expert annotator of argument structure working with the Argument Interchange Format (AIF) and Inference Anchoring Theory (IAT). Follow the instructions exactly and answer only with the requested JSON."),
			llm.User(rendered),
		},
	}
	out, resp, err := llm.CompleteJSON[T](ctx, a.cfg.Client, req, schemaName, schemaDesc)
	if err != nil {
		return zero, resp, fmt.Errorf("aif: %s (%s): %w", stage, p.Version, err)
	}
	return out, resp, nil
}

func rawOf(resp *llm.Response) json.RawMessage {
	if resp == nil {
		return nil
	}
	return resp.Raw
}

func usageOf(resp *llm.Response) llm.Usage {
	if resp == nil {
		return llm.Usage{}
	}
	return resp.Usage
}

// verify is stage 4. It reviews generated relations one at a time and returns only those to keep.
//
// The key point is making it a separate call. "Proposing and pruning" within the same call
// does not work in bulk generation (measured over-generation exceeds the number of hits);
// this stage measures what happens when proposing and pruning are split.
func (a *Annotator) verify(ctx context.Context, id string, props any, propText map[int]string, rels []RelationItemScored) ([]RelationItemScored, *llm.Response, error) {
	type promptRel struct {
		Index          int
		Src, Dst       int
		Type           string
		Rationale      string
		SrcText        string
		DstText        string
		DialogueOffset int
	}
	// Candidates passed to the prompt, with the src / dst text attached (numbers alone are not enough to judge).
	cand := make([]promptRel, 0, len(rels))
	for i, r := range rels {
		d := r.Src - r.Dst
		if d < 0 {
			d = -d
		}
		cand = append(cand, promptRel{
			Index: i, Src: r.Src, Dst: r.Dst, Type: r.Type, Rationale: r.Rationale,
			SrcText: propText[r.Src], DstText: propText[r.Dst], DialogueOffset: d,
		})
	}
	out, resp, err := call[VerifyResult](ctx, a, "verify", a.cfg.VerifyPrompt,
		map[string]any{"Propositions": props, "Candidates": cand},
		"aif_relation_verification", "Keep-or-drop decision for each candidate relation")
	if err != nil {
		return nil, nil, err
	}
	drop := map[int]bool{}
	for _, it := range out.Items {
		if it.Relation < 0 || it.Relation >= len(rels) {
			a.logger.Warn("verification refers to an unknown relation", "nodeset", id, "index", it.Relation)
			continue
		}
		if !it.Keep {
			drop[it.Relation] = true
		}
	}
	kept := make([]RelationItemScored, 0, len(rels))
	for i, r := range rels {
		if !drop[i] {
			kept = append(kept, r)
		}
	}
	return kept, resp, nil
}

// classifyCandidates is the stage 3 variant that judges candidate pairs one at a time.
//
// Candidates are "pairs whose proposition-rank distance is within CandidateMaxDistance".
// Measured on dev, k=4 covers 90.5% of gold relations with 46 pairs per nodeset,
// of which 17.6% actually hold a relation.
// Unlike generation, which pairs to look at is not left to the LLM. The candidate set caps
// what can be found, but in exchange every failure other than "missing an existing relation" disappears.
func (a *Annotator) classifyCandidates(ctx context.Context, id string, props any, propText map[int]string) ([]RelationItemScored, *llm.Response, error) {
	n := len(propText)
	type promptCand struct {
		Index    int
		A, B     int
		AText    string
		BText    string
		Distance int
	}
	var cands []promptCand
	for i := 0; i < n; i++ {
		for j := i + 1; j < n && j <= i+a.cfg.CandidateMaxDistance; j++ {
			cands = append(cands, promptCand{
				Index: len(cands), A: i, B: j,
				AText: propText[i], BText: propText[j], Distance: j - i,
			})
		}
	}
	if len(cands) == 0 {
		return nil, nil, nil
	}
	out, resp, err := call[CandidateResult](ctx, a, "candidate", a.cfg.CandidatePrompt,
		map[string]any{"Propositions": props, "Candidates": cands, "MaxDistance": a.cfg.CandidateMaxDistance},
		"aif_candidate_verdicts", "Verdict for each candidate proposition pair")
	if err != nil {
		return nil, nil, err
	}
	force := map[NodeType]string{
		TypeRA: "Arguing", TypeCA: "Disagreeing", TypeMA: "Restating",
	}
	seen := map[int]bool{}
	var rels []RelationItemScored
	for _, it := range out.Items {
		if it.Candidate < 0 || it.Candidate >= len(cands) || seen[it.Candidate] {
			a.logger.Warn("verdict refers to an unknown candidate", "nodeset", id, "index", it.Candidate)
			continue
		}
		seen[it.Candidate] = true
		t := NodeType(strings.TrimSpace(it.Relation))
		if !t.IsSNode() {
			continue // "none" also lands here
		}
		c := cands[it.Candidate]
		src, dst := c.A, c.B
		if strings.TrimSpace(it.Source) == "later" {
			src, dst = c.B, c.A
		}
		rels = append(rels, RelationItemScored{
			Src: src, Dst: dst, Type: string(t), Force: force[t],
			Rationale: strings.TrimSpace(it.Rationale), Confidence: strings.TrimSpace(it.Confidence),
		})
	}
	return rels, resp, nil
}
