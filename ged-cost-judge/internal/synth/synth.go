// Package synth uses an LLM to build source arguments (graph + text) (plan G1),
// realize edit plans as variant text (G3), and judge triplets (J).
//
// Generation and judgment always use separate requests with no conversation history (plan §2.2).
package synth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/llm"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
)

// Prompt is a prompt file's body and hash.
type Prompt struct {
	Name string
	Body string
	Hash string
}

// LoadPrompt reads dir/name.md.
func LoadPrompt(dir, name string) (Prompt, error) {
	b, err := os.ReadFile(filepath.Join(dir, name+".md"))
	if err != nil {
		return Prompt{}, err
	}
	h := sha256.Sum256(b)
	return Prompt{Name: name, Body: string(b), Hash: hex.EncodeToString(h[:6])}, nil
}

// Utterance is an utterance.
type Utterance struct {
	Speaker string `json:"speaker" jsonschema_description:"Speaker name or role"`
	Text    string `json:"text" jsonschema_description:"What the speaker says"`
}

// Discussion is a source argument or a variant (text and graph).
type Discussion struct {
	Utterances []Utterance `json:"utterances"`
	Graph      ged.Graph   `json:"graph"`
	// PropUtterance maps proposition → index of the utterance stating it (source arguments only).
	PropUtterance map[string]int `json:"prop_utterance,omitempty"`
}

// Render is the text shown for judgment.
func (d Discussion) Render() string {
	var b strings.Builder
	for _, u := range d.Utterances {
		fmt.Fprintf(&b, "%s: %s\n", u.Speaker, u.Text)
	}
	return b.String()
}

// Spec is the allocation of a source argument (the factors of plan §4.1).
type Spec struct {
	Format    string `json:"format"`
	Domain    string `json:"domain"`
	Speakers  int    `json:"speakers"`
	Props     int    `json:"props"`
	Structure string `json:"structure"`
}

// sourceOut is the structured output of source argument generation.
type sourceOut struct {
	Utterances   []Utterance `json:"utterances"`
	Propositions []struct {
		ID        string `json:"id" jsonschema_description:"p1, p2, ..."`
		Utterance int    `json:"utterance" jsonschema_description:"0-based index of the utterance that states it"`
		Text      string `json:"text" jsonschema_description:"Self-contained declarative sentence"`
	} `json:"propositions"`
	Relations []struct {
		Kind string `json:"kind" jsonschema:"enum=RA,enum=CA,enum=MA"`
		From string `json:"from"`
		To   string `json:"to"`
	} `json:"relations"`
}

// Generator bundles LLM calls.
type Generator struct {
	Client        llm.Client
	Model         string
	SourcePrompt  Prompt
	RealizePrompt Prompt
	JudgePrompt   Prompt
	VerifyPrompt  Prompt
}

// Usage aggregates cost (per call purpose).
type Usage map[string]llm.Usage

// GenerateSource builds a source argument. prefix is the proposition ID prefix (unique per source argument).
func (g *Generator) GenerateSource(ctx context.Context, spec Spec, prefix string, attempt int) (Discussion, *llm.Response, error) {
	user := fmt.Sprintf("Format: %s\nTopic area: %s\nNumber of speakers: %d\nTarget number of propositions: %d\nStructure: %s\n",
		spec.Format, spec.Domain, spec.Speakers, spec.Props, spec.Structure)
	if attempt > 0 {
		user += fmt.Sprintf("(attempt %d)\n", attempt+1)
	}
	out, resp, err := llm.CompleteJSON[sourceOut](ctx, g.Client, llm.Request{
		Model:    g.Model,
		Messages: []llm.Message{llm.System(g.SourcePrompt.Body), llm.User(user)},
		Tag:      "synth_source",
	}, "discussion", "A discussion with its argument structure")
	if err != nil {
		return Discussion{}, resp, err
	}
	d := Discussion{Utterances: out.Utterances, PropUtterance: map[string]int{}}
	ids := map[string]string{}
	for _, p := range out.Propositions {
		if _, dup := ids[p.ID]; dup || p.Utterance < 0 || p.Utterance >= len(out.Utterances) || strings.TrimSpace(p.Text) == "" {
			return Discussion{}, resp, fmt.Errorf("synth: invalid proposition %q", p.ID)
		}
		id := prefix + p.ID
		ids[p.ID] = id
		d.Graph.Props = append(d.Graph.Props, ged.Prop{ID: id, Text: p.Text})
		d.PropUtterance[id] = p.Utterance
	}
	seen := map[[2]string]bool{}
	for _, r := range out.Relations {
		f, ok1 := ids[r.From]
		t, ok2 := ids[r.To]
		if !ok1 || !ok2 || f == t {
			return Discussion{}, resp, fmt.Errorf("synth: invalid relation %s→%s", r.From, r.To)
		}
		k := [2]string{min(f, t), max(f, t)}
		if seen[k] {
			continue // keep only one relation per pair
		}
		seen[k] = true
		d.Graph.Edges = append(d.Graph.Edges, ged.Edge{From: f, To: t, Kind: r.Kind})
	}
	if len(d.Graph.Props) < 4 || len(d.Graph.Edges) < 2 {
		return Discussion{}, resp, errors.New("synth: too few propositions or relations")
	}
	return d, resp, nil
}

var verbs = map[string]string{ged.RA: "supports", ged.CA: "attacks", ged.MA: "restates"}

func short(id string) string {
	if i := strings.LastIndexByte(id, '.'); i >= 0 {
		return id[i+1:]
	}
	return id
}

// describe turns an edit into an English instruction for realization.
func describe(e perturb.Edit) string {
	switch e.Kind {
	case perturb.PPara:
		how := map[string]string{"weak": "light rewording", "medium": "clear rewording", "strong": "substantial rewording"}[e.Strength]
		return fmt.Sprintf("Paraphrase proposition %s (%s) without changing its meaning.", short(e.Prop), how)
	case perturb.PShift:
		how := map[string]string{"small": "a small change", "medium": "a moderate change", "large": "a large change (it should become a clearly different claim)"}[e.Strength]
		return fmt.Sprintf("Change the content of proposition %s to a different claim on the same topic that keeps all its relations (%s).", short(e.Prop), how)
	case perturb.PNeg:
		return fmt.Sprintf("Negate proposition %s: reverse its polarity or quantifier (e.g. \"should\" → \"should not\", \"all\" → \"some\"), keeping everything else, including its relations as stated.", short(e.Prop))
	case perturb.PDel, perturb.PDelR:
		return fmt.Sprintf("Remove proposition %s from the discussion entirely, together with any relation it takes part in.", short(e.Prop))
	case perturb.PAdd:
		return fmt.Sprintf("Add a new proposition %s: a claim on the same topic that has no relation to any other proposition (it neither supports, attacks nor restates anything).", short(e.Prop))
	case perturb.RDel:
		return fmt.Sprintf("Remove the relation \"%s %s %s\": both propositions stay stated, but %s is no longer presented as related to %s.",
			short(e.Edge.From), verbs[e.Edge.Kind], short(e.Edge.To), short(e.Edge.From), short(e.Edge.To))
	case perturb.RAdd:
		return fmt.Sprintf("Add the relation \"%s %s %s\" without changing what either proposition claims.",
			short(e.Edge.From), verbs[e.Edge.Kind], short(e.Edge.To))
	case perturb.RType:
		return fmt.Sprintf("Change the relation \"%s %s %s\" so that %s now %s %s, keeping what each proposition claims as far as possible.",
			short(e.Edge.From), verbs[e.Edge.Kind], short(e.Edge.To), short(e.Edge.From), verbs[e.NewKind], short(e.Edge.To))
	case perturb.RRev:
		return fmt.Sprintf("Reverse the relation \"%s %s %s\" so that %s now %s %s.",
			short(e.Edge.From), verbs[e.Edge.Kind], short(e.Edge.To), short(e.Edge.To), verbs[e.Edge.Kind], short(e.Edge.From))
	case perturb.XSurf:
		return "Reword one utterance of your choice without changing any proposition or relation."
	}
	return ""
}

type realizeOut struct {
	Changes []struct {
		Utterance int    `json:"utterance"`
		Op        string `json:"op" jsonschema:"enum=replace,enum=insert_after,enum=delete"`
		Speaker   string `json:"speaker"`
		Text      string `json:"text"`
		Role      string `json:"role" jsonschema:"enum=edit,enum=connective"`
	} `json:"changes"`
	Propositions []struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	} `json:"propositions"`
}

// RealizeStats is the result of mechanically checking a realization.
type RealizeStats struct {
	Changed     int `json:"changed"`
	Connective  int `json:"connective"`
	MissingText int `json:"missing_text"`
}

// Instructions are the edit instructions passed to realization and verification (speaker swaps are done by the harness, so they are excluded).
func Instructions(v perturb.Variant) []string {
	var instr []string
	for _, e := range v.Edits {
		if e.Kind != perturb.XSpk {
			instr = append(instr, fmt.Sprintf("%d. %s", len(instr)+1, describe(e)))
		}
	}
	return instr
}

// NeedsLLM reports whether the variant needs LLM realization (speaker swaps alone are handled by the harness).
func NeedsLLM(v perturb.Variant) bool { return len(Instructions(v)) > 0 }

func swapsSpeakers(v perturb.Variant) bool {
	for _, e := range v.Edits {
		if e.Kind == perturb.XSpk {
			return true
		}
	}
	return false
}

// describeSource writes the source argument's utterances, propositions and relations with numbers.
func describeSource(b *strings.Builder, utts []Utterance, src Discussion) {
	b.WriteString("Utterances:\n")
	for i, u := range utts {
		fmt.Fprintf(b, "[%d] %s: %s\n", i, u.Speaker, u.Text)
	}
	b.WriteString("\nPropositions:\n")
	for _, p := range src.Graph.Props {
		fmt.Fprintf(b, "%s (utterance %d): %s\n", short(p.ID), src.PropUtterance[p.ID], p.Text)
	}
	b.WriteString("\nRelations:\n")
	for _, e := range src.Graph.Edges {
		fmt.Fprintf(b, "%s %s %s\n", short(e.From), verbs[e.Kind], short(e.To))
	}
}

// Realize realizes variant v as text based on src. Unchanged utterances are assembled here verbatim (plan §4.4).
// attempt and feedback are passed on regeneration (0 and "" for the first attempt).
func (g *Generator) Realize(ctx context.Context, src Discussion, v perturb.Variant, prefix string, attempt int, feedback string) (Discussion, RealizeStats, *llm.Response, error) {
	out := Discussion{Graph: v.Graph.Clone()}
	instr := Instructions(v)
	needLLM := len(instr) > 0
	utts := slices.Clone(src.Utterances)
	if swapsSpeakers(v) {
		utts = swapSpeakers(utts)
	}
	srcText := map[string]string{}
	for _, p := range src.Graph.Props {
		srcText[p.ID] = p.Text
	}
	if !needLLM {
		out.Utterances = utts
		for i, p := range out.Graph.Props {
			out.Graph.Props[i].Text = srcText[p.ID]
		}
		return out, RealizeStats{}, nil, nil
	}

	var b strings.Builder
	describeSource(&b, utts, src)
	b.WriteString("\nEdits:\n" + strings.Join(instr, "\n") + "\n")
	if feedback != "" {
		fmt.Fprintf(&b, "\nA previous attempt (%d) was rejected: %s\n", attempt, feedback)
	}

	res, resp, err := llm.CompleteJSON[realizeOut](ctx, g.Client, llm.Request{
		Model:    g.Model,
		Messages: []llm.Message{llm.System(g.RealizePrompt.Body), llm.User(b.String())},
		Tag:      "realize",
	}, "edited_discussion", "Changed utterances and propositions")
	if err != nil {
		return Discussion{}, RealizeStats{}, resp, err
	}

	var st RealizeStats
	replace := map[int]Utterance{}
	deleted := map[int]bool{}
	inserts := map[int][]Utterance{}
	for _, c := range res.Changes {
		if c.Utterance < -1 || c.Utterance >= len(utts) {
			return Discussion{}, st, resp, fmt.Errorf("synth: change refers to utterance %d", c.Utterance)
		}
		st.Changed++
		if c.Role == "connective" {
			st.Connective++
		}
		u := Utterance{Speaker: c.Speaker, Text: c.Text}
		switch c.Op {
		case "replace":
			if c.Utterance < 0 {
				return Discussion{}, st, resp, errors.New("synth: replace of utterance -1")
			}
			if u.Speaker == "" {
				u.Speaker = utts[c.Utterance].Speaker
			}
			replace[c.Utterance] = u
		case "delete":
			if c.Utterance >= 0 {
				deleted[c.Utterance] = true
			}
		case "insert_after":
			inserts[c.Utterance] = append(inserts[c.Utterance], u)
		}
	}
	out.Utterances = append(out.Utterances, inserts[-1]...)
	for i, u := range utts {
		if !deleted[i] {
			if r, ok := replace[i]; ok {
				u = r
			}
			out.Utterances = append(out.Utterances, u)
		}
		out.Utterances = append(out.Utterances, inserts[i]...)
	}

	newText := map[string]string{}
	for _, p := range res.Propositions {
		newText[prefix+p.ID] = p.Text
		newText[p.ID] = p.Text
	}
	for i, p := range out.Graph.Props {
		t, changed := newText[p.ID]
		if !changed {
			t, changed = newText[short(p.ID)]
		}
		_, mustChange := v.Changed[p.ID]
		isNew := !strings.HasPrefix(p.ID, prefix)
		switch {
		case (mustChange || isNew) && changed && strings.TrimSpace(t) != "":
			out.Graph.Props[i].Text = t
		case mustChange || isNew:
			st.MissingText++
			out.Graph.Props[i].Text = srcText[p.ID]
		default:
			out.Graph.Props[i].Text = srcText[p.ID]
		}
	}
	return out, st, resp, nil
}

func swapSpeakers(us []Utterance) []Utterance {
	out := slices.Clone(us)
	var names []string
	for _, u := range us {
		if !slices.Contains(names, u.Speaker) {
			names = append(names, u.Speaker)
		}
	}
	if len(names) < 2 {
		return out
	}
	a, b := names[0], names[1]
	for i, u := range out {
		switch u.Speaker {
		case a:
			out[i].Speaker = b
		case b:
			out[i].Speaker = a
		}
	}
	return out
}

// Verification is the result of verifying a realization (plan G4; in the PoC, the same luna as generation does it).
type Verification struct {
	Edits []struct {
		Index    int  `json:"index" jsonschema_description:"Number of the requested edit"`
		Realized bool `json:"realized"`
	} `json:"edits"`
	UnintendedChange bool   `json:"unintended_change"`
	Note             string `json:"note"`
	Naturalness      string `json:"naturalness" jsonschema:"enum=1,enum=2,enum=3,enum=4,enum=5"`
}

// Natural is naturalness (1–5, or 0 if unreadable).
func (v Verification) Natural() int {
	if len(v.Naturalness) == 1 && v.Naturalness[0] >= '1' && v.Naturalness[0] <= '5' {
		return int(v.Naturalness[0] - '0')
	}
	return 0
}

// Problems returns the reasons for failing (empty if passed). n is the number of instructed edits.
// Criteria: every edit appears in the text, no uninstructed changes, naturalness 3 or more (plan §4.4).
func (v Verification) Problems(n int) []string {
	var out []string
	seen := map[int]bool{}
	for _, e := range v.Edits {
		seen[e.Index] = true
		if !e.Realized {
			out = append(out, fmt.Sprintf("edit %d is not clearly visible in the text", e.Index))
		}
	}
	for i := 1; i <= n; i++ {
		if !seen[i] {
			out = append(out, fmt.Sprintf("edit %d was not checked", i))
		}
	}
	if v.UnintendedChange {
		out = append(out, "it changes something no edit asked for ("+v.Note+")")
	}
	if v.Natural() < 3 {
		out = append(out, fmt.Sprintf("it reads unnaturally (naturalness %s of 5)", v.Naturalness))
	}
	return out
}

// Verify checks, in a separate request, whether the realized variant d implements variant v's edits.
func (g *Generator) Verify(ctx context.Context, src Discussion, d Discussion, v perturb.Variant) (Verification, *llm.Response, error) {
	var b strings.Builder
	b.WriteString("ORIGINAL DISCUSSION\n\n")
	describeSource(&b, src.Utterances, src)
	b.WriteString("\nEDITED DISCUSSION\n\n")
	b.WriteString(d.Render())
	b.WriteString("\nREQUESTED EDITS\n\n" + strings.Join(Instructions(v), "\n") + "\n")
	if swapsSpeakers(v) {
		b.WriteString("\nSpeaker names were swapped on purpose.\n")
	}
	return llm.CompleteJSON[Verification](ctx, g.Client, llm.Request{
		Model:    g.Model,
		Messages: []llm.Message{llm.System(g.VerifyPrompt.Body), llm.User(b.String())},
		Tag:      "verify",
	}, "verification", "Whether each requested edit is visible, unintended changes, naturalness")
}

type judgeOut struct {
	Answer string `json:"answer" jsonschema:"enum=1,enum=2,enum=same"`
}

// JudgeAnswer is the result of one judgment ("1" / "2" / "same").
func (g *Generator) JudgeAnswer(ctx context.Context, s, first, second Discussion) (string, *llm.Response, error) {
	user := "Original discussion S:\n" + s.Render() +
		"\nVersion 1:\n" + first.Render() +
		"\nVersion 2:\n" + second.Render()
	out, resp, err := llm.CompleteJSON[judgeOut](ctx, g.Client, llm.Request{
		Model:    g.Model,
		Messages: []llm.Message{llm.System(g.JudgePrompt.Body), llm.User(user)},
		Tag:      "judge",
	}, "judgment", "Which version better preserves the original")
	if err != nil {
		return "", resp, err
	}
	return out.Answer, resp, nil
}

// MarshalIndent is for writing JSON.
func MarshalIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}
