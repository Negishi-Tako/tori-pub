You are reviewing a draft AIF/IAT annotation of a dialogue.

Another annotator has proposed a list of argumentative relations between the
propositions below. Your job is **not** to add relations, and **not** to rewrite them.
Your job is to decide, for each proposed relation, whether an annotator following the
IAT guidelines would actually record it. Keep it or drop it.

## Propositions

The indices are in the order the propositions were uttered: `[0]` came first.

{{range .Propositions}}
[{{.Index}}] ({{.Force}}; {{.Speaker}}) {{.Text}}
{{end}}

## Candidate relations

Each candidate gives its index, its type, the two propositions it connects, how far
apart they are in the dialogue, and the reason the first annotator gave.

{{range .Candidates}}
({{.Index}}) **{{.Type}}** [{{.Src}}] → [{{.Dst}}] (distance {{.DialogueOffset}})
  src: {{.SrcText}}
  dst: {{.DstText}}
  reason given: {{.Rationale}}
{{end}}

## Drop a candidate when

1. **The two propositions say the same thing and nothing is being argued.** A speaker or
   the host repeating, quoting or reading back what was just said — to confirm it, to
   hand the floor over, or to introduce the next speaker — is not a rephrase. `MA` is for
   a speaker **reformulating content as a move in the argument**: narrowing it, widening
   it, drawing out what it commits you to, answering a question with it. Mere repetition
   of identical or near-identical wording is not a relation at all.
2. **The connection is topical, not argumentative.** The two propositions are about the
   same subject, or one simply continues the speaker's narrative, but neither gives a
   reason for the other, restates the other, or conflicts with the other.
3. **The relation is an artefact of adjacency.** The two propositions happen to be next
   to each other, or two or three apart, and the reason given amounts to "it follows on
   from" or "it adds to" the previous one. Elaboration that adds new information is not
   `RA` unless the earlier claim is genuinely better supported by the later one.
4. **The type is wrong in a way that makes the relation unrecoverable** — for example it
   is recorded as `RA` when the two propositions plainly say the same thing, or as `MA`
   when one is plainly a reason for the other. Drop it rather than keeping a wrong edge.

## Keep a candidate when

- It is a genuine reason, reformulation or conflict, **however far apart the two
  propositions are.** Distance is not a reason to drop: about one relation in five
  spans five or more locutions, and those are exactly the ones a reviewer is tempted to
  remove. Judge the content, not the gap.
- It is an answer instantiating an earlier question, or a chain of successive
  reformulations. Both are ordinary IAT annotations.
- You are unsure but the reading is defensible. **Drop only what you can say is wrong.**

Do not drop relations merely to make the list shorter, and do not aim for any particular
number kept. Judge each candidate on its own.

## Output

Return exactly one entry per candidate, in the same order, referring to each candidate
by its index.
