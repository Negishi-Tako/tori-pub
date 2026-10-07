You are annotating a dialogue under the Argument Interchange Format (AIF).

The propositional contents (I-nodes) of the dialogue have already been extracted.
Your task is to identify the **argumentative relations** between them. Each relation
becomes an S-node in AIF, anchored in the transition between the locutions that carried
the two propositions.

## Propositions

Each line gives the proposition index, the illocutionary force with which it was uttered,
the speaker, and its text. **The indices are in the order the propositions were uttered
in the dialogue**: `[0]` came first, and a higher index came later. This order matters
for the direction of some relations (see below).

{{range .Propositions}}
[{{.Index}}] ({{.Force}}; {{.Speaker}}) {{.Text}}
{{end}}

## Relation types (choose exactly one per relation)

- **RA** (Rule Application / *Default Inference*) — the source proposition is offered
  as a **reason supporting** the target proposition. `src` is the premise, `dst` is the
  conclusion. Ask: "src, therefore dst" — does that read correctly? If "dst, therefore
  src" reads better, swap them. **RA is the one relation whose direction is logical, not
  temporal**: a speaker may state the conclusion first and the reason after, or the
  reverse. Its illocutionary force is `Arguing`.

- **CA** (Conflict Application / *Default Conflict*) — the source proposition is
  **incompatible with** the target proposition: asserting one gives a reason to reject
  the other. **Direction is temporal: the later proposition attacks the earlier one.**
  `src` must therefore have the higher index. Its illocutionary force is `Disagreeing`,
  or `Challenging` when the conflict is voiced as a demand for justification.

- **MA** (Preference/Rephrase Application / *Default Rephrase*) — the source
  proposition **rephrases, restates or reformulates** the target proposition.
  **Direction is temporal: the later reformulation is `src`, the content being
  reformulated is `dst`.** `src` must therefore have the higher index. Its illocutionary
  force is `Restating`, or `Default Illocuting` for the question-answer case below.

## Two rules about MA that are easy to miss

1. **MA also holds between a question and its answer.** When a proposition uttered with
   a questioning force (`PureQuestioning`, `AssertiveQuestioning`,
   `RhetoricalQuestioning`) is subsequently answered, the answer instantiates the
   queried proposition: emit `MA` with the answer as `src`, the question's proposition
   as `dst`, and `Default Illocuting` as the force.
2. **Reformulation chains are common and must all be recorded.** In dialogue, speakers
   routinely restate the same content while narrowing, widening or adding precision to
   it ("people can meet outdoors" → "two households can meet outdoors" → "no more than
   eight people can meet outdoors"). Each successive restatement is its own MA pointing
   back to the one before it. Do not skip these as mere repetition, and do not record
   them as RA: adding precision to a claim is a reformulation, not a reason for it.

Ask yourself for each candidate pair: is the speaker **saying the same thing again in
different words** (MA), or **giving a reason to accept something else** (RA)? A passive
rewording, a generalisation, a specialisation, a summary and an answer to a question are
all MA. Only emit RA when `dst` would be less well supported without `src`.

## Rules

1. Only relate propositions that stand in a genuine relation **in this dialogue**.
   Topical similarity is not a relation. If two propositions are merely about the same
   subject, or one simply continues the narrative of the other, do not connect them.
2. The great majority of relations hold between propositions that are **close together**
   in the dialogue, most often between consecutive or near-consecutive ones. A relation
   between two distant propositions needs a clear reason.
3. A relation may hold between propositions from different speakers or from the same
   speaker. Elaborating one's own point across two consecutive locutions is usually RA
   (one gives a reason for the other) or MA (the later restates the earlier).
4. Do not create a relation between a proposition and itself.
5. Do not force every proposition into the graph. Propositions that merely add new
   information without supporting, attacking, or restating anything are left unconnected.
6. If several propositions jointly support one conclusion, emit one RA per premise with
   the same `dst`.

## How strongly does the target depend on the source

For every relation, grade the **dependency** on this five-point scale. Grade the relation
itself, not your own confidence. Apply the test written under each level and report which
test you applied in the `test` field.

**5 — explicit and constitutive.** The speaker marks the relation in words ("so",
"because", "therefore", "that's why", "in other words", "no, that's not right"), **or**
removing the source leaves the target with nothing holding it up.
*Test: is there a connective, or does the target collapse without the source?*

**4 — constitutive but unmarked.** No connective, but the target is put forward *on the
strength of* the source. Take the source away and the target changes role — it stops
being a conclusion, or stops being an answer, or stops being a correction.
*Test: does removing the source change what the target is doing in the dialogue?*

**3 — a real relation between two claims that each stand alone.** Both propositions are
independent assertions, and one nevertheless supports, restates or contradicts the other.
An annotator would record it. Neither collapses without the other.
*Test: would you record this if you were annotating, even though both stand alone?*

**2 — defensible only on a second reading.** The two propositions continue the same
topic, or the later one adds information to the earlier one, and you can construct a
reading on which one supports or restates the other — but nothing in the dialogue asks
for that reading.
*Test: did you have to argue yourself into it?*

**1 — not a relation. Do not emit these at all.** Identical or near-identical wording
repeated back; the host quoting a claim to hand over the floor; a speaker confirming what
was just said; narrative that simply moves on.
*Test: is the content the same, or merely adjacent?*

Two things to keep straight:

- **Emit everything you would grade 2 or higher, and grade it honestly.** Do not leave out
  a weak relation — record it as a 2. Do not promote a weak relation to 3 or 4 to justify
  recording it. The grades are used downstream to choose how strict the graph should be,
  so a wrong grade costs more than a weak relation.
- **Most relations you are tempted to record are 2s and 3s.** Levels 4 and 5 require the
  test to actually pass: a connective in the text, or a target that stops working without
  its source. Reserve them for those cases.

## Output

Return the list of relations. Return an empty list if there are none.
