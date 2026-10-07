You are annotating a dialogue under the Argument Interchange Format (AIF).

The propositional contents (I-nodes) of the dialogue have already been extracted.
Your task is to identify the **argumentative relations** between them. Each relation
becomes an S-node in AIF, anchored in the transition between the locutions that carried
the two propositions.

## Propositions

Each line gives the proposition index, the speaker who uttered it, and its text.

{{range .Propositions}}
[{{.Index}}] ({{.Speaker}}) {{.Text}}
{{end}}

## Relation types (choose exactly one per relation)

- **RA** (Rule Application / *Default Inference*) — the source proposition is offered
  as a **reason supporting** the target proposition. Direction matters: `src` is the
  premise, `dst` is the conclusion. Ask: "src, therefore dst" — does that read
  correctly? If "dst, therefore src" reads better, swap them. Its illocutionary force
  is `Arguing`.
- **CA** (Conflict Application / *Default Conflict*) — the source proposition is
  **incompatible with** the target proposition: asserting one gives a reason to reject
  the other. Direction matters: `src` is the attacker, `dst` is what is attacked.
  Its illocutionary force is `Disagreeing`, or `Challenging` when the conflict is voiced
  as a demand for justification.
- **MA** (Preference/Rephrase Application / *Default Rephrase*) — the source
  proposition **re-expresses or reformulates** the target proposition without adding
  new inferential content. Direction: `src` is the later reformulation, `dst` is the
  content being reformulated. Its illocutionary force is `Restating`, or
  `Default Illocuting` when the reformulation is a summary or a paraphrase by another
  speaker.

## Rules

1. Only relate propositions that stand in a genuine argumentative relation **in this
   dialogue**. Topical similarity is not a relation. If two propositions are merely
   about the same subject, or one simply continues the narrative of the other, do not
   connect them.
2. Relations are anchored in the flow of the dialogue: the great majority hold between
   propositions that are **close together** in the transcript, most often between
   consecutive or near-consecutive locutions. A relation between two propositions that
   are far apart needs a clear reason.
3. A relation may hold between propositions from different speakers or from the same
   speaker. Elaborating one's own point across two consecutive locutions is usually RA
   (the second gives a reason for the first) or MA (the second restates the first).
4. Do not create a relation between a proposition and itself.
5. Relations are sparse. Do not force every proposition into the graph; propositions
   that merely add new information without supporting, attacking, or restating anything
   should be left unconnected.
6. Prefer MA over RA when the second proposition adds no new reason and merely says the
   same thing differently. Prefer RA over MA when the target would be weakened without
   the source.
7. If several propositions jointly support one conclusion, emit one RA per premise with
   the same `dst`.

## Output

Return the list of relations. Return an empty list if there are none.
