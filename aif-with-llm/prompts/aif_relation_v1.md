You are annotating a dialogue under the Argument Interchange Format (AIF).

The propositional contents (I-nodes) of the dialogue have already been extracted.
Your task is to identify the **argumentative relations** between them. Each relation
becomes an S-node in AIF.

## Propositions

Each line gives the proposition index, the speaker who uttered it, and its text.

{{range .Propositions}}
[{{.Index}}] ({{.Speaker}}) {{.Text}}
{{end}}

## Relation types (choose exactly one per relation)

- **RA** (Rule Application / *Default Inference*) — the source proposition is offered
  as a **reason supporting** the target proposition. Direction matters: `src` is the
  premise, `dst` is the conclusion. Ask: "src, therefore dst" — does that read
  correctly? If "dst, therefore src" reads better, swap them.
- **CA** (Conflict Application / *Default Conflict*) — the source proposition is
  **incompatible with** the target proposition: asserting one gives a reason to reject
  the other. Direction matters: `src` is the attacker, `dst` is what is attacked.
  Use CA for explicit disagreement and for propositions that cannot both hold.
- **MA** (Preference/Rephrase Application / *Default Rephrase*) — the source
  proposition **re-expresses or reformulates** the target proposition without adding
  new inferential content. Direction: `src` is the later reformulation, `dst` is the
  content being reformulated.

## Rules

1. Only relate propositions that stand in a genuine argumentative relation **in this
   dialogue**. Topical similarity is not a relation. If two propositions are merely
   about the same subject, do not connect them.
2. A relation may hold between propositions from different speakers or from the same
   speaker.
3. Do not create a relation between a proposition and itself.
4. Relations are sparse. A typical dialogue of {{len .Propositions}} propositions has
   far fewer relations than propositions. Do not force every proposition into the graph.
5. Prefer MA over RA when the second proposition adds no new reason and merely says the
   same thing differently. Prefer RA over MA when the second proposition would be
   weakened without the first.
6. If several propositions jointly support one conclusion, emit one RA per premise with
   the same `dst`.
7. `illocutionary_force` records how the relation was performed in the dialogue: use
   `Arguing` for RA, `Disagreeing` for CA when the conflict is voiced as an explicit
   objection (otherwise `Default Illocuting`), and `Restating` for MA (or `Analysing`
   when the reformulation makes an implication explicit).

## Output

Return the list of relations. Return an empty list if there are none.
