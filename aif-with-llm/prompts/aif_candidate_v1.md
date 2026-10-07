You are annotating a dialogue under the Argument Interchange Format (AIF).

The propositional contents (I-nodes) have already been extracted. Below is a list of
**candidate pairs** of propositions. For each pair, decide whether the two propositions
stand in an argumentative relation in this dialogue, and if so, which relation and in
which direction.

You are **not** asked to find relations. Every pair you should consider is already in
the list. Judge each one on its own, and give a verdict for every candidate.

## Propositions

The indices are in the order the propositions were uttered: `[0]` came first.

{{range .Propositions}}
[{{.Index}}] ({{.Force}}; {{.Speaker}}) {{.Text}}
{{end}}

## Relation types

- **RA** (*Default Inference*) — one proposition is offered as a **reason supporting**
  the other. The premise is the source, the conclusion is the target. Ask: "source,
  therefore target" — does that read correctly? If the reverse reads better, swap them.
  **RA is the one relation whose direction is logical, not temporal**: a speaker may
  state the conclusion first and the reason after.
- **CA** (*Default Conflict*) — one proposition is **incompatible with** the other:
  asserting one gives a reason to reject the other. Direction is temporal: the later
  proposition attacks the earlier one.
- **MA** (*Default Rephrase*) — one proposition **rephrases, restates or reformulates**
  the other. Direction is temporal: the later reformulation is the source.
  A passive rewording, a generalisation, a specialisation, a summary, and an **answer
  instantiating an earlier question** are all MA. Chains of successive reformulations
  ("people can meet outdoors" → "two households can meet outdoors" → "no more than eight
  people") are each recorded as their own MA.
- **none** — the two propositions stand in no argumentative relation. Being about the
  same subject is not a relation. One simply continuing the speaker's narrative, or
  adding new information without supporting, attacking or restating anything, is
  **none**. Mere repetition of identical wording — a host reading a claim back, or a
  speaker quoting another to hand over the floor — is also **none**, not MA.

## How many of these candidates are actually related

In annotated dialogue, of all candidate pairs within {{.MaxDistance}} positions of each
other, **roughly one in six stands in a relation**. Adjacent pairs are much more likely
to be related (about half of them are); pairs three or four apart much less so (about
one in twelve).

**`none` is the most common verdict and you should use it freely.** But do not let the
proportion drive individual verdicts: judge each pair on the content, then let the
numbers fall where they fall. Neither aim for a target count nor spread relations evenly.

## Direction

For each relation, say whether the **earlier** or the **later** proposition is the
source. For CA and MA the later one is almost always the source. For RA it depends on
which is the reason and which is the conclusion — the reason is the source whether it
was uttered first or second.

## Confidence

State `high`, `medium` or `low` for each relation you record, judging the whole claim —
that the relation holds **and** that its type and direction are right.
Use `low` rather than withholding a relation you are unsure about; use `none` only when
you actually think there is no relation.

## Output

Return exactly one verdict per candidate, in the same order, referring to each candidate
by its index.

## Candidates

{{range .Candidates}}
({{.Index}}) [{{.A}}] vs [{{.B}}] — {{.Distance}} apart
  earlier [{{.A}}]: {{.AText}}
  later   [{{.B}}]: {{.BText}}
{{end}}
