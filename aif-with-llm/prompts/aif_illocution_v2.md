You are annotating a dialogue under Inference Anchoring Theory (IAT), the dialogical
layer of the Argument Interchange Format (AIF).

For each locution (L-node) below you must produce:

1. the **illocutionary force** with which the speaker puts that locution forward —
   this becomes a YA-node anchored in the locution; and
2. its **propositional content** — this becomes an I-node.

Important: you are annotating the force of a *single locution*. Forces that describe a
**relation between** locutions (Arguing, Restating, Analysing) are annotated separately,
on the transition between locutions, and must not be used here — even when the locution
obviously gives a reason for something said earlier. That inferential work is recorded
later, as an RA-node. Here, a locution that states something is `Asserting`.

## Locutions

{{range .Locutions}}
[{{.Index}}] {{.Speaker}}: {{.Text}}
{{end}}

## Illocutionary forces (choose exactly one)

- **Asserting** — the speaker puts forward a proposition as true or as their position.
  This is the default and covers the large majority of locutions, including those that
  give reasons, offer evidence, or state conclusions.
- **PureQuestioning** — a genuine information-seeking question, taking no position.
- **AssertiveQuestioning** — a question that presupposes or conveys a position the
  speaker holds.
- **RhetoricalQuestioning** — a question used to assert its own (usually negative)
  answer; no answer is expected.
- **Challenging** — the speaker demands grounds or justification for something said,
  rather than stating a position of their own.
- **Agreeing** — the locution's whole content is an explicit endorsement of what another
  speaker just said ("yes, exactly", "I agree with that").
- **Disagreeing** — the locution's whole content is an explicit rejection of what another
  speaker just said ("no, that is not right").
- **Default Illocuting** — the locution carries propositional content but none of the
  above fits.

Use `Agreeing` / `Disagreeing` only when the locution *is* the (dis)agreement move. If
the speaker states a substantive position that happens to conflict with someone else's,
that is `Asserting`; the conflict is recorded separately as a CA-node.

## Propositional content (the I-node)

Rewrite the locution as a standalone declarative proposition, following IAT's
reported-speech convention:

- Strip the performative frame: "I think the schools should reopen" → "the schools
  should reopen".
- Resolve first- and second-person reference and deixis to the named participants:
  "I have not worn one" (said by Camilla Tominey) → "Camilla Tominey has not worn a
  face mask".
- Resolve pronouns and ellipsis using the surrounding locutions.
- For questions, state the queried proposition, not the question: "who here has worn a
  mask?" → "a face mask has been worn by xxx so far". Use `xxx` for the unknown.
- Keep it lowercase, one sentence, no quotation marks, and no speaker attribution
  clause ("X says that ...") — the attribution is carried by the L-node.
- If the locution has no propositional content at all (pure procedural moves such as
  "Let's take the next question", greetings, or floor management), return an empty
  string for `proposition` and use `Default Illocuting` as the force.

## Output

Return exactly one entry per locution, in the same order, referring to each locution
by its index.
