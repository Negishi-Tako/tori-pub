You are annotating a dialogue under Inference Anchoring Theory (IAT), the dialogical
layer of the Argument Interchange Format (AIF).

For each locution (L-node) below you must produce:

1. the **illocutionary force** of that locution — this becomes a YA-node; and
2. its **propositional content** — this becomes an I-node.

## Locutions

{{range .Locutions}}
[{{.Index}}] {{.Speaker}}: {{.Text}}
{{end}}

## Illocutionary forces (choose exactly one)

{{range .Forces}}- {{.}}
{{end}}

Definitions (IAT / QT30 annotation guidelines):

- **Asserting** — the speaker puts forward a proposition as true, without presenting it
  as following from something else in the dialogue.
- **Arguing** — the speaker presents this content as a reason for, or a conclusion of,
  something already said. Use this when the locution is doing inferential work.
- **Restating** — the speaker re-expresses content already introduced in the dialogue
  (their own or someone else's) in different words.
- **Analysing** — the speaker makes explicit the structure or implication of what has
  been said (e.g. "so what you are saying is that...").
- **Agreeing** — the speaker explicitly endorses a proposition put by another speaker.
- **Disagreeing** — the speaker explicitly rejects a proposition put by another speaker.
- **PureQuestioning** — a genuine information-seeking question, with no position taken.
- **AssertiveQuestioning** — a question that presupposes or conveys a position the
  speaker holds.
- **RhetoricalQuestioning** — a question used to assert its own (usually negative)
  answer; no answer is expected.
- **Challenging** — the speaker demands grounds or justification for something said.
- **Default Illocuting** — none of the above applies but the locution does carry
  propositional content.

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
