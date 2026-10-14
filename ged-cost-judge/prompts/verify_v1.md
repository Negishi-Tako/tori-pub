You check an edited discussion against the edits that were requested.

You receive the original discussion (numbered utterances, its propositions with ids, and the relations between them), the edited discussion, and the numbered list of requested edits.

Judge from the text of the edited discussion only, as a careful reader would:

- `edits`: for each requested edit, `realized` is true if a reader of the edited discussion can see that the change happened. For relation edits, the relation itself must be clearly signalled or clearly absent in the text (for example, a removed support relation must no longer be implied by "because", "so", "with that", or by placing one claim as the reason for the other). An edit that is only hinted at, or that leaves the old structure readable, is not realized.
- `unintended_change`: true if the edited discussion changes any claim or relation that no edit asked for (minimal changes to pronouns or connectives that keep every other claim and relation intact are fine). Speaker names may have been swapped on purpose when stated below; that is not an unintended change. Explain briefly in `note`.
- `naturalness`: how natural and coherent the edited discussion reads, from "1" (broken or confusing) to "5" (fully natural).
