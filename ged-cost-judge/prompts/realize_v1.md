You edit a discussion so that its argument structure changes in exactly the ways requested, and in no other way.

You receive the discussion as numbered utterances, its propositions (claims) with the utterance that states each one, the relations between propositions, and a list of edits. Make every requested edit in the text.

Rules:

- Change only what an edit needs. You may also make minimal connective fixes (pronouns, connectives, references to an earlier utterance) in the utterance directly before or after an edited utterance, so the discussion still reads naturally. A connective fix must not change any claim or relation.
- Do not change the meaning of any proposition or relation that is not named in an edit. Keep the wording of untouched utterances.
- Return only the utterances you change in `changes`:
  - `op` "replace": `utterance` is the index of the utterance to rewrite; `text` is its full new text; `speaker` is its speaker (usually unchanged).
  - `op` "insert_after": add a new utterance after the utterance with index `utterance` (use -1 to insert at the very beginning); give `speaker` and `text`.
  - `op` "delete": remove the utterance with index `utterance` (leave `text` and `speaker` empty).
  - `role` is "edit" for a change that carries out an edit and "connective" for a minimal connective fix.
  - Indices always refer to the original numbering.
- In `propositions`, give the new self-contained text of every proposition whose content an edit changes or adds (paraphrased, changed, negated or added propositions). Leave other propositions out.
