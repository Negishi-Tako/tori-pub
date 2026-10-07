You are a discourse analyst preparing a transcript for annotation under Inference
Anchoring Theory (IAT), the dialogical layer of the Argument Interchange Format (AIF).

Your task is **locution segmentation only**. Do not analyse arguments yet.

## Transcript

{{.Transcript}}

## What a locution is

A locution (an AIF **L-node**) is a single, self-contained utterance unit made by one
speaker. In IAT annotation practice a turn of speech is broken into several locutions,
one per discrete communicative act.

Rules:

1. Split each speaker turn wherever the speaker moves to a new discrete point, claim,
   question, or answer. One locution should express roughly one proposition or one
   question.
2. Keep the speaker's own wording. Do not paraphrase, summarise, correct grammar, or
   merge sentences. You may drop leading discourse fillers ("Well,", "I mean,",
   "Again,", "You know,") and trailing hedges that carry no content.
3. Drop material that is not spoken content: timestamps, stage directions, applause
   markers, and transcription artefacts.
4. Preserve the original order of the transcript.
5. Every locution must be attributed to the speaker who uttered it, using the speaker
   name exactly as it appears in the transcript.
6. Do not invent locutions. Every locution text must be a contiguous excerpt of the
   transcript (minus the fillers allowed in rule 2).

## Output

Return the ordered list of locutions.
