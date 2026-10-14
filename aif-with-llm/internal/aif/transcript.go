package aif

import (
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// Excerpts one nodeset's span from a per-episode transcript.
//
// AIFdb subcorpus archives used to bundle per-nodeset raw text (nodesetNNN.txt),
// but as of 2026-10 there is only one per-episode transcript (<shortname>.txt).
// The e2e condition's input and the "has raw text" check need per-nodeset text, so we
// find where the gold L-nodes occur in the transcript and return the turns in that span, in transcript form.
//
// The transcript looks like this (noise lines such as "Part 1" and "Words: 273" appear between turns):
//
//	Claire Cooper
//	[0:01:27] Hello. Even if some children do get back to school ...
//
// The returned text has the same form (a speaker line + a "[time] text" line), with the text of the
// first and last turns cut at the span edges. The excerpt is not guaranteed to match the original
// nodesetNNN.txt byte for byte, so its origin is recorded in Document.TextSource.

// transcriptTurn is one turn of a transcript.
type transcriptTurn struct {
	Speaker string
	Stamp   string
	Body    string
}

var (
	turnStampRe = regexp.MustCompile(`^\[(\d+:)?\d{1,2}:\d{2}\]\s*`)
	// noiseLineRe matches non-text lines mixed into transcripts (part headings, word counts, time-only lines, emoji short names).
	noiseLineRe = regexp.MustCompile(`^(Part \d+|Words: \d+|\d{1,2}:\d{2}|[a-z0-9_+-]+)$`)
)

// parseTranscript splits a transcript into turns.
//
// A "[time] text" line starts a turn, and the non-empty line just before it is the speaker. A turn's text
// may span paragraphs (continuation lines without a time), and those are included in the text.
// The next turn's speaker line (one directly followed by a "[time]" line) and noise lines are not included.
func parseTranscript(text string) []transcriptTurn {
	var lines []string
	for _, raw := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		if line := strings.TrimSpace(raw); line != "" {
			lines = append(lines, line)
		}
	}
	isStamp := func(i int) bool { return i < len(lines) && turnStampRe.MatchString(lines[i]) }
	var turns []transcriptTurn
	for i, line := range lines {
		switch {
		case isStamp(i):
			speaker := ""
			if i > 0 && !isStamp(i-1) {
				speaker = lines[i-1]
			}
			loc := turnStampRe.FindStringIndex(line)
			turns = append(turns, transcriptTurn{
				Speaker: speaker,
				Stamp:   strings.TrimSpace(line[:loc[1]]),
				Body:    line[loc[1]:],
			})
		case isStamp(i + 1), noiseLineRe.MatchString(line), len(turns) == 0:
			// the next turn's speaker, noise, or lines before the first turn
		default:
			turns[len(turns)-1].Body += " " + line
		}
	}
	return turns
}

// normIndex is the text normalized to lowercase alphanumerics only, along with each character's original position.
type normIndex struct {
	text []rune
	turn []int // turn each normalized character belongs to
	off  []int // byte offset within that turn's text
}

func normalizeForSearch(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

func indexTurns(turns []transcriptTurn) normIndex {
	var ix normIndex
	for ti, t := range turns {
		for off, r := range t.Body {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				ix.text = append(ix.text, unicode.ToLower(r))
				ix.turn = append(ix.turn, ti)
				ix.off = append(ix.off, off)
			}
		}
	}
	return ix
}

// Key length for finding utterances in the transcript. Keys that are too short ("Yes", etc.) are not unique, so they are not used.
const (
	minLocateKey = 12
	maxLocateKey = 40
	// maxLocateGap is the maximum gap (in normalized characters) between match positions considered part of the same nodeset.
	// It is long enough not to break even if 10 or more consecutive utterances are not found.
	maxLocateGap = 3000
)

// ExcerptTranscript excerpts the span of episode (the episode transcript) in which g's L-nodes occur.
//
// It looks up each L-node's text (without the speaker) in the transcript and takes the span from the first to the last unique match.
// Annotators sometimes corrected transcription errors in L-node text ("Fog fog …", etc.),
// so 1,767 of 1,903 are found verbatim in test. Utterances not found are included in the excerpt
// if they fall inside the span (in test, 97% of the first 20 characters of L-nodes appear in the excerpt).
// ok=false is returned only when none are found. located / total are the number of L-nodes the span is based on
// and the total; callers record them (so how reliable the excerpt is can be checked later).
func ExcerptTranscript(episode string, g Graph) (text string, located, total int, ok bool) {
	turns := parseTranscript(episode)
	if len(turns) == 0 {
		return "", 0, 0, false
	}
	ix := indexTurns(turns)
	hay := string(ix.text)
	// strings.Index returns byte offsets, so build a table to convert to rune offsets (non-alphanumerics are dropped, but non-ASCII characters remain).
	byteToRune := make(map[int]int, len(ix.text))
	bpos := 0
	for i, r := range ix.text {
		byteToRune[bpos] = i
		bpos += len(string(r))
	}

	type span struct{ start, end int } // rune offsets; end is exclusive
	var spans []span
	locs := g.NodesOfType(TypeL)
	total = len(locs)
	for _, n := range locs {
		key, probe, found := "", "", false
		for _, body := range locutionBodies(n) {
			key = normalizeForSearch(body)
			if len([]rune(key)) < minLocateKey {
				continue
			}
			probe = key
			if r := []rune(key); len(r) > maxLocateKey {
				probe = string(r[:maxLocateKey])
			}
			if strings.Count(hay, probe) == 1 {
				found = true
				break
			}
		}
		if !found {
			continue
		}
		start := byteToRune[strings.Index(hay, probe)]
		end := start + len([]rune(probe))
		// If the whole text continues as-is, extend the end to the end of the text as well.
		if full := []rune(key); start+len(full) <= len(ix.text) && string(ix.text[start:start+len(full)]) == key {
			end = start + len(full)
		}
		spans = append(spans, span{start, end})
	}
	if total == 0 || len(spans) == 0 {
		return "", 0, total, false
	}
	// Of the unique match positions, use only the largest cluster of positions close to each other as the span.
	// If an utterance's wording happens to match elsewhere in the episode, the span would stretch thousands of characters
	// (in test 17975, one item matched a place 8,000 characters away).
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	bestLo, bestHi := 0, 1
	for lo := 0; lo < len(spans); {
		hi, e := lo+1, spans[lo].end
		for hi < len(spans) && spans[hi].start-e <= maxLocateGap {
			e = max(e, spans[hi].end)
			hi++
		}
		if hi-lo > bestHi-bestLo {
			bestLo, bestHi = lo, hi
		}
		lo = hi
	}
	cluster := spans[bestLo:bestHi]
	located = len(cluster)
	s, e := cluster[0].start, cluster[0].end
	for _, sp := range cluster {
		e = max(e, sp.end)
	}

	ts, os := ix.turn[s], ix.off[s]
	te := ix.turn[e-1]
	lastRuneOff := ix.off[e-1]
	var b strings.Builder
	for ti := ts; ti <= te; ti++ {
		t := turns[ti]
		body := t.Body
		lo, hi := 0, len(body)
		if ti == ts {
			lo = os
		}
		if ti == te {
			hi = lastRuneOff + len(string([]rune(body[lastRuneOff:])[0]))
			// Include sentence-final punctuation (. ? !) in the span, as a hint for utterance boundaries.
			for hi < len(body) && strings.ContainsRune(".?!\"'", rune(body[hi])) {
				hi++
			}
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(t.Speaker)
		b.WriteString("\n")
		b.WriteString(t.Stamp)
		b.WriteString(" ")
		b.WriteString(strings.TrimSpace(body[lo:hi]))
	}
	return b.String(), located, total, true
}

// speakerPrefixRe matches a leading "speaker:" in L-node text (separated by ": " rather than " : ").
var speakerPrefixRe = regexp.MustCompile(`^[A-Z][^:]{0,58}:\s+`)

// locutionBodies returns candidate texts of an L-node with the speaker removed, most likely first.
//
// AIFdb L-nodes are usually "speaker : text", but there are also "speaker: text" and
// "annotator: speaker : text" with the annotator's name prefixed. These are used only for searching,
// so over-stripped candidates may be listed too (only unique matches are taken).
func locutionBodies(n Node) []string {
	body := n.Text
	if sp := SpeakerOf(n); sp != "" && strings.HasPrefix(body, sp) {
		body = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(body[len(sp):]), ":"))
	}
	out := []string{body}
	for range 2 {
		loc := speakerPrefixRe.FindStringIndex(body)
		if loc == nil {
			break
		}
		body = body[loc[1]:]
		out = append(out, body)
	}
	return out
}
