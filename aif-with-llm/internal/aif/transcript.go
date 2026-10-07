package aif

import (
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// エピソード単位の書き起こしから、1 ノードセット分の区間を切り出す。
//
// AIFdb のサブコーパスのアーカイブには、以前はノードセットごとの素テキスト（nodesetNNN.txt）が
// 同梱されていたが、2026-10 時点ではエピソード単位の書き起こし（<shortname>.txt）1 本しか無い。
// e2e 条件の入力と「素テキストがあるか」の判定にはノードセット単位のテキストが要るので、
// ゴールドの L-node が書き起こしのどこにあるかを探し、その区間のターンを書き起こしの形のまま返す。
//
// 書き起こしの形は次のとおり（ターンの間に「Part 1」「Words: 273」等の雑音行が混ざる）:
//
//	Claire Cooper
//	[0:01:27] Hello. Even if some children do get back to school ...
//
// 返すテキストも同じ形（話者の行 + 「[時刻] 本文」の行）で、最初と最後のターンの本文は
// 区間の端で切る。切り出しは元の nodesetNNN.txt とバイト単位で一致する保証は無いので、
// Document.TextSource に由来を残す。

// transcriptTurn は書き起こしの 1 ターン。
type transcriptTurn struct {
	Speaker string
	Stamp   string
	Body    string
}

var (
	turnStampRe = regexp.MustCompile(`^\[(\d+:)?\d{1,2}:\d{2}\]\s*`)
	// noiseLineRe は書き起こしに混ざる本文でない行（パート見出し・語数・時刻だけの行・絵文字の短縮名）。
	noiseLineRe = regexp.MustCompile(`^(Part \d+|Words: \d+|\d{1,2}:\d{2}|[a-z0-9_+-]+)$`)
)

// parseTranscript は書き起こしをターンに分ける。
//
// 「[時刻] 本文」の行がターンの始まりで、その直前の空でない行が話者。1 ターンの本文は
// 段落をまたぐことがあり（時刻の付かない続きの行）、それも本文に含める。
// 次のターンの話者の行（直後が「[時刻]」の行）と雑音行は本文に含めない。
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
			// 次のターンの話者・雑音・最初のターンより前の行
		default:
			turns[len(turns)-1].Body += " " + line
		}
	}
	return turns
}

// normIndex は本文を英数字だけの小文字列に正規化したものと、その各文字の元の位置。
type normIndex struct {
	text []rune
	turn []int // 正規化後の各文字が属するターン
	off  []int // そのターンの本文中のバイト位置
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

// 発話を書き起こしの中で探すときの鍵の長さ。短すぎる鍵（"Yes" 等）は一意に定まらないので使わない。
const (
	minLocateKey = 12
	maxLocateKey = 40
	// maxLocateGap は、同じノードセットの発話とみなす一致位置どうしの最大の間隔（正規化後の文字数）。
	// 見つからない発話が 10 個以上続いても切れない長さにしてある。
	maxLocateGap = 3000
)

// ExcerptTranscript は episode（エピソードの書き起こし）から、g の L-node が出てくる区間を切り出す。
//
// 各 L-node の本文（話者を除く）を書き起こしの中で探し、一意に見つかったものの最初から最後までを区間とする。
// L-node の本文はアノテータが書き起こしの誤りを直していることがあり（"Fog fog …" 等）、
// 逐語で見つかるのは test で 1,903 個中 1,767 個である。見つからない発話は区間の内側にあれば
// 切り出しに含まれる（test では L-node の先頭 20 字の 97% が切り出しの中に現れる）。
// 1 つも見つからないときだけ ok=false を返す。located / total は区間の根拠にした L-node の数と総数で、
// 呼び出し側はこれを記録に残す（どれだけ確かな切り出しかを後から確かめられるように）。
func ExcerptTranscript(episode string, g Graph) (text string, located, total int, ok bool) {
	turns := parseTranscript(episode)
	if len(turns) == 0 {
		return "", 0, 0, false
	}
	ix := indexTurns(turns)
	hay := string(ix.text)
	// strings.Index はバイト位置を返すので、rune 位置に直す表を作る（英数字以外は落としてあるが非 ASCII の文字は残る）。
	byteToRune := make(map[int]int, len(ix.text))
	bpos := 0
	for i, r := range ix.text {
		byteToRune[bpos] = i
		bpos += len(string(r))
	}

	type span struct{ start, end int } // rune 位置、end は含まない
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
		// 本文全体がそのまま続いていれば、終わりも本文の末尾まで伸ばす。
		if full := []rune(key); start+len(full) <= len(ix.text) && string(ix.text[start:start+len(full)]) == key {
			end = start + len(full)
		}
		spans = append(spans, span{start, end})
	}
	if total == 0 || len(spans) == 0 {
		return "", 0, total, false
	}
	// 一意に見つかった位置のうち、互いに近いもののまとまりで最大のものだけを区間にする。
	// 発話の言い回しがエピソードの別の箇所とたまたま一致すると、区間が数千字先まで伸びてしまうため
	// （test の 17975 では 1 件が 8,000 字離れた箇所に一致していた）。
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
			// 文末の句読点（. ? !）は区間に含める。発話の切れ目の手がかりになるため。
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

// speakerPrefixRe は L-node の本文の先頭にある「話者:」（" : " でなく ": " で区切った形）。
var speakerPrefixRe = regexp.MustCompile(`^[A-Z][^:]{0,58}:\s+`)

// locutionBodies は L-node の本文から話者を除いた候補を、確からしい順に返す。
//
// AIFdb の L-node は「話者 : 本文」が基本だが、「話者: 本文」や、アノテータ名を前に付けた
// 「アノテータ: 話者 : 本文」もある。探索にだけ使うので、外しすぎた候補も並べてよい
// （一意に見つかったものだけを採る）。
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
