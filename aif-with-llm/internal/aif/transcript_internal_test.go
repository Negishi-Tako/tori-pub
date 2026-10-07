package aif

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// 2026-10 時点の AIFdb のエピソード書き起こしと同じ形（CRLF・雑音行・段落をまたぐターン）。
// 話者と本文は架空のもの（コーパス本文はリポジトリに含めない）。
const episodeFixture = "Part 1\r\nWords: 273\r\nAlex Moore\r\n[0:01:27] Hello. Even if the new bus timetable starts before the end of the spring season, most riders are unlikely to notice any change.\r\n\r\nRobin Hale\r\n[0:01:58] Sam, let's start with you. Some libraries are opening late on the first of May.\r\n \r\nAnd some are not opening late at all this season.\r\nwhite_check_mark\r\neyes\r\n\r\n7:37\r\nPart 2\r\nWords: 298\r\nSam Carter\r\n[0:02:07] It is an uneven pattern, Robin. The council announced a date to residents without any consultation.\r\n\r\nJo Baker\r\n[0:03:05] I have three library cards and one of them can be renewed next week.\r\n"

func TestParseTranscriptJoinsParagraphsAndDropsNoise(t *testing.T) {
	turns := parseTranscript(episodeFixture)
	if len(turns) != 4 {
		t.Fatalf("ターンは 4 つのはず: %d %+v", len(turns), turns)
	}
	fb := turns[1]
	if fb.Speaker != "Robin Hale" || !strings.HasSuffix(fb.Body, "And some are not opening late at all this season.") {
		t.Errorf("段落をまたぐ本文を 1 ターンにまとめるはず: %+v", fb)
	}
	for _, noise := range []string{"white_check_mark", "Part 2", "Words: 298", "7:37", "Sam Carter"} {
		if strings.Contains(fb.Body, noise) {
			t.Errorf("雑音行・次の話者 %q を本文に入れてはいけない: %q", noise, fb.Body)
		}
	}
	if turns[2].Speaker != "Sam Carter" {
		t.Errorf("雑音行の後でも話者を拾うはず: %+v", turns[2])
	}
}

func excerptGraph(texts ...string) Graph {
	g := Graph{ID: "e"}
	for i, t := range texts {
		g.Nodes = append(g.Nodes, Node{ID: "L" + string(rune('0'+i)), Type: TypeL, Text: t})
	}
	return g
}

// 区間は最初の発話の頭から最後の発話の末尾までで、ターンの形（話者・時刻）を保つこと。
func TestExcerptTranscriptCutsSpan(t *testing.T) {
	g := excerptGraph(
		"Robin Hale : Some libraries are opening late on the first of May",
		"Robin Hale : And some are not opening late at all this season",
		"Sam Carter : It is an uneven pattern, Robin",
	)
	got, located, total, ok := ExcerptTranscript(episodeFixture, g)
	if !ok || located != 3 || total != 3 {
		t.Fatalf("3 発話とも見つかるはず: ok=%v %d/%d", ok, located, total)
	}
	want := "Robin Hale\n[0:01:58] Some libraries are opening late on the first of May. And some are not opening late at all this season.\n\nSam Carter\n[0:02:07] It is an uneven pattern, Robin."
	if got != want {
		t.Errorf("切り出しが違う:\n got: %q\nwant: %q", got, want)
	}
	if strings.Contains(got, "Alex Moore") || strings.Contains(got, "Jo Baker") {
		t.Errorf("区間の外のターンを含めてはいけない: %q", got)
	}
}

// 「話者: 本文」「アノテータ: 話者 : 本文」の形でも見つけること。
func TestExcerptTranscriptHandlesSpeakerPrefixes(t *testing.T) {
	g := excerptGraph("Jo Baker: I have three library cards", "Annotator: Sam Carter : The council announced a date to residents")
	if _, located, _, ok := ExcerptTranscript(episodeFixture, g); !ok || located != 2 {
		t.Errorf("話者の書き方が違っても見つかるはず: ok=%v located=%d", ok, located)
	}
}

// 1 発話だけが遠く離れた箇所に一致しても、区間をそこまで伸ばさないこと。
func TestExcerptTranscriptIgnoresDistantOutlier(t *testing.T) {
	filler := strings.Repeat("the panel went on to discuss many unrelated matters at length. ", 80) // 区切りより十分長い
	episode := "A\n[0:00:01] Opening remarks about library opening plans today.\n\nB\n[0:00:10] " + filler +
		"\n\nC\n[0:09:00] Queues make people feel rushed in shops. People should decide for themselves.\n"
	g := excerptGraph("A : Opening remarks about library opening plans", "C : Queues make people feel rushed in shops", "C : People should decide for themselves")
	got, located, _, ok := ExcerptTranscript(episode, g)
	if !ok || located != 2 {
		t.Fatalf("近い 2 発話だけを根拠にするはず: ok=%v located=%d", ok, located)
	}
	if strings.Contains(got, "Opening") || strings.Contains(got, "unrelated") {
		t.Errorf("離れた一致まで区間を伸ばしてはいけない: %q", got)
	}
}

func TestExcerptTranscriptFailsWhenNothingFound(t *testing.T) {
	if _, _, _, ok := ExcerptTranscript(episodeFixture, excerptGraph("X : nothing like this appears anywhere in it")); ok {
		t.Error("1 発話も見つからなければ ok=false のはず")
	}
}

// 2026-10 時点の壊れたアーカイブ（JSON が 0 バイト・素テキストはエピソード単位）でも、
// JSON は www.aifdb.org/json/<id> から取り直し、素テキストは書き起こしから切り出すこと。
// 健全なアーカイブ（中身のある JSON・nodesetNNN.txt）はそのまま使うこと。
func TestReadArchiveFallsBackForBrokenArchive(t *testing.T) {
	nodeset := `{"nodes":[{"nodeID":"1","type":"L","text":"Robin Hale : Some libraries are opening late on the first of May"},` +
		`{"nodeID":"2","type":"L","text":"Sam Carter : It is an uneven pattern, Robin"},` +
		`{"nodeID":"3","type":"TA","text":"Default Transition"}],` +
		`"edges":[{"fromID":"1","toID":"3"},{"fromID":"3","toID":"2"}],"locutions":[]}`
	var hits []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, r.URL.Path)
		if r.URL.Path == "/json/100" {
			_, _ = w.Write([]byte(nodeset))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	archive := tarGz(t, map[string]string{
		"nodeset100.json":    "", // 壊れている
		"nodeset200.json":    nodeset,
		"nodeset200.txt":     "Robin Hale\n[0:00:01] original per-nodeset text",
		"cutietestrun_x.txt": episodeFixture,
	})
	c := NewClient(ClientConfig{NodesetBaseURL: srv.URL, Interval: time.Millisecond})
	docs, noText, err := c.readArchive(context.Background(), "cutietestrun_x", bytes.NewReader(archive))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 || len(noText) != 0 {
		t.Fatalf("2 件とも取れるはず: docs=%d noText=%v", len(docs), noText)
	}
	byID := map[string]Document{}
	for _, d := range docs {
		byID[d.NodesetID] = d
	}
	if d := byID["100"]; d.GraphSource != SourceNodesetJSON || d.TextSource != SourceEpisodeExcerpt || len(d.Graph.Nodes) != 3 ||
		!strings.HasPrefix(d.Text, "Robin Hale\n[0:01:58] Some libraries") || d.TextLocated != 2 {
		t.Errorf("壊れた JSON は取り直し、テキストは書き起こしから切り出すはず: %+v", d)
	}
	if d := byID["200"]; d.GraphSource != SourceArchive || d.TextSource != SourceArchive || d.Text != "Robin Hale\n[0:00:01] original per-nodeset text" {
		t.Errorf("健全なアーカイブの中身はそのまま使うはず: %+v", d)
	}
	if len(hits) != 1 || hits[0] != "/json/100" {
		t.Errorf("取り直すのは壊れた 1 件だけのはず: %v", hits)
	}
}

// 壊れた JSON を取り直せないときは、黙って落とさずエラーにすること（標本が変わるため）。
func TestReadArchiveErrorsWhenRefetchFails(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()
	c := NewClient(ClientConfig{NodesetBaseURL: srv.URL, Interval: time.Millisecond})
	if _, _, err := c.readArchive(context.Background(), "s", bytes.NewReader(tarGz(t, map[string]string{"nodeset100.json": ""}))); err == nil {
		t.Error("取り直しに失敗したらエラーのはず")
	}
}

func tarGz(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// 書き起こしが 1 本も無いアーカイブ（2026-10 時点の cutietestrun18June2020）でも、
// ノードセットは落とさずに返し、素テキストが無いことを記録すること。
func TestReadArchiveKeepsNodesetsWithoutText(t *testing.T) {
	nodeset := `{"nodes":[{"nodeID":"1","type":"L","text":"A : hello there everyone in the room"}],"edges":[],"locutions":[]}`
	c := NewClient(ClientConfig{Interval: time.Millisecond})
	docs, noText, err := c.readArchive(context.Background(), "s", bytes.NewReader(tarGz(t, map[string]string{"nodeset300.json": nodeset})))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || len(noText) != 1 || noText[0] != "300" || docs[0].Text != "" || docs[0].TextSource != "" {
		t.Errorf("テキストが無くてもノードセットは返し、no_text に記録するはず: docs=%+v noText=%v", docs, noText)
	}
}
