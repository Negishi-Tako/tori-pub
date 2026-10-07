package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/evaluation"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// API を叩かずに、取得済みデータ → 基準線の比較 → 集計の書き出しまでを通す。
func TestReportPipelineWithoutAPI(t *testing.T) {
	b, err := os.ReadFile("../../testdata/aif/fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	g, err := aif.ParseNodeset("fixture", b)
	if err != nil {
		t.Fatal(err)
	}
	dataDir, outDir := t.TempDir(), t.TempDir()
	if err := writeJSON(filepath.Join(dataDir, "nodeset-fixture.json"), aif.Document{NodesetID: "fixture", Graph: g}); err != nil {
		t.Fatal(err)
	}
	docs, err := loadDocs(dataDir)
	if err != nil || len(docs) != 1 {
		t.Fatalf("loadDocs: %v (%d docs)", err, len(docs))
	}
	if err := goldcheck(dataDir); err != nil {
		t.Fatal(err)
	}

	gold, _ := aif.Normalize(docs[0].Graph)
	pred := aif.BaselineGraph(gold.ID, locutionsOf(gold))
	var lines []string
	for _, m := range evaluation.AIFMetrics {
		opts := evaluation.DefaultAIFCompareOptions()
		opts.Metric = m
		c, err := evaluation.CompareAIF(context.Background(), pred, gold, &llm.HashEmbedder{D: 32}, opts)
		if err != nil {
			t.Fatal(err)
		}
		c.Condition = condBaseline
		line, _ := json.Marshal(Result{NodesetID: gold.ID, Condition: condBaseline, Comparison: c})
		lines = append(lines, string(line))
	}
	if err := writeFile(filepath.Join(outDir, "results.jsonl"), []byte(strings.Join(lines, "\n"))); err != nil {
		t.Fatal(err)
	}
	if err := report(outDir); err != nil {
		t.Fatalf("report: %v", err)
	}
	summary, err := os.ReadFile(filepath.Join(outDir, "summary.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"関係の micro 平均", "baseline · ged/0.3 / uniform"} {
		if !strings.Contains(string(summary), want) {
			t.Errorf("summary.md に %q が無い", want)
		}
	}
	for _, f := range []string{"results.csv", "stats.md", "evidence.md"} {
		if _, err := os.Stat(filepath.Join(outDir, f)); err != nil {
			t.Errorf("%s が書き出されていない: %v", f, err)
		}
	}
}

// 基準線の入力は、TA の連鎖順に並んだ「話者 : 本文」の本文部分であること。
func TestLocutionsOfStripsSpeakerAndFollowsTransitions(t *testing.T) {
	g := aif.BaselineGraph("x", []aif.LocutionInput{{Speaker: "Alice", Text: "first"}, {Speaker: "Bob", Text: "second"}})
	g.Nodes = append(g.Nodes[len(g.Nodes)/2:], g.Nodes[:len(g.Nodes)/2]...) // JSON の並びを崩す
	got := locutionsOf(g)
	if len(got) != 2 || got[0] != (aif.LocutionInput{Speaker: "Alice", Text: "first"}) || got[1].Text != "second" {
		t.Errorf("locutionsOf = %+v", got)
	}
}
