package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
)

// ---------- Fetching ----------

func fetch(ctx context.Context, logger *slog.Logger, corpus string, n int, dir string) error {
	c := aif.NewClient(aif.ClientConfig{})
	details, err := c.CorpusDetails(ctx, corpus)
	if err != nil {
		return err
	}
	subs, err := c.Subcorpora(ctx, corpus)
	if err != nil {
		return err
	}
	if len(subs) == 0 {
		subs = []aif.Corpus{{Shortname: corpus, Title: details.Title}}
	}
	if n > 0 && n < len(subs) {
		subs = subs[:n]
	}
	meta := map[string]any{
		"corpus":       corpus,
		"title":        details.Title,
		"reference":    details.Reference,
		"licence":      details.Licence,
		"fetched_at":   time.Now().Format(time.RFC3339),
		"subcorpora":   subs,
		"source":       aif.DefaultCorporaBaseURL,
		"nodeset_urls": aif.DefaultNodesetBaseURL + "/json/<nodesetID>",
	}
	// files fingerprints the fetched inputs. AIFdb's state can change between fetches (broken archives,
	// a changed raw-text format), so this lets us later match up which inputs an experiment used.
	type fileEntry struct {
		Subcorpus     string `json:"subcorpus"`
		GraphSHA256   string `json:"graph_sha256"`
		TextSHA256    string `json:"text_sha256"`
		GraphSource   string `json:"graph_source"`
		TextSource    string `json:"text_source"`
		TextLocated   int    `json:"text_located,omitempty"`
		TextLocutions int    `json:"text_locutions,omitempty"`
	}
	files := map[string]fileEntry{}
	sources := map[string]int{}
	total := 0
	var noTextAll []string
	for _, s := range subs {
		docs, noText, err := c.FetchSubcorpus(ctx, s.Shortname)
		if err != nil {
			return err
		}
		for _, d := range docs {
			if err := writeJSON(filepath.Join(dir, "nodeset"+d.NodesetID+".json"), d); err != nil {
				return err
			}
			gb, err := json.Marshal(d.Graph)
			if err != nil {
				return fmt.Errorf("aif-eval: %s の指紋: %w", d.NodesetID, err)
			}
			files[d.NodesetID] = fileEntry{
				Subcorpus: s.Shortname, GraphSHA256: sha256Hex(gb), TextSHA256: sha256Hex([]byte(d.Text)),
				GraphSource: d.GraphSource, TextSource: d.TextSource,
				TextLocated: d.TextLocated, TextLocutions: d.TextLocutions,
			}
			sources["graph:"+d.GraphSource]++
			sources["text:"+d.TextSource]++
		}
		total += len(docs)
		noTextAll = append(noTextAll, noText...)
		logger.Info("サブコーパスを取得した", "subcorpus", s.Shortname, "nodesets", len(docs), "素テキスト無し（e2e 不可）", len(noText))
	}
	meta["nodesets"] = total
	meta["files"] = files
	meta["sources"] = sources
	meta["no_text"] = noTextAll
	if err := writeJSON(filepath.Join(dir, "corpus.json"), meta); err != nil {
		return err
	}
	logger.Info("取得完了", "nodesets", total, "dir", dir, "sources", fmt.Sprint(sources))
	return nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// ---------- Loading and sampling ----------

// loadDocs loads the fetched nodesets in ID order.
func loadDocs(dir string) ([]aif.Document, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("aif-eval: %s の読み込み: %w", dir, err)
	}
	var out []aif.Document
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), "nodeset") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("aif-eval: %s の読み込み: %w", e.Name(), err)
		}
		var d aif.Document
		if err := json.Unmarshal(b, &d); err != nil {
			return nil, fmt.Errorf("aif-eval: %s の解析: %w", e.Name(), err)
		}
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].NodesetID < out[j].NodesetID })
	return out, nil
}

// sampleDocs deterministically picks the nodesets that match the criteria.
// The seed is fixed so that the same arguments give the same sample (reproducibility).
func sampleDocs(docs []aif.Document, opts runOptions) []aif.Document {
	var inSplit map[string]bool
	if sub, found := splits[opts.Split]; found {
		inSplit = map[string]bool{}
		for _, s := range sub {
			inSplit[s] = true
		}
	}
	var ok []aif.Document
	for _, d := range docs {
		if inSplit != nil && !inSplit[d.Corpus] {
			continue
		}
		g, _ := aif.Normalize(d.Graph)
		n := len(g.NodesOfType(aif.TypeL))
		if n < opts.MinLoc || n > opts.MaxLoc {
			continue
		}
		if len(g.Relations()) == 0 {
			continue // a nodeset with no relations cannot be used to evaluate relations
		}
		// Do not select on whether raw text exists. Only the e2e condition uses text, and if the sample
		// varied by condition the paired comparison across conditions would break. e2e on a nodeset
		// without text fails in build and is kept as an error row in the results.
		ok = append(ok, d)
	}
	// Shuffle with a deterministic pseudo-random order, then take the head.
	type keyed struct {
		key float64
		doc aif.Document
	}
	ks := make([]keyed, len(ok))
	for i, d := range ok {
		h := opts.Seed
		for _, r := range d.NodesetID {
			h = h*1099511628211 ^ int64(r)
		}
		ks[i] = keyed{key: math.Abs(float64(h%1000003)) / 1000003.0, doc: d}
	}
	// Break ties on equal keys (hash collisions) by ID so the order does not vary between runs.
	// sort.Slice is not stable, so two colliding entries straddling the sample boundary could change the sample.
	sort.Slice(ks, func(i, j int) bool {
		if ks[i].key != ks[j].key {
			return ks[i].key < ks[j].key
		}
		return ks[i].doc.NodesetID < ks[j].doc.NodesetID
	})
	if opts.Sample > 0 && opts.Sample < len(ks) {
		ks = ks[:opts.Sample]
	}
	out := make([]aif.Document, len(ks))
	for i, k := range ks {
		out[i] = k.doc
	}
	sort.Slice(out, func(i, j int) bool { return out[i].NodesetID < out[j].NodesetID })
	return out
}

// writeJSON writes v to path as indented JSON (creating parent directories).
func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return fmt.Errorf("aif-eval: %s の書き出し: %w", path, err)
	}
	return writeFile(path, b)
}

// writeFile writes b to path (creating parent directories).
func writeFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("aif-eval: %s の作成: %w", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("aif-eval: %s の保存: %w", path, err)
	}
	return nil
}
