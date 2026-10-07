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

// ---------- 取得 ----------

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
	// files は取得した入力の指紋。AIFdb の状態は取得の時点で変わりうる（アーカイブが壊れている・
	// 素テキストの形が変わる）ので、どの入力で実験したかを後から突き合わせられるようにする。
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

// ---------- 読み込みと標本抽出 ----------

// loadDocs は取得済みのノードセットを ID 順に読み込む。
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

// sampleDocs は条件に合うノードセットを決定的に選ぶ。
// 乱数種を固定するのは、同じ引数なら同じ標本になるようにするため（再現性）。
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
			continue // 関係が 1 本も無いノードセットは関係の評価に使えない
		}
		// 素テキストの有無では選ばない。テキストを使うのは e2e 条件だけで、条件によって標本が
		// 変わると条件間の対応のある比較が崩れるため。テキストの無いノードセットの e2e は
		// build がエラーにし、結果に error 行として残る。
		ok = append(ok, d)
	}
	// 決定的な擬似乱数で並べ替えてから先頭を取る。
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
	// 鍵が同じ（ハッシュの衝突）ときに順序が実行ごとに揺れないよう、ID で決着をつける。
	// sort.Slice は安定でないので、衝突した 2 件が標本の境界にまたがると標本が変わりうる。
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

// writeJSON は v をインデント付き JSON で path に書き出す（親ディレクトリも作る）。
func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return fmt.Errorf("aif-eval: %s の書き出し: %w", path, err)
	}
	return writeFile(path, b)
}

// writeFile は path に b を書き出す（親ディレクトリも作る）。
func writeFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("aif-eval: %s の作成: %w", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("aif-eval: %s の保存: %w", path, err)
	}
	return nil
}
