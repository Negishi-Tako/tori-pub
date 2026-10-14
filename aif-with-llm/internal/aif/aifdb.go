package aif

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"
	"time"
)

// Client for fetching from AIFdb corpora (https://corpora.aifdb.org/).
//
// The server is a public research service, so use concurrency 1 and space out requests. Bulk fetching uses
// one archive (tar.gz) per subcorpus instead of hitting nodesets one by one.
//
// However, archives are sometimes broken (as of 2026-10, nodeset JSON files are 0 bytes and the
// raw text is a single per-episode file rather than per-nodeset). Only then is each broken
// JSON refetched from AIFdb proper (www.aifdb.org/json/<id>), and the raw text is
// excerpted from the episode transcript (ExcerptTranscript).
const (
	DefaultCorporaBaseURL = "https://corpora.aifdb.org"
	DefaultNodesetBaseURL = "https://www.aifdb.org"
	DefaultInterval       = 2 * time.Second
)

// ClientConfig configures the AIFdb client.
type ClientConfig struct {
	BaseURL string
	// NodesetBaseURL is where per-nodeset JSON (<NodesetBaseURL>/json/<id>) is fetched from.
	NodesetBaseURL string
	Interval       time.Duration
	UserAgent      string
	HTTP           *http.Client
}

// Client is an HTTP client for AIFdb corpora.
type Client struct {
	base     string
	nodesets string
	interval time.Duration
	ua       string
	http     *http.Client
	last     time.Time
}

// NewClient creates a client.
func NewClient(cfg ClientConfig) *Client {
	base := cfg.BaseURL
	if base == "" {
		base = DefaultCorporaBaseURL
	}
	iv := cfg.Interval
	if iv <= 0 {
		iv = DefaultInterval
	}
	hc := cfg.HTTP
	if hc == nil {
		hc = &http.Client{Timeout: 180 * time.Second}
	}
	ua := cfg.UserAgent
	if ua == "" {
		ua = "aif-with-llm/0.1 (AIF annotation study)"
	}
	ns := cfg.NodesetBaseURL
	if ns == "" {
		ns = DefaultNodesetBaseURL
	}
	return &Client{base: strings.TrimRight(base, "/"), nodesets: strings.TrimRight(ns, "/"), interval: iv, ua: ua, http: hc}
}

// Corpus is metadata for a corpus or subcorpus.
type Corpus struct {
	CorpusID     int    `json:"corpusID"`
	Shortname    string `json:"shortname"`
	Title        string `json:"title"`
	Description  string `json:"description"`
	NodesetCount int    `json:"nodesetCount"`
	Reference    string `json:"reference"`
	Licence      string `json:"licence"`
}

// CorpusDetails returns metadata for one corpus (used for attribution).
func (c *Client) CorpusDetails(ctx context.Context, shortname string) (Corpus, error) {
	var out Corpus
	q := url.Values{"shortname": {shortname}}
	if err := c.getJSON(ctx, "/api/getCorpusDetails.php", q, &out); err != nil {
		return Corpus{}, fmt.Errorf("aif: コーパス %s のメタ情報: %w", shortname, err)
	}
	return out, nil
}

// Subcorpora returns the list of subcorpora under a corpus.
func (c *Client) Subcorpora(ctx context.Context, shortname string) ([]Corpus, error) {
	var out struct {
		Subcorpora []Corpus `json:"subcorpora"`
	}
	q := url.Values{"shortname": {shortname}}
	if err := c.getJSON(ctx, "/api/getCorpusSubcorpora.php", q, &out); err != nil {
		return nil, fmt.Errorf("aif: コーパス %s のサブコーパス: %w", shortname, err)
	}
	return out.Subcorpora, nil
}

// Document is one nodeset (= a gold AIF graph) together with its raw text.
//
// Text is the transcript before annotation (speaker[time] text...).
// That is the input given to the LLM; Graph is used only for scoring.
type Document struct {
	NodesetID string `json:"nodeset_id"`
	Corpus    string `json:"corpus"`
	Text      string `json:"text"`
	Graph     Graph  `json:"graph"`
	// GraphSource / TextSource record where each came from (SourceArchive / SourceNodesetJSON / SourceEpisodeExcerpt).
	// Inputs can vary with AIFdb's state at fetch time, so what was used is kept alongside the results.
	GraphSource string `json:"graph_source,omitempty"`
	TextSource  string `json:"text_source,omitempty"`
	// TextLocated / TextLocutions are, when TextSource is SourceEpisodeExcerpt, the number of L-nodes
	// that could anchor the excerpt (found uniquely in the transcript) and the total number.
	TextLocated   int `json:"text_located,omitempty"`
	TextLocutions int `json:"text_locutions,omitempty"`
}

// Values of Document.GraphSource / TextSource.
const (
	SourceArchive        = "archive"         // bundled in the subcorpus archive
	SourceNodesetJSON    = "nodeset-json"    // the archive's JSON was broken, so it was refetched from www.aifdb.org/json/<id>
	SourceEpisodeExcerpt = "episode-excerpt" // no per-nodeset raw text, so it was excerpted from the episode transcript
)

// FetchSubcorpus fetches a subcorpus once as tar.gz and returns
// nodeset JSON paired with the corresponding raw text.
//
// If the archive's JSON is empty, broken, or has no nodes, only that nodeset is refetched from
// www.aifdb.org/json/<id> (failing that, it is an error; silently dropping it would change the sample).
// If there is no per-nodeset raw text (nodesetNNN.txt) but there is a per-episode transcript (any other .txt),
// the nodeset's span is excerpted from it.
//
// Nodesets whose text cannot be built are still returned (empty Text and TextSource), and their IDs are returned
// as the second value. Only the e2e condition uses raw text; gold evaluation (gold-loc / baseline) does not need it.
// These used to be dropped here, but as of 2026-10 there is a subcorpus whose transcript is not bundled at all
// (cutietestrun18June2020), and dropping them would silently change the dev sample.
func (c *Client) FetchSubcorpus(ctx context.Context, shortname string) ([]Document, []string, error) {
	body, err := c.get(ctx, c.base, "/api/archive.php", url.Values{
		"shortname": {shortname},
		"filetype":  {"json"},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("aif: サブコーパス %s のアーカイブ取得: %w", shortname, err)
	}
	defer body.Close()
	return c.readArchive(ctx, shortname, body)
}

func (c *Client) readArchive(ctx context.Context, shortname string, body io.Reader) ([]Document, []string, error) {
	gz, err := gzip.NewReader(body)
	if err != nil {
		return nil, nil, fmt.Errorf("aif: サブコーパス %s の gzip 展開: %w", shortname, err)
	}
	defer func() { _ = gz.Close() }()

	graphs := map[string]Graph{}
	graphSrc := map[string]string{}
	var broken []string
	texts := map[string]string{}
	var episode []string
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("aif: サブコーパス %s の tar 展開: %w", shortname, err)
		}
		if h.Typeflag != tar.TypeReg {
			continue
		}
		name := path.Base(h.Name)
		b, err := io.ReadAll(io.LimitReader(tr, 32<<20))
		if err != nil {
			return nil, nil, fmt.Errorf("aif: %s の読み出し: %w", name, err)
		}
		id, perNodeset := nodesetIDOf(name)
		switch {
		case strings.HasSuffix(name, ".json") && perNodeset:
			g, err := ParseNodeset(id, b)
			if err != nil || len(g.Nodes) == 0 {
				broken = append(broken, id)
				continue
			}
			graphs[id], graphSrc[id] = g, SourceArchive
		case strings.HasSuffix(name, ".txt") && perNodeset:
			texts[id] = string(b)
		case strings.HasSuffix(name, ".txt"):
			episode = append(episode, string(b))
		}
	}

	sort.Strings(broken)
	for _, id := range broken {
		g, err := c.FetchNodeset(ctx, id)
		if err != nil {
			return nil, nil, fmt.Errorf("aif: サブコーパス %s のアーカイブの %s が壊れており、取り直しにも失敗した: %w", shortname, id, err)
		}
		graphs[id], graphSrc[id] = g, SourceNodesetJSON
	}

	ids := make([]string, 0, len(graphs))
	for id := range graphs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var docs []Document
	var noText []string
	for _, id := range ids {
		d := Document{NodesetID: id, Corpus: shortname, Graph: graphs[id], GraphSource: graphSrc[id]}
		if t := texts[id]; strings.TrimSpace(t) != "" {
			d.Text, d.TextSource = t, SourceArchive
		} else {
			norm, _ := Normalize(d.Graph)
			for _, ep := range episode {
				if t, located, total, ok := ExcerptTranscript(ep, norm); ok {
					d.Text, d.TextSource, d.TextLocated, d.TextLocutions = t, SourceEpisodeExcerpt, located, total
					break
				}
			}
		}
		if strings.TrimSpace(d.Text) == "" {
			noText = append(noText, id)
		}
		docs = append(docs, d)
	}
	return docs, noText, nil
}

// nodesetIDOf extracts the nodeset ID from a file name inside the archive.
// "nodeset17918.json" → ("17918", true). Files that are not per-nodeset give ("", false).
func nodesetIDOf(name string) (string, bool) {
	stem := strings.TrimSuffix(strings.TrimSuffix(name, ".json"), ".txt")
	id, ok := strings.CutPrefix(stem, "nodeset")
	if !ok || id == "" {
		return "", false
	}
	for _, r := range id {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	return id, true
}

// FetchNodeset fetches one nodeset's JSON from AIFdb proper.
func (c *Client) FetchNodeset(ctx context.Context, id string) (Graph, error) {
	body, err := c.get(ctx, c.nodesets, "/json/"+url.PathEscape(id), nil)
	if err != nil {
		return Graph{}, err
	}
	defer body.Close()
	b, err := io.ReadAll(io.LimitReader(body, 32<<20))
	if err != nil {
		return Graph{}, fmt.Errorf("aif: ノードセット %s の読み出し: %w", id, err)
	}
	g, err := ParseNodeset(id, b)
	if err != nil {
		return Graph{}, fmt.Errorf("aif: ノードセット %s: %w", id, err)
	}
	if len(g.Nodes) == 0 {
		return Graph{}, fmt.Errorf("aif: ノードセット %s にノードが無い", id)
	}
	return g, nil
}

// nodeset is the raw shape of AIFdb's nodeset JSON.
type nodeset struct {
	Nodes []struct {
		NodeID    string `json:"nodeID"`
		Text      string `json:"text"`
		Type      string `json:"type"`
		Scheme    string `json:"scheme"`
		Timestamp string `json:"timestamp"`
	} `json:"nodes"`
	Edges []struct {
		FromID string `json:"fromID"`
		ToID   string `json:"toID"`
	} `json:"edges"`
	Locutions []struct {
		NodeID   string `json:"nodeID"`
		PersonID string `json:"personID"`
	} `json:"locutions"`
}

// ParseNodeset converts AIFdb nodeset JSON into the canonical representation.
//
// AIFdb sometimes omits the default TA/RA scheme from scheme, so an
// empty value is filled with the default (so a missing scheme name is not counted as "a different scheme").
func ParseNodeset(id string, b []byte) (Graph, error) {
	var ns nodeset
	if err := json.Unmarshal(b, &ns); err != nil {
		return Graph{}, fmt.Errorf("aif: ノードセット JSON の解析: %w", err)
	}
	person := map[string]string{}
	for _, l := range ns.Locutions {
		person[l.NodeID] = l.PersonID
	}
	g := Graph{ID: id, Nodes: make([]Node, 0, len(ns.Nodes)), Edges: make([]Edge, 0, len(ns.Edges))}
	for _, n := range ns.Nodes {
		node := Node{
			ID:        n.NodeID,
			Type:      NodeType(strings.TrimSpace(n.Type)),
			Text:      strings.TrimSpace(n.Text),
			Scheme:    strings.TrimSpace(n.Scheme),
			Timestamp: n.Timestamp,
		}
		if node.Scheme == "" {
			node.Scheme = defaultScheme(node.Type)
		}
		if node.Type == TypeI || node.Type == TypeL {
			node.Scheme = ""
		}
		if node.Type == TypeL {
			node.Speaker = SpeakerOf(node)
			if node.Speaker == "" {
				node.Speaker = person[n.NodeID]
			}
		}
		g.Nodes = append(g.Nodes, node)
	}
	for _, e := range ns.Edges {
		g.Edges = append(g.Edges, Edge{FromID: e.FromID, ToID: e.ToID})
	}
	return g, nil
}

func defaultScheme(t NodeType) string {
	switch t {
	case TypeRA:
		return SchemeDefaultInference
	case TypeCA:
		return SchemeDefaultConflict
	case TypeMA:
		return SchemeDefaultRephrase
	case TypeTA:
		return SchemeDefaultTransition
	case TypeYA:
		return SchemeDefaultIllocuting
	}
	return ""
}

func (c *Client) getJSON(ctx context.Context, p string, q url.Values, dst any) error {
	body, err := c.get(ctx, c.base, p, q)
	if err != nil {
		return err
	}
	defer body.Close()
	b, err := io.ReadAll(io.LimitReader(body, 64<<20))
	if err != nil {
		return fmt.Errorf("aif: レスポンスの読み出し: %w", err)
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("aif: レスポンスの解析: %w", err)
	}
	return nil
}

func (c *Client) get(ctx context.Context, base, p string, q url.Values) (io.ReadCloser, error) {
	if d := time.Until(c.last.Add(c.interval)); d > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(d):
		}
	}
	u := base + p
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, fmt.Errorf("aif: リクエスト作成: %w", err)
	}
	req.Header.Set("User-Agent", c.ua)
	resp, err := c.http.Do(req)
	c.last = time.Now()
	if err != nil {
		return nil, fmt.Errorf("aif: GET %s: %w", u, err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("aif: GET %s: status %d", u, resp.StatusCode)
	}
	return resp.Body, nil
}
