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

// AIFdb corpora（https://corpora.aifdb.org/）から取得するクライアント。
//
// 相手は研究用の公開サービスなので、並列度 1・リクエスト間に間隔を空ける。まとめ取りは 1 サブコーパス =
// 1 アーカイブ（tar.gz）で済ませ、ノードセットを 1 件ずつ叩かない。
//
// ただしアーカイブが壊れていることがある（2026-10 時点では、ノードセットの JSON が 0 バイトで、
// 素テキストもノードセット単位ではなくエピソード単位の 1 本しか入っていない）。そのときだけ、
// 壊れた JSON を 1 件ずつ AIFdb 本体（www.aifdb.org/json/<id>）から取り直し、
// 素テキストはエピソードの書き起こしから切り出す（ExcerptTranscript）。
const (
	DefaultCorporaBaseURL = "https://corpora.aifdb.org"
	DefaultNodesetBaseURL = "https://www.aifdb.org"
	DefaultInterval       = 2 * time.Second
)

// ClientConfig は AIFdb クライアントの設定。
type ClientConfig struct {
	BaseURL string
	// NodesetBaseURL はノードセット単位の JSON（<NodesetBaseURL>/json/<id>）の取得先。
	NodesetBaseURL string
	Interval       time.Duration
	UserAgent      string
	HTTP           *http.Client
}

// Client は AIFdb corpora の HTTP クライアント。
type Client struct {
	base     string
	nodesets string
	interval time.Duration
	ua       string
	http     *http.Client
	last     time.Time
}

// NewClient はクライアントを作る。
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

// Corpus はコーパス・サブコーパスのメタ情報。
type Corpus struct {
	CorpusID     int    `json:"corpusID"`
	Shortname    string `json:"shortname"`
	Title        string `json:"title"`
	Description  string `json:"description"`
	NodesetCount int    `json:"nodesetCount"`
	Reference    string `json:"reference"`
	Licence      string `json:"licence"`
}

// CorpusDetails は 1 コーパスのメタ情報を返す（出典表記に使う）。
func (c *Client) CorpusDetails(ctx context.Context, shortname string) (Corpus, error) {
	var out Corpus
	q := url.Values{"shortname": {shortname}}
	if err := c.getJSON(ctx, "/api/getCorpusDetails.php", q, &out); err != nil {
		return Corpus{}, fmt.Errorf("aif: コーパス %s のメタ情報: %w", shortname, err)
	}
	return out, nil
}

// Subcorpora はコーパス配下のサブコーパス一覧を返す。
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

// Document は 1 ノードセット（＝ゴールドの AIF グラフ）と、その素のテキスト。
//
// Text はアノテーションが付く前の書き起こし（話者[時刻] 本文…）。
// LLM に渡す入力はこれで、Graph は答え合わせにしか使わない。
type Document struct {
	NodesetID string `json:"nodeset_id"`
	Corpus    string `json:"corpus"`
	Text      string `json:"text"`
	Graph     Graph  `json:"graph"`
	// GraphSource / TextSource はどこから得たか（SourceArchive / SourceNodesetJSON / SourceEpisodeExcerpt）。
	// 取得時の AIFdb の状態で入力が変わりうるので、何を使ったかを結果と一緒に残す。
	GraphSource string `json:"graph_source,omitempty"`
	TextSource  string `json:"text_source,omitempty"`
	// TextLocated / TextLocutions は、TextSource が SourceEpisodeExcerpt のとき、
	// 切り出しの根拠にできた（書き起こしの中で一意に見つかった）L-node の数と総数。
	TextLocated   int `json:"text_located,omitempty"`
	TextLocutions int `json:"text_locutions,omitempty"`
}

// Document.GraphSource / TextSource の値。
const (
	SourceArchive        = "archive"         // サブコーパスのアーカイブに同梱のもの
	SourceNodesetJSON    = "nodeset-json"    // アーカイブの JSON が壊れていたので www.aifdb.org/json/<id> から取り直した
	SourceEpisodeExcerpt = "episode-excerpt" // ノードセット単位の素テキストが無いので、エピソードの書き起こしから切り出した
)

// FetchSubcorpus はサブコーパスを tar.gz で 1 回だけ取得し、
// ノードセット JSON と対応する素テキストを組にして返す。
//
// アーカイブの JSON が空・壊れている、またはノードを 1 つも持たないときは、そのノードセットだけを
// www.aifdb.org/json/<id> から取り直す（取り直せなければエラーにする。黙って落とすと標本が変わる）。
// ノードセット単位の素テキスト（nodesetNNN.txt）が無く、エピソード単位の書き起こし（それ以外の .txt）が
// あるときは、そこからノードセットの区間を切り出す。
//
// テキストが作れないノードセットも落とさずに返し（Text は空、TextSource も空）、その ID を 2 つ目の戻り値で返す。
// 素テキストを使うのは e2e 条件だけで、ゴールドの評価（gold-loc / baseline）には要らないため。
// 以前はここで落としていたが、2026-10 時点では書き起こし自体が同梱されないサブコーパス
// （cutietestrun18June2020）があり、落とすと dev の標本が黙って変わってしまう。
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

// nodesetIDOf はアーカイブ内のファイル名からノードセット ID を取り出す。
// "nodeset17918.json" → ("17918", true)。ノードセット単位でないファイルは ("", false)。
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

// FetchNodeset は 1 ノードセットの JSON を AIFdb 本体から取得する。
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

// nodeset は AIFdb のノードセット JSON の生の形。
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

// ParseNodeset は AIFdb のノードセット JSON を正準表現に読み替える。
//
// AIFdb は TA/RA の既定スキームを scheme に入れない場合があるので、
// 空なら既定値で埋める（スキーム名の欠落を「別のスキーム」と数えないため）。
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
