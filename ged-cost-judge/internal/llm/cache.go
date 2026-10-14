package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"
)

// ErrCacheMiss はキャッシュに無かった場合。
var ErrCacheMiss = errors.New("llm: cache miss")

// CacheKey は Request から決定的なキーを作る。
//
// 結果に影響する値だけを含める。Tag は含めない（保存先ディレクトリで分ける）。
// スキーマ本文まで含めるのは、スキーマを変えたら出力も変わるため。
// **ここを変えると既存のキャッシュがすべて外れ、報告した数値を再現できなくなる。**
func CacheKey(req Request) string {
	type keyShape struct {
		Model       string    `json:"model"`
		Messages    []Message `json:"messages"`
		Temperature *float64  `json:"temperature,omitempty"`
		Seed        *int64    `json:"seed,omitempty"`
		MaxTokens   int64     `json:"max_tokens,omitempty"`
		SchemaName  string    `json:"schema_name,omitempty"`
		SchemaJSON  string    `json:"schema_json,omitempty"`
		Strict      bool      `json:"strict,omitempty"`
	}
	k := keyShape{
		Model:       req.Model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
		Seed:        req.Seed,
		MaxTokens:   req.MaxTokens,
	}
	if req.Schema != nil {
		k.SchemaName = req.Schema.Name
		k.SchemaJSON = string(req.Schema.JSON)
		k.Strict = req.Schema.Strict
	}
	// encoding/json は struct フィールド順・map キー順を固定するので決定的。
	b, err := json.Marshal(k)
	if err != nil {
		// Message/float は必ず marshal できるため、ここに来るのは実装バグ。
		panic(fmt.Sprintf("llm: cache key marshal: %v", err))
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

var safeTag = regexp.MustCompile(`[^a-zA-Z0-9_.-]+`)

// FileCache はファイルシステム上の補完結果のキャッシュ。
type FileCache struct {
	dir string
}

func NewFileCache(dir string) *FileCache { return &FileCache{dir: dir} }

func (c *FileCache) path(tag, key string) string {
	t := safeTag.ReplaceAllString(tag, "_")
	if t == "" {
		t = "default"
	}
	return filepath.Join(c.dir, t, key+".json")
}

// Get は保存済みの応答を返す。無ければ ErrCacheMiss。
func (c *FileCache) Get(tag, key string) (*Response, error) {
	b, err := os.ReadFile(c.path(tag, key))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrCacheMiss
		}
		return nil, fmt.Errorf("llm: read cache: %w", err)
	}
	var resp Response
	if err := json.Unmarshal(b, &resp); err != nil {
		return nil, fmt.Errorf("llm: parse cache %s: %w", c.path(tag, key), err)
	}
	return &resp, nil
}

func (c *FileCache) Put(tag, key string, resp *Response) error {
	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		return fmt.Errorf("llm: marshal cache entry: %w", err)
	}
	return writeFileAtomic(c.path(tag, key), b)
}

// writeFileAtomic は一時ファイル経由で path を置き換える。
// 同時実行や中断で壊れた JSON をキャッシュに残さないため。
func writeFileAtomic(path string, b []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("llm: create cache dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("llm: create temp cache file: %w", err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("llm: write cache: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("llm: close cache: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("llm: commit cache: %w", err)
	}
	return nil
}

// CachingClient は Client のキャッシュ装飾。ヒットしたら inner を呼ばない。
type CachingClient struct {
	inner Client
	cache *FileCache
}

var _ Client = (*CachingClient)(nil)

func NewCachingClient(inner Client, cache *FileCache) *CachingClient {
	return &CachingClient{inner: inner, cache: cache}
}

func (c *CachingClient) Complete(ctx context.Context, req Request) (*Response, error) {
	key := CacheKey(req)
	cached, err := c.cache.Get(req.Tag, key)
	switch {
	case err == nil:
		hit := *cached
		hit.Cached = true
		// キャッシュヒットは課金されないので、コスト集計に混ぜない。
		hit.Usage.CostUSD = 0
		return &hit, nil
	case !errors.Is(err, ErrCacheMiss):
		return nil, err
	}
	start := time.Now()
	resp, err := c.inner.Complete(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.LatencyMS == 0 {
		resp.LatencyMS = time.Since(start).Milliseconds()
	}
	if err := c.cache.Put(req.Tag, key, resp); err != nil {
		return nil, err
	}
	return resp, nil
}
