package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// CachingEmbedder は Embedder のキャッシュ装飾。
//
// キャッシュしないと評価をやり直すたびに再課金されるうえ、API の戻り値は同じ入力でも
// ビット単位で同一とは限らず、Hungarian 法の割当が揺れて指標が再現しなくなる。
//
// キーは sha256(モデル名 + 次元数 + テキスト)。モデルか次元が変われば別キーになる。
type CachingEmbedder struct {
	inner Embedder
	dir   string
}

var _ Embedder = (*CachingEmbedder)(nil)

func NewCachingEmbedder(inner Embedder, dir string) *CachingEmbedder {
	return &CachingEmbedder{inner: inner, dir: dir}
}

func (e *CachingEmbedder) Model() string { return e.inner.Model() }
func (e *CachingEmbedder) Dim() int      { return e.inner.Dim() }

func (e *CachingEmbedder) key(text string) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s\x1f%d\x1f%s", e.inner.Model(), e.inner.Dim(), text)
	return hex.EncodeToString(h.Sum(nil))
}

func (e *CachingEmbedder) path(key string) string {
	// 1 ディレクトリにファイルを詰め込みすぎないよう 2 文字で分ける。
	return filepath.Join(e.dir, safeTag.ReplaceAllString(e.inner.Model(), "_"), key[:2], key+".json")
}

// Embed はキャッシュに無いテキストだけを inner に問い合わせる。
func (e *CachingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, len(texts))
	var (
		missIdx  []int
		missText []string
	)
	for i, t := range texts {
		if v, ok := e.load(e.key(t)); ok {
			out[i] = v
			continue
		}
		missIdx = append(missIdx, i)
		missText = append(missText, t)
	}
	if len(missText) == 0 {
		return out, nil
	}
	vecs, err := e.inner.Embed(ctx, missText)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(missText) {
		return nil, fmt.Errorf("llm: embedder returned %d vectors for %d texts", len(vecs), len(missText))
	}
	for i, v := range vecs {
		out[missIdx[i]] = v
		if err := e.store(e.key(missText[i]), v); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (e *CachingEmbedder) load(key string) ([]float32, bool) {
	b, err := os.ReadFile(e.path(key))
	if err != nil {
		return nil, false
	}
	var v []float32
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, false
	}
	if len(v) != e.inner.Dim() {
		// 次元が合わないキャッシュは壊れているので使わない。
		return nil, false
	}
	return v, true
}

func (e *CachingEmbedder) store(key string, v []float32) error {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("llm: marshal embedding: %w", err)
	}
	return writeFileAtomic(e.path(key), b)
}
