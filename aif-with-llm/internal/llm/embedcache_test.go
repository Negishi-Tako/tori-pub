package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// countingEmbedder records the texts it was called with.
type countingEmbedder struct {
	inner llm.Embedder
	seen  []string
	err   error
}

func (e *countingEmbedder) Model() string { return e.inner.Model() }
func (e *countingEmbedder) Dim() int      { return e.inner.Dim() }

func (e *countingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	e.seen = append(e.seen, texts...)
	return e.inner.Embed(ctx, texts)
}

func TestCachingEmbedder(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	inner := &countingEmbedder{inner: &llm.HashEmbedder{Name: "test-model", D: 8}}
	e := llm.NewCachingEmbedder(inner, dir)

	first, err := e.Embed(ctx, []string{"環境委員会", "予算委員会"})
	if err != nil {
		t.Fatalf("first embed: %v", err)
	}
	if len(inner.seen) != 2 {
		t.Fatalf("upstream saw %d texts, want 2", len(inner.seen))
	}

	// The second time, everything comes from the cache.
	second, err := e.Embed(ctx, []string{"環境委員会", "予算委員会"})
	if err != nil {
		t.Fatalf("second embed: %v", err)
	}
	if len(inner.seen) != 2 {
		t.Errorf("upstream was called again: %v", inner.seen)
	}
	for i := range first {
		for j := range first[i] {
			if first[i][j] != second[i][j] {
				t.Fatal("cached vectors differ from the originals")
			}
		}
	}

	// With a mix (known + unknown), only the unknown ones are queried.
	inner.seen = nil
	mixed, err := e.Embed(ctx, []string{"環境委員会", "新しい文", "予算委員会"})
	if err != nil {
		t.Fatalf("mixed embed: %v", err)
	}
	if len(inner.seen) != 1 || inner.seen[0] != "新しい文" {
		t.Errorf("upstream saw %v, want only the new text", inner.seen)
	}
	if len(mixed) != 3 {
		t.Fatalf("got %d vectors, want 3", len(mixed))
	}
	// The order must match the input (if it shifts, lookup results break).
	for j := range mixed[0] {
		if mixed[0][j] != first[0][j] {
			t.Fatal("cached vector was placed at the wrong index")
		}
	}
}

// Upstream errors must not be swallowed, and empty input must not call upstream.
func TestCachingEmbedderEdgeCases(t *testing.T) {
	failing := &countingEmbedder{inner: &llm.HashEmbedder{D: 4}, err: errors.New("boom")}
	if _, err := llm.NewCachingEmbedder(failing, t.TempDir()).Embed(context.Background(), []string{"x"}); err == nil {
		t.Error("upstream errors must not be swallowed")
	}
	if got, err := llm.NewCachingEmbedder(failing, t.TempDir()).Embed(context.Background(), nil); err != nil || got != nil {
		t.Errorf("empty input = %v, %v", got, err)
	}
}
