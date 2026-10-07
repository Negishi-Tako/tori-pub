package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// countingEmbedder は呼ばれたテキストを記録する。
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

	// 2 回目は全部キャッシュから返る。
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

	// 混在（既知 + 未知）では未知のぶんだけ問い合わせる。
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
	// 順序が入力どおりに戻ること（ここがずれると検索結果が壊れる）。
	for j := range mixed[0] {
		if mixed[0][j] != first[0][j] {
			t.Fatal("cached vector was placed at the wrong index")
		}
	}
}

// 上流のエラーを握り潰さないこと・空入力では上流を呼ばないこと。
func TestCachingEmbedderEdgeCases(t *testing.T) {
	failing := &countingEmbedder{inner: &llm.HashEmbedder{D: 4}, err: errors.New("boom")}
	if _, err := llm.NewCachingEmbedder(failing, t.TempDir()).Embed(context.Background(), []string{"x"}); err == nil {
		t.Error("upstream errors must not be swallowed")
	}
	if got, err := llm.NewCachingEmbedder(failing, t.TempDir()).Embed(context.Background(), nil); err != nil || got != nil {
		t.Errorf("empty input = %v, %v", got, err)
	}
}
