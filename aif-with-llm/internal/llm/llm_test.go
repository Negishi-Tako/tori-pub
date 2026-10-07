package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

type sampleOut struct {
	Label      string   `json:"label"`
	Confidence float64  `json:"confidence"`
	Tags       []string `json:"tags"`
	Nested     struct {
		Note string `json:"note"`
	} `json:"nested"`
}

func TestSchemaForIsStrict(t *testing.T) {
	schema, err := llm.SchemaFor[sampleOut]("sample", "テスト用")
	if err != nil {
		t.Fatalf("SchemaFor: %v", err)
	}
	if !schema.Strict {
		t.Error("schema should be strict")
	}
	var m map[string]any
	if err := json.Unmarshal(schema.JSON, &m); err != nil {
		t.Fatalf("schema is not valid json: %v", err)
	}
	// OpenAI の strict モードは additionalProperties:false と
	// 全プロパティの required を要求する。ここが崩れると API が 400 を返す。
	if m["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", m["additionalProperties"])
	}
	required, ok := m["required"].([]any)
	if !ok {
		t.Fatalf("required is missing: %v", m)
	}
	if len(required) != 4 {
		t.Errorf("required = %v, want all 4 properties", required)
	}
	// $schema はサーバ側で嫌われるので落としてあること。
	if _, ok := m["$schema"]; ok {
		t.Error("$schema should be removed")
	}
	// 入れ子の object も同様に整形されていること。
	props := m["properties"].(map[string]any)
	nested := props["nested"].(map[string]any)
	if nested["additionalProperties"] != false {
		t.Errorf("nested additionalProperties = %v, want false", nested["additionalProperties"])
	}
}

func TestSchemaForIsDeterministic(t *testing.T) {
	// スキーマが揺れるとキャッシュキーが変わり、無駄に課金される。
	a, err := llm.SchemaFor[sampleOut]("sample", "テスト用")
	if err != nil {
		t.Fatal(err)
	}
	for range 5 {
		b, err := llm.SchemaFor[sampleOut]("sample", "テスト用")
		if err != nil {
			t.Fatal(err)
		}
		if string(a.JSON) != string(b.JSON) {
			t.Fatalf("schema is not deterministic:\n%s\n%s", a.JSON, b.JSON)
		}
	}
}

func TestCacheKeyDependsOnRequest(t *testing.T) {
	base := llm.Request{
		Model:    "gpt-5-mini",
		Messages: []llm.Message{llm.User("こんにちは")},
	}
	key := llm.CacheKey(base)

	same := base
	same.Tag = "別のタグ" // Tag は結果に影響しないのでキーを変えない
	if llm.CacheKey(same) != key {
		t.Error("Tag should not affect the cache key")
	}

	changed := base
	changed.Messages = []llm.Message{llm.User("こんばんは")}
	if llm.CacheKey(changed) == key {
		t.Error("different messages must produce a different key")
	}

	temp := 0.5
	withTemp := base
	withTemp.Temperature = &temp
	if llm.CacheKey(withTemp) == key {
		t.Error("temperature must affect the cache key")
	}
}

func TestCachingClientRecordsAndReplays(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	calls := 0
	upstream := &llm.FuncClient{Fn: func(context.Context, llm.Request) (*llm.Response, error) {
		calls++
		return &llm.Response{
			Text:  `{"ok":true}`,
			Model: "gpt-5-mini",
			Usage: llm.Usage{InputTokens: 10, OutputTokens: 5, CostUSD: 0.001},
		}, nil
	}}
	req := llm.Request{Model: "gpt-5-mini", Tag: "test", Messages: []llm.Message{llm.User("q")}}

	client := llm.NewCachingClient(upstream, llm.NewFileCache(dir))
	if _, err := client.Complete(ctx, req); err != nil {
		t.Fatalf("first call: %v", err)
	}
	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if calls != 1 {
		t.Errorf("upstream called %d times, want 1", calls)
	}
	if !resp.Cached {
		t.Error("second response should be marked as cached")
	}
	if resp.Usage.CostUSD != 0 {
		t.Errorf("cached response cost = %f, want 0 (it was not billed)", resp.Usage.CostUSD)
	}

}

func TestAPIErrorRetryable(t *testing.T) {
	cases := map[int]bool{429: true, 500: true, 503: true, 400: false, 401: false, 404: false}
	for status, want := range cases {
		err := error(&llm.APIError{StatusCode: status})
		if got := llm.IsRetryable(err); got != want {
			t.Errorf("status %d retryable = %v, want %v", status, got, want)
		}
	}
	if llm.IsRetryable(errors.New("plain")) {
		t.Error("plain errors must not be retryable")
	}
}

func TestPricingChatCost(t *testing.T) {
	p := llm.DefaultPricing()
	// gpt-5-mini: input $0.25 / cached $0.025 / output $2.00 per 1M
	cost := p.ChatCost("gpt-5-mini", llm.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if want := 2.25; cost != want {
		t.Errorf("ChatCost = %f, want %f", cost, want)
	}
	// 日付サフィックス付きのモデル名でも前方一致で単価が引けること。
	if got := p.ChatCost("gpt-5-mini-2026-01-01", llm.Usage{OutputTokens: 1_000_000}); got != 2.0 {
		t.Errorf("dated model cost = %f, want 2.0", got)
	}
	// キャッシュヒット分は安い単価で計算されること。
	cached := p.ChatCost("gpt-5-mini", llm.Usage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000})
	if cached != 0.025 {
		t.Errorf("cached input cost = %f, want 0.025", cached)
	}
	// 未知モデルは 0 を返すこと。
	if got := p.ChatCost("mystery-model", llm.Usage{InputTokens: 100}); got != 0 {
		t.Errorf("unknown model cost = %f, want 0", got)
	}
}
