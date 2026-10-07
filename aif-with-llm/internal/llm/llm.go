// Package llm は LLM / 埋め込みモデルへのアクセスを一箇所に集める。
//
// 規約:
//   - 他パッケージから openai SDK を直接 import しない。必ずこの Client 経由。
//   - 構造化出力は必ず JSON Schema 制約付き（Structured Outputs）で行う。
//   - 生レスポンス（Response.Raw）は捨てない（キャッシュにそのまま残る）。
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// Role はメッセージの役割。
type Role string

const (
	RoleSystem Role = "system"
	RoleUser   Role = "user"
)

// Message は 1 メッセージ。
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

func System(content string) Message { return Message{Role: RoleSystem, Content: content} }
func User(content string) Message   { return Message{Role: RoleUser, Content: content} }

// Schema は Structured Outputs に渡す JSON Schema。
type Schema struct {
	// Name は a-z / A-Z / 0-9 / _ / - のみ、64 文字以内（OpenAI の制約）。
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      bool            `json:"strict"`
	JSON        json.RawMessage `json:"json"`
}

// Request は 1 回の補完要求。キャッシュキーはこの構造体から決まるため、
// 「同じ Request なら同じ結果」であってほしい値だけを入れる。
type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	// Temperature / Seed は nil なら未指定（推論系モデルは温度指定を受け付けないため）。
	Temperature *float64 `json:"temperature,omitempty"`
	Seed        *int64   `json:"seed,omitempty"`
	MaxTokens   int64    `json:"max_tokens,omitempty"`
	// Schema が非 nil なら Structured Outputs を使う。
	Schema *Schema `json:"schema,omitempty"`
	// Tag は用途名（"aif_relation" 等）。
	// キャッシュの名前空間とコスト集計のラベルに使う。結果には影響しない。
	Tag string `json:"-"`
}

func (r Request) Validate() error {
	if r.Model == "" {
		return errors.New("llm: model is empty")
	}
	if len(r.Messages) == 0 {
		return errors.New("llm: messages is empty")
	}
	return nil
}

// Usage はトークン使用量と推定コスト。
type Usage struct {
	InputTokens int64 `json:"input_tokens"`
	// CachedInputTokens は OpenAI 側の prompt cache でヒットした入力トークン。
	CachedInputTokens int64   `json:"cached_input_tokens"`
	OutputTokens      int64   `json:"output_tokens"`
	CostUSD           float64 `json:"cost_usd"`
}

func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:       u.InputTokens + other.InputTokens,
		CachedInputTokens: u.CachedInputTokens + other.CachedInputTokens,
		OutputTokens:      u.OutputTokens + other.OutputTokens,
		CostUSD:           u.CostUSD + other.CostUSD,
	}
}

// Response は補完結果。
type Response struct {
	Text         string          `json:"text"`
	Model        string          `json:"model"`
	FinishReason string          `json:"finish_reason"`
	Usage        Usage           `json:"usage"`
	Raw          json.RawMessage `json:"raw,omitempty"`
	// Cached はローカルキャッシュから返した場合 true（コストは 0 として扱う）。
	Cached    bool  `json:"cached"`
	LatencyMS int64 `json:"latency_ms"`
}

// Client は LLM 補完の抽象。実装は OpenAI と、キャッシュ・レート制御の装飾。
type Client interface {
	Complete(ctx context.Context, req Request) (*Response, error)
}

// Embedder は埋め込み推論の抽象。
type Embedder interface {
	// Embed は texts と同じ順序・同じ長さのベクトル列を返す。
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Model はモデル名（埋め込みキャッシュのキーに入る）。
	Model() string
	// Dim はベクトル次元数（埋め込みキャッシュのキーに入る）。
	Dim() int
}

// APIError は上流 API のエラー。リトライ判定に使う。
type APIError struct {
	StatusCode int
	Message    string
	Err        error
}

func (e *APIError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("llm: api error (status %d)", e.StatusCode)
	}
	return fmt.Sprintf("llm: api error (status %d): %s", e.StatusCode, e.Message)
}

func (e *APIError) Unwrap() error { return e.Err }

// Retryable は一時的な失敗か判定する。429 と 5xx、408/409 を対象にする。
func (e *APIError) Retryable() bool {
	switch e.StatusCode {
	case 408, 409, 429:
		return true
	}
	return e.StatusCode >= 500
}

// IsRetryable はエラー鎖のどこかに再試行可能な APIError があるか調べる。
func IsRetryable(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable()
	}
	return false
}
