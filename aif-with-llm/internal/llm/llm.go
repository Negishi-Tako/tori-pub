// Package llm centralizes access to LLMs and embedding models.
//
// Conventions:
//   - Other packages must not import the openai SDK directly. Always go through this Client.
//   - Structured output always uses JSON Schema constraints (Structured Outputs).
//   - Raw responses (Response.Raw) are never discarded (they remain in the cache as-is).
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// Role is a message role.
type Role string

const (
	RoleSystem Role = "system"
	RoleUser   Role = "user"
)

// Message is a single message.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

func System(content string) Message { return Message{Role: RoleSystem, Content: content} }
func User(content string) Message   { return Message{Role: RoleUser, Content: content} }

// Schema is the JSON Schema passed to Structured Outputs.
type Schema struct {
	// Name may contain only a-z / A-Z / 0-9 / _ / -, up to 64 characters (an OpenAI constraint).
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      bool            `json:"strict"`
	JSON        json.RawMessage `json:"json"`
}

// Request is a single completion request. The cache key is derived from this struct, so
// it should contain only values for which "same Request means same result" is desired.
type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	// Temperature / Seed are unspecified when nil (reasoning models do not accept a temperature).
	Temperature *float64 `json:"temperature,omitempty"`
	Seed        *int64   `json:"seed,omitempty"`
	MaxTokens   int64    `json:"max_tokens,omitempty"`
	// If Schema is non-nil, Structured Outputs is used.
	Schema *Schema `json:"schema,omitempty"`
	// Tag is the purpose name ("aif_relation", etc.).
	// Used as the cache namespace and the cost-aggregation label. It does not affect the result.
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

// Usage is token usage and estimated cost.
type Usage struct {
	InputTokens int64 `json:"input_tokens"`
	// CachedInputTokens are input tokens that hit OpenAI's prompt cache.
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

// Response is a completion result.
type Response struct {
	Text         string          `json:"text"`
	Model        string          `json:"model"`
	FinishReason string          `json:"finish_reason"`
	Usage        Usage           `json:"usage"`
	Raw          json.RawMessage `json:"raw,omitempty"`
	// Cached is true when returned from the local cache (cost is treated as 0).
	Cached    bool  `json:"cached"`
	LatencyMS int64 `json:"latency_ms"`
}

// Client abstracts LLM completion. Implementations are OpenAI plus caching and rate-control decorators.
type Client interface {
	Complete(ctx context.Context, req Request) (*Response, error)
}

// Embedder abstracts embedding inference.
type Embedder interface {
	// Embed returns vectors in the same order and of the same length as texts.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Model is the model name (part of the embedding cache key).
	Model() string
	// Dim is the vector dimension (part of the embedding cache key).
	Dim() int
}

// APIError is an upstream API error. Used to decide whether to retry.
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

// Retryable reports whether the failure is transient. Covers 429, 5xx and 408/409.
func (e *APIError) Retryable() bool {
	switch e.StatusCode {
	case 408, 409, 429:
		return true
	}
	return e.StatusCode >= 500
}

// IsRetryable reports whether a retryable APIError appears anywhere in the error chain.
func IsRetryable(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable()
	}
	return false
}
