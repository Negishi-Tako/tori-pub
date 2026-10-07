package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// OpenAIConfig は OpenAI クライアントの設定。
type OpenAIConfig struct {
	APIKey  string
	BaseURL string // 空なら公式エンドポイント。互換 API を使うときだけ設定する
	// MaxRetries は SDK 内蔵リトライの回数（Retry-After を尊重する）。
	MaxRetries int
	Timeout    time.Duration
	// Pricing はコスト推定表。nil なら DefaultPricing を使う。
	Pricing *PriceTable
}

// OpenAIClient は Client の OpenAI 実装。
type OpenAIClient struct {
	api     openai.Client
	pricing *PriceTable
}

var _ Client = (*OpenAIClient)(nil)

// NewOpenAI は OpenAI クライアントを作る。
func NewOpenAI(cfg OpenAIConfig) (*OpenAIClient, error) {
	api, err := cfg.newAPI()
	if err != nil {
		return nil, err
	}
	pricing := cfg.Pricing
	if pricing == nil {
		pricing = DefaultPricing()
	}
	return &OpenAIClient{api: api, pricing: pricing}, nil
}

// newAPI は SDK のクライアントを作る。APIKey が空ならエラー
// （環境変数の暗黙読み込みに頼ると、キー未設定に気づくのが実行の後半になる）。
func (cfg OpenAIConfig) newAPI() (openai.Client, error) {
	if cfg.APIKey == "" {
		return openai.Client{}, errors.New("llm: OPENAI_API_KEY is not set")
	}
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.MaxRetries > 0 {
		opts = append(opts, option.WithMaxRetries(cfg.MaxRetries))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, option.WithRequestTimeout(cfg.Timeout))
	}
	return openai.NewClient(opts...), nil
}

func (c *OpenAIClient) Complete(ctx context.Context, req Request) (*Response, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	params := openai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)),
	}
	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			params.Messages = append(params.Messages, openai.SystemMessage(m.Content))
		case RoleUser:
			params.Messages = append(params.Messages, openai.UserMessage(m.Content))
		default:
			return nil, fmt.Errorf("llm: unknown role %q", m.Role)
		}
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.Seed != nil {
		params.Seed = openai.Int(*req.Seed)
	}
	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = openai.Int(req.MaxTokens)
	}
	if req.Schema != nil {
		// Schema.JSON は any として渡す必要がある（SDK が再度 marshal する）。
		var schemaObj any
		if err := json.Unmarshal(req.Schema.JSON, &schemaObj); err != nil {
			return nil, fmt.Errorf("llm: invalid json schema %q: %w", req.Schema.Name, err)
		}
		js := shared.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   req.Schema.Name,
			Schema: schemaObj,
			Strict: openai.Bool(req.Schema.Strict),
		}
		if req.Schema.Description != "" {
			js.Description = openai.String(req.Schema.Description)
		}
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{JSONSchema: js},
		}
	}

	start := time.Now()
	completion, err := c.api.Chat.Completions.New(ctx, params)
	latency := time.Since(start)
	if err != nil {
		return nil, wrapOpenAIError(err)
	}
	if len(completion.Choices) == 0 {
		return nil, errors.New("llm: response has no choices")
	}
	choice := completion.Choices[0]
	if choice.Message.Refusal != "" {
		return nil, fmt.Errorf("llm: model refused to answer: %s", choice.Message.Refusal)
	}
	usage := Usage{
		InputTokens:       completion.Usage.PromptTokens,
		CachedInputTokens: completion.Usage.PromptTokensDetails.CachedTokens,
		OutputTokens:      completion.Usage.CompletionTokens,
	}
	usage.CostUSD = c.pricing.ChatCost(completion.Model, usage)

	return &Response{
		Text:         choice.Message.Content,
		Model:        completion.Model,
		FinishReason: choice.FinishReason,
		Usage:        usage,
		Raw:          json.RawMessage(completion.RawJSON()),
		LatencyMS:    latency.Milliseconds(),
	}, nil
}

// OpenAIEmbedder は Embedder の OpenAI Embeddings API 実装。
type OpenAIEmbedder struct {
	api   openai.Client
	model string
	dim   int
	// batchSize は 1 リクエストに詰めるテキスト数。API 上限は 2048 入力。
	batchSize int
}

var _ Embedder = (*OpenAIEmbedder)(nil)

// OpenAIEmbedderConfig は埋め込みクライアントの設定。
type OpenAIEmbedderConfig struct {
	OpenAIConfig
	// Model は "text-embedding-3-small" 等。
	Model string
	// Dim は 0 なら既定次元。text-embedding-3-* は短縮次元を指定できる。
	Dim       int
	BatchSize int
}

func NewOpenAIEmbedder(cfg OpenAIEmbedderConfig) (*OpenAIEmbedder, error) {
	if cfg.Model == "" {
		return nil, errors.New("llm: embedding model is empty")
	}
	dim := cfg.Dim
	if dim == 0 {
		known, ok := defaultEmbeddingDims[cfg.Model]
		if !ok {
			return nil, fmt.Errorf("llm: unknown embedding model %q: specify Dim explicitly", cfg.Model)
		}
		dim = known
	}
	api, err := cfg.newAPI()
	if err != nil {
		return nil, err
	}
	batch := cfg.BatchSize
	if batch <= 0 {
		batch = 96
	}
	return &OpenAIEmbedder{api: api, model: cfg.Model, dim: dim, batchSize: batch}, nil
}

func (e *OpenAIEmbedder) Model() string { return e.model }
func (e *OpenAIEmbedder) Dim() int      { return e.dim }

// maxEmbedRunes は埋め込み 1 件あたりの文字数の安全上限。
//
// text-embedding-3-small の上限は 8192 トークンだが、こちらにトークナイザは無い。
// 1 文字あたり最大 2 トークン程度まで振れることを見込み、8192 / 2 に余裕を持たせた値にしてある。
// 先頭を残して切り詰める簡易対策で、厳密なトークン数管理ではない（AIF の命題・発話は十分短い）。
const maxEmbedRunes = 3500

func truncateForEmbedding(s string) string {
	r := []rune(s)
	if len(r) <= maxEmbedRunes {
		return s
	}
	return string(r[:maxEmbedRunes])
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	texts = slices.Clone(texts)
	for i, t := range texts {
		texts[i] = truncateForEmbedding(t)
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.batchSize {
		end := min(start+e.batchSize, len(texts))
		batch := texts[start:end]
		params := openai.EmbeddingNewParams{
			Model: e.model,
			Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: batch},
		}
		if e.dim > 0 {
			params.Dimensions = openai.Int(int64(e.dim))
		}
		resp, err := e.api.Embeddings.New(ctx, params)
		if err != nil {
			return nil, wrapOpenAIError(err)
		}
		if len(resp.Data) != len(batch) {
			return nil, fmt.Errorf("llm: embedding count mismatch: got %d, want %d", len(resp.Data), len(batch))
		}
		// Data の順序は index フィールドに従う（保証されているが念のため並べ替える）。
		vecs := make([][]float32, len(batch))
		for _, d := range resp.Data {
			if d.Index < 0 || int(d.Index) >= len(batch) {
				return nil, fmt.Errorf("llm: embedding index out of range: %d", d.Index)
			}
			v := make([]float32, len(d.Embedding))
			for i, f := range d.Embedding {
				v[i] = float32(f)
			}
			if len(v) != e.dim {
				return nil, fmt.Errorf("llm: embedding dim mismatch for %q: got %d, want %d", e.model, len(v), e.dim)
			}
			vecs[d.Index] = v
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func wrapOpenAIError(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return &APIError{StatusCode: apiErr.StatusCode, Message: apiErr.Message, Err: err}
	}
	return fmt.Errorf("llm: openai request failed: %w", err)
}
