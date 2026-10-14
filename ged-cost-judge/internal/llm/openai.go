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

// OpenAIConfig configures the OpenAI client.
type OpenAIConfig struct {
	APIKey  string
	BaseURL string // empty means the official endpoint; set only when using a compatible API
	// MaxRetries is the number of SDK built-in retries (Retry-After is honored).
	MaxRetries int
	Timeout    time.Duration
	// Pricing is the cost estimation table. nil means DefaultPricing.
	Pricing *PriceTable
}

// OpenAIClient is the OpenAI implementation of Client.
type OpenAIClient struct {
	api     openai.Client
	pricing *PriceTable
}

var _ Client = (*OpenAIClient)(nil)

// NewOpenAI creates an OpenAI client.
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

// newAPI creates the SDK client. An empty APIKey is an error
// (relying on implicit environment variable loading means a missing key is noticed only late in a run).
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
		// Schema.JSON must be passed as any (the SDK marshals it again).
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

// OpenAIEmbedder is the OpenAI Embeddings API implementation of Embedder.
type OpenAIEmbedder struct {
	api   openai.Client
	model string
	dim   int
	// batchSize is the number of texts packed into one request. The API limit is 2048 inputs.
	batchSize int
}

var _ Embedder = (*OpenAIEmbedder)(nil)

// OpenAIEmbedderConfig configures the embedding client.
type OpenAIEmbedderConfig struct {
	OpenAIConfig
	// Model is e.g. "text-embedding-3-small".
	Model string
	// Dim of 0 means the default dimension. text-embedding-3-* accepts shortened dimensions.
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

// maxEmbedRunes is a safe upper bound on characters per embedding.
//
// text-embedding-3-small's limit is 8192 tokens, but we have no tokenizer here.
// Allowing for up to about 2 tokens per character, the value is 8192 / 2 with some margin.
// It is a simple safeguard that truncates keeping the head, not exact token accounting (AIF propositions and utterances are short enough).
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
		// Data order follows the index field (guaranteed, but sorted just in case).
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
