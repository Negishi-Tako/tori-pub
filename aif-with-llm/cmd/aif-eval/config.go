package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/prompts"
)

// config holds the settings read from environment variables (and .env). This experiment reads only what is here.
type config struct {
	OpenAIAPIKey  string
	OpenAIBaseURL string
	ModelMining   string
	PricingFile   string
	LLMCacheDir   string
	PromptsDir    string
	LLMRateLimit  float64 // requests/sec
	LLMBurst      int
	LLMMaxRetries int

	EmbeddingModel string
	EmbeddingDim   int
}

func loadConfig() (*config, error) {
	// A missing .env is not an error (settings may come from environment variables).
	_ = godotenv.Load()
	c := &config{
		OpenAIAPIKey:   env("OPENAI_API_KEY", ""),
		OpenAIBaseURL:  env("OPENAI_BASE_URL", ""),
		ModelMining:    env("MODEL_MINING", "gpt-5.6-luna"),
		PricingFile:    env("PRICING_FILE", ""),
		LLMCacheDir:    env("LLM_CACHE_DIR", ".cache/llm"),
		PromptsDir:     env("PROMPTS_DIR", "prompts"),
		EmbeddingModel: env("EMBEDDING_MODEL", "text-embedding-3-small"),
	}
	var err error
	if c.EmbeddingDim, err = envNum("EMBEDDING_DIM", 0, strconv.Atoi); err != nil {
		return nil, err
	}
	if c.LLMRateLimit, err = envNum("LLM_RATE_LIMIT_RPS", 4, func(s string) (float64, error) { return strconv.ParseFloat(s, 64) }); err != nil {
		return nil, err
	}
	if c.LLMBurst, err = envNum("LLM_BURST", 4, strconv.Atoi); err != nil {
		return nil, err
	}
	if c.LLMMaxRetries, err = envNum("LLM_MAX_RETRIES", 4, strconv.Atoi); err != nil {
		return nil, err
	}
	if c.OpenAIAPIKey == "" {
		return nil, errors.New("config: OPENAI_API_KEY is not set (put it in .env)")
	}
	if c.LLMRateLimit <= 0 {
		return nil, errors.New("config: LLM_RATE_LIMIT_RPS must be > 0")
	}
	return c, nil
}

func env(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envNum[T int | float64](key string, fallback T, parse func(string) (T, error)) (T, error) {
	v := env(key, "")
	if v == "" {
		return fallback, nil
	}
	n, err := parse(v)
	if err != nil {
		return 0, fmt.Errorf("config: %s must be a number: %w", key, err)
	}
	return n, nil
}

func (c *config) openAI(timeout time.Duration, pricing *llm.PriceTable) llm.OpenAIConfig {
	return llm.OpenAIConfig{
		APIKey:     c.OpenAIAPIKey,
		BaseURL:    c.OpenAIBaseURL,
		MaxRetries: c.LLMMaxRetries,
		Timeout:    timeout,
		Pricing:    pricing,
	}
}

// newLLMClient returns a Client decorated in the order "cache → rate control and retry → OpenAI".
// The cache is outermost, so cache hits do not wait on rate control.
func (c *config) newLLMClient() (llm.Client, error) {
	pricing, err := llm.LoadPricing(c.PricingFile)
	if err != nil {
		return nil, err
	}
	base, err := llm.NewOpenAI(c.openAI(5*time.Minute, pricing))
	if err != nil {
		return nil, err
	}
	var client llm.Client = llm.NewThrottledClient(base, c.LLMRateLimit, c.LLMBurst, llm.DefaultRetryConfig())
	if c.LLMCacheDir != "" {
		client = llm.NewCachingClient(client, llm.NewFileCache(c.LLMCacheDir))
	}
	return client, nil
}

// newEmbedder returns OpenAI embeddings, cached when LLM_CACHE_DIR is set.
//
// The API does not always return bit-identical embeddings for the same input, which makes
// the Hungarian assignment wobble slightly and the metrics irreproducible, so they are always cached.
func (c *config) newEmbedder() (llm.Embedder, error) {
	emb, err := llm.NewOpenAIEmbedder(llm.OpenAIEmbedderConfig{
		OpenAIConfig: c.openAI(2*time.Minute, nil),
		Model:        c.EmbeddingModel,
		Dim:          c.EmbeddingDim,
	})
	if err != nil {
		return nil, err
	}
	if c.LLMCacheDir == "" {
		return emb, nil
	}
	return llm.NewCachingEmbedder(emb, filepath.Join(c.LLMCacheDir, "embed")), nil
}

func (c *config) newPromptStore() *prompts.Store {
	return prompts.NewStore(c.PromptsDir)
}
