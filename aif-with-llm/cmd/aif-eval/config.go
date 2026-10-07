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

// config は環境変数（と .env）から読む設定。本実験が読むのはここにあるものだけ。
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
	// .env が無くてもエラーにしない（環境変数で渡してもよい）。
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

// newLLMClient は「キャッシュ → レート制御・再試行 → OpenAI」の順に装飾した Client を返す。
// キャッシュを最外に置くので、ヒットしたときはレート制御を待たない。
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

// newEmbedder は OpenAI の埋め込みを返す。LLM_CACHE_DIR があればキャッシュする。
//
// 埋め込みは同じ入力でも API の戻り値がビット単位で同一とは限らず、
// Hungarian 法の割当がわずかに揺れて指標が再現しなくなるため、必ずキャッシュする。
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
