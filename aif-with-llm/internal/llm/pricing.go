package llm

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"strings"
)

// ModelPrice は 100 万トークンあたりの USD 単価。
type ModelPrice struct {
	Input       float64 `json:"input"`
	CachedInput float64 `json:"cached_input"`
	Output      float64 `json:"output"`
}

// PriceTable はモデル名 → 単価。実行ログに出すコストの推定に使う。
type PriceTable struct {
	prices map[string]ModelPrice
}

// defaultPrices は https://developers.openai.com/api/docs/pricing の値
// （2026-08-13 時点で確認）。値上げ・新モデルに追随する必要があるため、
// 実験前に PRICING_FILE で上書きできるようにしてある。
var defaultPrices = map[string]ModelPrice{
	"gpt-5.6-sol":            {Input: 5.00, CachedInput: 0.50, Output: 30.00},
	"gpt-5.6-terra":          {Input: 2.00, CachedInput: 0.20, Output: 12.00},
	"gpt-5.6-luna":           {Input: 0.20, CachedInput: 0.02, Output: 1.20},
	"gpt-5.5":                {Input: 5.00, CachedInput: 0.50, Output: 30.00},
	"gpt-5.4":                {Input: 2.50, CachedInput: 0.25, Output: 15.00},
	"gpt-5.2":                {Input: 1.75, CachedInput: 0.175, Output: 14.00},
	"gpt-5.1":                {Input: 1.25, CachedInput: 0.125, Output: 10.00},
	"gpt-5":                  {Input: 1.25, CachedInput: 0.125, Output: 10.00},
	"gpt-5-mini":             {Input: 0.25, CachedInput: 0.025, Output: 2.00},
	"gpt-5-nano":             {Input: 0.05, CachedInput: 0.005, Output: 0.40},
	"gpt-4.1":                {Input: 2.00, CachedInput: 0.50, Output: 8.00},
	"gpt-4.1-mini":           {Input: 0.40, CachedInput: 0.10, Output: 1.60},
	"gpt-4o":                 {Input: 2.50, CachedInput: 1.25, Output: 10.00},
	"gpt-4o-mini":            {Input: 0.15, CachedInput: 0.075, Output: 0.60},
	"text-embedding-3-small": {Input: 0.02},
	"text-embedding-3-large": {Input: 0.13},
}

// defaultEmbeddingDims は既定の埋め込み次元（EMBEDDING_DIM=0 のときに使う）。
var defaultEmbeddingDims = map[string]int{
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
}

func DefaultPricing() *PriceTable {
	return &PriceTable{prices: maps.Clone(defaultPrices)}
}

// LoadPricing は JSON ファイル（{"model": {"input":..,"cached_input":..,"output":..}}）で
// 既定表を上書きする。path が空なら既定表をそのまま返す。
func LoadPricing(path string) (*PriceTable, error) {
	t := DefaultPricing()
	if path == "" {
		return t, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("llm: read pricing file %s: %w", path, err)
	}
	var overrides map[string]ModelPrice
	if err := json.Unmarshal(b, &overrides); err != nil {
		return nil, fmt.Errorf("llm: parse pricing file %s: %w", path, err)
	}
	maps.Copy(t.prices, overrides)
	return t, nil
}

// Price はモデルの単価を返す。API が返す "gpt-5-mini-2026-01-01" のような
// 日付サフィックス付きモデル名は、最長前方一致でベースモデルに寄せる。
func (t *PriceTable) Price(model string) (ModelPrice, bool) {
	if p, ok := t.prices[model]; ok {
		return p, true
	}
	var (
		best    ModelPrice
		bestLen int
	)
	for name, p := range t.prices {
		if strings.HasPrefix(model, name) && len(name) > bestLen {
			best, bestLen = p, len(name)
		}
	}
	return best, bestLen > 0
}

// ChatCost は使用量から推定コスト（USD）を計算する。単価不明なら 0 を返す。
func (t *PriceTable) ChatCost(model string, u Usage) float64 {
	p, ok := t.Price(model)
	if !ok {
		return 0
	}
	const perToken = 1_000_000.0
	fresh := max(u.InputTokens-u.CachedInputTokens, 0)
	cachedRate := p.CachedInput
	if cachedRate == 0 {
		cachedRate = p.Input
	}
	return (float64(fresh)*p.Input +
		float64(u.CachedInputTokens)*cachedRate +
		float64(u.OutputTokens)*p.Output) / perToken
}
