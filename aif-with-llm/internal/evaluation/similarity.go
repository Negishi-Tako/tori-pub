package evaluation

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
)

// PRF は precision / recall / F1。
type PRF struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	TP        int     `json:"tp"`
	FP        int     `json:"fp"`
	FN        int     `json:"fn"`
}

func newPRF(tp, fp, fn int) PRF {
	p := PRF{TP: tp, FP: fp, FN: fn}
	if tp+fp > 0 {
		p.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		p.Recall = float64(tp) / float64(tp+fn)
	}
	if p.Precision+p.Recall > 0 {
		p.F1 = 2 * p.Precision * p.Recall / (p.Precision + p.Recall)
	}
	return p
}

// CosineSimilarity はコサイン類似度。長さが違う場合は 0 を返す。
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// simTable はノード ID の組に対するコサイン類似度。
type simTable map[[2]string]float64

func (s simTable) get(a, b string) float64 {
	return s[[2]string{a, b}] // 無ければ 0
}

// simTables は比較に使う 3 つの類似度表。
//
//	cross    生成 × ゴールド（ノードの対応付け＝GED の置換コスト）
//	predSelf 生成 × 生成 の I-node どうし（過剰生成の説明に使う）
//	goldSelf ゴールド × ゴールド の I-node どうし（取りこぼしの説明に使う）
type simTables struct {
	cross    simTable
	predSelf simTable
	goldSelf simTable
}

// textSimilarity は両グラフのテキストを持つノード（I / L）を 1 回でまとめて
// 埋め込み、総当たりのコサイン類似度表を作る。
func textSimilarity(ctx context.Context, pred, gold aif.Graph, emb llm.Embedder) (simTables, error) {
	textual := func(g aif.Graph) []aif.Node {
		var out []aif.Node
		for _, n := range g.Nodes {
			if (n.Type == aif.TypeI || n.Type == aif.TypeL) && strings.TrimSpace(n.Text) != "" {
				out = append(out, n)
			}
		}
		return out
	}
	pn, gn := textual(pred), textual(gold)
	out := simTables{cross: simTable{}, predSelf: simTable{}, goldSelf: simTable{}}
	if len(pn) == 0 || len(gn) == 0 {
		return out, nil
	}
	texts := make([]string, 0, len(pn)+len(gn))
	for _, n := range slices.Concat(pn, gn) {
		texts = append(texts, n.Text)
	}
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return simTables{}, fmt.Errorf("evaluation: embed AIF node texts: %w", err)
	}
	if len(vecs) != len(texts) {
		return simTables{}, fmt.Errorf("evaluation: embedder returned %d vectors for %d texts", len(vecs), len(texts))
	}
	predVecs, goldVecs := vecs[:len(pn)], vecs[len(pn):]
	for i, a := range pn {
		for j, b := range gn {
			if a.Type == b.Type {
				out.cross[[2]string{a.ID, b.ID}] = CosineSimilarity(predVecs[i], goldVecs[j])
			}
		}
	}
	self := func(nodes []aif.Node, vecs [][]float32, tbl simTable) {
		for i, a := range nodes {
			for j, b := range nodes {
				if i != j && a.Type == aif.TypeI && b.Type == aif.TypeI {
					tbl[[2]string{a.ID, b.ID}] = CosineSimilarity(vecs[i], vecs[j])
				}
			}
		}
	}
	self(pn, predVecs, out.predSelf)
	self(gn, goldVecs, out.goldSelf)
	return out, nil
}
