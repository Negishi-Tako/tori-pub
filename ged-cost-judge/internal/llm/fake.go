package llm

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
)

// テスト用の実装。実 API を叩かずに配線と集計を確かめるために使う。

// FuncClient は関数 1 つで応答を決める Client。
type FuncClient struct {
	Fn func(ctx context.Context, req Request) (*Response, error)
}

var _ Client = (*FuncClient)(nil)

func (c *FuncClient) Complete(ctx context.Context, req Request) (*Response, error) {
	return c.Fn(ctx, req)
}

// HashEmbedder はテキストのハッシュから安定したベクトルを作る決定的な埋め込み。
// 意味的な性質は無い（同じ文は cos 1、違う文はほぼ無相関）。
type HashEmbedder struct {
	Name string
	D    int
}

var _ Embedder = (*HashEmbedder)(nil)

func (e *HashEmbedder) Model() string {
	if e.Name == "" {
		return "hash-test"
	}
	return e.Name
}

func (e *HashEmbedder) Dim() int {
	if e.D == 0 {
		return 16
	}
	return e.D
}

func (e *HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	dim := e.Dim()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		vec := make([]float32, dim)
		var norm float64
		for j := range dim {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], uint32(j))
			sum := sha256.Sum256(append([]byte(t), buf[:]...))
			v := float64(binary.LittleEndian.Uint32(sum[:4]))/float64(math.MaxUint32)*2 - 1
			vec[j] = float32(v)
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range vec {
				vec[j] = float32(float64(vec[j]) / norm)
			}
		}
		out[i] = vec
	}
	return out, nil
}
