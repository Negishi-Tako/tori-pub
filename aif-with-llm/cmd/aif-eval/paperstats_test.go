package main

import (
	"math"
	"path/filepath"
	"testing"
)

// 論文の数値が公開した集計値から再現できること（Holm は 5 比較の族に掛ける）。
func TestPaperStatsReproducesPaper(t *testing.T) {
	tbl, err := loadPaperMetrics(filepath.Join("..", "..", paperMetricsDir))
	if err != nil {
		t.Fatal(err)
	}
	rs, err := runPaperComparisons(tbl, paperFamily, true)
	if err != nil {
		t.Fatal(err)
	}
	// 論文には 3 桁に丸めて載せている。
	want := []struct{ diff, lo, hi, holm float64 }{
		{+0.0478, +0.0127, +0.0830, 0.0070}, // A − K, undirected F1
		{+0.1107, +0.0731, +0.1499, 0.0000}, // E − A, strict F1
		{+0.0311, +0.0150, +0.0477, 0.0016}, // E − N, proposition F1
		{+0.0145, +0.0015, +0.0274, 0.0445}, // E − N, dialogue-act accuracy
		{-0.1022, -0.1321, -0.0725, 0.0000}, // E − N, S_gold
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 0.0001 }
	for i, w := range want {
		r := rs[i]
		if r.Paired.N != 124 {
			t.Errorf("%s: n = %d, want 124", r.Comparison.Label, r.Paired.N)
		}
		if !near(r.Paired.MeanDiff, w.diff) || !near(r.Paired.CILow, w.lo) || !near(r.Paired.CIHigh, w.hi) || !near(r.PHolm, w.holm) {
			t.Errorf("%s: got %+.4f [%+.4f, %+.4f] pHolm %.4f, want %+.4f [%+.4f, %+.4f] pHolm %.4f",
				r.Comparison.Label, r.Paired.MeanDiff, r.Paired.CILow, r.Paired.CIHigh, r.PHolm, w.diff, w.lo, w.hi, w.holm)
		}
	}
}

// 大きいモデルの比較（60 件、補正なし）も論文の数値と一致すること。
func TestPaperStatsLargerModel(t *testing.T) {
	tbl, err := loadPaperMetrics(filepath.Join("..", "..", paperMetricsDir))
	if err != nil {
		t.Fatal(err)
	}
	rs, err := runPaperComparisons(tbl, paperLargerModel, false)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct{ diff, lo, hi float64 }{
		{+0.011, -0.019, +0.041}, // strict F1
		{+0.003, -0.033, +0.038}, // undirected F1
		{+0.003, -0.014, +0.021}, // proposition F1
		{-0.005, -0.029, +0.017}, // S_gold
		{-0.014, -0.029, -0.001}, // dialogue-act accuracy
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 0.0005 }
	for i, w := range want {
		r := rs[i]
		if r.Paired.N != 60 {
			t.Errorf("%s: n = %d, want 60", r.Comparison.Label, r.Paired.N)
		}
		if !near(r.Paired.MeanDiff, w.diff) || !near(r.Paired.CILow, w.lo) || !near(r.Paired.CIHigh, w.hi) || r.PHolm != -1 {
			t.Errorf("%s: got %+.4f [%+.4f, %+.4f] pHolm %.4f, want %+.3f [%+.3f, %+.3f] uncorrected",
				r.Comparison.Label, r.Paired.MeanDiff, r.Paired.CILow, r.Paired.CIHigh, r.PHolm, w.diff, w.lo, w.hi)
		}
	}
}
