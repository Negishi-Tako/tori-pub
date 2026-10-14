package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/stats"
)

// 論文（TJ-SIF 2026）の検定を data/metrics/ の集計値だけから再現する。
//
// stats.md（-report）は 1 回の実行の中で条件対ごとに 6 指標をまたいで Holm を掛ける探索用の集計で、
// 実行をまたぐ比較（A − K など）も無向 F1・命題 F1 も含まない。論文の主張は次の 5 比較を
// 1 つの族として Holm 補正したもので、その計算はここにある。
const (
	paperMetric  = "aif-ged/0.4" // 論文の S_gold は 0.4（命題グラフでは 0.3 と同じ値）
	paperWeights = "uniform"
	condE        = "gold-loc@v3/5.6-luna"
	condK        = "gold-loc@w1/5.6-luna"
	condA        = "gold-loc@adj/5.6-luna"
	condN        = condBaseline
	condETerra   = "gold-loc@v3/5.6-terra"
)

type paperComparison struct {
	Label  string // 論文での書き方
	A, B   string // 条件（差は A − B）
	Column string // data/metrics の列
}

// paperFamily は論文で Holm 補正を掛ける 5 比較。
var paperFamily = []paperComparison{
	{"A − K, undirected F1", condA, condK, "rel_undirected_f1"},
	{"E − A, strict F1", condE, condA, "rel_typed_f1"},
	{"E − N, proposition F1", condE, condN, "prop_f1"},
	{"E − N, dialogue-act accuracy", condE, condN, "force_acc"},
	{"E − N, S_gold", condE, condN, "reduced_sim_gold"},
}

// paperLargerModel は大きいモデルの比較（60 件。族の外なので補正しない）。
var paperLargerModel = []paperComparison{
	{"terra − luna, strict F1", condETerra, condE, "rel_typed_f1"},
	{"terra − luna, undirected F1", condETerra, condE, "rel_undirected_f1"},
	{"terra − luna, proposition F1", condETerra, condE, "prop_f1"},
	{"terra − luna, S_gold", condETerra, condE, "reduced_sim_gold"},
	{"terra − luna, dialogue-act accuracy", condETerra, condE, "force_acc"},
}

// paperMetricsDir は論文の数値の出どころ。差し替えられないよう定数にしている
// （公開した集計値と別の CSV から「論文の検定」を出せてしまうと、論文との対応が崩れる）。
// テスト（paperstats_test.go）も同じ場所を読むので、ここを書き換えたらテストが論文との一致を確かめる。
const paperMetricsDir = "data/metrics"

var paperMetricFiles = []string{"rerun_n124_v3.csv", "rerun_n124_w1.csv", "rerun_n124_adj.csv", "rerun_n60_terra.csv"}

// metricTable は 条件 → ノードセット → 列 → 値。
type metricTable map[string]map[string]map[string]float64

func loadPaperMetrics(dir string) (metricTable, error) {
	t := metricTable{}
	for _, name := range paperMetricFiles {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, fmt.Errorf("aif-eval: %w", err)
		}
		recs, err := csv.NewReader(f).ReadAll()
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("aif-eval: %s の解析: %w", name, err)
		}
		if len(recs) == 0 {
			continue
		}
		head := recs[0]
		col := map[string]int{}
		for i, h := range head {
			col[h] = i
		}
		for _, r := range recs[1:] {
			if r[col["metric"]] != paperMetric || r[col["weights"]] != paperWeights {
				continue
			}
			cond, id := r[col["condition"]], r[col["nodeset"]]
			if t[cond] == nil {
				t[cond] = map[string]map[string]float64{}
			}
			vals := map[string]float64{}
			for i, h := range head {
				if v, err := strconv.ParseFloat(r[i], 64); err == nil {
					vals[h] = v
				}
			}
			t[cond][id] = vals
		}
	}
	return t, nil
}

// pairedSeries は両条件にあるノードセットだけを、ノードセット ID 順に並べて返す。
func (t metricTable) pairedSeries(c paperComparison) (xs, ys []float64, err error) {
	if t[c.A] == nil || t[c.B] == nil {
		return nil, nil, fmt.Errorf("aif-eval: 条件 %s または %s が data/metrics に無い", c.A, c.B)
	}
	var ids []string
	for id := range t[c.A] {
		if _, ok := t[c.B][id]; ok {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	for _, id := range ids {
		a, okA := t[c.A][id][c.Column]
		b, okB := t[c.B][id][c.Column]
		if !okA || !okB {
			return nil, nil, fmt.Errorf("aif-eval: nodeset %s に列 %s が無い", id, c.Column)
		}
		xs, ys = append(xs, a), append(ys, b)
	}
	return xs, ys, nil
}

type paperResult struct {
	Comparison paperComparison
	Paired     stats.PairedComparison
	MeanA      float64
	MeanB      float64
	PHolm      float64 // 族の外（補正しない比較）では -1
}

func runPaperComparisons(t metricTable, cs []paperComparison, holm bool) ([]paperResult, error) {
	var out []paperResult
	var ps []float64
	for _, c := range cs {
		xs, ys, err := t.pairedSeries(c)
		if err != nil {
			return nil, err
		}
		pc, err := stats.ComparePaired(xs, ys, 10000, 20260909)
		if err != nil {
			return nil, err
		}
		out = append(out, paperResult{Comparison: c, Paired: pc,
			MeanA: stats.Describe(xs).Mean, MeanB: stats.Describe(ys).Mean, PHolm: -1})
		ps = append(ps, pc.PValue)
	}
	if holm {
		for i, p := range stats.HolmAdjust(ps) {
			out[i].PHolm = p
		}
	}
	return out, nil
}

// paperStats は論文の検定を metricsDir の CSV から計算して w に書く。
func paperStats(metricsDir string, w io.Writer) error {
	t, err := loadPaperMetrics(metricsDir)
	if err != nil {
		return err
	}
	family, err := runPaperComparisons(t, paperFamily, true)
	if err != nil {
		return err
	}
	larger, err := runPaperComparisons(t, paperLargerModel, false)
	if err != nil {
		return err
	}
	var sb strings.Builder
	sb.WriteString("# Paper statistics (TJ-SIF 2026)\n\n")
	fmt.Fprintf(&sb, "Source: %s/{%s}, metric %s, %s weights.\n", metricsDir, strings.Join(paperMetricFiles, ","), paperMetric, paperWeights)
	sb.WriteString("Paired per nodeset; bootstrap 95% CI (10,000 resamples, fixed seed); two-sided Wilcoxon signed-rank;\n")
	sb.WriteString("Cliff's delta. Holm correction over the family of the five comparisons below.\n\n")
	writePaperTable(&sb, family)
	sb.WriteString("\n## Larger model (60 nodesets; outside the family, not corrected)\n\n")
	writePaperTable(&sb, larger)
	_, err = io.WriteString(w, sb.String())
	return err
}

func writePaperTable(sb *strings.Builder, rs []paperResult) {
	sb.WriteString("| Comparison | n | Mean A | Mean B | A − B | 95% CI | p | p (Holm) | Cliff's δ |\n")
	sb.WriteString("| --- | --: | --: | --: | --: | --- | --: | --: | --- |\n")
	for _, r := range rs {
		holm := "–"
		if r.PHolm >= 0 {
			holm = fmt.Sprintf("%.4f", r.PHolm)
		}
		pc := r.Paired
		fmt.Fprintf(sb, "| %s | %d | %.3f | %.3f | %+.3f | [%+.3f, %+.3f] | %.4f | %s | %+.2f (%s) |\n",
			r.Comparison.Label, pc.N, r.MeanA, r.MeanB, pc.MeanDiff, pc.CILow, pc.CIHigh, pc.PValue, holm,
			pc.CliffsDelta, pc.Magnitude)
	}
}
