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

// Reproduces the paper's (TJ-SIF 2026) tests from the aggregates in data/metrics/ alone.
//
// stats.md (-report) is an exploratory summary that applies Holm across 6 metrics per condition pair
// within a single run, and includes neither cross-run comparisons (such as A − K) nor undirected F1 or
// proposition F1. The paper's claims Holm-correct the following 5 comparisons as one family; that is computed here.
const (
	paperMetric  = "aif-ged/0.4" // the paper's S_gold is 0.4 (same value as 0.3 on proposition graphs)
	paperWeights = "uniform"
	condE        = "gold-loc@v3/5.6-luna"
	condK        = "gold-loc@w1/5.6-luna"
	condA        = "gold-loc@adj/5.6-luna"
	condN        = condBaseline
	condETerra   = "gold-loc@v3/5.6-terra"
)

type paperComparison struct {
	Label  string // how the paper writes it
	A, B   string // conditions (difference is A − B)
	Column string // column in data/metrics
}

// paperFamily is the 5 comparisons the paper Holm-corrects.
var paperFamily = []paperComparison{
	{"A − K, undirected F1", condA, condK, "rel_undirected_f1"},
	{"E − A, strict F1", condE, condA, "rel_typed_f1"},
	{"E − N, proposition F1", condE, condN, "prop_f1"},
	{"E − N, dialogue-act accuracy", condE, condN, "force_acc"},
	{"E − N, S_gold", condE, condN, "reduced_sim_gold"},
}

// paperLargerModel is the larger-model comparison (60 items; outside the family, so uncorrected).
var paperLargerModel = []paperComparison{
	{"terra − luna, strict F1", condETerra, condE, "rel_typed_f1"},
	{"terra − luna, undirected F1", condETerra, condE, "rel_undirected_f1"},
	{"terra − luna, proposition F1", condETerra, condE, "prop_f1"},
	{"terra − luna, S_gold", condETerra, condE, "reduced_sim_gold"},
	{"terra − luna, dialogue-act accuracy", condETerra, condE, "force_acc"},
}

// paperMetricsDir is where the paper's numbers come from. It is a constant so it cannot be swapped
// (if "the paper's tests" could be produced from a CSV other than the published aggregates, the link to the paper would break).
// The test (paperstats_test.go) reads the same place, so if this changes, the test checks agreement with the paper.
const paperMetricsDir = "data/metrics"

var paperMetricFiles = []string{"rerun_n124_v3.csv", "rerun_n124_w1.csv", "rerun_n124_adj.csv", "rerun_n60_terra.csv"}

// metricTable maps condition → nodeset → column → value.
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

// pairedSeries returns only the nodesets present in both conditions, ordered by nodeset ID.
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
	PHolm      float64 // -1 outside the family (uncorrected comparisons)
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

// paperStats computes the paper's tests from the CSVs in metricsDir and writes them to w.
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
