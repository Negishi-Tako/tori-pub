package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/evaluation"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/stats"
)

// ---------- 集計 ----------

func report(outDir string) error {
	b, err := os.ReadFile(filepath.Join(outDir, "results.jsonl"))
	if err != nil {
		return fmt.Errorf("aif-eval: 結果の読み込み（先に -run）: %w", err)
	}
	var results []Result
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var r Result
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return fmt.Errorf("aif-eval: 結果行の解析: %w", err)
		}
		results = append(results, r)
	}
	if err := writeCSV(filepath.Join(outDir, "results.csv"), results); err != nil {
		return err
	}
	if err := writeSummary(filepath.Join(outDir, "summary.md"), results); err != nil {
		return err
	}
	if err := writeEvidence(filepath.Join(outDir, "evidence.md"), results); err != nil {
		return err
	}
	if err := writeStats(filepath.Join(outDir, "stats.md"), results); err != nil {
		return err
	}
	fmt.Printf("書き出した: %s/results.csv, summary.md, evidence.md, stats.md\n", outDir)
	return nil
}

// condKey は集計・検定で条件を識別する鍵。指標の版が違えば別系列として扱う
// （同じグラフでも指標が違えば別の測り方なので、混ぜて平均してはいけない）。
func condKey(c *evaluation.AIFComparison) string {
	return c.Condition + " · " + shortMetric(c.Metric)
}

func shortMetric(m string) string {
	return strings.TrimPrefix(m, "aif-")
}

var csvHeader = []string{
	"nodeset", "condition", "prompt_set", "model", "metric", "weights",
	"full_ged", "full_norm_ged", "full_sim", "full_sim_gold",
	"reduced_ged", "reduced_norm_ged", "reduced_sim", "reduced_sim_gold",
	"prop_pred", "prop_gold", "prop_ratio", "prop_mean_sim", "prop_f1",
	"loc_ratio", "loc_mean_sim", "loc_f1",
	"rel_pred", "rel_gold", "rel_typed_f1", "rel_untyped_f1", "rel_undirected_f1", "rel_reversed",
	"force_acc", "force_compared", "pred_violations",
}

func csvRow(r Result) []string {
	c := r.Comparison
	f := func(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }
	return []string{
		// 条件名は c.Condition（プロンプト版とモデルを含む形）を使う。
		// r.Condition は生の条件名なので、CSV 単体では版を区別できない。
		r.NodesetID, c.Condition, r.PromptSet, r.Model, c.Metric, c.Weights,
		f(c.FullGED.GED), f(c.FullGED.NormalizedGED), f(c.FullSimilarity), f(c.FullSimilarityGold),
		f(c.ReducedGED.GED), f(c.ReducedGED.NormalizedGED), f(c.ReducedSimilarity), f(c.ReducedSimilarityGold),
		strconv.Itoa(c.CountsPred[aif.TypeI]), strconv.Itoa(c.CountsGold[aif.TypeI]),
		f(c.Propositions.CountRatio), f(c.Propositions.MeanSimilarity), f(c.Propositions.PRF.F1),
		f(c.Locutions.CountRatio), f(c.Locutions.MeanSimilarity), f(c.Locutions.PRF.F1),
		strconv.Itoa(c.Relations.PredTotal), strconv.Itoa(c.Relations.GoldTotal),
		f(c.Relations.Typed.F1), f(c.Relations.Untyped.F1), f(c.Relations.Undirected.F1),
		strconv.Itoa(c.Relations.Reversed),
		f(c.Forces.Accuracy), strconv.Itoa(c.Forces.Compared), strconv.Itoa(len(r.PredViolations)),
	}
}

func writeCSV(path string, results []Result) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("aif-eval: CSV の作成: %w", err)
	}
	defer func() { _ = f.Close() }()
	w := csv.NewWriter(f)
	if err := w.Write(csvHeader); err != nil {
		return fmt.Errorf("aif-eval: CSV の書き出し: %w", err)
	}
	for _, r := range results {
		if r.Comparison == nil {
			continue
		}
		if err := w.Write(csvRow(r)); err != nil {
			return fmt.Errorf("aif-eval: CSV の書き出し: %w", err)
		}
	}
	w.Flush()
	return w.Error()
}

// agg は条件 × 重み ごとの平均。
type agg struct {
	n                                        int
	fullSim, reducedSim                      float64
	fullSimGold, reducedSimGold              float64
	forceAccSum                              float64
	f1ByThreshold                            map[string]float64
	propRatio, propSim, propF1               float64
	locRatio, locSim, locF1                  float64
	typedF1, untypedF1, undirF1              float64
	typedTP, untypedTP, undirTP              int // micro 平均（全ノードセットの TP をプール）用
	reversed, relPred, relGold, violations   int
	forceAcc                                 float64
	forceCompared, forceExact                int
	pairCorrect, pairSpurious, pairMissed    float64
	pairCorrectN, pairSpuriousN, pairMissedN int
	byDistance                               map[string]evaluation.DistanceBucket
	dropped                                  int
	typeConfusion                            map[string]map[string]int
	forceConfusion                           map[string]map[string]int
}

func aggregate(results []Result) (map[string]*agg, []string) {
	m := map[string]*agg{}
	var keys []string
	for _, r := range results {
		if r.Comparison == nil {
			continue
		}
		c := r.Comparison
		k := condKey(c) + " / " + c.Weights
		a, ok := m[k]
		if !ok {
			a = &agg{
				byDistance:     map[string]evaluation.DistanceBucket{},
				typeConfusion:  map[string]map[string]int{},
				forceConfusion: map[string]map[string]int{},
				f1ByThreshold:  map[string]float64{},
			}
			m[k] = a
			keys = append(keys, k)
		}
		a.n++
		a.fullSim += c.FullSimilarity
		a.reducedSim += c.ReducedSimilarity
		a.fullSimGold += c.FullSimilarityGold
		a.reducedSimGold += c.ReducedSimilarityGold
		// YA 一致率はノードセットごとの平均を主に使う（検定と定義を揃えるため）。
		// プール値（forceExact / forceCompared）も併記する。
		a.forceAccSum += c.Forces.Accuracy
		for t, v := range c.Propositions.F1ByThreshold {
			a.f1ByThreshold[t] += v
		}
		a.propRatio += c.Propositions.CountRatio
		a.propSim += c.Propositions.MeanSimilarity
		a.propF1 += c.Propositions.PRF.F1
		a.locRatio += c.Locutions.CountRatio
		a.locSim += c.Locutions.MeanSimilarity
		a.locF1 += c.Locutions.PRF.F1
		a.typedF1 += c.Relations.Typed.F1
		a.untypedF1 += c.Relations.Untyped.F1
		a.undirF1 += c.Relations.Undirected.F1
		a.typedTP += c.Relations.Typed.TP
		a.untypedTP += c.Relations.Untyped.TP
		a.undirTP += c.Relations.Undirected.TP
		a.reversed += c.Relations.Reversed
		a.relPred += c.Relations.PredTotal
		a.relGold += c.Relations.GoldTotal
		a.violations += len(r.PredViolations)
		a.forceCompared += c.Forces.Compared
		a.forceExact += c.Forces.Exact
		p := c.RelationPairSimilarity
		a.pairCorrect += p.CorrectMean * float64(p.CorrectN)
		a.pairCorrectN += p.CorrectN
		a.pairSpurious += p.SpuriousMean * float64(p.SpuriousN)
		a.pairSpuriousN += p.SpuriousN
		a.pairMissed += p.MissedMean * float64(p.MissedN)
		a.pairMissedN += p.MissedN
		a.dropped += r.Dropped
		for band, b := range c.Relations.ByDistance {
			a.byDistance[band] = a.byDistance[band].Add(b)
		}
		mergeCounts(a.typeConfusion, c.Relations.TypeConfusion)
		mergeCounts(a.forceConfusion, c.Forces.Confusion)
	}
	for _, a := range m {
		if a.forceCompared > 0 {
			a.forceAcc = float64(a.forceExact) / float64(a.forceCompared)
		}
	}
	sort.Strings(keys)
	return m, keys
}

func writeSummary(path string, results []Result) error {
	m, keys := aggregate(results)
	if len(keys) == 0 {
		return fmt.Errorf("aif-eval: 比較に成功した結果行が 1 つも無い（results.jsonl の error を確認）")
	}
	var sb strings.Builder
	sb.WriteString("# AIF アノテーション精度の集計\n\n")
	fmt.Fprintf(&sb, "結果行: %d\n\n", len(results))
	sb.WriteString("## 条件 × 重み ごとの平均\n\n")
	sb.WriteString("**gold基準**（`GED / gold を作るコスト`）が主指標。分母が条件によらず一定なので条件比較に使える。\n")
	sb.WriteString("**対称**（`GED / (pred を消すコスト + gold を作るコスト)`）は 0.2 までの主指標で、\n")
	sb.WriteString("分母が pred の大きさを含むため「多く出した方が高く出る」。参考として併記する。\n\n")
	sb.WriteString("| 条件 / 重み | n | 全体GED(gold基準) | 命題GED(gold基準) | 全体GED(対称) | 命題GED(対称) | 発話数比 | 発話平均cos | 発話F1 | 命題数比 | 命題平均cos | 命題F1 | 関係F1(種別) | 関係F1(向きのみ) | 関係F1(無向) | 逆向き | 生成関係数 | 正解関係数 | YA一致率(平均) | YA一致率(プール) | 違反 |\n")
	sb.WriteString("| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |\n")
	for _, k := range keys {
		a := m[k]
		n := float64(a.n)
		fmt.Fprintf(&sb, "| %s | %d | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.3f | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %d | %d | %.3f | %.3f | %d |\n",
			k, a.n, a.fullSimGold/n, a.reducedSimGold/n, a.fullSim/n, a.reducedSim/n,
			a.locRatio/n, a.locSim/n, a.locF1/n, a.propRatio/n, a.propSim/n, a.propF1/n,
			a.typedF1/n, a.untypedF1/n, a.undirF1/n, a.reversed, a.relPred, a.relGold,
			a.forceAccSum/n, a.forceAcc, a.violations)
	}

	sb.WriteString("\n## 関係の micro 平均\n\n")
	sb.WriteString("全ノードセットの TP / 生成本数 / 正解本数をプールして求める（レポートの AP / AR）。\n")
	sb.WriteString("上の表の F1 はノードセットごとの平均（macro）で、関係を 1 本も出さないノードセットを\n")
	sb.WriteString("精度 0 と数えるため、強く絞り込む条件を不当に低く見せる。\n\n")
	sb.WriteString("| 条件 / 重み | 厳密 AP | 厳密 AR | 厳密 F1 | 向きのみ AP | 向きのみ AR | 向きのみ F1 | 無向 AP | 無向 AR | 無向 F1 |\n")
	sb.WriteString("| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |\n")
	for _, k := range keys {
		a := m[k]
		fmt.Fprintf(&sb, "| %s |", k)
		for _, tp := range []int{a.typedTP, a.untypedTP, a.undirTP} {
			p := micro(tp, a.relPred, a.relGold)
			fmt.Fprintf(&sb, " %.3f | %.3f | %.3f |", p.Precision, p.Recall, p.F1)
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n## 命題 F1 の閾値依存\n\n")
	sb.WriteString("既定の閾値 0.55 は、同一ノードセット内で*実際に関係がある*命題対の平均 cos\n")
	sb.WriteString("（0.44〜0.48）のすぐ上でしかない。1 点の値で「命題層は実用水準」と言えないので曲線で見る。\n\n")
	ths := slices.Sorted(maps.Keys(m[keys[0]].f1ByThreshold))
	sb.WriteString("| 条件 / 重み |")
	for _, t := range ths {
		fmt.Fprintf(&sb, " cos≥%s |", t)
	}
	sb.WriteString("\n| --- |")
	for range ths {
		sb.WriteString(" --: |")
	}
	sb.WriteString("\n")
	for _, k := range keys {
		a := m[k]
		fmt.Fprintf(&sb, "| %s |", k)
		for _, t := range ths {
			fmt.Fprintf(&sb, " %.3f |", a.f1ByThreshold[t]/float64(a.n))
		}
		sb.WriteString("\n")
	}
	sb.WriteString("\n## 発話間距離の帯ごとの関係の当たり方\n\n")
	sb.WriteString("ゴールドの関係の約 7 割は隣接発話どうしなので、全体の F1 は隣接の山に支配され、\n")
	sb.WriteString("「遠い関係を拾えていない」という失敗はここでしか見えない。帯 1 は距離 0（同じ発話の命題どうし）を含む。\n")
	sb.WriteString("P = 当たり / 生成（その帯で生成した本数）、R = 当たり / 正解（その帯の正解本数）。\n\n")
	sb.WriteString("| 条件 / 重み |")
	for _, band := range aif.DistanceBands {
		fmt.Fprintf(&sb, " 距離%s 正解 | 生成 | R | P |", band)
	}
	sb.WriteString("\n| --- |")
	for range aif.DistanceBands {
		sb.WriteString(" --: | --: | --: | --: |")
	}
	sb.WriteString("\n")
	for _, k := range keys {
		a := m[k]
		fmt.Fprintf(&sb, "| %s |", k)
		for _, band := range aif.DistanceBands {
			b := a.byDistance[band]
			fmt.Fprintf(&sb, " %d | %d | %s | %s |", b.GoldTotal, b.PredTotal,
				ratio(float64(b.TP), b.GoldTotal), ratio(float64(b.TPPred), b.PredTotal))
		}
		sb.WriteString("\n")
	}
	sb.WriteString("\n検証パス（stage 4）が捨てた関係: ")
	for _, k := range keys {
		if m[k].dropped > 0 {
			fmt.Fprintf(&sb, "%s=%d 本  ", k, m[k].dropped)
		}
	}
	sb.WriteString("\n")

	sb.WriteString("\n## 関係で結ばれた命題どうしの平均 cos\n\n")
	sb.WriteString("「過剰生成された関係は、単に話題が近い命題どうしを結んでいるだけではないか」を見る。\n")
	sb.WriteString("当たった関係と過剰生成で平均 cos が変わらなければ、埋め込みの近さでは関係の有無を説明できない。\n\n")
	sb.WriteString("| 条件 / 重み | 当たった関係 | n | 過剰生成 | n | 取りこぼし | n |\n| --- | --: | --: | --: | --: | --: | --: |\n")
	for _, k := range keys {
		a := m[k]
		fmt.Fprintf(&sb, "| %s | %s | %d | %s | %d | %s | %d |\n", k,
			ratio(a.pairCorrect, a.pairCorrectN), a.pairCorrectN,
			ratio(a.pairSpurious, a.pairSpuriousN), a.pairSpuriousN,
			ratio(a.pairMissed, a.pairMissedN), a.pairMissedN)
	}

	sb.WriteString("\n## 関係種別の混同（生成 → 正解、向きが一致した関係のみ）\n\n")
	for _, k := range keys {
		a := m[k]
		if len(a.typeConfusion) == 0 {
			continue
		}
		sb.WriteString("### " + k + "\n\n| 生成 \\ 正解 | RA | CA | MA |\n| --- | --: | --: | --: |\n")
		for _, p := range []string{"RA", "CA", "MA"} {
			row := a.typeConfusion[p]
			fmt.Fprintf(&sb, "| %s | %d | %d | %d |\n", p, row["RA"], row["CA"], row["MA"])
		}
		sb.WriteString("\n")
	}
	sb.WriteString("## 発話行為（YA）の混同（対応付いた命題のみ）\n\n")
	for _, k := range keys {
		a := m[k]
		if len(a.forceConfusion) == 0 {
			continue
		}
		sb.WriteString("### " + k + "\n\n| 生成 \\ 正解 | 件数 |\n| --- | --- |\n")
		for _, p := range slices.Sorted(maps.Keys(a.forceConfusion)) {
			var parts []string
			for _, g := range slices.Sorted(maps.Keys(a.forceConfusion[p])) {
				parts = append(parts, fmt.Sprintf("%s=%d", g, a.forceConfusion[p][g]))
			}
			fmt.Fprintf(&sb, "| %s | %s |\n", p, strings.Join(parts, ", "))
		}
		sb.WriteString("\n")
	}
	return writeFile(path, []byte(sb.String()))
}

// statMetrics は条件間で検定する指標。増やしすぎると多重比較補正で何も出なくなるので、
// 「主張したいこと」に対応する 6 本に絞る。
// GED は gold 基準の正規化を使う。対称正規化は分母が pred の大きさを含むので、
// 「関係を多く出す」だけで値が上がり、条件間の検定に使うと改善を偽って検出する。
var statMetrics = []struct {
	name string
	get  func(*evaluation.AIFComparison) float64
}{
	{"命題GED類似度(gold基準)", func(c *evaluation.AIFComparison) float64 { return c.ReducedSimilarityGold }},
	{"全体GED類似度(gold基準)", func(c *evaluation.AIFComparison) float64 { return c.FullSimilarityGold }},
	{"関係F1(種別)", func(c *evaluation.AIFComparison) float64 { return c.Relations.Typed.F1 }},
	{"命題平均cos", func(c *evaluation.AIFComparison) float64 { return c.Propositions.MeanSimilarity }},
	{"発話平均cos", func(c *evaluation.AIFComparison) float64 { return c.Locutions.MeanSimilarity }},
	{"YA一致率", func(c *evaluation.AIFComparison) float64 { return c.Forces.Accuracy }},
}

// writeStats は条件間の対応のある比較（同じノードセット同士）を書き出す。
//
// 対応を取る単位はノードセット。片方の条件が失敗した組は比較から外す
// （本 PJ の Run 比較と同じ扱い）。多重比較補正は 1 つの条件対について
// 全指標をまたいで掛ける。
func writeStats(path string, results []Result) error {
	// 条件 -> ノードセット -> 指標値（重みは uniform のみ。対応付けは重みで変わらない）
	byCond := map[string]map[string]*evaluation.AIFComparison{}
	var conds []string
	for _, r := range results {
		c := r.Comparison
		if c == nil || c.Weights != "uniform" {
			continue
		}
		if _, ok := byCond[condKey(c)]; !ok {
			byCond[condKey(c)] = map[string]*evaluation.AIFComparison{}
			conds = append(conds, condKey(c))
		}
		byCond[condKey(c)][c.NodesetID] = c
	}
	sort.Strings(conds)

	var sb strings.Builder
	sb.WriteString("# 条件間の対応のある比較\n\n")
	sb.WriteString("同じノードセット同士で条件を突き合わせる。差の信頼区間は対応ありブートストラップ（10,000 反復・seed 固定）、\n")
	sb.WriteString("p 値は Wilcoxon 符号順位検定（両側）、効果量は Cliff's delta。\n")
	sb.WriteString("多重比較補正（Holm）は 1 つの条件対について全指標をまたいで掛けている。\n\n")
	for i := range conds {
		for j := range conds {
			if i == j {
				continue
			}
			a, b := conds[i], conds[j]
			var ids []string
			for id := range byCond[a] {
				if _, ok := byCond[b][id]; ok {
					ids = append(ids, id)
				}
			}
			if len(ids) < 3 {
				continue
			}
			sort.Strings(ids)
			var rows []stats.PairedComparison
			var names []string
			var ps []float64
			for _, m := range statMetrics {
				xs := make([]float64, len(ids))
				ys := make([]float64, len(ids))
				for k, id := range ids {
					xs[k] = m.get(byCond[a][id])
					ys[k] = m.get(byCond[b][id])
				}
				pc, err := stats.ComparePaired(xs, ys, 10000, 20260909)
				if err != nil {
					return fmt.Errorf("aif-eval: %s vs %s の %s: %w", a, b, m.name, err)
				}
				rows = append(rows, pc)
				names = append(names, m.name)
				ps = append(ps, pc.PValue)
			}
			adj := stats.HolmAdjust(ps)
			fmt.Fprintf(&sb, "## %s − %s（n=%d）\n\n", a, b, len(ids))
			sb.WriteString("| 指標 | 平均差 | 95%CI | p | p(Holm) | Cliff's delta |\n| --- | --: | --- | --: | --: | --- |\n")
			for k, pc := range rows {
				fmt.Fprintf(&sb, "| %s | %+.3f | [%+.3f, %+.3f] | %.4f | %.4f | %+.2f (%s) |\n",
					names[k], pc.MeanDiff, pc.CILow, pc.CIHigh, pc.PValue, adj[k], pc.CliffsDelta, pc.Magnitude)
			}
			sb.WriteString("\n")
		}
	}
	return writeFile(path, []byte(sb.String()))
}

// writeEvidence は GED の数値を裏付ける具体例を書き出す。
// レポートの定性分析はこのファイルから引く。
func writeEvidence(path string, results []Result) error {
	var sb strings.Builder
	sb.WriteString("# 定性的な証拠\n\n")
	sb.WriteString("GED / F1 の数値がどの具体例から来ているかを、ノードセットごとに並べたもの。\n")
	sb.WriteString("重みは uniform・指標は " + evaluation.MetricGEDv2 + " の結果のみを載せる\n")
	sb.WriteString("（重みや指標を変えても、命題の対応付けと関係の診断は同じため）。\n\n")
	for _, r := range results {
		c := r.Comparison
		if c == nil || c.Weights != "uniform" || c.Metric != evaluation.MetricGEDv2 {
			continue
		}
		e := c.Evidence
		fmt.Fprintf(&sb, "## nodeset %s / %s\n\n", c.NodesetID, c.Condition)
		fmt.Fprintf(&sb, "- 命題 GED 類似度 %.3f / 関係 F1（種別まで一致）%.3f / 逆向き %d 本\n\n",
			c.ReducedSimilarity, c.Relations.Typed.F1, c.Relations.Reversed)
		writePairs(&sb, "### よく一致した命題", e.BestPairs)
		writePairs(&sb, "### 一致しなかった命題", e.WorstPairs)
		writeRels(&sb, "### 当たった関係", e.Correct)
		writeRels(&sb, "### 向きが逆だった関係", e.Reversed)
		writeRels(&sb, "### 種別を取り違えた関係", e.Mistyped)
		writeRels(&sb, "### 正解に無い関係（過剰生成）", e.Spurious)
		writeRels(&sb, "### 取りこぼした関係", e.Missed)
	}
	return writeFile(path, []byte(sb.String()))
}

func writePairs(sb *strings.Builder, title string, pairs []evaluation.PropositionPair) {
	if len(pairs) == 0 {
		return
	}
	sb.WriteString(title + "\n\n")
	for _, p := range pairs {
		fmt.Fprintf(sb, "- cos %.3f\n  - 生成: %s\n  - 正解: %s\n", p.Similarity, p.PredText, p.GoldText)
	}
	sb.WriteString("\n")
}

func writeRels(sb *strings.Builder, title string, rels []evaluation.RelationEvidence) {
	if len(rels) == 0 {
		return
	}
	sb.WriteString(title + "\n\n")
	for _, r := range rels {
		label := r.Type
		if r.GoldType != "" && r.GoldType != r.Type {
			label = r.Type + " → 正解 " + r.GoldType
		}
		// cos は生成側の src が正解側の何に対応付いたかの類似度。gold 側の行には無い。
		head := fmt.Sprintf("- **%s**（%s, src↔dst cos %.3f）", label, r.Side, r.PairSimilarity)
		if r.Side == "pred" {
			head = fmt.Sprintf("- **%s**（%s, src↔dst cos %.3f / src の対応 cos %.3f）",
				label, r.Side, r.PairSimilarity, r.SrcMatchSimilarity)
		}
		fmt.Fprintf(sb, "%s\n  - src: %s\n  - dst: %s\n", head, r.SrcText, r.DstText)
		if r.SrcMatch != "" || r.DstMatch != "" {
			fmt.Fprintf(sb, "  - 対応先: src=%s / dst=%s\n", r.SrcMatch, r.DstMatch)
		}
		if r.Rationale != "" {
			sb.WriteString("  - LLM の根拠: " + r.Rationale + "\n")
		}
	}
	sb.WriteString("\n")
}

// mergeCounts は混同行列 src を dst に足し込む。
func mergeCounts(dst, src map[string]map[string]int) {
	for p, row := range src {
		if dst[p] == nil {
			dst[p] = map[string]int{}
		}
		for g, v := range row {
			dst[p][g] += v
		}
	}
}

// ratio は x / y を小数 3 桁で返す。y が 0 なら "-"。
func ratio(x float64, y int) string {
	if y == 0 {
		return "-"
	}
	return fmt.Sprintf("%.3f", x/float64(y))
}

// micro は全ノードセットをプールした TP・生成本数・正解本数から P/R/F1 を求める。
func micro(tp, pred, gold int) evaluation.PRF {
	var p evaluation.PRF
	if pred > 0 {
		p.Precision = float64(tp) / float64(pred)
	}
	if gold > 0 {
		p.Recall = float64(tp) / float64(gold)
	}
	if p.Precision+p.Recall > 0 {
		p.F1 = 2 * p.Precision * p.Recall / (p.Precision + p.Recall)
	}
	return p
}
