package main

import (
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/fit"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
)

// judgeDiagnostics reports the judge quality criteria (plan §5.4) per judge.
func judgeDiagnostics(w io.Writer, names []string, types []perturb.TripletType, trueSign []int, js []fit.Judgment) {
	fmt.Fprintln(w, "\n## 判定者の診断（計画 §5.4）")
	fmt.Fprintln(w, "\n| 判定者 | 判定数 | 同程度の割合 | 提示順を入れ替えても同じ | T1 正答率 |")
	fmt.Fprintln(w, "| --- | --: | --: | --: | --: |")
	for j, name := range names {
		var n, tie int
		byT := map[int][]fit.Answer{}
		t1n, t1hit := 0, 0
		for _, k := range js {
			if k.Judge != j {
				continue
			}
			n++
			if k.Answer == fit.AnswerTie {
				tie++
			}
			byT[k.Triplet] = append(byT[k.Triplet], k.Answer)
			if types[k.Triplet] == perturb.T1Monotone && trueSign[k.Triplet] != 0 {
				t1n++
				want := fit.AnswerA
				if trueSign[k.Triplet] < 0 {
					want = fit.AnswerB
				}
				if k.Answer == want {
					t1hit++
				}
			}
		}
		same, pairs := 0, 0
		for _, as := range byT {
			if len(as) == 2 {
				pairs++
				if as[0] == as[1] {
					same++
				}
			}
		}
		fmt.Fprintf(w, "| %s | %d | %.1f%% | %s | %s |\n", name, n, pct(tie, n), ratio(same, pairs), ratio(t1hit, t1n))
	}
}

func pct(a, b int) float64 {
	if b == 0 {
		return math.NaN()
	}
	return 100 * float64(a) / float64(b)
}

func ratio(a, b int) string {
	if b == 0 {
		return "—"
	}
	return fmt.Sprintf("%.1f%%（%d/%d）", pct(a, b), a, b)
}

// paramTable reports per-parameter estimates, intervals and decisions (plan §7.4).
func paramTable(w io.Writer, th0 ged.Theta, truth *ged.Theta, trueNeg *float64, est fit.Params, boot [][]float64, m float64) {
	fmt.Fprintln(w, "\n## パラメータの推定（計画 §6・§7.4）")
	fmt.Fprintf(w, "\n判断の幅 m = ±%.2f（置換関数の節点は幅で判断しない）。区間はクラスタブートストラップ %d 回のパーセンタイル。\n", m, len(boot))
	fmt.Fprint(w, "SD は外れた回に引きずられないよう、分位点から求めた値（(84% 点 − 16% 点) / 2）。\n\n")
	head := "| パラメータ | 現行値 θ₀ |"
	sep := "| --- | --: |"
	if truth != nil {
		head += " 真の値 θ* |"
		sep += " --: |"
	}
	fmt.Fprintln(w, head+" 推定値 | SD | 90% 区間 | 95% 区間 | 判断 |")
	fmt.Fprintln(w, sep+" --: | --: | --- | --- | --- |")
	for i := 0; i <= ged.NumParams; i++ {
		var col []float64
		for _, b := range boot {
			col = append(col, b[i])
		}
		slices.Sort(col)
		name, cur, e := "θ_neg（極性反転）", math.NaN(), est.ThetaNeg
		if i < ged.NumParams {
			name, cur, e = ged.ParamNames[i], th0[i], est.Theta[i]
		}
		lo90, hi90 := fit.Quantile(col, 0.05), fit.Quantile(col, 0.95)
		lo95, hi95 := fit.Quantile(col, 0.025), fit.Quantile(col, 0.975)
		decision := "—"
		isSub := i >= ged.PSub09 && i <= ged.PSub03
		if i < ged.NumParams && !isSub && len(col) > 0 {
			switch {
			case lo90 >= cur-m && hi90 <= cur+m:
				decision = "維持"
			case (lo95 > cur || hi95 < cur) && math.Abs(e-cur) >= m:
				decision = "変更"
			default:
				decision = "未決"
			}
		}
		row := fmt.Sprintf("| %s | %s |", name, num(cur))
		if truth != nil {
			tv := *trueNeg
			if i < ged.NumParams {
				tv = truth[i]
			}
			row += fmt.Sprintf(" %.2f |", tv)
		}
		sd := (fit.Quantile(col, 0.8413) - fit.Quantile(col, 0.1587)) / 2
		row += fmt.Sprintf(" %.2f | %.3f | [%.2f, %.2f] | [%.2f, %.2f] | %s |", e, sd, lo90, hi90, lo95, hi95, decision)
		fmt.Fprintln(w, row)
	}
}

func num(v float64) string {
	if math.IsNaN(v) {
		return "—"
	}
	return fmt.Sprintf("%.2f", v)
}

// auxTable is the per-judge auxiliary parameters.
func auxTable(w io.Writer, names []string, est fit.Params, truth []judgeTruth) {
	fmt.Fprintln(w, "\n## 判定者の補助パラメータ")
	if truth != nil {
		fmt.Fprintln(w, "\n| 判定者 | β（真） | β | τ（真） | τ | π（真） | π |")
		fmt.Fprintln(w, "| --- | --: | --: | --: | --: | --: | --: |")
		for j, n := range names {
			fmt.Fprintf(w, "| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n", n, truth[j].beta, est.Beta[j], truth[j].tau, est.Tau[j], truth[j].pi, est.Pi[j])
		}
		fmt.Fprintln(w, "\n（合成の判定には三つ組ごとの共有のずれ ε_t が入るので、推定の β は真の β より小さく出るのが正常。）")
		return
	}
	fmt.Fprintln(w, "\n| 判定者 | β 鋭さ | τ 同程度の幅 | π 位置バイアス |")
	fmt.Fprintln(w, "| --- | --: | --: | --: |")
	for j, n := range names {
		fmt.Fprintf(w, "| %s | %.2f | %.2f | %.2f |\n", n, est.Beta[j], est.Tau[j], est.Pi[j])
	}
}

type judgeTruth struct{ beta, tau, pi float64 }

// agreementTable is the agreement rate per rule (plan §7.3).
func agreementTable(w io.Writer, rows []agreeRow, js []fit.Judgment, negMask []bool) {
	fmt.Fprintln(w, "\n## 規則ごとの一致率（計画 §7.3）")
	fmt.Fprintln(w, "\nΔ の符号が判定の多数（A / B）と一致する割合。多数が「同程度」の三つ組は除く。")
	fmt.Fprintln(w, "\n| 規則 | 全体 | 極性反転なし | 極性反転あり |")
	fmt.Fprintln(w, "| --- | --- | --- | --- |")
	noNeg := make([]bool, len(negMask))
	for i, v := range negMask {
		noNeg[i] = !v
	}
	for _, r := range rows {
		a, n := fit.Agreement(r.delta, js, nil)
		b, nb := fit.Agreement(r.delta, js, noNeg)
		c, nc := fit.Agreement(r.delta, js, negMask)
		fmt.Fprintf(w, "| %s | %s | %s | %s |\n", r.name, agr(a, n), agr(b, nb), agr(c, nc))
	}
}

type agreeRow struct {
	name  string
	delta []float64
}

func agr(v float64, n int) string {
	if n == 0 {
		return "—"
	}
	return fmt.Sprintf("%.1f%%（n=%d）", 100*v, n)
}
