package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/fit"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/simulate"
)

type simOptions struct {
	sources, boot int
	seed          uint64
	lambda, m     float64
	sigma         float64
	null          bool
	out           string
}

func runSimulate(o simOptions) error {
	cfg := simulate.Default()
	cfg.Sources, cfg.Seed, cfg.Sigma = o.sources, o.seed, o.sigma
	if o.null {
		cfg.True = ged.Current() // a world where the current values are correct (checks whether "keep" can be concluded)
	}
	trips, js, metas, err := simulate.Generate(cfg)
	if err != nil {
		return err
	}
	names := make([]string, len(cfg.Judges))
	truth := make([]judgeTruth, len(cfg.Judges))
	for j, jd := range cfg.Judges {
		names[j] = fmt.Sprintf("合成 %d", j+1)
		truth[j] = judgeTruth{jd.Beta, jd.Tau, jd.Pi}
	}
	fmt.Fprintf(os.Stderr, "generated %d triplets, %d judgments\n", len(trips), len(js))

	opts := fit.Options{Judges: len(cfg.Judges), Theta0: ged.Current(), Lambda: o.lambda, FitNeg: true, MaxOuter: 20}
	res, err := fit.Fit(trips, js, opts, nil)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fit: outer=%d converged=%v nll=%.1f\n", res.Outer, res.Converged, res.NegLogLik)
	boot, err := fit.Bootstrap(trips, js, opts, res.Params, o.boot, o.seed+100)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "bootstrap done")
	cv, err := fit.CrossValidate(trips, js, opts, 5, o.seed)
	if err != nil {
		return err
	}

	d0, err := fit.Delta(trips, ged.Current())
	if err != nil {
		return err
	}
	dEst, err := fit.Delta(trips, res.Params.Theta)
	if err != nil {
		return err
	}
	dTrue, err := fit.Delta(trips, cfg.True)
	if err != nil {
		return err
	}
	types := make([]perturb.TripletType, len(trips))
	sign := make([]int, len(trips))
	neg := make([]bool, len(trips))
	for i, m := range metas {
		types[i] = m.Type
		switch {
		case m.TrueDelta > 1e-9:
			sign[i] = 1
		case m.TrueDelta < -1e-9:
			sign[i] = -1
		}
		neg[i] = trips[i].NegDiff != 0
	}

	var b bytes.Buffer
	title := "推定はコストを取り戻せるか（θ* を現行値からずらした世界）"
	if o.null {
		title = "現行値が正しい世界で「維持」と判断できるか（θ* = θ₀）"
	}
	fmt.Fprintln(&b, "# シミュレーション（API 不使用）: "+title)
	fmt.Fprintf(&b, "\n- 三つ組 %d（元の議論 %d × %d）、判定 %d（判定者 %d × 提示順 2）\n", len(trips), cfg.Sources, cfg.PerSrc, len(js), len(cfg.Judges))
	fmt.Fprintf(&b, "- 共有のずれ σ = %.2f、無作為に答える確率 %.0f%%、seed = %d\n", cfg.Sigma, 100*cfg.Random, cfg.Seed)
	fmt.Fprintf(&b, "- 正則化 λ = %.2f（θ₀ を中心）、交互最適化 %d 回（収束: %v、循環: %v）\n", o.lambda, res.Outer, res.Converged, res.Cycled)
	paramTable(&b, ged.Current(), &cfg.True, &cfg.TrueNeg, res.Params, boot, o.m)
	auxTable(&b, names, res.Params, truth)
	agreementTable(&b, []agreeRow{
		{"R0 現行のコスト表", d0},
		{"R2 学習したコスト（全データ、学習に使ったもの）", dEst},
		{"R2 学習したコスト（5 分割交差検証）", cv},
		{"真のコスト θ*（極性反転は見えない）", dTrue},
	}, js, neg)
	judgeDiagnostics(&b, names, types, sign, js)
	if err := os.WriteFile(o.out, b.Bytes(), 0o644); err != nil {
		return err
	}
	fmt.Print(b.String())
	return nil
}
