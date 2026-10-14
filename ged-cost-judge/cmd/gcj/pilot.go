package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/joho/godotenv"

	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/fit"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/ged"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/llm"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/perturb"
	"github.com/negishi-tako/tori-pub/ged-cost-judge/internal/synth"
)

type pilotOptions struct {
	sources, boot int
	seed          uint64
	lambda, m     float64
	out           string
	budget        float64
	// verify は書き下しの検証つきの手順（realize_v2 + verify_v1、T3 の割り付け）を使う。
	verify bool
}

var (
	formats = []string{
		"TV debate with a moderator", "meeting", "one-on-one argument",
		"online forum thread", "opinion essay by a single author", "interview",
	}
	domains = []string{
		"politics", "economy", "healthcare", "education", "environment", "technology",
		"law and justice", "sports", "local government", "business management", "science", "daily life",
	}
	structures = []string{
		"chain (each claim supports the next)",
		"convergent (several premises support one conclusion)",
		"divergent (one claim supports several conclusions)",
		"conflict-heavy (at least 3 attacks)",
		"restatement-heavy (at least 3 restatements)",
	}
)

// meter は API の費用を数え、上限を超えたら以降の呼び出しを止める。
type meter struct {
	mu     sync.Mutex
	limit  float64
	byTag  map[string]llm.Usage
	calls  map[string]int
	cached map[string]int
}

var errBudget = errors.New("pilot: API の費用が上限に達した")

func (m *meter) add(tag string, r *llm.Response) {
	if r == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.byTag[tag] = m.byTag[tag].Add(r.Usage)
	if r.Cached {
		m.cached[tag]++
	} else {
		m.calls[tag]++
	}
}

func (m *meter) total() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	t := 0.0
	for _, u := range m.byTag {
		t += u.CostUSD
	}
	return t
}

func (m *meter) check() error {
	if m.total() >= m.limit {
		return errBudget
	}
	return nil
}

func parallel(n, workers int, f func(i int) error) []error {
	errs := make([]error, n)
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for i := range n {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			errs[i] = f(i)
		}(i)
	}
	wg.Wait()
	return errs
}

type pilotTriplet struct {
	Source int                 `json:"source"`
	Type   perturb.TripletType `json:"type"`
	A, B   perturb.Variant     `json:"-"`
	TextA  synth.Discussion    `json:"a"`
	TextB  synth.Discussion    `json:"b"`
	EditsA []perturb.Edit      `json:"edits_a"`
	EditsB []perturb.Edit      `json:"edits_b"`
	StatA  synth.RealizeStats  `json:"realize_a"`
	StatB  synth.RealizeStats  `json:"realize_b"`
	Raw    []string            `json:"judge_raw"` // 提示順 +1, −1 の生の回答
	// Unrealized は、命題グラフを変える編集があるのにテキストが元の議論と同じ変種を含むこと。
	Unrealized bool   `json:"unrealized"`
	Err        string `json:"error,omitempty"`

	// 以下は検証つきの手順（-verify）だけで埋まる。
	Replans    int          `json:"replans,omitempty"`
	NearTarget *bool        `json:"near_target,omitempty"`
	PredDelta  float64      `json:"pred_delta,omitempty"`
	CheckA     variantCheck `json:"check_a,omitzero"`
	CheckB     variantCheck `json:"check_b,omitzero"`
}

func runPilot(o pilotOptions) error {
	_ = godotenv.Load(".env")
	_ = godotenv.Load("../aif-with-llm/.env") // 前実験の .env（OpenAI のキー）を流用する
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return errors.New("pilot: OPENAI_API_KEY が無い")
	}
	model := cmp(os.Getenv("GCJ_MODEL"), "gpt-5.6-luna")
	ctx := context.Background()

	base, err := llm.NewOpenAI(llm.OpenAIConfig{APIKey: key, MaxRetries: 2, Timeout: 5 * time.Minute})
	if err != nil {
		return err
	}
	client := llm.NewCachingClient(llm.NewThrottledClient(base, 4, 4, llm.DefaultRetryConfig()), llm.NewFileCache(".cache/llm"))
	embBase, err := llm.NewOpenAIEmbedder(llm.OpenAIEmbedderConfig{
		OpenAIConfig: llm.OpenAIConfig{APIKey: key, MaxRetries: 2, Timeout: 2 * time.Minute},
		Model:        "text-embedding-3-small",
	})
	if err != nil {
		return err
	}
	emb := llm.NewCachingEmbedder(embBase, ".cache/embed")

	g := &synth.Generator{Client: client, Model: model}
	for _, p := range []struct {
		dst  *synth.Prompt
		name string
	}{{&g.SourcePrompt, "synth_source_v1"}, {&g.RealizePrompt, map[bool]string{false: "realize_v1", true: "realize_v2"}[o.verify]},
		{&g.JudgePrompt, "judge_v1"}, {&g.VerifyPrompt, "verify_v1"}} {
		if *p.dst, err = synth.LoadPrompt("prompts", p.name); err != nil {
			return err
		}
	}
	mt := &meter{limit: o.budget, byTag: map[string]llm.Usage{}, calls: map[string]int{}, cached: map[string]int{}}
	rng := rand.New(rand.NewPCG(o.seed, 0xb11d))

	// ---- G1 元の議論 ----
	specs := make([]synth.Spec, o.sources)
	for i := range specs {
		f := formats[rng.IntN(len(formats))]
		sp := 2 + rng.IntN(3)
		if f == formats[4] {
			sp = 1
		}
		props := 6 + rng.IntN(5)
		if rng.IntN(2) == 1 {
			props = 11 + rng.IntN(8)
		}
		specs[i] = synth.Spec{Format: f, Domain: domains[rng.IntN(len(domains))], Speakers: sp, Props: props,
			Structure: structures[rng.IntN(len(structures))]}
	}
	sources := make([]synth.Discussion, o.sources)
	srcErr := parallel(o.sources, 8, func(i int) error {
		var last error
		for attempt := range 3 {
			if err := mt.check(); err != nil {
				return err
			}
			d, resp, err := g.GenerateSource(ctx, specs[i], fmt.Sprintf("s%d.", i), attempt)
			mt.add("synth_source", resp)
			if err == nil {
				sources[i] = d
				return nil
			}
			last = err
		}
		return last
	})
	okSrc := 0
	for _, e := range srcErr {
		if e == nil {
			okSrc++
		}
	}
	fmt.Fprintf(os.Stderr, "sources: %d/%d ok, cost $%.3f\n", okSrc, o.sources, mt.total())

	var trips []*pilotTriplet
	var vstats *verifyStats
	if o.verify {
		// ---- G2〜G4 編集計画・書き下し・検証（枠ごと）----
		trips, vstats = buildVerified(ctx, g, mt, sources, srcErr, rng, o.seed)
	} else {
		// ---- G2 編集計画 ----
		for i, src := range sources {
			if srcErr[i] != nil {
				continue
			}
			for range 4 {
				typ := perturb.DrawType(rng)
				ka, kb := perturb.Plan(rng, typ)
				t := &pilotTriplet{Source: i, Type: typ}
				if typ == perturb.T1Monotone {
					steps := perturb.Build(rng, src.Graph, kb)
					t.A, t.B = steps[min(len(ka), len(steps))-1], steps[len(steps)-1]
				} else {
					t.A = lastOf(perturb.Build(rng, src.Graph, ka))
					t.B = lastOf(perturb.Build(rng, src.Graph, kb))
				}
				t.EditsA, t.EditsB = t.A.Edits, t.B.Edits
				trips = append(trips, t)
			}
		}

		// ---- G3 書き下し ----
		parallel(len(trips), 8, func(i int) error {
			t := trips[i]
			src := sources[t.Source]
			prefix := fmt.Sprintf("s%d.", t.Source)
			for _, side := range []struct {
				v    perturb.Variant
				dst  *synth.Discussion
				stat *synth.RealizeStats
			}{{t.A, &t.TextA, &t.StatA}, {t.B, &t.TextB, &t.StatB}} {
				if err := mt.check(); err != nil {
					t.Err = err.Error()
					return err
				}
				d, st, resp, err := g.Realize(ctx, src, side.v, prefix, 0, "")
				mt.add("realize", resp)
				if err != nil {
					t.Err = "realize: " + err.Error()
					return err
				}
				*side.dst, *side.stat = d, st
			}
			return nil
		})
	}
	fmt.Fprintf(os.Stderr, "realized, cost $%.3f\n", mt.total())

	// ---- 埋め込み（命題の cos）----
	textSet := map[string]bool{}
	for _, s := range sources {
		for _, p := range s.Graph.Props {
			textSet[p.Text] = true
		}
	}
	for _, t := range trips {
		if t.Err != "" {
			continue
		}
		for _, p := range append(slices.Clone(t.TextA.Graph.Props), t.TextB.Graph.Props...) {
			textSet[p.Text] = true
		}
	}
	texts := make([]string, 0, len(textSet))
	for s := range textSet {
		if s != "" {
			texts = append(texts, s)
		}
	}
	sort.Strings(texts)
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("pilot: embed: %w", err)
	}
	vec := make(map[string][]float32, len(texts))
	for i, s := range texts {
		vec[s] = vecs[i]
	}
	simFor := func(src ged.Graph, v ged.Graph) ged.Sim {
		ref := map[string]string{}
		for _, p := range src.Props {
			ref[p.ID] = p.Text
		}
		table := map[[2]string]float64{}
		for _, r := range src.Props {
			for _, p := range v.Props {
				c := 1.0
				if r.Text != p.Text {
					c = cosine(vec[r.Text], vec[p.Text])
				}
				table[[2]string{r.ID, p.ID}] = c
			}
		}
		return func(a, b string) float64 { return table[[2]string{a, b}] }
	}

	// ---- J 判定（提示順 2 通り）----
	parallel(len(trips), 8, func(i int) error {
		t := trips[i]
		if t.Err != "" {
			return nil
		}
		s := sources[t.Source]
		t.Raw = make([]string, 2)
		for k, pair := range [][2]synth.Discussion{{t.TextA, t.TextB}, {t.TextB, t.TextA}} {
			if err := mt.check(); err != nil {
				t.Err = err.Error()
				return err
			}
			ans, resp, err := g.JudgeAnswer(ctx, s, pair[0], pair[1])
			mt.add("judge", resp)
			if err != nil {
				t.Err = "judge: " + err.Error()
				return err
			}
			t.Raw[k] = ans
		}
		return nil
	})
	fmt.Fprintf(os.Stderr, "judged, cost $%.3f\n", mt.total())

	for _, t := range trips {
		if t.Err != "" {
			continue
		}
		src := sources[t.Source]
		for _, side := range []struct {
			v perturb.Variant
			d synth.Discussion
		}{{t.A, t.TextA}, {t.B, t.TextB}} {
			if changesGraph(side.v.Edits) && slices.Equal(side.d.Utterances, src.Utterances) {
				t.Unrealized = true
			}
		}
	}

	// ---- 推定 ----
	var ft []fit.Triplet
	var js []fit.Judgment
	var kept []*pilotTriplet
	for _, t := range trips {
		if t.Err != "" {
			continue
		}
		s := sources[t.Source]
		idx := len(ft)
		ft = append(ft, fit.Triplet{
			Source: t.Source, Ref: s.Graph, A: t.TextA.Graph, B: t.TextB.Graph,
			SimA: simFor(s.Graph, t.TextA.Graph), SimB: simFor(s.Graph, t.TextB.Graph),
			NegDiff: float64(t.B.Neg - t.A.Neg),
		})
		kept = append(kept, t)
		for k, ord := range []float64{1, -1} {
			js = append(js, fit.Judgment{Triplet: idx, Judge: 0, Order: ord, Answer: mapAnswer(t.Raw[k], ord)})
		}
	}
	if len(ft) < 10 {
		return fmt.Errorf("pilot: 使える三つ組が %d しかない", len(ft))
	}
	opts := fit.Options{Judges: 1, Theta0: ged.Current(), Lambda: o.lambda, FitNeg: true, MaxOuter: 20, AuxRidge: 0.5}
	res, err := fit.Fit(ft, js, opts, nil)
	if err != nil {
		return err
	}
	boot, err := fit.Bootstrap(ft, js, opts, res.Params, o.boot, o.seed+100)
	if err != nil {
		return err
	}
	cv, err := fit.CrossValidate(ft, js, opts, 5, o.seed)
	if err != nil {
		return err
	}
	d0, _ := fit.Delta(ft, ged.Current())
	dEst, _ := fit.Delta(ft, res.Params.Theta)
	naive := ged.Current()
	naive[ged.PTypeRACA], naive[ged.PTypeRAMA], naive[ged.PTypeCAMA], naive[ged.PDirection] = 2, 2, 2, 2
	dNaive, _ := fit.Delta(ft, naive)

	types := make([]perturb.TripletType, len(kept))
	sign := make([]int, len(kept))
	neg := make([]bool, len(kept))
	for i, t := range kept {
		types[i] = t.Type
		if t.Type == perturb.T1Monotone && math.Abs(d0[i]) > 1e-9 {
			sign[i] = 1 // A の編集 ⊂ B の編集なので、A が近いのが正解
		}
		neg[i] = ft[i].NegDiff != 0
	}

	if err := writeProfiles(filepath.Join(o.out, "profile.csv"), ft, js, opts, res.Params); err != nil {
		return err
	}

	// 機械検査で書き下しの失敗を含む三つ組を除いた場合（計画 G4 の代わりの簡易検査）。
	var keepIdx []int
	for i, t := range kept {
		if !t.Unrealized {
			keepIdx = append(keepIdx, i)
		}
	}
	ft2, js2 := pick(ft, js, keepIdx)
	res2, err := fit.Fit(ft2, js2, opts, nil)
	if err != nil {
		return err
	}
	boot2, err := fit.Bootstrap(ft2, js2, opts, res2.Params, o.boot, o.seed+200)
	if err != nil {
		return err
	}
	cv2, err := fit.CrossValidate(ft2, js2, opts, 5, o.seed)
	if err != nil {
		return err
	}
	d02, _ := fit.Delta(ft2, ged.Current())
	dNaive2, _ := fit.Delta(ft2, naive)
	neg2 := make([]bool, len(ft2))
	for i := range ft2 {
		neg2[i] = ft2[i].NegDiff != 0
	}

	// ---- 報告 ----
	var b bytes.Buffer
	fmt.Fprintf(&b, "# PoC: %s で通しの実行（元の議論 %d 件）\n", model, o.sources)
	fmt.Fprintf(&b, "\n- 元の議論: %d/%d 件を生成（構造の検査に通ったもの）\n", okSrc, o.sources)
	nErr := len(trips) - len(kept)
	fmt.Fprintf(&b, "- 三つ組: %d 組を計画し、%d 組を書き下し・判定まで通した（失敗 %d）\n", len(trips), len(kept), nErr)
	fmt.Fprintf(&b, "- 判定: %d 回（判定者 1 × 提示順 2）\n", len(js))
	fmt.Fprintf(&b, "- プロンプト: %s=%s, %s=%s, %s=%s", g.SourcePrompt.Name, g.SourcePrompt.Hash, g.RealizePrompt.Name, g.RealizePrompt.Hash, g.JudgePrompt.Name, g.JudgePrompt.Hash)
	if o.verify {
		fmt.Fprintf(&b, ", %s=%s", g.VerifyPrompt.Name, g.VerifyPrompt.Hash)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "- 正則化 λ = %.2f（θ₀ を中心）、補助パラメータ（log β・log τ・π）に係数 %.2f の L2、交互最適化 %d 回（収束: %v、循環: %v）\n", o.lambda, opts.AuxRidge, res.Outer, res.Converged, res.Cycled)

	fmt.Fprintln(&b, "\n## 費用")
	fmt.Fprintln(&b, "\n| 用途 | API 呼び出し | キャッシュ | 入力トークン | 出力トークン | 費用 |")
	fmt.Fprintln(&b, "| --- | --: | --: | --: | --: | --: |")
	tags := []string{"synth_source", "realize", "verify", "judge"}
	var tot llm.Usage
	for _, tg := range tags {
		u := mt.byTag[tg]
		tot = tot.Add(u)
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d | $%.4f |\n", tg, mt.calls[tg], mt.cached[tg], u.InputTokens, u.OutputTokens, u.CostUSD)
	}
	fmt.Fprintf(&b, "| 計 | | | %d | %d | $%.4f |\n", tot.InputTokens, tot.OutputTokens, tot.CostUSD)
	fmt.Fprintln(&b, "\n埋め込み（text-embedding-3-small）は数セント未満なので表に入れていない。キャッシュから返した呼び出しは $0 として数える。")
	writeCostProjection(&b, mt)
	if vstats != nil {
		writeVerifyReport(&b, vstats, kept, d0)
	}

	var connective, missing, changed, realized int
	for _, t := range kept {
		for _, st := range []synth.RealizeStats{t.StatA, t.StatB} {
			if st.Changed > 0 {
				realized++
			}
			changed += st.Changed
			connective += st.Connective
			missing += st.MissingText
		}
	}
	fmt.Fprintln(&b, "\n## 書き下しの機械検査")
	fmt.Fprintf(&b, "\n- LLM で書き下した変種 %d 個、変更した発話 %d（うち接続の手直し %d）\n", realized, changed, connective)
	fmt.Fprintf(&b, "- 命題の新しい本文が返らなかったもの: %d（元の本文で代用した）\n", missing)
	unrealized := map[perturb.Kind][2]int{}
	for _, t := range kept {
		src := sources[t.Source]
		for _, side := range []struct {
			v perturb.Variant
			d synth.Discussion
		}{{t.A, t.TextA}, {t.B, t.TextB}} {
			same := slices.Equal(side.d.Utterances, src.Utterances)
			seen := map[perturb.Kind]bool{}
			for _, e := range side.v.Edits {
				if seen[e.Kind] || e.Kind == perturb.XSpk {
					continue
				}
				seen[e.Kind] = true
				c := unrealized[e.Kind]
				c[0]++
				if same {
					c[1]++
				}
				unrealized[e.Kind] = c
			}
		}
	}
	fmt.Fprintln(&b, "- 計画にある別系列のモデルによる検証と自然さの評価（G4）は、この PoC では行っていない。代わりに「命題グラフを変える編集があるのに、テキストが元の議論と 1 文字も変わっていない変種」を数えた（書き下しの失敗の下限）")
	fmt.Fprintln(&b, "\n| 編集 | その編集を含む変種 | テキストが元のまま |")
	fmt.Fprintln(&b, "| --- | --: | --: |")
	for _, k := range slices.Sorted(maps.Keys(unrealized)) {
		c := unrealized[k]
		fmt.Fprintf(&b, "| %s | %d | %d（%.0f%%） |\n", k, c[0], c[1], pct(c[1], c[0]))
	}
	fmt.Fprintf(&b, "\nこの失敗を含む三つ組は %d / %d 組。\n", len(kept)-len(keepIdx), len(kept))

	judgeDiagnostics(&b, []string{model}, types, sign, js)
	agreementTable(&b, []agreeRow{
		{"R0 現行のコスト表", d0},
		{"R1 素朴な一様コスト（種別違い・向き違い = 2.0）", dNaive},
		{"R2 学習したコスト（学習に使ったデータ）", dEst},
		{"R2 学習したコスト（5 分割交差検証）", cv},
	}, js, neg)
	paramTable(&b, ged.Current(), nil, nil, res.Params, boot, o.m)
	auxTable(&b, []string{model}, res.Params, nil)
	writeTypeBreakdown(&b, kept, js, d0)

	fmt.Fprintf(&b, "\n# 書き下しの失敗を除いた場合（%d 組）\n", len(ft2))
	fmt.Fprintf(&b, "\n交互最適化 %d 回（収束: %v、循環: %v）\n", res2.Outer, res2.Converged, res2.Cycled)
	agreementTable(&b, []agreeRow{
		{"R0 現行のコスト表", d02},
		{"R1 素朴な一様コスト（種別違い・向き違い = 2.0）", dNaive2},
		{"R2 学習したコスト（5 分割交差検証）", cv2},
	}, js2, neg2)
	paramTable(&b, ged.Current(), nil, nil, res2.Params, boot2, o.m)

	if err := os.WriteFile(filepath.Join(o.out, "pilot.md"), b.Bytes(), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(o.out, "pilot_sources.json"), synth.MarshalIndent(sources), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(o.out, "pilot_triplets.json"), synth.MarshalIndent(trips), 0o644); err != nil {
		return err
	}
	fmt.Print(b.String())
	if fails := errors.Join(srcErr...); fails != nil && errors.Is(fails, errBudget) {
		return errBudget
	}
	return nil
}

func writeCostProjection(b *bytes.Buffer, mt *meter) {
	j := mt.byTag["judge"]
	r := mt.byTag["realize"]
	s := mt.byTag["synth_source"]
	nj, nr, ns := mt.calls["judge"], mt.calls["realize"], mt.calls["synth_source"]
	if nj == 0 || nr == 0 || ns == 0 {
		return
	}
	per := func(u llm.Usage, n int) (float64, float64) {
		return float64(u.InputTokens) / float64(n), float64(u.OutputTokens) / float64(n)
	}
	ji, jo := per(j, nj)
	ri, ro := per(r, nr)
	si, so := per(s, ns)
	fmt.Fprintln(b, "\n### 1 回あたりの実測トークン（計画 §10.2 の見積もりとの比較）")
	fmt.Fprintln(b, "\n| 段階 | 実測 入力 / 出力 | 計画の見積もり |")
	fmt.Fprintln(b, "| --- | --- | --- |")
	fmt.Fprintf(b, "| G1 元の議論の生成 | %.0f / %.0f | 1.5k / 5k |\n", si, so)
	fmt.Fprintf(b, "| G3 書き下し | %.0f / %.0f | 4k / 3k |\n", ri, ro)
	fmt.Fprintf(b, "| J 判定 | %.0f / %.0f | 4.5k / 1.5k |\n", ji, jo)
	// 第 1 段（1,000 三つ組）の費用を実測トークンで置き換えて見積もる（§10.1 の回数）。
	price := map[string][2]float64{
		"luna":   {0.20, 1.20},
		"sonnet": {1.0, 5.0}, // Sonnet 5.5 のバッチ単価（$2 / $10 の半額）
		"haiku":  {0.5, 2.5}, // Haiku 4.5 のバッチ単価（$1 / $5 の半額）
	}
	cost := func(model string, n float64, in, out float64) float64 {
		p := price[model]
		return n * (in*p[0] + out*p[1]) / 1e6
	}
	// 検証（G1・G4）は書き下しと同程度の入力・生成の 1/2 の出力と仮定する。
	luna := cost("luna", 120, si, so) + cost("luna", 120+1100, ri, ro/2) + cost("luna", 1100, ri, ro) + cost("luna", 2000, ji, jo)
	sonnet := cost("sonnet", 120, si, so) + cost("sonnet", 120+1100, ri, ro/2) + cost("sonnet", 1100, ri, ro) + cost("sonnet", 2000, ji, jo)
	haiku := cost("haiku", 2000, ji, jo)
	fmt.Fprintln(b, "\n### 1,000 三つ組（第 1 段）の費用の見積もり（実測トークンで置き換え）")
	fmt.Fprintln(b, "\n他モデルも luna と同じトークン数を使うと仮定した概算（Claude 側はトークナイザも推論の量も違うので幅がある）。")
	fmt.Fprintln(b, "\n| 内訳 | 見積もり | 計画の値 |")
	fmt.Fprintln(b, "| --- | --: | --: |")
	fmt.Fprintf(b, "| OpenAI luna | $%.1f | 約 $17 |\n", luna)
	fmt.Fprintf(b, "| Anthropic Sonnet 5.5（バッチ） | $%.1f | 約 $63 |\n", sonnet)
	fmt.Fprintf(b, "| Anthropic Haiku 4.5（バッチ） | $%.1f | 約 $12 |\n", haiku)
	fmt.Fprintf(b, "| 計 | $%.1f | 約 $92 |\n", luna+sonnet+haiku)
}

func writeTypeBreakdown(b *bytes.Buffer, kept []*pilotTriplet, js []fit.Judgment, d0 []float64) {
	fmt.Fprintln(b, "\n## 三つ組の種類ごとの回答の分布")
	fmt.Fprintln(b, "\n| 種類 | 三つ組 | A | B | 同程度 | R0 の一致率 |")
	fmt.Fprintln(b, "| --- | --: | --: | --: | --: | --- |")
	for _, typ := range []perturb.TripletType{perturb.T1Monotone, perturb.T2SameKind, perturb.T3Contrast, perturb.T4Invariant} {
		mask := make([]bool, len(kept))
		n := 0
		var cnt [3]int
		for i, t := range kept {
			if t.Type == typ {
				mask[i] = true
				n++
			}
		}
		for _, k := range js {
			if mask[k.Triplet] {
				cnt[k.Answer]++
			}
		}
		a, na := fit.Agreement(d0, js, mask)
		fmt.Fprintf(b, "| %s | %d | %d | %d | %d | %s |\n", typ, n, cnt[0], cnt[1], cnt[2], agr(a, na))
	}
}

// writeProfiles は値段を 1 つずつ動かしたときの対数尤度を書き出す。
// slice は他のパラメータを推定値に固定した切り口、profile は他を当てはめ直したプロファイル尤度。
// どちらも各点で命題の対応を解き直す（本物の GED での値）。
func writeProfiles(path string, ft []fit.Triplet, js []fit.Judgment, opts fit.Options, est fit.Params) error {
	var b bytes.Buffer
	b.WriteString("param,value,slice_loglik,profile_loglik,assign_changes\n")
	for p := range ged.NumParams {
		var prevSig []string
		for k := 0; k <= 60; k++ {
			v := float64(k) * 0.05
			sp := est
			sp.Theta[p] = v
			nll, err := fit.NegLogLik(ft, js, sp)
			if err != nil {
				return err
			}
			po := opts
			po.Fixed = map[int]float64{p: v}
			pr, err := fit.Fit(ft, js, po, &est)
			if err != nil {
				return err
			}
			sigs, err := fit.Signatures(ft, sp.Theta)
			if err != nil {
				return err
			}
			changes := 0
			if prevSig != nil {
				for i := range sigs {
					if sigs[i] != prevSig[i] {
						changes++
					}
				}
			}
			prevSig = sigs
			fmt.Fprintf(&b, "%s,%.2f,%.4f,%.4f,%d\n", ged.ParamNames[p], v, -nll, -pr.NegLogLik, changes)
		}
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

func changesGraph(es []perturb.Edit) bool {
	for _, e := range es {
		if e.Kind != perturb.XSpk && e.Kind != perturb.XSurf {
			return true
		}
	}
	return false
}

// pick は idx の三つ組だけを残し、判定の添字を付け直す。
func pick(ft []fit.Triplet, js []fit.Judgment, idx []int) ([]fit.Triplet, []fit.Judgment) {
	pos := map[int]int{}
	var out []fit.Triplet
	for _, i := range idx {
		pos[i] = len(out)
		out = append(out, ft[i])
	}
	var oj []fit.Judgment
	for _, k := range js {
		if p, ok := pos[k.Triplet]; ok {
			k.Triplet = p
			oj = append(oj, k)
		}
	}
	return out, oj
}

func mapAnswer(raw string, order float64) fit.Answer {
	switch {
	case raw == "same":
		return fit.AnswerTie
	case (raw == "1") == (order > 0):
		return fit.AnswerA
	default:
		return fit.AnswerB
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}

func lastOf(vs []perturb.Variant) perturb.Variant { return vs[len(vs)-1] }

func cmp(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
