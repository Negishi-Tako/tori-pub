// Command aif-eval は「LLM に素の AIF を書かせて、AIFdb のゴールドと突き合わせる」
// 実験を回す CLI。
//
// 手順:
//
//	aif-eval -fetch -corpus qt30 -subcorpora 6       # ゴールドと素テキストを取得
//	aif-eval -run -split test -sample 124 -prompt-set v3
//	aif-eval -report                                 # 集計 + 検定 + 証拠の書き出し
//	aif-eval -goldcheck                              # ゴールド自体の AIF メタモデル違反を数える
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/aif"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/evaluation"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

// 条件名。必ず「LLM 無し」の基準線を 1 本置く。
const (
	condBaseline = "baseline" // LLM を使わない下限（発話 = 命題、関係なし）
	condGoldLoc  = "gold-loc" // ゴールドの L-node を与え、YA/I/RA/CA/MA だけ LLM に任せる
	condE2E      = "e2e"      // 素の書き起こしから全部 LLM に任せる
)

func run() error {
	var (
		doFetch     = flag.Bool("fetch", false, "AIFdb からゴールドと素テキストを取得する")
		doRun       = flag.Bool("run", false, "アノテーションと比較を実行する")
		doReport    = flag.Bool("report", false, "結果を集計してレポート素材を書き出す")
		doGoldcheck = flag.Bool("goldcheck", false, "ゴールド自体の AIF メタモデル違反を数える")
		corpus      = flag.String("corpus", "qt30", "AIFdb のコーパス shortname")
		subcorpora  = flag.Int("subcorpora", 3, "取得するサブコーパス数（先頭から）")
		dataDir     = flag.String("data", "data/aifdb", "取得したノードセットの置き場")
		outDir      = flag.String("out", "out/aif", "結果の書き出し先")
		sample      = flag.Int("sample", 12, "評価するノードセット数")
		minLoc      = flag.Int("min-locutions", 8, "ゴールドの L-node がこの数未満のノードセットは使わない")
		maxLoc      = flag.Int("max-locutions", 60, "ゴールドの L-node がこの数を超えるノードセットは使わない")
		conditions  = flag.String("conditions", condBaseline+","+condGoldLoc+","+condE2E, "実行する条件（カンマ区切り）")
		model       = flag.String("model", "", "モデル（既定は MODEL_MINING）")
		threshold   = flag.Float64("threshold", 0.55, "命題が対応付いたと数えるコサイン類似度の下限")
		seed        = flag.Int64("seed", 20260909, "サンプリングの乱数種（決定的に選ぶ）")
		promptSet   = flag.String("prompt-set", "v3", "プロンプト一式の版（"+strings.Join(slices.Sorted(maps.Keys(promptSets)), " / ")+"）")
		metrics     = flag.String("metrics", strings.Join(evaluation.AIFMetrics, ","), "計算する GED 実装の版（カンマ区切り）")
		split       = flag.String("split", splitAll, "使う分割（dev / test / all）。詳細は splits を参照")
	)
	flag.Parse()

	if !*doFetch && !*doRun && !*doReport && !*doGoldcheck {
		flag.Usage()
		return errors.New("-fetch / -run / -report / -goldcheck のいずれかを指定してください")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *doFetch {
		if err := fetch(ctx, logger, *corpus, *subcorpora, *dataDir); err != nil {
			return err
		}
	}
	if *doGoldcheck {
		if err := goldcheck(*dataDir); err != nil {
			return err
		}
	}
	if *doRun {
		cfg, err := loadConfig()
		if err != nil {
			return err
		}
		if *model == "" {
			*model = cfg.ModelMining
		}
		if err := runExperiment(ctx, logger, cfg, runOptions{
			DataDir:    *dataDir,
			OutDir:     *outDir,
			Sample:     *sample,
			MinLoc:     *minLoc,
			MaxLoc:     *maxLoc,
			Conditions: splitList(*conditions),
			Model:      *model,
			Threshold:  *threshold,
			Seed:       *seed,
			PromptSet:  *promptSet,
			Metrics:    splitList(*metrics),
			Split:      *split,
		}); err != nil {
			return err
		}
	}
	if *doReport {
		if err := report(*outDir); err != nil {
			return err
		}
	}
	return nil
}

// splitList はカンマ区切りの値を、前後の空白と空要素を除いて返す。
func splitList(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}

// goldcheck はゴールド（正規化後）の AIF メタモデル違反を規則ごとに数える。
// レポートに「正解データにもこれだけ逸脱がある」を書くための点検用。
func goldcheck(dataDir string) error {
	docs, err := loadDocs(dataDir)
	if err != nil {
		return err
	}
	counts := map[string]int{}
	total := 0
	for _, d := range docs {
		g, _ := aif.Normalize(d.Graph)
		for _, v := range aif.Validate(g) {
			counts[v.Rule]++
			total++
		}
	}
	for _, rule := range slices.Sorted(maps.Keys(counts)) {
		fmt.Println(rule, counts[rule])
	}
	fmt.Println("total", total, "over", len(docs), "nodesets")
	return nil
}
