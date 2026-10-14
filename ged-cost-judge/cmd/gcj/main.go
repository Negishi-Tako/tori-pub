// Command gcj is the CLI for the experiment that estimates GED edit costs from judgments.
//
//	gcj -simulate   check with synthetic judgments whether estimation can recover the costs (no API, plan P1)
//	gcj -pilot      build source arguments, variants and judgments with luna, and estimate end to end (PoC)
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

func main() {
	var (
		doSim   = flag.Bool("simulate", false, "合成の判定でパラメータの回収を確かめる（API 不使用）")
		doPilot = flag.Bool("pilot", false, "luna で通しの PoC を実行する（API を使う）")
		sources = flag.Int("sources", 250, "元の議論の数（1 件につき三つ組 4 つ）")
		seed    = flag.Uint64("seed", 1, "乱数の seed")
		boot    = flag.Int("boot", 100, "クラスタブートストラップの回数")
		lambda  = flag.Float64("lambda", 0.5, "θ₀ を中心とする L2 正則化の係数")
		margin  = flag.Float64("m", 0.3, "判断の幅 m")
		sigma   = flag.Float64("sigma", 1, "合成: 三つ組ごとに判定者が共有するずれ σ")
		null    = flag.Bool("null", false, "合成: 真のコストを現行値にする")
		verify  = flag.Bool("verify", false, "PoC: 書き下しの検証つきの手順を使う（realize_v2 + verify_v1、T3 の割り付け）")
		outDir  = flag.String("out", "out", "出力先")
		budget  = flag.Float64("budget", 3, "PoC: API の費用の上限（USD）。超えたら以降の呼び出しを止める")
	)
	flag.Parse()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	var err error
	switch {
	case *doSim:
		err = runSimulate(simOptions{
			sources: *sources, seed: *seed, boot: *boot, lambda: *lambda, m: *margin, sigma: *sigma, null: *null,
			out: filepath.Join(*outDir, map[bool]string{false: "simulate.md", true: "simulate_null.md"}[*null]),
		})
	case *doPilot:
		err = runPilot(pilotOptions{
			sources: *sources, seed: *seed, boot: *boot, lambda: *lambda, m: *margin, out: *outDir, budget: *budget, verify: *verify,
		})
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "done in %s\n", time.Since(start).Round(time.Millisecond))
}
