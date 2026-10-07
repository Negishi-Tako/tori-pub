SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help: ## このヘルプを表示
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: ## fmt → vet → test
	gofmt -l cmd internal | tee /dev/stderr | (! read)
	go vet ./...
	go test ./...

.PHONY: aif-fetch
aif-fetch: ## AIFdb からゴールドと素テキストを取得（例: ARGS="-corpus qt30 -subcorpora 6"）
	go run ./cmd/aif-eval -fetch $(ARGS)

.PHONY: aif-run
aif-run: ## 抽出と評価を実行（例: ARGS="-split test -sample 124 -prompt-set v3"）
	go run ./cmd/aif-eval -run $(ARGS)

.PHONY: aif-report
aif-report: ## 集計・検定・証拠を書き出す（例: ARGS="-out out/aif"）
	go run ./cmd/aif-eval -report $(ARGS)

.PHONY: aif-goldcheck
aif-goldcheck: ## ゴールド自体の AIF メタモデル適合を数える
	go run ./cmd/aif-eval -goldcheck $(ARGS)

.PHONY: figures
figures: ## レポートの図を再生成（out/aif-n124-*/ の生出力が必要。Python + matplotlib）
	python3 scripts/make_report_figs.py
