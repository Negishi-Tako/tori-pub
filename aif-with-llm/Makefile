SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: ## fmt → vet → test
	gofmt -l cmd internal | tee /dev/stderr | (! read)
	go vet ./...
	go test ./...

.PHONY: aif-fetch
aif-fetch: ## Fetch gold annotations and raw text from AIFdb (e.g. ARGS="-corpus qt30 -subcorpora 6")
	go run ./cmd/aif-eval -fetch $(ARGS)

.PHONY: aif-run
aif-run: ## Run extraction and evaluation (e.g. ARGS="-split test -sample 124 -prompt-set v3")
	go run ./cmd/aif-eval -run $(ARGS)

.PHONY: aif-report
aif-report: ## Write aggregates, tests and evidence (e.g. ARGS="-out out/aif")
	go run ./cmd/aif-eval -report $(ARGS)

.PHONY: aif-goldcheck
aif-goldcheck: ## Count AIF metamodel conformance of the gold data itself
	go run ./cmd/aif-eval -goldcheck $(ARGS)

.PHONY: paper-stats
paper-stats: ## Reproduce the paper's tests (5 comparisons, Holm correction) from the aggregates in data/metrics/
	go run ./cmd/aif-eval -paperstats

.PHONY: figures
figures: ## Regenerate the report figures (needs raw output in out/aif-n124-*/; Python + matplotlib)
	python3 scripts/make_report_figs.py
