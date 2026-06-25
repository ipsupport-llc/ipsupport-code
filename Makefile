PKG := ./cmd/agent
BIN := ipsupport-code
PLATFORMS := linux/amd64 darwin/amd64 darwin/arm64

.PHONY: build release test vet tidy clean

build: ## host binary for local testing
	go build -o dist/$(BIN) $(PKG)

release: ## static binaries for every target into dist/
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; \
	  echo "→ $$os/$$arch"; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -o dist/$(BIN)-$$os-$$arch $(PKG); \
	done

test: ## run all tests
	go test ./...

vet: ## go vet
	go vet ./...

tidy: ## tidy modules
	go mod tidy

clean: ## remove build artifacts
	rm -rf dist
