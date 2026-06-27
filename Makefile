PKG := ./cmd/agent
BIN := ipsupport-code
PLATFORMS := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64

# Version stamped into the binary: the current tag, else the short commit.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: build release test race vet fmt fmtcheck tidy clean

build: ## host binary for local testing
	go build -ldflags "$(LDFLAGS)" -o dist/$(BIN) $(PKG)

release: ## stripped static binaries for every target into dist/
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; ext=$$( [ "$$os" = windows ] && echo .exe || echo ); \
	  echo "→ $$os/$$arch"; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -trimpath -ldflags "$(LDFLAGS)" -o dist/$(BIN)-$$os-$$arch$$ext $(PKG); \
	done

test: ## run all tests
	go test ./...

race: ## run all tests with the race detector
	go test -race ./...

vet: ## go vet
	go vet ./...

fmt: ## format the code
	gofmt -w .

fmtcheck: ## fail if anything is unformatted
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then echo "gofmt needed on:"; echo "$$unformatted"; exit 1; fi

tidy: ## tidy modules
	go mod tidy

clean: ## remove build artifacts
	rm -rf dist
