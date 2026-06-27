PKG := ./cmd/agent
BIN := ipsupport-code
PLATFORMS := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64

# Version stamped into the binary: the current tag, else the short commit.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: build release archives test race vet fmt fmtcheck tidy clean

build: ## host binary for local testing
	go build -ldflags "$(LDFLAGS)" -o dist/$(BIN) $(PKG)

release: ## stripped static binaries for every target into dist/
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; ext=$$( [ "$$os" = windows ] && echo .exe || echo ); \
	  echo "→ $$os/$$arch"; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -trimpath -ldflags "$(LDFLAGS)" -o dist/$(BIN)-$$os-$$arch$$ext $(PKG); \
	done

archives: release ## package the release binaries into .tar.gz/.zip + SHA-256 checksums
	@cd dist && rm -f *.tar.gz *.zip checksums.txt && \
	for f in $(BIN)-*; do \
	  oa=$${f#$(BIN)-}; \
	  case $$f in \
	    *.exe) oa=$${oa%.exe}; cp "$$f" $(BIN).exe; zip -q "$(BIN)_$(VERSION)_$$oa.zip" $(BIN).exe; rm $(BIN).exe ;; \
	    *) cp "$$f" $(BIN); tar -czf "$(BIN)_$(VERSION)_$$oa.tar.gz" $(BIN); rm $(BIN) ;; \
	  esac; \
	done && \
	sha256sum $(BIN)_*.tar.gz $(BIN)_*.zip > checksums.txt && \
	echo "→ archives + checksums.txt in dist/"

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
