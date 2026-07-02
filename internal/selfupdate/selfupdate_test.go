package selfupdate

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Apply must never install a binary it couldn't verify: if the checksums file
// can't be fetched, it errors out instead of silently skipping verification. The
// asset here is deliberately not a valid archive, so even a regression can't reach
// replaceExecutable (which would clobber the test binary).
func TestApplyRefusesWhenChecksumsUnavailable(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/asset", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "not-an-archive") })
	mux.HandleFunc("/sums", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusInternalServerError) })

	rel := Release{Version: "v9", AssetName: "a.tar.gz", AssetURL: srv.URL + "/asset", SumsURL: srv.URL + "/sums"}
	if _, err := Apply(context.Background(), rel, srv.Client()); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("Apply err = %v, want a checksum-fetch failure (never install unverified)", err)
	}
}

func makeTarGz(t *testing.T, name string, data []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(data))}); err != nil {
		t.Fatal(err)
	}
	tw.Write(data)
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

func TestLatestExtractAndVerify(t *testing.T) {
	bin := []byte("FAKE-IPSUPPORT-CODE-BINARY")
	archive := makeTarGz(t, "ipsupport-code", bin)
	sum := sha256.Sum256(archive)
	oa := osArch()
	assetName := "ipsupport-code_v1.2.3_" + oa + ".tar.gz"
	checksums := hex.EncodeToString(sum[:]) + "  " + assetName + "\n"

	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/asset", func(w http.ResponseWriter, _ *http.Request) { w.Write(archive) })
	mux.HandleFunc("/sums", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, checksums) })
	mux.HandleFunc("/repos/o/r/releases/latest", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, `{"assets":[`+
			`{"name":"ipsupport-code_v1.2.3_other-arch.tar.gz","browser_download_url":"%s/nope"},`+
			`{"name":%q,"browser_download_url":"%s/asset"},`+
			`{"name":"checksums.txt","browser_download_url":"%s/sums"}]}`,
			srv.URL, assetName, srv.URL, srv.URL)
	})

	old := apiBase
	apiBase = srv.URL
	defer func() { apiBase = old }()

	rel, err := Latest(context.Background(), "o/r", Stable, srv.Client())
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if rel.Version != "v1.2.3" {
		t.Errorf("version = %q, want v1.2.3", rel.Version)
	}
	if rel.AssetName != assetName || rel.SumsURL == "" {
		t.Errorf("resolved wrong asset/sums: %+v", rel)
	}

	data, err := get(context.Background(), srv.Client(), rel.AssetURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyChecksum(data, checksums, rel.AssetName); err != nil {
		t.Errorf("checksum should match: %v", err)
	}
	if err := verifyChecksum([]byte("tampered"), checksums, rel.AssetName); err == nil {
		t.Error("checksum of tampered data should fail")
	}
	got, err := extractBinary(data, "ipsupport-code")
	if err != nil || !bytes.Equal(got, bin) {
		t.Errorf("extractBinary = %q, %v; want the original binary", got, err)
	}
}

func TestLatestNoAssetForPlatform(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/repos/o/r/releases/tags/nightly", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"assets":[{"name":"ipsupport-code_x_some-other.tar.gz","browser_download_url":"u"}]}`)
	})
	old := apiBase
	apiBase = srv.URL
	defer func() { apiBase = old }()

	if _, err := Latest(context.Background(), "o/r", Nightly, srv.Client()); err == nil {
		t.Error("expected an error when no asset matches this platform")
	}
}
