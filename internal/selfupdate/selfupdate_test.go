package selfupdate

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
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

// An incomplete/broken release with no checksums.txt asset at all (SumsURL
// left empty by Latest) must refuse to install rather than silently skip
// verification — the old `if rel.SumsURL != ""` guard let this through.
func TestApplyRefusesWhenChecksumsAssetMissing(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/asset", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "not-an-archive") })

	rel := Release{Version: "v9", AssetName: "a.tar.gz", AssetURL: srv.URL + "/asset"} // no SumsURL
	if _, err := Apply(context.Background(), rel, srv.Client()); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("Apply err = %v, want a refusal for a missing checksums.txt (never install unverified)", err)
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
	// The archive this platform's release really carries: a .zip holding the
	// .exe on Windows — a tar.gz named .zip would test nothing Apply does there.
	archive, extract := makeTarGz(t, "ipsupport-code", bin), func(b []byte) ([]byte, error) { return extractBinary(b, "ipsupport-code") }
	if runtime.GOOS == "windows" {
		archive, extract = makeZip(t, "ipsupport-code.exe", bin), func(b []byte) ([]byte, error) { return extractZip(b, "ipsupport-code.exe") }
	}
	sum := sha256.Sum256(archive)
	assetName := "ipsupport-code_v1.2.3" + assetSuffix(runtime.GOOS, runtime.GOARCH)
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
	got, err := extract(data)
	if err != nil || !bytes.Equal(got, bin) {
		t.Errorf("extracted %q, %v; want the original binary", got, err)
	}
}

func TestAssetSuffix(t *testing.T) {
	for _, tc := range []struct{ goos, goarch, want string }{
		{"linux", "amd64", "_linux-amd64.tar.gz"},
		{"darwin", "arm64", "_darwin-arm64.tar.gz"},
		{"windows", "amd64", "_windows-amd64.zip"},
		{"windows", "arm64", "_windows-arm64.zip"},
	} {
		if got := assetSuffix(tc.goos, tc.goarch); got != tc.want {
			t.Errorf("assetSuffix(%s, %s) = %q, want %q", tc.goos, tc.goarch, got, tc.want)
		}
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

// Only a regular, non-empty file of a size we will read whole is a binary:
// a directory or symlink entry yielded zero bytes and an oversized one was cut
// short — and either was then written over the working executable.
func TestExtractRefusesWhatIsNotAWholeBinary(t *testing.T) {
	entry := func(h *tar.Header, body []byte) []byte {
		var buf bytes.Buffer
		gz := gzip.NewWriter(&buf)
		tw := tar.NewWriter(gz)
		_ = tw.WriteHeader(h)
		_, _ = tw.Write(body)
		_ = tw.Flush() // an oversized header with no body: the header bytes are what matters
		_ = gz.Close()
		return buf.Bytes()
	}
	for what, data := range map[string][]byte{
		"directory": entry(&tar.Header{Name: "ipsupport-code/", Typeflag: tar.TypeDir, Mode: 0o755}, nil),
		"symlink":   entry(&tar.Header{Name: "ipsupport-code", Typeflag: tar.TypeSymlink, Linkname: "/bin/sh"}, nil),
		"empty":     entry(&tar.Header{Name: "ipsupport-code", Typeflag: tar.TypeReg, Mode: 0o755}, nil),
		"oversized": entry(&tar.Header{Name: "ipsupport-code", Typeflag: tar.TypeReg, Mode: 0o755, Size: maxBinaryBytes + 1}, nil),
	} {
		if got, err := extractBinary(data, "ipsupport-code"); err == nil {
			t.Errorf("%s entry: extracted %d bytes with no error", what, len(got))
		}
	}
}

// On Windows the release is a .zip, and an x64 binary running under emulation
// on an ARM64 machine moves to the native build when the release has one.
func TestLatestOnWindowsPicksTheNativeZip(t *testing.T) {
	oldOS, oldArch := goos, nativeArch
	defer func() { goos, nativeArch = oldOS, oldArch }()
	goos = "windows"
	serve := func(names ...string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var as []string
			for _, n := range append(names, "checksums.txt") {
				as = append(as, fmt.Sprintf(`{"name":%q,"browser_download_url":"http://x/%s"}`, n, n))
			}
			fmt.Fprintf(w, `{"assets":[%s]}`, strings.Join(as, ","))
		}))
	}
	for _, c := range []struct {
		native string
		assets []string
		want   string
	}{
		{runtime.GOARCH, []string{"ipsupport-code_v1_windows-" + runtime.GOARCH + ".zip", "ipsupport-code_v1_linux-" + runtime.GOARCH + ".tar.gz"}, "ipsupport-code_v1_windows-" + runtime.GOARCH + ".zip"},
		{"arm64", []string{"ipsupport-code_v1_windows-amd64.zip", "ipsupport-code_v1_windows-arm64.zip"}, "ipsupport-code_v1_windows-arm64.zip"},
		{"arm64", []string{"ipsupport-code_v1_windows-" + runtime.GOARCH + ".zip"}, "ipsupport-code_v1_windows-" + runtime.GOARCH + ".zip"}, // older release: no arm64 build
	} {
		nativeArch = c.native
		srv := serve(c.assets...)
		old := apiBase
		apiBase = srv.URL
		rel, err := Latest(context.Background(), "o/r", Stable, srv.Client())
		apiBase = old
		srv.Close()
		if err != nil || rel.AssetName != c.want || rel.Version != "v1" {
			t.Errorf("native %s, assets %v: got %+v, %v; want %s", c.native, c.assets, rel, err, c.want)
		}
	}
}

func makeZip(t *testing.T, name string, data []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(data)
	zw.Close()
	return buf.Bytes()
}

// The Windows archive gives back the whole .exe, and nothing that is not one.
func TestExtractFromAZip(t *testing.T) {
	bin := []byte("MZ-FAKE-EXE")
	got, err := extractZip(makeZip(t, "ipsupport-code.exe", bin), "ipsupport-code.exe")
	if err != nil || string(got) != string(bin) {
		t.Fatalf("got %q, %v", got, err)
	}
	if _, err := extractZip(makeZip(t, "ipsupport-code.exe/", nil), "ipsupport-code.exe"); err == nil {
		t.Error("a directory entry was extracted")
	}
	if _, err := extractZip(makeZip(t, "ipsupport-code.exe", nil), "ipsupport-code.exe"); err == nil {
		t.Error("an empty entry was extracted")
	}
	if _, err := extractZip(makeZip(t, "other.exe", bin), "ipsupport-code.exe"); err == nil {
		t.Error("a missing binary was not reported")
	}
}

// A running .exe cannot be overwritten on Windows, but it can be renamed: the
// old one moves aside, the new one takes its name, and the next start removes
// the old one.
func TestReplaceInUseMovesTheOldOneAside(t *testing.T) {
	exe := filepath.Join(t.TempDir(), "ipsupport-code.exe")
	os.WriteFile(exe, []byte("old"), 0o755)
	if err := replaceInUse(exe, []byte("new")); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(exe); string(b) != "new" {
		t.Fatalf("exe holds %q, want the new build", b)
	}
	if b, _ := os.ReadFile(exe + ".old"); string(b) != "old" {
		t.Fatalf(".old holds %q, want the previous build", b)
	}
	if _, err := os.Stat(exe + ".new"); !os.IsNotExist(err) {
		t.Fatalf(".new left behind: %v", err)
	}
	removeOld(exe)
	if _, err := os.Stat(exe + ".old"); !os.IsNotExist(err) {
		t.Fatalf(".old still there after cleanup: %v", err)
	}
}

// A binary Homebrew installed is updated by brew, not by replacing it.
func TestBrewInstallIsUpdatedByBrew(t *testing.T) {
	for _, c := range []struct{ exe, want string }{
		{"/opt/homebrew/Cellar/ipsupport-code/0.62.21/bin/ipsupport-code", "brew upgrade ipsupport-code"}, // Apple Silicon
		{"/usr/local/Cellar/ipsupport-code/0.62.21/bin/ipsupport-code", "brew upgrade ipsupport-code"},    // Intel
		{"/home/linuxbrew/.linuxbrew/Cellar/ipsupport-code/0.62.21/bin/ipsupport-code", "brew upgrade ipsupport-code"},
		{"/opt/homebrew/bin/ipco", "brew upgrade ipsupport-code"}, // a brew link that didn't resolve (no such file here)
		{"/usr/local/bin/ipsupport-code", ""},                     // a plain file the installer put there
		{"/home/me/.local/bin/ipsupport-code", ""},
	} {
		if got := brewUpgradeFor(c.exe); got != c.want {
			t.Errorf("brewUpgradeFor(%q) = %q, want %q", c.exe, got, c.want)
		}
	}
}

// Through brew's links — bin/ipco → Cellar, opt/ → Cellar — the install is
// still brew's.
func TestBrewInstallIsFoundThroughItsLinks(t *testing.T) {
	root := t.TempDir()
	cellar := filepath.Join(root, "Cellar", "ipsupport-code", "0.62.22", "bin")
	if err := os.MkdirAll(cellar, 0o755); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(cellar, "ipsupport-code")
	os.WriteFile(bin, []byte("x"), 0o755)
	if err := os.Symlink("ipsupport-code", filepath.Join(cellar, "ipco")); err != nil {
		t.Skipf("no symlinks here: %v", err) // Windows without the privilege
	}
	os.MkdirAll(filepath.Join(root, "bin"), 0o755)
	os.Symlink(filepath.Join(cellar, "ipco"), filepath.Join(root, "bin", "ipco"))
	os.MkdirAll(filepath.Join(root, "opt"), 0o755)
	os.Symlink(filepath.Join(root, "Cellar", "ipsupport-code", "0.62.22"), filepath.Join(root, "opt", "ipsupport-code"))
	for _, p := range []string{filepath.Join(root, "bin", "ipco"), filepath.Join(root, "opt", "ipsupport-code", "bin", "ipsupport-code")} {
		if got := brewUpgradeFor(p); got == "" {
			t.Errorf("%s: not recognised as a brew install", p)
		}
	}
}

// A plain binary someone copied under brew's prefix resolves to itself, not
// into the Cellar: it isn't brew's, and brew can't update it.
func TestAPlainFileUnderBrewsPrefixIsNotBrews(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "opt", "homebrew", "bin", "ipsupport-code")
	os.MkdirAll(filepath.Dir(bin), 0o755)
	os.WriteFile(bin, []byte("x"), 0o755)
	if got := brewUpgradeFor(bin); got != "" {
		t.Fatalf("brewUpgradeFor(%q) = %q, want \"\"", bin, got)
	}
}
