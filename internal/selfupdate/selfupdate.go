// Package selfupdate downloads a newer ipsupport-code binary from GitHub Releases
// and replaces the running executable in place. It picks the release channel
// (stable = the latest tagged release, nightly = the rolling pre-release) and the
// asset for this machine's OS/arch, verifies the SHA-256, and atomically swaps
// the binary.
package selfupdate

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Repo is the GitHub "owner/name" releases are pulled from.
const Repo = "ipsupport-llc/ipsupport-code"

// apiBase is the GitHub API root; a var so tests can point it at a fake server.
var apiBase = "https://api.github.com"

// Channels.
const (
	Stable  = "stable"
	Nightly = "nightly"
)

// Release is the resolved newest build on a channel for this OS/arch.
type Release struct {
	Version   string // e.g. "v0.1.0" or "nightly-20260627-9aa8de9"
	AssetName string // the .tar.gz asset name
	AssetURL  string
	SumsURL   string
}

// Latest resolves the newest release on the channel and the asset for this
// machine. A nil client uses http.DefaultClient.
func Latest(ctx context.Context, repo, channel string, hc *http.Client) (Release, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	path := "/releases/latest"
	if channel == Nightly {
		path = "/releases/tags/nightly"
	}
	body, err := get(ctx, hc, apiBase+"/repos/"+repo+path)
	if err != nil {
		return Release{}, err
	}
	var raw struct {
		Assets []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return Release{}, err
	}

	suffix := "_" + osArch() + ".tar.gz"
	var rel Release
	for _, a := range raw.Assets {
		switch {
		case a.Name == "checksums.txt":
			rel.SumsURL = a.URL
		case strings.HasSuffix(a.Name, suffix):
			rel.AssetName = a.Name
			rel.AssetURL = a.URL
			rel.Version = strings.TrimPrefix(strings.TrimSuffix(a.Name, suffix), "ipsupport-code_")
		}
	}
	if rel.AssetURL == "" {
		return rel, fmt.Errorf("no %s asset in the %s release", osArch(), channel)
	}
	return rel, nil
}

// Apply downloads the release asset, verifies its checksum, and replaces the
// running executable. It returns the path of the replaced binary.
func Apply(ctx context.Context, rel Release, hc *http.Client) (string, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	if runtime.GOOS == "windows" {
		return "", fmt.Errorf("self-update isn't supported on Windows yet; download the .zip from the Releases page")
	}
	data, err := get(ctx, hc, rel.AssetURL)
	if err != nil {
		return "", err
	}
	if rel.SumsURL != "" {
		sums, err := get(ctx, hc, rel.SumsURL)
		if err == nil {
			if err := verifyChecksum(data, string(sums), rel.AssetName); err != nil {
				return "", err
			}
		}
	}
	bin, err := extractBinary(data, "ipsupport-code")
	if err != nil {
		return "", err
	}
	return replaceExecutable(bin)
}

func osArch() string { return runtime.GOOS + "-" + runtime.GOARCH }

func get(ctx context.Context, hc *http.Client, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: http %d", url, resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 100<<20)) // 100 MiB cap
}

func verifyChecksum(data []byte, sums, name string) error {
	want := ""
	for _, line := range strings.Split(sums, "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && f[1] == name {
			want = f[0]
		}
	}
	if want == "" {
		return fmt.Errorf("no checksum listed for %s", name)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != want {
		return fmt.Errorf("checksum mismatch for %s", name)
	}
	return nil
}

func extractBinary(gzData []byte, name string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(gzData))
	if err != nil {
		return nil, err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if filepath.Base(h.Name) == name {
			return io.ReadAll(tr)
		}
	}
	return nil, fmt.Errorf("%q not found in the archive", name)
}

// replaceExecutable writes the new binary next to the current one and renames it
// over the top — atomic on the same filesystem, and safe while running on Unix
// (the live process keeps the old inode until it exits).
func replaceExecutable(bin []byte) (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	tmp, err := os.CreateTemp(filepath.Dir(exe), ".ipsupport-code-update-*")
	if err != nil {
		return "", fmt.Errorf("can't write next to %s: %w", exe, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename succeeds
	if _, err := tmp.Write(bin); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Chmod(tmpName, 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(tmpName, exe); err != nil {
		return "", fmt.Errorf("can't replace %s: %w", exe, err)
	}
	return exe, nil
}
