// Package selfupdate downloads a newer ipsupport-code binary from GitHub Releases
// and replaces the running executable in place. It picks the release channel
// (stable = the latest tagged release, nightly = the rolling pre-release) and the
// asset for this machine's OS/arch, verifies the SHA-256, and atomically swaps
// the binary.
package selfupdate

import (
	"archive/tar"
	"archive/zip"
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
	"path"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
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

	// The machine's own build first: an x64 binary under emulation on ARM64
	// moves to the native one. A release without it (older ones carry no
	// windows-arm64) falls back to this binary's own.
	ext := ".tar.gz"
	if goos == "windows" {
		ext = ".zip"
	}
	var rel Release
	for _, arch := range []string{nativeArch, runtime.GOARCH} {
		suffix := "_" + goos + "-" + arch + ext
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
		if rel.AssetURL != "" {
			break
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
	if rel.SumsURL == "" {
		return "", fmt.Errorf("release %s has no checksums.txt asset — refusing to install an unverified binary", rel.Version)
	}
	data, err := get(ctx, hc, rel.AssetURL)
	if err != nil {
		return "", err
	}
	sums, err := get(ctx, hc, rel.SumsURL)
	if err != nil {
		return "", fmt.Errorf("fetch checksums: %w", err) // never install an unverified binary
	}
	if err := verifyChecksum(data, string(sums), rel.AssetName); err != nil {
		return "", err
	}
	if goos == "windows" {
		bin, err := extractZip(data, "ipsupport-code.exe")
		if err != nil {
			return "", err
		}
		exe, err := executable()
		if err != nil {
			return "", err
		}
		return exe, replaceInUse(exe, bin)
	}
	bin, err := extractBinary(data, "ipsupport-code")
	if err != nil {
		return "", err
	}
	return replaceExecutable(bin)
}

func osArch() string { return goos + "-" + runtime.GOARCH }

// goos is runtime.GOOS, a variable so a test can play Windows.
var goos = runtime.GOOS

// nativeArch is the machine's own architecture, which differs from
// runtime.GOARCH for an x64 binary under emulation on Windows on ARM (see
// arch_windows.go). A variable so a test can play one.
var nativeArch = runtime.GOARCH

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

// maxBinaryBytes caps the decompressed binary read so a malformed/hostile archive
// can't expand into an OOM (the compressed download is already capped at 100 MiB).
const maxBinaryBytes = 300 << 20

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
		if filepath.Base(h.Name) != name {
			continue
		}
		// Whatever this returns is written over the working executable, so it
		// must be the whole of a real file: a directory or link entry reads as
		// zero bytes, and a read cut at the cap is a truncated binary.
		if h.Typeflag != tar.TypeReg {
			return nil, fmt.Errorf("%q in the archive is not a regular file", name)
		}
		if h.Size <= 0 || h.Size > maxBinaryBytes {
			return nil, fmt.Errorf("%q in the archive is %d bytes — refusing (limit %d MiB)", name, h.Size, maxBinaryBytes>>20)
		}
		bin, err := io.ReadAll(io.LimitReader(tr, maxBinaryBytes)) // cap the decompressed read (gzip-bomb guard)
		if err != nil {
			return nil, err
		}
		if int64(len(bin)) != h.Size {
			return nil, fmt.Errorf("%q in the archive is truncated (%d of %d bytes)", name, len(bin), h.Size)
		}
		return bin, nil
	}
	return nil, fmt.Errorf("%q not found in the archive", name)
}

// extractZip returns the named file from a .zip — the Windows archive — under
// the same rules as extractBinary: a regular, non-empty file read whole.
func extractZip(data []byte, name string) ([]byte, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	for _, f := range zr.File {
		if filepath.Base(f.Name) != name && path.Base(f.Name) != name {
			continue
		}
		if !f.Mode().IsRegular() {
			return nil, fmt.Errorf("%q in the archive is not a regular file", name)
		}
		if f.UncompressedSize64 == 0 || f.UncompressedSize64 > maxBinaryBytes {
			return nil, fmt.Errorf("%q in the archive is %d bytes — refusing (limit %d MiB)", name, f.UncompressedSize64, maxBinaryBytes>>20)
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		bin, err := io.ReadAll(io.LimitReader(rc, maxBinaryBytes))
		rc.Close()
		if err != nil {
			return nil, err
		}
		if uint64(len(bin)) != f.UncompressedSize64 {
			return nil, fmt.Errorf("%q in the archive is truncated (%d of %d bytes)", name, len(bin), f.UncompressedSize64)
		}
		return bin, nil
	}
	return nil, fmt.Errorf("%q not found in the archive", name)
}

// executable is the path of the running binary, symlinks resolved.
func executable() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	return exe, nil
}

// replaceInUse installs bin at exe while exe is running. Windows will not let a
// running .exe be overwritten or replaced by a rename, but it will let it be
// renamed: so the old one moves aside to exe.old, the new one is written in its
// place, and RemoveOld deletes the old one on the next start. If the write
// fails, the old one is moved back.
func replaceInUse(exe string, bin []byte) error {
	old := exe + ".old"
	_ = os.Remove(old) // left by an earlier update
	if err := os.Rename(exe, old); err != nil {
		return fmt.Errorf("can't move the running %s aside: %w", exe, err)
	}
	if err := atomicfile.Write(exe, bin, 0o755); err != nil {
		_ = os.Rename(old, exe)
		return fmt.Errorf("can't write %s: %w", exe, err)
	}
	return nil
}

// RemoveOld deletes the binary an update on Windows moved aside, once it is no
// longer running. Best effort; a no-op when there is none.
func RemoveOld() {
	if exe, err := executable(); err == nil {
		removeOld(exe)
	}
}

func removeOld(exe string) { _ = os.Remove(exe + ".old") }

// replaceExecutable writes the new binary next to the current one and renames it
// over the top — atomic on the same filesystem, and safe while running on Unix
// (the live process keeps the old inode until it exits).
func replaceExecutable(bin []byte) (string, error) {
	exe, err := executable()
	if err != nil {
		return "", err
	}
	// Atomic temp+rename over the running exe (safe on Unix — the live process keeps
	// the old inode until it exits); executable perms.
	if err := atomicfile.Write(exe, bin, 0o755); err != nil {
		return "", fmt.Errorf("can't replace %s: %w", exe, err)
	}
	return exe, nil
}
