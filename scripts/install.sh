#!/usr/bin/env sh
# Download an ipsupport-code binary for this machine from GitHub Releases.
#
#   ./install.sh                # newest nightly build (tracks main)
#   ./install.sh latest         # newest stable release
#   ./install.sh v0.1.0         # a specific tag
#   ./install.sh nightly ~/bin/ipsupport-code   # custom destination
#
# Or straight from the web:
#   curl -fsSL https://raw.githubusercontent.com/ipsupport-llc/ipsupport-code/main/scripts/install.sh | sh
set -eu

REPO="ipsupport-llc/ipsupport-code"
TAG="${1:-nightly}"
DEST="${2:-./ipsupport-code}"

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$arch" in
  arm64 | aarch64) arch=arm64 ;;
  x86_64 | amd64) arch=amd64 ;;
  *) echo "unsupported arch: $arch" >&2; exit 1 ;;
esac
case "$os" in
  darwin | linux) ;;
  *) echo "this script handles macOS/Linux; for Windows grab the .zip from Releases" >&2; exit 1 ;;
esac

if [ "$TAG" = "latest" ]; then
  api="https://api.github.com/repos/$REPO/releases/latest"
else
  api="https://api.github.com/repos/$REPO/releases/tags/$TAG"
fi

echo "→ resolving $TAG release for ${os}-${arch}…"
json=$(curl -fsSL "$api")
url=$(printf '%s' "$json" | grep -o "https://[^\"]*_${os}-${arch}\.tar\.gz" | head -n1)
sums=$(printf '%s' "$json" | grep -o "https://[^\"]*checksums\.txt" | head -n1)
[ -n "$url" ] || { echo "no ${os}-${arch} asset in the $TAG release" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
file=$(basename "$url")

echo "→ downloading $file"
curl -fsSL "$url" -o "$tmp/$file"

if [ -n "$sums" ]; then
  curl -fsSL "$sums" -o "$tmp/checksums.txt"
  expected=$(grep " $file\$" "$tmp/checksums.txt" | awk '{print $1}')
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp/$file" | awk '{print $1}')
  else
    actual=$(shasum -a 256 "$tmp/$file" | awk '{print $1}')
  fi
  [ -n "$expected" ] && [ "$expected" = "$actual" ] && echo "→ checksum OK" ||
    { echo "checksum mismatch for $file" >&2; exit 1; }
fi

tar -xzf "$tmp/$file" -C "$tmp"
mkdir -p "$(dirname "$DEST")"
mv "$tmp/ipsupport-code" "$DEST"
chmod +x "$DEST"
[ "$os" = darwin ] && xattr -d com.apple.quarantine "$DEST" 2>/dev/null || true

echo "→ installed: $DEST"
"$DEST" -version
