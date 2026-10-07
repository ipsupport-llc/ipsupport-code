package telemetry

import (
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/hostarch"
)

// CollectInfo describes this machine for a report. Every field is
// best-effort: one the platform can't tell is left empty and not sent.
func CollectInfo() Info {
	// The machine's architecture, not the build's: an x64 build under
	// emulation on ARM would otherwise count the machine as x64.
	info := Info{OS: runtime.GOOS, Arch: hostarch.Native(), Locale: locale()}
	info.OSVersion, info.Chip, info.MemoryGB = platformInfo()
	return info
}

// locale is the user's language setting from the environment, as POSIX
// tools read it; "C" and "POSIX" mean none.
func locale() string {
	for _, k := range []string{"LC_ALL", "LC_MESSAGES", "LANG"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" && v != "C" && v != "POSIX" {
			return v
		}
	}
	return ""
}

// memorySizes are the sizes machines ship with — the server's list (its
// MemorySizes), which its metrics fold anything else into "other" by.
var memorySizes = []int{2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512}

// gb is installed memory in whole gigabytes. The OS reports a little less
// than is installed — the kernel keeps some, firmware some more: a 32 GB
// Linux box says 31.2 GiB. So a total within 10% under a size machines ship
// with is that size; anything else is rounded.
func gb(bytes uint64) int {
	f := float64(bytes) / (1 << 30)
	for _, s := range memorySizes {
		if f <= float64(s) && f >= float64(s)*0.9 {
			return s
		}
	}
	return int(math.Round(f))
}

var leadingVersion = regexp.MustCompile(`^\d+(\.\d+){0,2}`)

// versionPrefix keeps the numeric major.minor.patch of a version string:
// "6.8.0-45-generic" -> "6.8.0".
func versionPrefix(s string) string { return leadingVersion.FindString(strings.TrimSpace(s)) }

// appleChip keeps an Apple silicon brand string and drops anything else:
// only "Apple M…" means something to the server.
func appleChip(brand string) string {
	brand = strings.TrimSpace(brand)
	if strings.HasPrefix(brand, "Apple M") {
		return brand
	}
	return ""
}
