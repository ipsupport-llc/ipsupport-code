package main

import (
	"encoding/json"
	"log/slog"
	"os"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
)

// Every file in a workspace's state directory can have a second writer: two
// named sessions on one checkout, or simply the same command run twice. The
// stores built for that already take a cross-process lock and merge under it
// (internal/knowledge, internal/usage, internal/config); the helpers here are
// for the rest, which were writing whatever this process happened to be holding.
//
// Two guarantees, and the weaker one is not enough on its own:
//
//   - writeShared serializes the write. Right for a file holding ONE value — a
//     standing goal, a saved conversation — where the last writer should win.
//   - updateSharedStrings serializes the write AND derives the new content from
//     what is on disk at that moment rather than from what this process read
//     minutes ago. Required for a COLLECTION both processes add to, where a
//     plain lock still lets the later save drop the other's entries.
//
// A lock we cannot take is never a reason to lose the write: both fall back to
// writing unlocked and say so in the log.

// writeShared writes data under a cross-process lock on path.
func writeShared(path string, data []byte, perm os.FileMode) error {
	unlock, err := filelock.Lock(path)
	if err != nil {
		slog.Warn("state lock failed — writing unlocked", "path", path, "err", err)
		return atomicfile.Write(path, data, perm)
	}
	defer unlock()
	return atomicfile.Write(path, data, perm)
}

// updateSharedStrings read-modify-writes a JSON string list under one lock:
// apply is handed what is on disk right now and returns what to write, or
// write=false to leave the file alone. Returns the list that is now current,
// which the caller should adopt — it may contain another process's entries.
func updateSharedStrings(path string, perm os.FileMode, apply func(onDisk []string) (out []string, write bool)) []string {
	if unlock, err := filelock.Lock(path); err != nil {
		slog.Warn("state lock failed — merging unlocked", "path", path, "err", err)
	} else {
		defer unlock()
	}
	var onDisk []string
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &onDisk)
	}
	out, write := apply(onDisk)
	if write {
		if data, err := json.Marshal(out); err == nil {
			if err := atomicfile.Write(path, data, perm); err != nil {
				slog.Warn("shared state not saved", "path", path, "err", err)
			}
		}
	}
	return out
}
