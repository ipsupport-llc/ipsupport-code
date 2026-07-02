package atomicfile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteCreatesDirSetsPermOverwritesAndLeavesNoTemp(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "sub", "f.json") // parent dir doesn't exist yet

	if err := Write(p, []byte("hello"), 0o600); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if b, err := os.ReadFile(p); err != nil || string(b) != "hello" {
		t.Fatalf("read = %q, %v", b, err)
	}
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
		t.Errorf("perm = %v, want 0600", fi.Mode().Perm())
	}

	if err := Write(p, []byte("world"), 0o644); err != nil { // overwrite
		t.Fatalf("overwrite: %v", err)
	}
	if b, _ := os.ReadFile(p); string(b) != "world" {
		t.Errorf("after overwrite = %q, want world", b)
	}

	ents, _ := os.ReadDir(filepath.Dir(p))
	for _, e := range ents {
		if strings.HasPrefix(e.Name(), ".tmp-") {
			t.Errorf("left a temp file behind: %s", e.Name())
		}
	}
}
