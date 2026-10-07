package tool

import (
	"slices"
	"testing"
)

// A tool installed after start shows up in the registry's PATH, not ours: it
// is added at the end, and nothing we already had moves or doubles.
func TestMergePathAddsWhatWasInstalledSince(t *testing.T) {
	cur := `C:\venv\Scripts;C:\Windows\system32;C:\Program Files\Git\cmd\`
	machine := `C:\Windows\system32;C:\Program Files\nodejs\`
	user := `c:\program files\git\cmd;C:\Users\u\AppData\Local\Programs\Python\Python312\;;`
	got := mergePath(cur, machine, user)
	want := `C:\venv\Scripts;C:\Windows\system32;C:\Program Files\Git\cmd\;C:\Program Files\nodejs\;C:\Users\u\AppData\Local\Programs\Python\Python312\`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestWithEnvReplacesWithoutCase(t *testing.T) {
	got := withEnv([]string{"Path=old", "HOME=x", "PATH=older"}, "Path", "new")
	if !slices.Equal(got, []string{"HOME=x", "Path=new"}) {
		t.Fatalf("got %q", got)
	}
}
