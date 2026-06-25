package policy

import (
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

func eng(t *testing.T, c config.Config) *Engine {
	t.Helper()
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestRunDenyBeatsAllow(t *testing.T) {
	c := config.Default()
	c.Run = config.RunPolicy{Default: "ask", Allow: []string{"rm*"}, Deny: []string{"rm -rf*"}}
	e := eng(t, c)
	if got := e.Run("rm -rf /tmp/x"); got != Deny {
		t.Errorf("Run(rm -rf /tmp/x) = %v, want Deny", got)
	}
}

func TestRunAllowAndDefault(t *testing.T) {
	c := config.Default()
	c.Run = config.RunPolicy{Default: "ask", Allow: []string{"ls*", "git status"}}
	e := eng(t, c)
	if got := e.Run("ls -la"); got != Allow {
		t.Errorf("Run(ls -la) = %v, want Allow", got)
	}
	if got := e.Run("git push"); got != Ask {
		t.Errorf("Run(git push) = %v, want Ask (default)", got)
	}
}

func TestWriteGlobsAndJail(t *testing.T) {
	dir := t.TempDir()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{
		Default:    "ask",
		Jail:       ".",
		AllowWrite: []string{"**/*.go"},
		DenyWrite:  []string{"**/*secret*"},
	}
	e := eng(t, c)

	if d, err := e.Write("pkg/main.go"); err != nil || d != Allow {
		t.Errorf("Write(pkg/main.go) = %v,%v want Allow,nil", d, err)
	}
	if d, err := e.Write("config/my_secret.txt"); err != nil || d != Deny {
		t.Errorf("Write(my_secret) = %v,%v want Deny,nil", d, err)
	}
	if d, err := e.Write("notes/todo.txt"); err != nil || d != Ask {
		t.Errorf("Write(notes/todo) = %v,%v want Ask,nil", d, err)
	}
	if _, err := e.Write("../escape.txt"); err == nil {
		t.Error("Write(../escape.txt) expected jail-escape error, got nil")
	}
}

func TestJailDisabled(t *testing.T) {
	dir := t.TempDir()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: ""}
	e := eng(t, c)

	if err := e.Read("/etc/hosts"); err != nil {
		t.Errorf("Read(/etc/hosts) with jail disabled errored: %v", err)
	}
	if d, err := e.Write("/tmp/anywhere.txt"); err != nil || d != Allow {
		t.Errorf("Write(/tmp/anywhere) with jail disabled = %v,%v want Allow,nil", d, err)
	}
}
