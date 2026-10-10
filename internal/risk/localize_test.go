package risk

import "testing"

// localizeVectors are pinned on both sides: scripts/train_risk.py asserts the
// same pairs (LOCALIZE_VECTORS).
var localizeVectors = []struct{ text, ws, want string }{
	{"wc -l /app/data.txt", "/app", "wc -l ./data.txt"},
	{"cd /app && make", "/app", "cd . && make"},
	{"cat /application/x /x/app/y", "/app", "cat /application/x /x/app/y"},
	{`cat "/app/a b.txt"`, "/app/", `cat "./a b.txt"`},
	{"ls /app", "/app", "ls ."},
	{"cp /app/x /etc/x", "/app", "cp ./x /etc/x"},
	{`type C:\Users\dev\project\report.txt`, `C:\Users\dev\project`, `type .\report.txt`},
	{"cat ~/project/a.md ~/projects/b", "~/project", "cat ./a.md ~/projects/b"},
	{"rm -rf /app", "", "rm -rf /app"},
	{"rm -rf /", "/", "rm -rf /"},
	{`cat "/app backup/data"`, "/app", `cat "/app backup/data"`}, // a quoted space is part of the name
	{"cat /app,old/data", "/app", "cat /app,old/data"},
	{`cd "/app" && ls`, "/app", `cd "." && ls`},
	{`type c:/USERS/dev/project/a.txt`, `C:\Users\dev\project`, `type ./a.txt`}, // Windows: case and / don't matter
	{"cat é/a", "é", "cat é/a"},
	{`rm "/app"x`, "/app", `rm "/app"x`},            // the word is /appx: outside
	{`echo "a\"" /app/b`, "/app", `echo "a\"" ./b`}, // an escaped quote doesn't end the string                                                 // too short to be a workspace
}

func TestLocalize(t *testing.T) {
	for _, v := range localizeVectors {
		if got := Localize(v.text, v.ws); got != v.want {
			t.Errorf("Localize(%q, %q) = %q, want %q", v.text, v.ws, got, v.want)
		}
	}
}
