package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// usage prints the top-level help: the task form, the subcommands, then flags.
func printUsage() {
	fmt.Fprint(os.Stderr, `ipsupport-code — local coding agent

usage:
  ipsupport-code [flags] [task]         run the agent (TUI in a terminal; one-shot if [task] is given)
  ipsupport-code <command> [args]

commands:
  update [stable|nightly]     download & install a newer build (optionally switch channel)
  config get|set|unset|list   read or change persisted settings (add --local for the workspace)
  version                     print version and exit
  init                        (re-)run first-time setup and exit
  help                        print this help

flags:
`)
	flag.PrintDefaults()
}

// runConfig implements the `config` subcommand: get/set/unset/list over the
// persisted configuration. It writes the global user config by default; --local
// targets the workspace's .agent/config.json (where run/file policy usually
// lives). get and list always report the effective, merged value.
func runConfig(workspace string, args []string) {
	fs := flag.NewFlagSet("config", flag.ExitOnError)
	local := fs.Bool("local", false, "target the workspace .agent/config.json instead of the global user config")
	fs.Usage = configUsage
	_ = fs.Parse(args)
	rest := fs.Args()
	if len(rest) == 0 {
		configUsage()
		os.Exit(2)
	}

	path, perm := config.GlobalPath(), os.FileMode(0o600)
	if *local {
		path = filepath.Join(workspace, ".agent", "config.json")
		perm = 0o644
	}

	switch rest[0] {
	case "get":
		if len(rest) != 2 {
			fmt.Fprintln(os.Stderr, "usage: config get <key>")
			os.Exit(2)
		}
		cfg, err := config.Load(workspace)
		if err != nil {
			fatal(err)
		}
		v, ok := config.LookupPath(cfg, rest[1])
		if !ok {
			fmt.Fprintf(os.Stderr, "no such key: %s\n", rest[1])
			os.Exit(1)
		}
		b, _ := json.Marshal(v)
		fmt.Println(string(b))

	case "set":
		if len(rest) != 3 {
			fmt.Fprintln(os.Stderr, "usage: config set <key> <value>")
			os.Exit(2)
		}
		if err := config.SetFileValue(path, perm, rest[1], rest[2]); err != nil {
			fatal(err)
		}
		fmt.Printf("set %s  (%s)\n", rest[1], path) // don't echo the value — it may be a secret
		if w := shadowWarning(workspace, *local, rest[1]); w != "" {
			fmt.Fprintln(os.Stderr, w)
		}

	case "unset":
		if len(rest) != 2 {
			fmt.Fprintln(os.Stderr, "usage: config unset <key>")
			os.Exit(2)
		}
		if err := config.UnsetFileValue(path, perm, rest[1]); err != nil {
			fatal(err)
		}
		fmt.Printf("unset %s  (%s)\n", rest[1], path)
		if w := shadowWarning(workspace, *local, rest[1]); w != "" {
			fmt.Fprintln(os.Stderr, w)
		}

	case "list":
		cfg, err := config.Load(workspace)
		if err != nil {
			fatal(err)
		}
		for _, line := range config.Flatten(cfg) {
			fmt.Println(line)
		}

	default:
		fmt.Fprintf(os.Stderr, "unknown config command: %s\n", rest[0])
		configUsage()
		os.Exit(2)
	}
}

// shadowWarning returns a note (or "" for none) for when the key just changed
// is ALSO defined in the OTHER config layer, which would win the merge — so
// the change that just wrote successfully has no visible effect until it's
// also changed there. The workspace file always wins over the global one, so
// this only fires the direction that can actually be silently shadowed: a
// global set/unset (wroteLocal=false) while the workspace file also defines
// the key. The reverse (a --local write) is never shadowed.
func shadowWarning(workspace string, wroteLocal bool, key string) string {
	if wroteLocal {
		return ""
	}
	wsPath := filepath.Join(workspace, ".agent", "config.json")
	if _, ok := config.LookupPathInFile(wsPath, key); ok {
		return fmt.Sprintf("note: %s is ALSO set in %s, which takes precedence over the global config — that value is what's actually in effect. Use --local to change it there instead.", key, wsPath)
	}
	return ""
}

func configUsage() {
	fmt.Fprint(os.Stderr, `usage: ipsupport-code config [--local] <command>

  get <key>            print the effective value at a dotted key (e.g. update_check, llm.model)
  set <key> <value>    set a key; JSON values keep their type, else a bare string
  unset <key>          remove a key (falls back to its default / lower layer)
  list                 print the effective config, one dotted key per line

  --local              read/write the workspace .agent/config.json (default: the global user config)
`)
}

// fatal prints err and exits non-zero. Used by the one-shot subcommands where
// there's no TUI to surface the error in.
func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
