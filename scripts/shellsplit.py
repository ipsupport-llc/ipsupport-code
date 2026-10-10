"""Mirror of internal/shellsplit: cut a command line into the commands a shell
would run, by that shell's own rules.

The risk model scores a shell line by its parts — its code (the line without
heredoc bodies, here-strings and comments), each command in it, and each group
of commands joined by | && || — and the trainer must cut lines exactly as the
Go scorer does, or it trains and measures on text inference never produces. So
this is a line-by-line port, and it is guarded like the feature hash:
internal/shellsplit/testdata/golden.jsonl holds what the Go code returns for
every line the trainer reads, the Go tests check Go against it, and
check_golden() below checks this file against it on every training run. Edit
one side alone and both fail.

Python strings index by code point, as the Go code indexes []rune.
"""
import json
import pathlib
import posixpath
import unicodedata

SH, POWERSHELL, CMD = 0, 1, 2

_GO_SPACE = set("\t\n\v\f\r \x85\xa0")


def _is_go_space(c):
    return c in _GO_SPACE or unicodedata.category(c) in ("Zs", "Zl", "Zp")


def trim(s):
    """strings.TrimSpace: Python's strip() also drops \\x1c-\\x1f, Go's does not."""
    i, j = 0, len(s)
    while i < j and _is_go_space(s[i]):
        i += 1
    while j > i and _is_go_space(s[j - 1]):
        j -= 1
    return s[i:j]


MAX_DEPTH = 32  # mirrors maxDepth


def parse(dialect, line, depth=0):
    """(commands, groups, code, incomplete) — see internal/shellsplit.Parse."""
    if depth >= MAX_DEPTH:
        s = trim(line)
        return ([s] if s else []), [], trim(line), True
    r = _Result(dialect, depth)
    if dialect == POWERSHELL:
        _split_powershell(line, r)
    elif dialect == CMD:
        _split_cmd(line, r)
    else:
        _split_sh(line, r)
    r.end_group()
    c = r.code
    if c.endswith(" ;"):
        c = c[:-2]
    return r.cmds + r.inner, r.groups + r.inner_g, trim(c), r.incomplete


def split(dialect, line):
    return parse(dialect, line)[0]


def code(dialect, line):
    return parse(dialect, line)[2]


_JOINS = {"|", "||", "&&", "|&"}


class _Result:
    def __init__(self, dialect=SH, depth=0):
        self.d = dialect
        self.depth = depth
        self.incomplete = False
        self.inner = []
        self.inner_g = []
        self.cmds = []
        self.groups = []
        self.code = ""
        self.group = []
        self.gtext = ""

    def cut(self, b, op):
        s = trim(b.text())
        if s:
            self.cmds.append(s)
            self.inner_code(s)
            if self.code:
                self.code += " "
            self.code += s
            if self.group:
                self.gtext += " "
            self.gtext += s
            self.group.append(s)
        if op and self.code:
            self.code += " " + op
        if op in _JOINS:
            if self.group:
                self.gtext += " " + op
        else:
            self.end_group()
        b.reset()

    def end_group(self):
        if len(self.group) > 1:
            self.groups.append(trim(trim(self.gtext).rstrip("|&")))
        self.group = []
        self.gtext = ""

    def nest(self, nested, ngroups, dialect, code_):
        c, g, _, inc = parse(dialect, code_, self.depth + 1)
        nested += c
        ngroups += g
        self.incomplete = self.incomplete or inc

    def inner_code(self, cmd):
        """Mirrors innerCode: sh -c '…', pwsh -Command "…", cmd /c …, python -c '…', eval …"""
        if self.d == CMD:
            ws = _fields(cmd)
            for i, w in enumerate(ws):
                if w.lower() in ("/c", "/k") and i > 0 and _go_path_base(ws[i - 1].replace("\\", "/")).lower() == "cmd":
                    self.nest(self.inner, self.inner_g, CMD, " ".join(ws[i + 1:]))
                    return
            return
        ws = _words(cmd)
        at = _program_at(ws)
        if at < 0:
            return
        prog, rest = ws[at], ws[at + 1:]
        if prog == "eval":
            self.nest(self.inner, self.inner_g, self.d, " ".join(rest))
            return
        if not (prog in _SHELLS or prog in ("pwsh", "powershell", "cmd") or prog in _INTERPRETERS):
            return
        for i, w in enumerate(rest):
            if i + 1 >= len(rest):
                break
            body = rest[i + 1]
            if prog in _SHELLS and w.startswith("-") and not w.startswith("--") and "c" in w:
                self.nest(self.inner, self.inner_g, SH, body)
                return
            if prog in ("pwsh", "powershell") and w.lower() in ("-c", "-command"):
                self.nest(self.inner, self.inner_g, POWERSHELL, " ".join(rest[i + 1:]))
                return
            if prog == "cmd" and w.lower() in ("/c", "/k"):
                self.nest(self.inner, self.inner_g, CMD, " ".join(rest[i + 1:]))
                return
            if prog in _INTERPRETERS and w in ("-c", "-e", "--eval"):
                t = trim(body)
                if t:
                    self.inner.append(t)
                return


class _Buf:
    def __init__(self):
        self.parts = []
        self.last = ""
        self.last_esc = False

    def write(self, ch, esc):
        self.parts.append(ch)
        self.last, self.last_esc = ch, esc

    def write_str(self, s):
        if not s:
            return
        self.parts.append(s)
        self.last, self.last_esc = s[-1], False

    def reset(self):
        self.parts = []
        self.last, self.last_esc = "", False

    def text(self):
        return "".join(self.parts)

    def empty(self):
        return not any(self.parts)

    def after(self, chars):
        return not self.empty() and not self.last_esc and self.last != "" and self.last in chars


def _index_from(rs, i, ch):
    while i < len(rs):
        if rs[i] == ch:
            return i
        i += 1
    return len(rs)


def _close_paren(rs, i, dialect):
    depth = 1
    while i < len(rs):
        r = rs[i]
        if (dialect == SH and r == "\\") or (dialect == POWERSHELL and r == "`"):
            i += 1
        elif r in "'\"":
            i = _index_from(rs, i + 1, r)
        elif r == "(":
            depth += 1
        elif r == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(rs)


def _close_brace(rs, i):
    depth = 1
    while i < len(rs):
        r = rs[i]
        if r == "`":
            i += 1
        elif r in "'\"":
            i = _index_from(rs, i + 1, r)
        elif r == "{":
            depth += 1
        elif r == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(rs)


# ── sh ──────────────────────────────────────────────────────────────────────

_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "ash", "mksh", "ssh", "su"}
_INTERPRETERS = {"python", "python3", "python2", "perl", "ruby", "node", "php", "lua", "deno", "bun",
                 "osascript", "tclsh", "Rscript"}
_WRAPPERS = {"sudo", "env", "exec", "nohup", "time", "command", "doas",
             "timeout", "nice", "ionice", "stdbuf", "setsid", "xargs", "builtin"}


def _fields(s):
    """strings.Fields: runs of Go white space."""
    out, cur = [], []
    for c in s:
        if _is_go_space(c):
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


def _go_path_base(f):
    """path.Base for a non-empty field."""
    f = f.rstrip("/")
    if f == "":
        return "/"
    return f[f.rfind("/") + 1:]


def _words(cmd):
    """Mirrors words: a simple command's words with sh quote removal."""
    out, w, inw = [], [], False
    rs = cmd
    i = 0
    while i < len(rs):
        c = rs[i]
        if c in " \t\n":
            if inw:
                out.append("".join(w))
                w = []
                inw = False
        elif c == "'":
            inw = True
            j = _index_from(rs, i + 1, "'")
            w.append(rs[i + 1:min(j, len(rs))])
            i = j
        elif c == '"':
            inw = True
            i += 1
            while i < len(rs) and rs[i] != '"':
                if rs[i] == "\\" and i + 1 < len(rs) and rs[i + 1] in "\"\\$`":
                    i += 1
                w.append(rs[i])
                i += 1
        elif c == "\\" and i + 1 < len(rs):
            inw = True
            i += 1
            w.append(rs[i])
        else:
            inw = True
            w.append(c)
        i += 1
    if inw:
        out.append("".join(w))
    return out


def _redirection(w):
    i = 0
    while i < len(w) and "0" <= w[i] <= "9":
        i += 1
    return 0 < i < len(w) and w[i] in "<>"


def _duration(w):
    t = w.rstrip("smhd")
    return t != "" and all(("0" <= c <= "9") or c == "." for c in t)


def _program_at(ws):
    i = 0
    while i < len(ws):
        w = ws[i]
        if w in ("<<", "<<-", "<", ">", ">>", "2>", "&>"):
            i += 2
            continue
        if w.startswith("<") or w.startswith(">") or _redirection(w) or w.startswith("-") or "=" in w or _duration(w):
            i += 1
            continue
        base = _go_path_base(w)
        if base in _WRAPPERS:
            i += 1
            continue
        ws[i] = base
        return i
    return -1


def _program(cmd):
    ws = _words(cmd)
    at = _program_at(ws)
    return ws[at] if at >= 0 else ""


def _split_sh(line, out):
    rs = line
    b = _Buf()
    heredocs = []
    nested, ngroups = [], []

    def cut(op):
        out.cut(b, op)

    def word_start():
        return b.empty() or b.after(" \t\n")

    def blank_next(i):
        return i + 1 >= len(rs) or rs[i + 1] in " \t\n;&|)"

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "\\" and i + 1 < len(rs):
            if rs[i + 1] == "\n":
                i += 2
                continue
            b.write(r, True)
            i += 1
            b.write(rs[i], True)
        elif r == "'":
            j = min(_index_from(rs, i + 1, "'") + 1, len(rs))
            b.write_str(rs[i:j])
            i = j - 1
        elif r == '"':
            j = i + 1
            while j < len(rs) and rs[j] != '"':
                if rs[j] == "\\":
                    j += 1
                elif rs[j] == "$" and j + 1 < len(rs) and rs[j + 1] == "(":
                    end = _close_paren(rs, j + 2, SH)
                    out.nest(nested, ngroups, SH, rs[j + 2:min(end, len(rs))])
                    j = end
                elif rs[j] == "`":
                    end = _index_from(rs, j + 1, "`")
                    out.nest(nested, ngroups, SH, rs[j + 1:min(end, len(rs))])
                    j = end
                j += 1
            j = min(j + 1, len(rs))
            b.write_str(rs[i:j])
            i = j - 1
        elif r == "$" and i + 1 < len(rs) and rs[i + 1] == "(" and not (i + 2 < len(rs) and rs[i + 2] == "("):
            end = _close_paren(rs, i + 2, SH)
            out.nest(nested, ngroups, SH, rs[i + 2:min(end, len(rs))])
            b.write_str(rs[i:min(end + 1, len(rs))])
            i = end
        elif r == "`":
            end = _index_from(rs, i + 1, "`")
            out.nest(nested, ngroups, SH, rs[i + 1:min(end, len(rs))])
            b.write_str(rs[i:min(end + 1, len(rs))])
            i = end
        elif r == "#" and word_start():
            while i < len(rs) and rs[i] != "\n":
                i += 1
            i -= 1
        elif r == "<" and i + 2 < len(rs) and rs[i + 1] == "<" and rs[i + 2] == "<":
            b.write_str("<<<")
            i += 2
        elif r == "<" and i + 1 < len(rs) and rs[i + 1] == "<":
            h, nxt = _read_heredoc(rs, i + 2)
            h["cmd"] = len(out.cmds)
            heredocs.append(h)
            b.write_str(rs[i:nxt])
            i = nxt - 1
        elif r == "\n":
            progs = {}
            if heredocs:
                progs[len(out.cmds)] = _program(b.text())
            cut(";")
            for h in heredocs:
                if h["cmd"] in progs:
                    prog = progs[h["cmd"]]
                elif h["cmd"] < len(out.cmds):
                    prog = progs[h["cmd"]] = _program(out.cmds[h["cmd"]])
                else:
                    prog = ""
                start = i + 1
                end, nxt = _heredoc_body(rs, start, h)
                body = rs[start:end]
                if not h["quoted"]:
                    _substitutions(body, out, nested, ngroups)
                if prog in _SHELLS:
                    out.nest(nested, ngroups, SH, body)
                elif prog in _INTERPRETERS:
                    s = trim(body)
                    if s:
                        nested.append(s)
                i = nxt - 1
            heredocs = []
        elif r in ";()":
            cut(r)
        elif r in "{}" and word_start() and blank_next(i):
            cut(r)
        elif r == "|":
            op = "|"
            if i + 1 < len(rs) and rs[i + 1] in "|&":
                i += 1
                op += rs[i]
            cut(op)
        elif r == "&":
            if i + 1 < len(rs) and rs[i + 1] == "&":
                i += 1
                cut("&&")
            elif (i + 1 < len(rs) and rs[i + 1] == ">") or b.after("<>"):
                b.write(r, False)
            else:
                cut("&")
        else:
            b.write(r, False)
        i += 1
    cut("")
    out.cmds += nested
    out.groups += ngroups


def _substitutions(body, out, nested, ngroups):
    rs = body
    i = 0
    while i < len(rs):
        if rs[i] == "\\":
            i += 1
        elif rs[i] == "$" and i + 1 < len(rs) and rs[i + 1] == "(":
            end = _close_paren(rs, i + 2, SH)
            out.nest(nested, ngroups, SH, rs[i + 2:min(end, len(rs))])
            i = end
        elif rs[i] == "`":
            end = _index_from(rs, i + 1, "`")
            out.nest(nested, ngroups, SH, rs[i + 1:min(end, len(rs))])
            i = end
        i += 1


def _read_heredoc(rs, i):
    h = {"word": "", "tabs": False, "quoted": False, "cmd": 0}
    if i < len(rs) and rs[i] == "-":
        h["tabs"] = True
        i += 1
    while i < len(rs) and rs[i] in " \t":
        i += 1
    w = []
    while i < len(rs) and rs[i] not in " \t\n;&|<>()":
        c = rs[i]
        if c == "'":
            h["quoted"] = True
            j = _index_from(rs, i + 1, "'")
            w.append(rs[i + 1:min(j, len(rs))])
            i = min(j + 1, len(rs))
            continue
        if c == '"':
            h["quoted"] = True
            i += 1
            while i < len(rs) and rs[i] != '"':
                if rs[i] == "\\" and i + 1 < len(rs) and rs[i + 1] in "\"\\$`":
                    i += 1
                w.append(rs[i])
                i += 1
            i = min(i + 1, len(rs))
            continue
        if c == "\\":
            if i + 1 < len(rs) and rs[i + 1] == "\n":
                i += 2
                continue
            h["quoted"] = True
            i += 1
            if i >= len(rs):
                continue
        w.append(rs[i])
        i += 1
    h["word"] = "".join(w)
    return h, i


def _heredoc_body(rs, i, h):
    while i < len(rs):
        e = _index_from(rs, i, "\n")
        l = rs[i:e]
        while not h["quoted"] and l.endswith("\\") and e < len(rs):
            n = _index_from(rs, e + 1, "\n")
            l = l[:-1] + rs[e + 1:n]
            e = n
        if h["tabs"]:
            l = l.lstrip("\t")
        if l == h["word"]:
            return i, min(e + 1, len(rs))
        i = e + 1
    return len(rs), len(rs)


# ── PowerShell ──────────────────────────────────────────────────────────────

def _split_powershell(line, out):
    rs = line
    b = _Buf()
    nested, ngroups = [], []

    def cut(op):
        out.cut(b, op)

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "`" and i + 1 < len(rs):
            b.write(r, True)
            i += 1
            b.write(rs[i], True)
        elif r == "@" and i + 1 < len(rs) and rs[i + 1] in "'\"" and _here_string_start(rs, i + 2):
            q = rs[i + 1]
            end = _here_string_end(rs, i + 2, q)
            if q == '"':
                _subexpressions(rs[i + 2:min(end, len(rs))], out, nested, ngroups)
            b.write_str("@" + q + "…" + q + "@")
            i = end
        elif r == "'":
            j = i + 1
            while j < len(rs):
                if rs[j] == "'":
                    if j + 1 < len(rs) and rs[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            j = min(j + 1, len(rs))
            b.write_str(rs[i:j])
            i = j - 1
        elif r == '"':
            j = i + 1
            while j < len(rs) and rs[j] != '"':
                if rs[j] == "`":
                    j += 2
                elif rs[j] == "$" and j + 1 < len(rs) and rs[j + 1] == "(":
                    end = _close_paren(rs, j + 2, POWERSHELL)
                    out.nest(nested, ngroups, POWERSHELL, rs[j + 2:min(end, len(rs))])
                    j = end + 1
                else:
                    j += 1
            j = min(j + 1, len(rs))
            b.write_str(rs[i:j])
            i = j - 1
        elif r == "<" and i + 1 < len(rs) and rs[i + 1] == "#":
            j = i + 2
            while j + 1 < len(rs) and not (rs[j] == "#" and rs[j + 1] == ">"):
                j += 1
            i = j + 1
        elif r == "#" and (b.empty() or b.after(" \t")):
            while i < len(rs) and rs[i] != "\n":
                i += 1
            i -= 1
        elif r in "({":
            end = _close_paren(rs, i + 1, POWERSHELL) if r == "(" else _close_brace(rs, i + 1)
            out.nest(nested, ngroups, POWERSHELL, rs[i + 1:min(end, len(rs))])
            b.write_str(rs[i:min(end + 1, len(rs))])
            i = end
        elif r in "\n;":
            cut(";")
        elif r == "|":
            op = "|"
            if i + 1 < len(rs) and rs[i + 1] == "|":
                i += 1
                op = "||"
            cut(op)
        elif r == "&" and i + 1 < len(rs) and rs[i + 1] == "&":
            i += 1
            cut("&&")
        else:
            b.write(r, False)
        i += 1
    cut("")
    out.cmds += nested
    out.groups += ngroups


def _here_string_start(rs, i):
    while i < len(rs) and rs[i] != "\n":
        if rs[i] not in " \t\r":
            return False
        i += 1
    return i < len(rs)


def _here_string_end(rs, i, q):
    while i + 1 < len(rs):
        if rs[i] == "\n" and i + 2 < len(rs) and rs[i + 1] == q and rs[i + 2] == "@":
            return i + 2
        i += 1
    return len(rs)


def _subexpressions(rs, out, nested, ngroups):
    i = 0
    while i + 1 < len(rs):
        if rs[i] == "`":
            i += 2
            continue
        if rs[i] == "$" and rs[i + 1] == "(":
            end = _close_paren(rs, i + 2, POWERSHELL)
            out.nest(nested, ngroups, POWERSHELL, rs[i + 2:min(end, len(rs))])
            i = end
        i += 1


# ── cmd.exe ─────────────────────────────────────────────────────────────────

def _split_cmd(line, out):
    rs = line
    b = _Buf()

    def cut(op):
        out.cut(b, op)

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "^" and i + 1 < len(rs):
            b.write(r, True)
            i += 1
            b.write(rs[i], True)
        elif r == '"':
            j = min(_index_from(rs, i + 1, '"') + 1, len(rs))
            b.write_str(rs[i:j])
            i = j - 1
        elif r == "\n":
            cut("&")
        elif r in "()":
            cut(r)
        elif r in "&|":
            if r == "&" and b.after("<>"):
                b.write(r, False)
                i += 1
                continue
            op = r
            if i + 1 < len(rs) and rs[i + 1] == r:
                i += 1
                op += op
            cut(op)
        else:
            b.write(r, False)
        i += 1
    cut("")


# ── the guard ───────────────────────────────────────────────────────────────

GOLDEN = pathlib.Path(__file__).parent.parent / "internal" / "shellsplit" / "testdata" / "golden.jsonl"


def _load(path):
    known = {}
    for l in path.read_text().splitlines():
        if l.strip():
            g = json.loads(l)
            known[(g["dialect"], g["line"])] = g
    return known


def check_golden(lines=(), path=GOLDEN, extra=()):
    """Every line in the Go-made golden file(s) cut the same way here, and every
    (dialect, line) in lines that the files leave out — Go found it trivial,
    one command cut into itself — trivial here too; or exit. extra are golden
    files Go made for data outside the repository (see TestGolden's -extra)."""
    known = _load(path)
    for p in extra:
        known.update(_load(pathlib.Path(p)))
    bad = []
    for g in known.values():
        c, gr, co, _ = parse(g["dialect"], g["line"])
        if c != g["parts"] or gr != g.get("groups", []) or co != g["code"]:
            bad.append((g, {"parts": c, "groups": gr, "code": co}))
    for d, line in lines:
        if (d, line) in known:
            continue
        s = trim(line)
        c, gr, co, _ = parse(d, line)
        if c != [s] or gr or co != s:
            bad.append(({"dialect": d, "line": line, "parts": "[not in a golden file: trivial in Go]", "code": ""},
                        {"parts": c, "groups": gr, "code": co}))
    if bad:
        g, got = bad[0]
        raise SystemExit(f"scripts/shellsplit.py disagrees with internal/shellsplit on {len(bad)} line(s) — "
                         f"first: {g['line']!r}\n  go:     {g['parts']} | {g['code']!r}\n"
                         f"  python: {got['parts']} {got['groups']} | {got['code']!r}\n"
                         "(the repository's data changed? go test ./internal/shellsplit -run TestGolden -update\n"
                         " data of your own? go test ./internal/shellsplit -run TestGolden -args -extra a.jsonl,b.jsonl"
                         " -out /tmp/extra-golden.jsonl, then train_risk.py --golden /tmp/extra-golden.jsonl)")
    return len(known) + len(lines)
