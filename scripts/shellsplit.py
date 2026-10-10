"""Mirror of internal/shellsplit: cut a command line into the commands a shell
would run, by that shell's own rules.

The risk model scores a shell line by its parts — its code (the line without
heredoc bodies, here-strings and comments) and each command in it — and the
trainer must cut lines exactly as the Go scorer does, or it trains and measures
on text inference never produces. So this is a line-by-line port, and it is
guarded like the feature hash: internal/shellsplit/testdata/golden.jsonl holds
what the Go code returns for a corpus of lines, the Go tests check Go against
it, and check_golden() below checks this file against it on every training run.
Edit one side alone and both fail.

Python strings index by code point, as the Go code indexes []rune.
"""
import json
import pathlib

SH, POWERSHELL, CMD = 0, 1, 2


def split(dialect, line):
    """The commands in line, in order, trimmed; empty ones dropped. Commands
    nested in another follow the one that holds them."""
    return _parse(dialect, line)[0]


def code(dialect, line):
    """line without its data, with its commands and the operators between them."""
    c = _parse(dialect, line)[1]
    if c.endswith(" ;"):
        c = c[:-2]
    return c.strip()


class _Result:
    def __init__(self):
        self.cmds = []
        self.code = ""

    def cut(self, b, op):
        s = "".join(b).strip()
        if s:
            self.cmds.append(s)
            if self.code:
                self.code += " "
            self.code += s
        if op and self.code:
            self.code += " " + op
        b.clear()


def _parse(dialect, line):
    r = _Result()
    if dialect == POWERSHELL:
        _split_powershell(line, r)
    elif dialect == CMD:
        _split_cmd(line, r)
    else:
        _split_sh(line, r)
    return r.cmds, r.code


def _last(b):
    s = "".join(b)
    return s[-1] if s else ""


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

def _split_sh(line, out):
    rs = line
    b = []
    heredocs = []
    nested = []

    def cut(op):
        out.cut(b, op)

    def word_start():
        return not b or _last(b) in " \t\n;&|()"

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "\\" and i + 1 < len(rs):
            if rs[i + 1] == "\n":
                i += 2
                continue
            b.append(r)
            i += 1
            b.append(rs[i])
        elif r == "'":
            j = min(_index_from(rs, i + 1, "'") + 1, len(rs))
            b.append(rs[i:j])
            i = j - 1
        elif r == '"':
            j = i + 1
            while j < len(rs) and rs[j] != '"':
                if rs[j] == "\\":
                    j += 1
                elif rs[j] == "$" and j + 1 < len(rs) and rs[j + 1] == "(":
                    end = _close_paren(rs, j + 2, SH)
                    nested += split(SH, rs[j + 2:min(end, len(rs))])
                    j = end
                elif rs[j] == "`":
                    end = _index_from(rs, j + 1, "`")
                    nested += split(SH, rs[j + 1:min(end, len(rs))])
                    j = end
                j += 1
            j = min(j + 1, len(rs))
            b.append(rs[i:j])
            i = j - 1
        elif r == "$" and i + 1 < len(rs) and rs[i + 1] == "(" and not (i + 2 < len(rs) and rs[i + 2] == "("):
            end = _close_paren(rs, i + 2, SH)
            nested += split(SH, rs[i + 2:min(end, len(rs))])
            b.append(rs[i:min(end + 1, len(rs))])
            i = end
        elif r == "`":
            end = _index_from(rs, i + 1, "`")
            nested += split(SH, rs[i + 1:min(end, len(rs))])
            b.append(rs[i:min(end + 1, len(rs))])
            i = end
        elif r == "#" and word_start():
            while i < len(rs) and rs[i] != "\n":
                i += 1
            i -= 1
        elif r == "<" and i + 1 < len(rs) and rs[i + 1] == "<" and not (i + 2 < len(rs) and rs[i + 2] == "<"):
            h, nxt = _read_heredoc(rs, i + 2)
            heredocs.append(h)
            b.append(rs[i:nxt])
            i = nxt - 1
        elif r == "\n":
            cut(";")
            for h in heredocs:
                i = _skip_heredoc_body(rs, i + 1, h) - 1
            heredocs = []
        elif r in ";()" or (r in "{}" and word_start()):
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
            elif (i + 1 < len(rs) and rs[i + 1] == ">") or (b and _last(b) in "<>"):
                b.append(r)
            else:
                cut("&")
        else:
            b.append(r)
        i += 1
    cut("")
    out.cmds += nested


def _read_heredoc(rs, i):
    tabs = False
    if i < len(rs) and rs[i] == "-":
        tabs = True
        i += 1
    while i < len(rs) and rs[i] in " \t":
        i += 1
    w = []
    while i < len(rs) and rs[i] not in " \t\n;&|<>()":
        c = rs[i]
        if c in "'\"":
            j = _index_from(rs, i + 1, c)
            w.append(rs[i + 1:min(j, len(rs))])
            i = min(j + 1, len(rs))
            continue
        if c == "\\":
            i += 1
            if i >= len(rs):
                continue
        w.append(rs[i])
        i += 1
    return ("".join(w), tabs), i


def _skip_heredoc_body(rs, i, h):
    word, tabs = h
    while i < len(rs):
        end = _index_from(rs, i, "\n")
        l = rs[i:end]
        if tabs:
            l = l.lstrip("\t")
        if l == word:
            return min(end + 1, len(rs))
        i = end + 1
    return len(rs)


# ── PowerShell ──────────────────────────────────────────────────────────────

def _split_powershell(line, out):
    rs = line
    b = []
    nested = []

    def cut(op):
        out.cut(b, op)

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "`" and i + 1 < len(rs):
            b.append(r)
            i += 1
            b.append(rs[i])
        elif r == "@" and i + 1 < len(rs) and rs[i + 1] in "'\"" and _here_string_start(rs, i + 2):
            q = rs[i + 1]
            end = _here_string_end(rs, i + 2, q)
            if q == '"':
                nested += _subexpressions(rs[i + 2:min(end, len(rs))])
            b.append("@" + q + "…" + q + "@")
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
            b.append(rs[i:j])
            i = j - 1
        elif r == '"':
            j = i + 1
            while j < len(rs) and rs[j] != '"':
                if rs[j] == "`":
                    j += 1
                j += 1
            nested += _subexpressions(rs[i + 1:min(j, len(rs))])
            j = min(j + 1, len(rs))
            b.append(rs[i:j])
            i = j - 1
        elif r == "<" and i + 1 < len(rs) and rs[i + 1] == "#":
            j = i + 2
            while j + 1 < len(rs) and not (rs[j] == "#" and rs[j + 1] == ">"):
                j += 1
            i = j + 1
        elif r == "#" and (not b or _last(b) in " \t;|&(){}"):
            while i < len(rs) and rs[i] != "\n":
                i += 1
            i -= 1
        elif r in "({":
            end = _close_paren(rs, i + 1, POWERSHELL) if r == "(" else _close_brace(rs, i + 1)
            nested += split(POWERSHELL, rs[i + 1:min(end, len(rs))])
            b.append(rs[i:min(end + 1, len(rs))])
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
            b.append(r)
        i += 1
    cut("")
    out.cmds += nested


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


def _subexpressions(rs):
    out = []
    i = 0
    while i + 1 < len(rs):
        if rs[i] == "`":
            i += 2
            continue
        if rs[i] == "$" and rs[i + 1] == "(":
            end = _close_paren(rs, i + 2, POWERSHELL)
            out += split(POWERSHELL, rs[i + 2:min(end, len(rs))])
            i = end
        i += 1
    return out


# ── cmd.exe ─────────────────────────────────────────────────────────────────

def _split_cmd(line, out):
    rs = line
    b = []

    def cut(op):
        out.cut(b, op)

    i = 0
    while i < len(rs):
        r = rs[i]
        if r == "^" and i + 1 < len(rs):
            b.append(r)
            i += 1
            b.append(rs[i])
        elif r == '"':
            j = min(_index_from(rs, i + 1, '"') + 1, len(rs))
            b.append(rs[i:j])
            i = j - 1
        elif r == "\n":
            cut("&")
        elif r in "()":
            cut(r)
        elif r in "&|":
            if r == "&" and b and _last(b) in "<>":
                b.append(r)
                i += 1
                continue
            op = r
            if i + 1 < len(rs) and rs[i + 1] == r:
                i += 1
                op += op
            cut(op)
        else:
            b.append(r)
        i += 1
    cut("")


# ── the guard ───────────────────────────────────────────────────────────────

GOLDEN = pathlib.Path(__file__).parent.parent / "internal" / "shellsplit" / "testdata" / "golden.jsonl"


def check_golden(lines=(), path=GOLDEN):
    """Every line in the Go-made golden file cut the same way here, and every
    (dialect, line) in lines that the file leaves out — Go found it trivial, one
    command cut into itself — trivial here too; or exit. Returns the count."""
    bad = []
    known = {}
    for l in path.read_text().splitlines():
        if l.strip():
            g = json.loads(l)
            known[(g["dialect"], g["line"])] = g
    for g in known.values():
        got = {"parts": split(g["dialect"], g["line"]), "code": code(g["dialect"], g["line"])}
        if got["parts"] != g["parts"] or got["code"] != g["code"]:
            bad.append((g, got))
    for d, line in lines:
        if (d, line) in known:
            continue
        s = line.strip()
        got = {"parts": split(d, line), "code": code(d, line)}
        if got["parts"] != [s] or got["code"] != s:
            bad.append(({"dialect": d, "line": line, "parts": "[not in the golden file: trivial in Go]", "code": ""}, got))
    if bad:
        g, got = bad[0]
        raise SystemExit(f"scripts/shellsplit.py disagrees with internal/shellsplit on {len(bad)} line(s) — "
                         f"first: {g['line']!r}\n  go:     {g['parts']} | {g['code']!r}\n"
                         f"  python: {got['parts']} | {got['code']!r}\n"
                         "(data changed? go test ./internal/shellsplit -run TestGolden -update)")
    return len(known) + len(lines)
