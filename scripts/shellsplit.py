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
import base64
import binascii
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
MAX_PARSES = 1024  # mirrors maxParses


def parse(dialect, line, depth=0, budget=None):
    """(commands, groups, code, incomplete) — see internal/shellsplit.Parse."""
    if budget is None:
        budget = [MAX_PARSES]
    budget[0] -= 1
    if depth >= MAX_DEPTH or budget[0] < 0:
        s = trim(line)
        return ([s] if s else []), [], trim(line), True
    r = _Result(dialect, depth, budget)
    if dialect == POWERSHELL:
        _split_powershell(line, r)
    elif dialect == CMD:
        _split_cmd(line, r)
    else:
        _split_sh(line, r)
    r.end_group()
    c = "".join(r.code)
    if c.endswith(" ;"):
        c = c[:-2]
    return r.cmds + r.inner, r.groups + r.inner_g, trim(c), r.incomplete


def split(dialect, line):
    return parse(dialect, line)[0]


def code(dialect, line):
    return parse(dialect, line)[2]


_JOINS = {"|", "||", "&&", "|&"}


class _Result:
    def __init__(self, dialect=SH, depth=0, budget=None):
        self.d = dialect
        self.depth = depth
        self.budget = budget if budget is not None else [MAX_PARSES]
        self.incomplete = False
        self.inner = []
        self.inner_g = []
        self.cmds = []
        self.piped = []
        self.groups = []
        self.code = []  # pieces, joined once: += on a str is quadratic
        self.group = []
        self.gtext = []

    def cut(self, b, op):
        s = trim(b.text())
        if s:
            self.cmds.append(s)
            self.piped.append(op in ("|", "|&"))
            self.inner_code(s)
            if self.code:
                self.code.append(" ")
            self.code.append(s)
            if self.group:
                self.gtext.append(" ")
            self.gtext.append(s)
            self.group.append(s)
        if op and self.code:
            self.code.append(" " + op)
        if op in _JOINS:
            if self.group:
                self.gtext.append(" " + op)
        else:
            self.end_group()
        b.reset()

    def end_group(self):
        if len(self.group) > 1:
            self.groups.append(trim(trim("".join(self.gtext)).rstrip("|&")))
        self.group = []
        self.gtext = []

    def nest(self, nested, ngroups, dialect, code_):
        c, g, _, inc = parse(dialect, code_, self.depth + 1, self.budget)
        nested += c
        ngroups += g
        self.incomplete = self.incomplete or inc

    def inner_code(self, cmd):
        """Mirrors innerCode: one pass; a flag belongs to the latest interpreter named."""
        ws = _fields(cmd) if self.d == CMD else _words(cmd)
        prog = ""
        i = 0
        while i < len(ws):
            w = ws[i]
            run = _word_runner(w)
            if _operator(w):
                i += 1
            elif run == "eval":
                self.nest(self.inner, self.inner_g, self.d, " ".join(ws[i + 1:]))
                return
            elif run:
                prog = run
            elif prog and i + 1 < len(ws) and self.hand_off(prog, w, ws, i + 1):
                return
            i += 1

    def hand_off(self, prog, w, ws, at):
        """Mirrors handOff; the flag's argument is ws[at], what follows it ws[at:]."""
        ps = prog in ("pwsh", "powershell")
        if prog in _SHELLS and w.startswith("-") and not w.startswith("--") and "c" in w:
            self.nest(self.inner, self.inner_g, SH, ws[at])
        elif ps and w.lower() in ("-c", "-command"):
            self.nest(self.inner, self.inner_g, POWERSHELL, " ".join(ws[at:]))
        elif ps and _encoded_flag(w):
            code_ = _decode_utf16(ws[at])
            if code_ is not None:
                self.nest(self.inner, self.inner_g, POWERSHELL, code_)
            else:
                self.incomplete = True
        elif prog == "cmd" and w.lower() in ("/c", "/k"):
            self.nest(self.inner, self.inner_g, CMD, " ".join(ws[at:]))
        elif prog in _INTERPRETERS and w in ("-c", "-e", "--eval"):
            t = trim(ws[at])
            if t:
                self.inner.append(t)
        else:
            return False
        return True

    def readers(self, first):
        """Mirrors readers: what reads each command's heredoc, through pipes."""
        if first >= len(self.cmds):
            return []
        fed = [0] * (len(self.cmds) - first)
        for j in range(len(self.cmds) - 1, first - 1, -1):
            f = _feeds(self.cmds[j])
            if self.piped[j]:
                if j + 1 < len(self.cmds):
                    f |= fed[j + 1 - first]
                else:
                    self.incomplete = True
            fed[j - first] = f
        return fed


def _encoded_flag(w):
    """Mirrors encodedFlag: -EncodedCommand, any prefix of it, -ec; - or /."""
    l = w.lower()
    if len(l) < 2 or l[0] not in "-/":
        return False
    return l[1:] == "ec" or "encodedcommand".startswith(l[1:])


def _decode_utf16(s):
    """Mirrors decodeUTF16: base64 (Go's StdEncoding skips CR and LF) of UTF-16LE."""
    try:
        b = base64.b64decode(s.replace("\r", "").replace("\n", ""), validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(b) % 2:
        return None
    return b.decode("utf-16-le", errors="replace")


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
_INTERPRETERS = {"python", "python3", "python2", "perl", "ruby", "node", "nodejs", "php", "lua", "deno", "bun",
                 "osascript", "tclsh", "Rscript"}


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
    """path.Base."""
    if f == "":
        return "."
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


def _runner(w):
    """Mirrors runner: the interpreter a word names, by base name, or ""."""
    b = _name(w)
    l = b.lower()
    if l in ("pwsh", "powershell", "cmd"):
        return l
    v = b.rstrip("0123456789.")
    if v:
        b = v
    if b == "eval" or b in _SHELLS or b in _INTERPRETERS:
        return b
    return ""


def _name(w):
    """Mirrors name: base name, without quotes or .exe."""
    b = _go_path_base(w.strip('"').replace("\\", "/"))
    if len(b) > 4 and b[-4:].lower() == ".exe":
        b = b[:-4]
    return b


def _operator(w):
    """Mirrors operator: a redirection operator alone, waiting for its operand."""
    if not any(c in "<>" for c in w) or w.strip("0123456789&<>-|") != "":
        return False
    return w.endswith("<<-") or w[-1] in "<>&|"


def _word_runner(w):
    """Mirrors wordRunner: runner of the word before an attached redirection."""
    cut = [k for k in (w.find("<"), w.find(">")) if k >= 0]
    if cut:
        w = w[:min(cut)].rstrip("0123456789&")
    return _runner(w)


_SCRIPT, _CODE, _UNKNOWN = 1, 2, 4

# Mirrors dataReaders.
_DATA_READERS = {":", "cat", "tee", "git", "gh", "grep", "egrep", "fgrep", "rg", "sort", "uniq", "wc", "head",
                 "tail", "cut", "tr", "base64", "jq", "yq", "diff", "cmp", "patch", "column", "fold", "nl", "od",
                 "xxd", "hexdump", "less", "more", "read", "echo", "printf", "tac", "rev", "paste", "envsubst",
                 "md5sum", "sha256sum", "shasum", "sed", "awk"}


def _feeds(cmd):
    """Mirrors feeds: interpreters among the words add what is scored; a program
    neither an interpreter nor a data reader makes the reader unknown."""
    f = 0
    prog, found = "", False
    ws = _words(cmd)
    i = 0
    while i < len(ws):
        w = ws[i]
        if _operator(w):
            i += 2
            continue
        r = _word_runner(w)
        if r in _SHELLS:
            f |= _SCRIPT
        elif r in _INTERPRETERS:
            f |= _CODE
        if not found and not _assignment(w):
            prog, found = _program(w)
        i += 1
    if not found or (ws and _operator(ws[-1])):
        f |= _UNKNOWN
    elif prog in _DATA_READERS and prog in ("sed", "awk") and _script_file(ws):
        f |= _UNKNOWN
    elif prog in _DATA_READERS or _word_runner(prog):
        pass
    else:
        f |= _UNKNOWN
    return f


def _program(w):
    """Mirrors program."""
    cut = [k for k in (w.find("<"), w.find(">")) if k >= 0]
    if cut:
        k = min(cut)
        if w[:k].strip("0123456789&") == "":
            return "", False
        w = w[:k]
    return _name(w), True


def _assignment(w):
    """Mirrors assignment: NAME=value."""
    i = w.find("=")
    if i <= 0:
        return False
    for j, c in enumerate(w[:i]):
        if not (c == "_" or "a" <= c <= "z" or "A" <= c <= "Z" or (j > 0 and "0" <= c <= "9")):
            return False
    return True


def _script_file(ws):
    """Mirrors scriptFile."""
    return any(w.startswith("--file") or (w.startswith("-") and not w.startswith("--") and "f" in w) for w in ws)


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
            cut(";")
            fed = out.readers(heredocs[0]["cmd"]) if heredocs else []
            for h in heredocs:
                f = fed[h["cmd"] - heredocs[0]["cmd"]] if h["cmd"] < len(out.cmds) else 0
                start = i + 1
                end, nxt = _heredoc_body(rs, start, h)
                body = rs[start:end]
                if not h["quoted"]:
                    _substitutions(body, out, nested, ngroups)
                if f & _UNKNOWN and not f & (_SCRIPT | _CODE):
                    out.incomplete = True
                if f & _SCRIPT:
                    out.nest(nested, ngroups, SH, body)
                if f & _CODE:
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
                if rs[i] == "\\" and i + 1 < len(rs) and rs[i + 1] == "\n":
                    i += 2
                    continue
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
        while not h["quoted"] and (len(l) - len(l.rstrip("\\"))) % 2 == 1 and e < len(rs):
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
        c, gr, co, inc = parse(g["dialect"], g["line"])
        if c != g["parts"] or gr != g.get("groups", []) or co != g["code"] or inc != g.get("incomplete", False):
            bad.append((g, {"parts": c, "groups": gr, "code": co, "incomplete": inc}))
    for d, line in lines:
        if (d, line) in known:
            continue
        s = trim(line)
        c, gr, co, inc = parse(d, line)
        if c != [s] or gr or co != s or inc:
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
