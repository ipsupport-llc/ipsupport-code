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
MAX_LINKS = 1 << 16  # mirrors maxLinks


def parse(dialect, line, depth=0, budget=None):
    """(commands, groups, code, incomplete) — see internal/shellsplit.Parse."""
    return _parse(dialect, line, depth, budget)[:4]


def _parse(dialect, line, depth=0, budget=None):
    """parse, and the _Result fields every nesting hands on: untracked, var_prog, dropped."""
    if budget is None:
        budget = [MAX_PARSES]
    budget[0] -= 1
    if depth >= MAX_DEPTH or budget[0] < 0:
        s = trim(line)
        return ([s] if s else []), [], trim(line), True, False, False, False
    r = _Result(dialect, depth, budget)
    if dialect == POWERSHELL:
        _split_powershell(line, r)
    elif dialect == CMD:
        _split_cmd(line, r)
    else:
        _split_sh(line, r)
    r.end_group()
    if r.dropped and (r.untracked or r.var_prog):
        r.incomplete = True
    c = "".join(r.code)
    if c.endswith(" ;"):
        c = c[:-2]
    return r.cmds + r.inner, r.groups + r.inner_g, trim(c), r.incomplete, r.untracked, r.var_prog, r.dropped


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
        self.pipe_to = []
        self.to_block = []
        self.blocks = []
        self.feeding = []
        self.links = 0
        self.input = []
        self.func_name = False
        self.untracked = False
        self.var_prog = False
        self.dropped = False
        self.opens = []  # [start, end word, feeders], innermost last
        self.closed = None
        self.groups = []
        self.code = []  # pieces, joined once: += on a str is quadratic
        self.group = []
        self.gtext = []

    def cut(self, b, op):
        s = trim(b.text())
        pipe = op in ("|", "|&")
        if s:
            self.cmds.append(s)
            self.piped.append(pipe)
            self.pipe_to.append([])
            self.to_block.append([])
            self.input.append(False)
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
        self.block(s, op, pipe)
        b.reset()

    def block(self, s, op, pipe):
        """Mirrors block: blocks, and what each cut sends on (| |& > >( >( )."""
        ended = []
        if self.closed is not None:
            ended.append(self.closed)
        self.closed = None
        if s and self.d == SH:
            k = _keyword(s)
            if _CLOSERS.get(k):
                self.opens.append([len(self.cmds) - 1, _CLOSERS[k], []])
            elif self.opens and k == self.opens[-1][1]:
                ended.append((self.opens[-1][0], len(self.cmds) - 1))
                self.opens.pop()
        feeders = []
        if pipe or (op == "(" and s.endswith(">")):
            feeders = list(ended)
            if s:
                feeders.append((len(self.cmds) - 1, len(self.cmds)))
            for f in feeders:
                self.read_by(f[0], f[1], len(self.cmds))
            if op == "(" and self.d == SH and _lead(s) == "exec":
                self.incomplete = True
        if self.d == SH:
            k = _keyword(s)
            lead = _lead(s)
            if (self.func_name and op == ")" and s == "") or k in ("function", "coproc") or lead == "alias" or _expands(lead):
                self.untracked = True
            if _expands(lead) or (lead in _WRAPPERS and _wraps_variable(s)):
                self.var_prog = True
            self.func_name = op == "(" and len(_fields(s)) == 1 and not any(c in s for c in "<>=")
        pending = self.feeding
        self.feeding = []
        if op in ("(", "{"):
            o = [len(self.cmds), ")", feeders, op == "(" and s.endswith("<")]
            if not feeders and s == "":
                o[2] = pending
            self.opens.append(o)
        elif op in (")", "}") and self.opens and self.opens[-1][1] == ")":
            o = self.opens.pop()
            self.closed = (o[0], len(self.cmds))
            self.blocks.append((o[0], len(self.cmds)))
            if o[3] and self.link(len(self.cmds) - o[0]):
                for j in range(o[0], len(self.cmds)):
                    self.input[j] = True
            for f in o[2]:
                if not self.link(f[1] - f[0] + len(self.cmds) - o[0]):
                    break
                for j in range(f[0], f[1]):
                    self.to_block[j].append(len(self.blocks) - 1)
        else:
            self.feeding = feeders

    def link(self, n):
        """Mirrors link: room for n more reader links, under MAX_LINKS."""
        self.links += n
        if self.links > MAX_LINKS:
            self.incomplete = True
            return False
        return True

    def read_by(self, frm, to, t):
        """Mirrors readBy."""
        if not self.link(to - frm):
            return
        for j in range(frm, to):
            self.pipe_to[j].append(t)

    def end_group(self):
        if len(self.group) > 1:
            self.groups.append(trim(trim("".join(self.gtext)).rstrip("|&")))
        self.group = []
        self.gtext = []

    def nest(self, nested, ngroups, dialect, code_):
        c, g, _, inc, untracked, var_prog, dropped = _parse(dialect, code_, self.depth + 1, self.budget)
        nested += c
        ngroups += g
        self.incomplete = self.incomplete or inc
        self.var_prog = self.var_prog or var_prog
        self.dropped = self.dropped or dropped
        return untracked

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
                arg = " ".join(ws[i + 1:])
                if self.nest(self.inner, self.inner_g, self.d, arg):
                    self.untracked = True
                if self.d == SH and _expands(arg):
                    self.incomplete = True
                return
            elif run:
                prog = run
            elif prog and self.hand_off(prog, w, ws, i + 1):
                return
            i += 1

    def hand_off(self, prog, w, ws, at):
        """Mirrors handOff; the flag's argument is ws[at], what follows it ws[at:]."""
        ps = prog in ("pwsh", "powershell")
        rest = " ".join(ws[at:])
        code_, attached = _interpreter_code(prog, w)
        if at >= len(ws) and not attached:
            return False
        if prog in _SHELLS and w.startswith("-") and not w.startswith("--") and "c" in w:
            self.nest(self.inner, self.inner_g, SH, ws[at])
            rest = ws[at]
        elif ps and w.lower() in ("-c", "-command"):
            self.nest(self.inner, self.inner_g, POWERSHELL, rest)
        elif ps and _encoded_flag(w):
            code_ = _decode_utf16(ws[at])
            if code_ is not None:
                self.nest(self.inner, self.inner_g, POWERSHELL, code_)
            else:
                self.incomplete = True
        elif prog == "cmd" and w.lower() in ("/c", "/k"):
            self.nest(self.inner, self.inner_g, CMD, rest)
        elif prog in _INTERPRETERS and code_ is not None:
            if not attached:
                code_ = ws[at]
            t = trim(code_)
            if t:
                self.inner.append(t)
            if attached:
                rest = code_ + " " + rest
        else:
            return False
        if self.d == SH and _expands(rest):
            self.var_prog = True
        return True

    def readers(self, first):
        """Mirrors readers: through pipes and piped blocks; open ones are unknown."""
        if first >= len(self.cmds):
            return []
        n = len(self.cmds)
        open_ = self.opens[0][0] if self.opens else n
        fed = [0] * (n - first)

        def into(t):
            return fed[t - first] if t < n else _UNKNOWN

        block_fed = {}

        def into_block(i):
            if i not in block_fed:
                v = 0
                for t in range(self.blocks[i][0], self.blocks[i][1]):
                    v |= into(t)
                block_fed[i] = v
            return block_fed[i]

        for j in range(n - 1, first - 1, -1):
            f = _feeds(self.cmds[j])
            if self.piped[j]:
                f |= into(j + 1)
            for t in self.pipe_to[j]:
                f |= into(t)
            for i in self.to_block[j]:
                f |= into_block(i)
            if j >= open_ or self.input[j] or self.untracked:
                f |= _UNKNOWN
            fed[j - first] = f
        return fed


_CLOSERS = {"case": "esac", "if": "fi", "for": "done", "while": "done", "until": "done", "select": "done"}


def _expands(s):
    """Mirrors expands: a $ that isn't a $( ) substitution."""
    for i, c in enumerate(s):
        if c == "$" and not (i + 1 < len(s) and s[i + 1] == "(" and not (i + 2 < len(s) and s[i + 2] == "(")):
            return True
    return False


def _lead(s):
    """Mirrors lead: the builtin a simple command runs."""
    ws = _words(s)
    i = 0
    while i < len(ws):
        w = ws[i]
        if _operator(w):
            i += 2
            continue
        if w in ("!", "time", "command", "builtin", "-p", "-v", "-V", "--", "if", "then", "do", "else", "elif",
                 "while", "until") or _assignment(w):
            i += 1
            continue
        p, ok = _program(w)
        if ok:
            return p
        i += 1
    return ""



# Mirrors wrappers.
_WRAPPERS = {"exec", "env", "sudo", "doas", "su", "ssh", "timeout", "nice", "nohup", "stdbuf", "setsid", "ionice",
             "chrt", "taskset", "xargs", "find", "watch", "flock", "chroot", "runuser", "unshare", "nsenter",
             "strace", "ltrace", "parallel"}


def _wraps_variable(s):
    """Mirrors wrapsVariable."""
    ws = _words(s)
    i, past = 0, False
    while i < len(ws):
        w = ws[i]
        if _operator(w):
            i += 2
            continue
        if past and "<" not in w and ">" not in w and _expands(w):
            return True
        p, ok = _program(w)
        past = past or (ok and p in _WRAPPERS)
        i += 1
    return False


# Mirrors codeFlags.
_CODE_FLAGS = {"python": "c", "python2": "c", "python3": "c", "perl": "eE", "ruby": "e", "node": "ep", "nodejs": "ep",
               "bun": "ep", "php": "rBRE", "lua": "e", "osascript": "e", "Rscript": "e"}

# Mirrors codeOptions.
_CODE_OPTIONS = {"node": ["--print"], "nodejs": ["--print"], "bun": ["--print"], "php": ["--run"]}


def _interpreter_code(prog, w):
    """Mirrors interpreterCode: (code, attached), code None when w hands none."""
    if prog not in _INTERPRETERS:
        return None, False
    for o in ["--eval"] + _CODE_OPTIONS.get(prog, []):
        if w == o:
            return "", False
        if w.startswith(o + "="):
            return w[len(o) + 1:], True
    if len(w) < 2 or w[0] != "-" or w[1] == "-":
        return None, False
    for j in range(1, len(w)):
        c = w[j]
        if c in _CODE_FLAGS.get(prog, ""):
            return ("", False) if j == len(w) - 1 else (w[j + 1:], True)
        if not ("a" <= c <= "z" or "A" <= c <= "Z") or c in "WXMmr":
            return None, False
    return None, False


def _abbreviates(w, flag, least):
    """Mirrors abbreviates."""
    n = w.partition("=")[0]
    return len(n) >= least and flag.startswith(n)


def _keyword(s):
    """Mirrors keyword: the first word past ! and time (-p, --)."""
    for w in _fields(s):
        if w not in ("!", "time", "-p", "--"):
            return w
    return ""


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


_SCRIPT, _CODE, _UNKNOWN, _PS_SCRIPT, _CMD_SCRIPT = 1, 2, 4, 8, 16
_SCORED = _SCRIPT | _CODE | _PS_SCRIPT | _CMD_SCRIPT

# Mirrors dataReaders.
_DATA_READERS = {":", "cat", "tee", "git", "gh", "grep", "egrep", "fgrep", "rg", "sort", "uniq", "wc", "head",
                 "tail", "cut", "tr", "base64", "jq", "yq", "diff", "cmp", "column", "fold", "nl", "od",
                 "xxd", "hexdump", "more", "read", "echo", "printf", "tac", "rev", "paste", "envsubst",
                 "md5sum", "sha256sum", "shasum"}


# Mirrors readerRuns.
_READER_RUNS = {"sort": ("--compress-program", 4), "rg": ("--pre", 5)}


def _feeds(cmd):
    """Mirrors feeds: interpreters among the words add what is scored; a program
    neither an interpreter nor a data reader makes the reader unknown."""
    f = 0
    prog, found, at, env = "", False, 0, False
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
        elif r in ("pwsh", "powershell"):
            f |= _PS_SCRIPT
        elif r == "cmd":
            f |= _CMD_SCRIPT
        if not found and not _assignment(w):
            prog, found = _program(w)
            at = i
        env = env or (not found and _assignment(w) and (w.startswith("GIT_") or w.startswith("GH_")))
        i += 1
    if not found:
        f |= _UNKNOWN
    elif prog in ("git", "gh") and (env or not _data_subcommand(prog, ws[at + 1:])):
        f |= _UNKNOWN
    elif prog in _READER_RUNS and any(_abbreviates(w, *_READER_RUNS[prog]) for w in ws[at + 1:]):
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


# Mirrors dataSubcommands.
_DATA_SUBCOMMANDS = {
    "git": {"commit", "tag", "notes", "hash-object", "cat-file", "check-ignore", "check-attr", "check-mailmap",
            "rev-list", "log", "diff", "show", "status", "stripspace", "interpret-trailers", "mktree", "mktag",
            "commit-tree", "rev-parse", "ls-files", "grep", "shortlog", "blame"},
    "gh": {"api", "issue", "pr", "release", "gist", "secret", "variable", "repo", "label", "workflow", "run",
           "search"},
}


def _data_subcommand(prog, ws):
    """Mirrors dataSubcommand."""
    i = 0
    while i < len(ws):
        w = ws[i]
        if _operator(w):
            i += 1
        elif w.startswith("-c") or w.startswith("--config-env") or w.startswith("--exec-path"):
            return False
        elif w in ("-C", "--git-dir", "--work-tree", "--namespace", "-R", "--repo"):
            i += 1
        elif w.startswith("-"):
            pass
        else:
            sub, ok = _program(w)
            if prog == "git":
                return ok and sub in _DATA_SUBCOMMANDS[prog] and not (sub in _EDITS and _git_editor(ws[i + 1:]))
            return ok and sub in _DATA_SUBCOMMANDS[prog] and not any(x in ("-e", "--edit", "--editor") for x in ws[i + 1:])
        i += 1
    return False



# Mirrors edits.
_EDITS = {"commit", "tag", "notes"}


def _git_editor(ws):
    """Mirrors gitEditor."""
    i = 0
    while i < len(ws):
        w = ws[i]
        if w == "--":
            break
        if w.startswith("--"):
            if _abbreviates(w, "--edit", 3) or _abbreviates(w, "--template", 3):
                return True
        elif w.startswith("-"):
            for j, c in enumerate(w[1:], 1):
                if c in "et":
                    return True
                if c in "mFCcuS":
                    if j == len(w) - 1 and c in "mFCc":
                        i += 1
                    break
        i += 1
    return False

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
                if f & _UNKNOWN and not f & _SCORED:
                    out.incomplete = True
                elif not f & _SCORED:
                    out.dropped = True
                if f & _SCRIPT:
                    out.nest(nested, ngroups, SH, body)
                if f & _CODE:
                    s = trim(body)
                    if s:
                        nested.append(s)
                if f & _PS_SCRIPT:
                    out.nest(nested, ngroups, POWERSHELL, body)
                if f & _CMD_SCRIPT:
                    out.nest(nested, ngroups, CMD, body)
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
    hs = []  # (body, the index of the command it is in)

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
            stop = end - 1 if end < len(rs) else len(rs)
            hs.append((rs[i + 2:stop], len(out.cmds)))
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
    for body, k in hs:
        if _ps_reads(out, nested, k):
            continue
        t = trim(body)
        if t:
            nested.append(t)
            out.nest(nested, ngroups, POWERSHELL, body)
    out.cmds += nested
    out.groups += ngroups


def _ps_reads(out, nested, k):
    """Mirrors psReads."""
    if k >= len(out.cmds):
        return True
    j = k
    while True:
        if not _ps_data_command(out.cmds[j]):
            return False
        if not out.piped[j] or j + 1 >= len(out.cmds):
            break
        j += 1
    every = out.cmds + nested
    if any(_ps_redefines(c) for c in every):
        return False
    if _ps_assignment(out.cmds[k]) is None:
        return True
    return all(_ps_data_command(c) for c in every)


def _ps_assignment(cmd):
    """Mirrors psAssignment: the words assigned, or None."""
    ws = _fields(cmd)
    if not ws:
        return None
    v, eq, after = ws[0].partition("=")
    if eq:
        return _fields(after) + ws[1:] if _ps_variable(v.rstrip("+-*/")) else None
    if len(ws) < 2 or not _ps_variable(ws[0]):
        return None
    op = ws[1].lstrip("+-*/")
    if op.startswith("="):
        return _fields(op[1:]) + ws[2:]
    return None


# Mirrors psDataCommands.
_PS_DATA_COMMANDS = {"set-content", "add-content", "out-file", "new-item", "write-output", "write-host", "write",
                     "echo", "write-verbose", "write-information", "out-string", "out-null", "set-clipboard", "sc",
                     "ac", "convertfrom-json", "convertfrom-csv", "convertfrom-stringdata", "select-string",
                     "measure-object", "sort-object", "git", "gh"}


def _ps_data_command(cmd):
    """Mirrors psDataCommand."""
    ws = _fields(cmd)
    rest = _ps_assignment(cmd)
    if rest is not None:
        ws = rest
    if not ws or _ps_variable(ws[0]) or ws[0][0] in "'\"@":
        return True
    prog = _name(ws[0]).lower()
    if prog in ("git", "gh"):
        return _data_subcommand(prog, ws[1:])
    return prog in _PS_DATA_COMMANDS


def _ps_redefines(cmd):
    """Mirrors psRedefines."""
    ws = _fields(cmd.lower())
    while ws and ws[0] in ("&", "."):
        ws = ws[1:]
    return bool(ws) and (ws[0] in _PS_DEFINERS or any("alias:" in w or "function:" in w for w in ws))


# Mirrors psDefiners.
_PS_DEFINERS = {"set-alias", "sal", "new-alias", "nal", "import-alias", "ipal", "function", "filter"}


def _ps_variable(w):
    """Mirrors psVariable."""
    if len(w) < 2 or w[0] != "$":
        return False
    return all(c == "_" or c == ":" or "a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9" for c in w[1:])


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
