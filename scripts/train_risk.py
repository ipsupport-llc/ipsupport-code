#!/usr/bin/env python3
"""Train the tool-call risk classifier and write internal/risk/model.bin.

Offline and standalone on purpose: the agent ships no Python, so this runs on
a developer's machine and its only output is a weights file the Go build
embeds. Pure stdlib — no numpy, no sklearn — because the dataset is small
enough that sparse SGD in plain Python trains it in under a second, and a
training step nobody can run without installing a stack is a training step
nobody re-runs.

    python3 scripts/gen_risk_dataset.py     # regenerate the data
    python3 scripts/train_risk.py           # retrain + write the weights
    go test ./internal/risk/                # the Go side must agree

Everything that decides how text becomes features — the hash, the tokenizer,
the n-gram ranges, the rendered call text — is duplicated here from
internal/risk. That duplication is the price of not shipping a Python runtime,
so it is guarded: HASH_VECTORS below is asserted on every run against values
produced by the Go implementation, and internal/risk/features_test.go pins the
same ones. If either side is edited alone, both fail.
"""
import argparse, json, math, pathlib, random, struct, sys, unicodedata
from collections import Counter

# ── feature config (written into the model file, read back by Go) ────────────
DIM        = 1 << 15          # 32768 features -> 6 labels * 32768 * 4B = 786KB
SEED       = 20260921
WORD_MIN, WORD_MAX = 1, 2
CHAR_MIN, CHAR_MAX = 3, 5
LOWERCASE  = True
LABELS     = ["destructive", "sandbox_escape", "credential_access",
              "network", "external_side_effect", "safe"]
# Labels that describe the call without being a reason for concern. "network" is
# the case: reaching the internet is a property, not a risk — the risky half of
# it is external_side_effect. Left in the headline number, every `curl` to a
# documentation page scored 1.00, which is how a risk signal becomes noise.
# Written into the model file, so a replacement model declares its own.
INFORMATIONAL = {"network"}

# ── training ────────────────────────────────────────────────────────────────
EPOCHS, LR, L2 = 200, 0.25, 1e-6

# Produced by the Go implementation (internal/risk, TestHashVectors).
HASH_VECTORS = {
    "w1:rm":                (13286, -1),
    "w2:rm -rf":            (31782, -1),
    "c3:rm ":               (23190, -1),
    "c4:sudo":              (21484, +1),
    "w1:~/.aws/credentials": (9074, -1),
    "w1:безопасно":         (29652, -1),
}

OFFSET32, PRIME32, MASK32 = 2166136261, 16777619, 0xFFFFFFFF


def hash_idx(s: str):
    """FNV-1a over the seed then the feature's UTF-8 bytes. Mirrors hashIdx."""
    h = OFFSET32
    for b in SEED.to_bytes(4, "little"):
        h = ((h ^ b) * PRIME32) & MASK32
    for b in s.encode("utf-8"):
        h = ((h ^ b) * PRIME32) & MASK32
    sign = -1 if h & 1 else 1
    return (h >> 1) % DIM, sign


KEEP = set("/._-~:\\")


def tokenize(text: str):
    """Mirrors Tokenize: letters, digits, and the punctuation a command line
    carries meaning in. Everything else is a separator."""
    if LOWERCASE:
        text = text.lower()
    out, cur = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "N") or ch in KEEP:
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def featurize(text: str):
    """Mirrors Featurize. Returns {index: value}."""
    vec = {}
    def add(f):
        i, sign = hash_idx(f)
        vec[i] = vec.get(i, 0.0) + sign
    toks = tokenize(text)
    for n in range(WORD_MIN, WORD_MAX + 1):
        for i in range(0, len(toks) - n + 1):
            add("w" + chr(ord("0") + n) + ":" + " ".join(toks[i:i + n]))
    t = text.lower() if LOWERCASE else text
    for n in range(CHAR_MIN, CHAR_MAX + 1):
        for i in range(0, len(t) - n + 1):
            add("c" + chr(ord("0") + n) + ":" + t[i:i + n])
    return vec


MAX_VALUE, MAX_TEXT = 400, 2000


def call_text(tool, action, params):
    """Mirrors CallText: sorted params, capped values, capped whole."""
    parts = [tool] + ([action] if action else [])
    s = " ".join(parts)
    for k in sorted(params):
        v = str(params[k]).strip()
        if not v:
            continue
        s += " " + k + "=" + v[:MAX_VALUE]
        if len(s) > MAX_TEXT:
            break
    return s[:MAX_TEXT]


def check_vectors():
    bad = [(f, hash_idx(f), want) for f, want in HASH_VECTORS.items() if hash_idx(f) != want]
    if bad:
        for f, got, want in bad:
            print(f"  {f!r}: python {got} != go {want}", file=sys.stderr)
        sys.exit("hash vectors disagree with the Go implementation — one side was edited alone")
    print(f"hash vectors: {len(HASH_VECTORS)} ok (matches internal/risk)")


def report(title, rows, W, b):
    """Per-label precision/recall. Printed for the holdout first, because that is
    the number that decides whether the signal is worth anything."""
    print(f"\n{title} ({len(rows)} examples):")
    for li, lab in enumerate(LABELS):
        tp = fp = fn = 0
        for x, y in rows:
            z = b[li] + sum(W[li][j] * v for j, v in x.items())
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            hit, want = p >= 0.5, y[li] >= 0.5
            tp += hit and want
            fp += hit and not want
            fn += (not hit) and want
        if tp + fn == 0 and tp + fp == 0:
            print(f"  {lab:22} no examples either way")
            continue
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        print(f"  {lab:22} precision {prec:.2f}  recall {rec:.2f}  (tp={tp} fp={fp} fn={fn})")


def read_model(path):
    """Read a model file back, so training can CONTINUE from it instead of
    starting at zero. Mirrors Model.Write / Load (internal/risk/model.go).

    Starting from the shipped weights is what makes this fine-tuning rather than
    retraining: the base took the whole synthetic dataset to learn what a
    credential path looks like, and a handful of corrections collected from real
    approvals cannot rediscover that from scratch. It refuses a file whose
    feature config or label set differs — the weights would mean something else.
    """
    b = path.read_bytes()
    o = 0
    def take(n):
        nonlocal o
        s_ = b[o:o + n]; o += n
        if len(s_) != n:
            sys.exit(f"{path}: truncated")
        return s_
    if take(8) != b"IPSRISK\x01":
        sys.exit(f"{path}: not a model file")
    (ver,) = struct.unpack("<H", take(2))
    if ver != 1:
        sys.exit(f"{path}: format v{ver}, this script writes v1")
    dim, seed = struct.unpack("<II", take(8))
    wmin, wmax, cmin, cmax = struct.unpack("<BBBB", take(4))
    (flags,) = struct.unpack("<B", take(1))
    (nlab,) = struct.unpack("<H", take(2))
    labels, info = [], []
    for _ in range(nlab):
        (ln,) = struct.unpack("<H", take(2))
        labels.append(take(ln).decode("utf-8"))
        info.append(struct.unpack("<B", take(1))[0] & 1 == 1)
    got = (dim, seed, wmin, wmax, cmin, cmax, bool(flags & 1), labels)
    want = (DIM, SEED, WORD_MIN, WORD_MAX, CHAR_MIN, CHAR_MAX, LOWERCASE, LABELS)
    if got != want:
        sys.exit(f"{path} was built with a different feature config or label set — "
                 f"its weights index a different feature space, so continuing from it "
                 f"would be meaningless.\n  file:   {got}\n  script: {want}")
    bias = list(struct.unpack(f"<{nlab}f", take(4 * nlab)))
    W = [list(struct.unpack(f"<{dim}f", take(4 * dim))) for _ in range(nlab)]
    if o != len(b):
        sys.exit(f"{path}: {len(b) - o} trailing bytes")
    return W, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", metavar="model.bin",
                    help="continue training from these weights instead of from zero")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("extra", nargs="*", metavar="dataset.jsonl",
                    help="extra datasets to train on, e.g. a workspace's "
                         "risk-feedback.jsonl — the corrections real approvals collected")
    args = ap.parse_args()

    check_vectors()
    here = pathlib.Path(__file__).parent
    rows = [json.loads(l) for l in (here / "risk_dataset.jsonl").read_text().splitlines() if l.strip()]
    base = len(rows)
    for extra in args.extra:
        add = [json.loads(l) for l in pathlib.Path(extra).read_text().splitlines() if l.strip()]
        # Corrections from real use are the point of collecting them, so they are
        # never held out: there are few, they are in-domain, and measuring on
        # them would measure the wrong thing.
        for r in add:
            r["split"] = "train"
        rows += add
        print(f"+ {len(add)} examples from {extra}")
    print(f"dataset: {len(rows)} examples ({base} synthetic), {DIM} features, {len(LABELS)} labels")

    data = []
    for r in rows:
        x = featurize(call_text(r["tool"], r.get("action", ""), r.get("params", {})))
        y = [1.0 if l in r["labels"] else 0.0 for l in LABELS]
        data.append((x, y))

    # The split comes from the dataset, which holds out PATHS rather than rows:
    # a fifth of every path class never appears in training under any verb.
    # Splitting rows at random would leave "~/.ssh/id_rsa" in both halves and
    # measure memorisation; holding out the path asks whether an unseen secret is
    # recognized as one, which is the only question worth answering.
    train = [d for d, r in zip(data, rows) if r.get("split") != "holdout"]
    hold = [d for d, r in zip(data, rows) if r.get("split") == "holdout"]
    print(f"split: {len(train)} train, {len(hold)} held out (by path, not by row)")

    data = train
    # Positive-class weighting, per label. The dataset is imbalanced on purpose —
    # most real tool calls are ordinary, and a training set that pretended
    # otherwise would teach the model the wrong prior. But an unweighted fit on
    # it drifts toward "everything is safe": adding thirty ordinary build
    # directories once dropped credential_access recall from 0.91 to 0.63 without
    # touching a single credential example. Weighting each label's positives by
    # how rare they are keeps the data honest and the gradient balanced.
    pos_w = []
    for li in range(len(LABELS)):
        pos = sum(1 for _, y in data if y[li] >= 0.5) or 1
        neg = len(data) - pos
        pos_w.append(min(20.0, max(1.0, neg / pos)))
    print("positive-class weights: " + "  ".join(f"{l}={w:.1f}" for l, w in zip(LABELS, pos_w)))

    if args.start:
        W, b = read_model(pathlib.Path(args.start))
        print(f"continuing from {args.start} (fine-tuning, not retraining)")
    else:
        W = [[0.0] * DIM for _ in LABELS]
        b = [0.0] * len(LABELS)
    rng = random.Random(SEED)
    order = list(range(len(data)))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        lr = LR * (1.0 - epoch / args.epochs)          # linear decay
        for i in order:
            x, y = data[i]
            for li in range(len(LABELS)):
                row = W[li]
                z = b[li] + sum(row[j] * v for j, v in x.items())
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g = (p - y[li]) * lr
                if y[li] >= 0.5:
                    g *= pos_w[li]
                if g == 0.0:
                    continue
                b[li] -= g
                for j, v in x.items():
                    row[j] -= g * v + L2 * row[j]

    # The headline number, which is what the log actually shows and a gate would
    # actually use: "any risky label at or over 0.5". Per-label precision does
    # not answer "how often does this cry wolf on ordinary work" — a call can be
    # a false positive for one label and correct overall.
    RISKY = [i for i, l in enumerate(LABELS) if l != "safe" and l not in INFORMATIONAL]
    holdrows_ = [r for r in rows if r.get("split") == "holdout"]
    alarm_on_safe, safe_total, missed_risky, risky_total = [], 0, [], 0
    for (x, y), r in zip(hold, holdrows_):
        top = 0.0
        for li in RISKY:
            z = b[li] + sum(W[li][j] * v for j, v in x.items())
            top = max(top, 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
        truly_risky = any(y[li] >= 0.5 for li in RISKY)
        text = call_text(r["tool"], r.get("action", ""), r.get("params", {}))
        if truly_risky:
            risky_total += 1
            if top < 0.5:
                missed_risky.append((top, text))
        else:
            safe_total += 1
            if top >= 0.5:
                alarm_on_safe.append((top, text))
    print(f"\nheadline score on the holdout (max risky label, threshold 0.50):")
    print(f"  false alarms   {len(alarm_on_safe):3}/{safe_total:<4} ordinary calls  ({100*len(alarm_on_safe)/max(1,safe_total):.1f}%)")
    print(f"  missed         {len(missed_risky):3}/{risky_total:<4} risky calls     ({100*len(missed_risky)/max(1,risky_total):.1f}%)")
    for title, items in (("every false alarm", alarm_on_safe), ("worst misses", sorted(missed_risky, reverse=True)[:8])):
        if not items:
            continue
        print(f"  {title}:")
        for sc, text in sorted(items, reverse=True)[:14]:
            print(f"    {sc:.2f}  {text[:76]}")

    report("held out", hold, W, b)
    # Name the misses. A precision number says how often it cries wolf; the
    # actual wolves are what tells you whether the dataset or the model is wrong.
    holdrows = [r for r in rows if r.get("split") == "holdout"]
    print("\nworst holdout mistakes (up to 4 per label):")
    for li, lab in enumerate(LABELS):
        wrong = []
        for (x, y), r in zip(hold, holdrows):
            z = b[li] + sum(W[li][j] * v for j, v in x.items())
            pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            if (pr >= 0.5) != (y[li] >= 0.5):
                kind = "false alarm" if pr >= 0.5 else "missed"
                wrong.append((abs(pr - 0.5), kind, pr, call_text(r["tool"], r.get("action", ""), r.get("params", {}))))
        if not wrong:
            continue
        wrong.sort(reverse=True)
        print(f"  {lab}:")
        for _, kind, pr, text in wrong[:4]:
            print(f"    {kind:11} {pr:.2f}  {text[:78]}")
    report("training data (memorised — for contrast only)", data, W, b)

    out = here.parent / "internal" / "risk" / "model.bin"
    with out.open("wb") as f:
        f.write(b"IPSRISK\x01")
        f.write(struct.pack("<H", 1))                       # format version
        f.write(struct.pack("<II", DIM, SEED))
        f.write(struct.pack("<BBBB", WORD_MIN, WORD_MAX, CHAR_MIN, CHAR_MAX))
        f.write(struct.pack("<B", 1 if LOWERCASE else 0))
        f.write(struct.pack("<H", len(LABELS)))
        for l in LABELS:
            e = l.encode("utf-8")
            f.write(struct.pack("<H", len(e))); f.write(e)
            f.write(struct.pack("<B", 1 if l in INFORMATIONAL else 0))
        for v in b:
            f.write(struct.pack("<f", v))
        for row in W:
            f.write(struct.pack(f"<{DIM}f", *row))
    print(f"\nwrote {out} — {out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
