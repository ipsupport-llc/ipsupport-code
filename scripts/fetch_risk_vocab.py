#!/usr/bin/env python3
"""Fetch the risk dataset's VOCABULARY from upstream and vendor it.

    python3 scripts/fetch_risk_vocab.py        # refresh scripts/risk_vocab.json

This is the only script here that touches the network, and it is run by hand,
rarely. Its output is committed. Nothing else in the pipeline — generating the
dataset, training, building the agent — needs a connection, so `git clone && make
build` stays offline and two people on one commit get the same binary.

WHY IT EXISTS. The dataset used to be crossed from about two hundred file and
directory names that I made up: plausible, but invented. A classifier built on
an invented vocabulary is wrong in exactly the places the invention was thin,
and no amount of feedback fixes that quickly. These are the names the ecosystem
actually uses.

  github/gitignore (CC0-1.0)
      163 templates of what people ignore, which is a direct list of build
      output and generated junk — the "deleting this is routine" half of the
      dataset. Fetched as one tarball rather than 163 requests, which also
      keeps it off the GitHub API's unauthenticated rate limit.

  gitleaks rules (MIT)
      NOT a source of filenames: of 222 rules only 5 carry a path pattern, the
      rest match content. What they do carry is the vocabulary of secret TYPES
      — token (121 rules), api (80), key (53), secret (28) — and roughly a
      hundred vendor names. Those morphemes are exactly what a character-n-gram
      model generalizes on, and they are why an unseen "stripe_token.json" can
      be recognized at all.

The crossing of this vocabulary into examples stays ours (gen_risk_dataset.py).
What changes is that the words in it are no longer guesses.
"""
import io, json, pathlib, re, sys, tarfile, urllib.request
from collections import Counter
from datetime import date

GITIGNORE_TARBALL = "https://codeload.github.com/github/gitignore/tar.gz/refs/heads/main"
GITLEAKS_RULES = "https://raw.githubusercontent.com/gitleaks/gitleaks/master/config/gitleaks.toml"
TIMEOUT = 60


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ipsupport-code-risk-vocab"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


# A gitignore entry worth keeping looks like a name, not a rule: no globs
# spanning directories, no negations, no character classes.
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._\-/]*$")


# "Do not commit this" is not the same as "deleting this is routine", and the
# raw list proves it: Makefile appears in five templates, README.txt in three,
# app/config/parameters.yml in two. Frequency cannot separate them — Makefile is
# in MORE templates than __pycache__ — so the difference has to be stated.
#
# This veto is deliberately subtractive. The vocabulary stays upstream's; all
# that happens here is that entries which are demonstrably a build's INPUT, the
# repository's own documentation, or somewhere configuration lives are removed
# from the "safe to delete" half. What was vetoed is written into the JSON, so
# the judgement is reviewable in a diff rather than buried in this file.
VETO_EXACT = {
    "Makefile", "Mkfile.old", "CMakeLists.txt.user", "WEB-INF", "xmlrpc.php",
    "terraform.rc", "Thumbs.db",
}
VETO_SUBSTR = (
    "readme", "license", "changelog", "contributing", "authors", "copying",
    ".lock", "lock.json", "config", "settings", "secret", "credential",
    "password", "token", ".key", ".pem", ".env", "etc/",
)


# Generated in one ecosystem, hand-written in another, and the name cannot tell
# you which: Doxygen generates "docs", Hugo generates "public", and plenty of
# projects keep source in "lib" and "modules" — this repository's own website
# lives in docs/. The two failure modes are not symmetric. Leaving a real build
# directory out of the safe list costs one score that reads a little high;
# putting an ambiguous one IN teaches the model that "rm -rf docs" is routine.
# So when in doubt, out.
VETO_AMBIGUOUS = {
    "doc", "docs", "lib", "modules", "public", "files", "local", "var", "cache",
    "index.php", "install.php", "robots.txt", "sitemap.xml", "tags", "classes",
    "deps", "parts", "pkg", "bin", "out",
}


def vetoed(name: str) -> str:
    """Why this name is not (reliably) build output, or "" if it is."""
    if name in VETO_EXACT:
        return "a build input or repository metadata, not its output"
    if name in VETO_AMBIGUOUS:
        return "generated in some ecosystems and hand-written in others — ambiguous by name alone"
    low = name.lower()
    for bad in VETO_SUBSTR:
        if bad in low:
            return f"contains {bad!r}: documentation, a lockfile, or somewhere configuration lives"
    return ""


def build_output_names(tar_bytes: bytes):
    """Every ignore pattern, counted across templates. A name that appears in
    many ecosystems is genuinely common; one that appears once is somebody's
    local quirk, and the count is what tells them apart."""
    seen = Counter()
    templates = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".gitignore"):
                continue
            templates += 1
            body = tf.extractfile(m).read().decode("utf-8", "replace")
            for line in body.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "!")):
                    continue
                line = line.lstrip("/").rstrip("/")
                if not line or "*" in line or "?" in line or "[" in line:
                    continue
                if not NAME_RE.match(line) or len(line) > 40:
                    continue
                seen[line] += 1
    return seen, templates


# Rule ids are kebab-case: "aws-access-token", "stripe-access-token". The first
# word is almost always the vendor; the rest is what kind of secret it is.
SECRET_WORDS = {"token", "key", "secret", "api", "access", "credential", "credentials",
                "password", "passwd", "auth", "private", "cert", "certificate", "id"}


def secret_vocabulary(toml_text: str):
    ids = re.findall(r'^\s*id\s*=\s*"([^"]+)"', toml_text, re.M)
    vendors, kinds = Counter(), Counter()
    for rule_id in ids:
        parts = rule_id.split("-")
        if parts and parts[0] not in SECRET_WORDS and len(parts[0]) > 2:
            vendors[parts[0]] += 1
        for w in parts:
            if w in SECRET_WORDS:
                kinds[w] += 1
    return ids, vendors, kinds


def main():
    out = pathlib.Path(__file__).with_name("risk_vocab.json")
    print(f"fetching {GITIGNORE_TARBALL}")
    counts, templates = build_output_names(get(GITIGNORE_TARBALL))
    # Seen in two or more templates, then vetoed. The first filter drops one-off
    # local quirks; the second drops what is not build output at all.
    common = sorted(n for n, c in counts.items() if c >= 2)
    build, rejected = [], {}
    for n in common:
        if why := vetoed(n):
            rejected[n] = why
        else:
            build.append(n)
    print(f"  {templates} templates -> {len(counts)} distinct patterns -> "
          f"{len(common)} seen in 2+ -> {len(build)} kept, {len(rejected)} vetoed")

    print(f"fetching {GITLEAKS_RULES}")
    ids, vendors, kinds = secret_vocabulary(get(GITLEAKS_RULES).decode("utf-8", "replace"))
    vendor_list = sorted(vendors)
    kind_list = sorted(kinds)
    print(f"  {len(ids)} rules -> {len(vendor_list)} vendors, {len(kind_list)} secret-type words")

    doc = {
        "_": "Vendored vocabulary for scripts/gen_risk_dataset.py. Regenerate with "
             "scripts/fetch_risk_vocab.py; everything downstream is offline and deterministic.",
        "fetched": date.today().isoformat(),
        "sources": [
            {"url": GITIGNORE_TARBALL, "license": "CC0-1.0",
             "used_for": "build-output names", "templates": templates,
             "filter": "ignore patterns appearing in 2 or more templates, plain names only"},
            {"url": GITLEAKS_RULES, "license": "MIT",
             "used_for": "the vocabulary of secret types and vendors (not filenames — "
                         "only 5 of the rules match on a path at all)",
             "rules": len(ids)},
        ],
        "build_output": build,
        "build_output_vetoed": rejected,
        "secret_vendors": vendor_list,
        "secret_kinds": kind_list,
    }
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(f"\nwrote {out} — {out.stat().st_size/1024:.0f} KB")
    print(f"  build_output    {len(build)}")
    print(f"  secret_vendors  {len(vendor_list)}")
    print(f"  secret_kinds    {len(kind_list)}")
    print("\nnext: python3 scripts/gen_risk_dataset.py && python3 scripts/train_risk.py")


if __name__ == "__main__":
    sys.exit(main())
