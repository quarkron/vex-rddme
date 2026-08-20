#!/usr/bin/env python3
"""Check operator-facing text against Simplified Technical English rules.

Standard library only. No model, no dataset at runtime, no network.

## What this checks, and what it deliberately does not

Simplified Technical English (ASD-STE100) is a controlled language for *procedural*
writing: short sentences, one instruction each, active voice, a restricted vocabulary.
That is exactly the register of a guard message, which a reader meets at the moment
something failed and needs to know what to do.

It is *not* the register of an explanatory docstring. This package's docstrings say why
-- "the log(1-xi3) term is what makes this BMCSL rather than the Rosenfeld-1989 SPT
form" -- and the sentence rules forbid the subordinate clauses that carry that meaning.
So the default targets are guard messages and the README. Docstrings are out of scope
by design, not by oversight.

## The substitution list is physics-safe

The reference STE dataset for aerospace maintenance yields 280 word substitutions.
Most cannot be used here, because they collide with this codebase:

    mask -> apply        would break feasible_mask
    function -> operate  would break "the free-energy function"
    fit -> install       would break "fit the curve"
    raise -> increase    would break "raise a GuardViolation"
    press -> push        would break "pressure"
    set -> ...           a Python builtin and a maths term

The list below is therefore hand-curated: only substitutions whose source word has no
technical meaning in this domain. Every exclusion above is deliberate.

Usage:
    python tools/ste_lint.py                 # check guard messages + README
    python tools/ste_lint.py --strict        # exit 1 on any violation
    python tools/ste_lint.py --max-words 25  # relax the sentence-length rule
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

# --------------------------------------------------------------------------- rules

MAX_WORDS_DEFAULT = 20

# Wordy phrase -> plain replacement. Multi-word entries are checked first so that
# "in order to" is caught before "order" could be considered.
SUBSTITUTIONS = {
    # wordy connectives and prepositions
    "in order to": "to",
    "so as to": "to",
    "due to the fact that": "because",
    "in the event that": "if",
    "prior to": "before",
    "subsequent to": "after",
    "in the case of": "for",
    "with regard to": "about",
    "in terms of": "for",
    "a number of": "some",
    "the majority of": "most",
    "at this point in time": "now",
    "is able to": "can",
    "are able to": "can",
    "has the ability to": "can",
    # long verbs with plain equivalents
    "utilise": "use",
    "utilize": "use",
    "commence": "start",
    "terminate": "stop",
    "endeavour": "try",
    "endeavor": "try",
    "ascertain": "find",
    "facilitate": "help",
    "necessitate": "need",
    "initiate": "start",
    "attempt": "try",
    "assist": "help",
    "purchase": "buy",
    "demonstrate": "show",
    # long modifiers
    "approximately": "about",
    "additional": "more",
    "sufficient": "enough",
    "numerous": "many",
    "subsequently": "then",
    "consequently": "so",
    "nevertheless": "but",
    "furthermore": "also",
    "however": "but",
    "absolutely": "fully",
    "entirely": "fully",
    "ancillary": "auxiliary",
}

# Words whose STE substitution is REJECTED for this codebase, with the reason. Kept
# here so the exclusion is documented rather than merely absent, and asserted by the
# test suite.
REJECTED_SUBSTITUTIONS = {
    "mask": "identifier: feasible_mask, can_leave mask",
    "function": "maths: the free-energy function",
    "fit": "maths: fit the curve",
    "raise": "Python: raise a GuardViolation",
    "press": "physics: pressure",
    "set": "Python builtin and maths term",
    "may": "correct for permission; 'can' changes the meaning",
    "certain": "'certain voxels' is not 'sure'",
    "operate": "no operators in this domain",
}

BE_FORMS = r"(?:is|are|was|were|be|been|being)"
PASSIVE = re.compile(rf"\b{BE_FORMS}\s+(?:\w+\s+)?(\w+(?:ed|en))\b", re.I)

# Words that end in -ed or -en but are not past participles. Without these the
# regex reports "is then" as a passive construction, because "then" ends in "en".
NOT_PARTICIPLES = {
    "then", "when", "even", "often", "open", "golden", "sudden", "seldom",
    "red", "bed", "need", "speed", "indeed", "seed", "exceed", "proceed",
}

# Past participles that are adjectives here, not passive constructions.
PASSIVE_OK = {
    "excluded", "based", "excited", "limited", "related", "given", "fixed",
    "seeded", "packed", "weighted", "reversed", "shifted", "coupled", "needed",
    "expected", "reduced", "increased", "used", "unchanged", "defined",
    "unbounded", "advised", "detailed", "sampled", "correlated", "installed",
    "representable", "committed", "written", "chosen", "taken", "driven",
}

IMPERATIVE_VERBS = {
    "reduce", "raise", "lower", "increase", "decrease", "use", "set", "run",
    "check", "examine", "add", "remove", "enlarge", "pick", "choose", "do",
    "call", "pass", "write", "read", "make", "keep", "put", "give", "split",
    "apply", "install", "start", "stop", "try", "see", "note", "avoid",
}

MULTI_ACTION = re.compile(r",\s+(?:or|and)\s+\w+|\s+(?:or|and)\s+(?:reduce|raise|lower|increase|decrease|use|set|enlarge|check|examine|remove|add)\b", re.I)


def sentences(text: str):
    """Split into sentences, treating bullet lines as sentences in their own right."""
    out = []
    for block in text.split("\n"):
        b = block.strip()
        if not b:
            continue
        b = re.sub(r"^[-*•]\s*", "", b)
        for s in re.split(r"(?<=[.!?:])\s+", b):
            s = s.strip()
            if s:
                out.append(s)
    return out


def words(s: str):
    # A {placeholder} from an f-string counts as one word, not zero.
    s = re.sub(r"\{[^{}]*\}", "X", s)
    return re.findall(r"[^\s]+", s)


def check_text(text: str, where: str, max_words: int):
    """Return a list of (where, rule, detail) violations."""
    v = []
    for s in sentences(text):
        w = words(s)
        if len(w) > max_words:
            v.append((where, "sentence-length", f"{len(w)} words (limit {max_words}): {s[:80]}"))

        low = s.lower()
        for phrase, plain in SUBSTITUTIONS.items():
            if re.search(rf"\b{re.escape(phrase)}\b", low):
                v.append((where, "word-choice", f"'{phrase}' -> '{plain}': {s[:70]}"))

        for m in PASSIVE.finditer(s):
            participle = m.group(1).lower()
            if participle not in PASSIVE_OK and participle not in NOT_PARTICIPLES:
                v.append((where, "passive-voice", f"'{m.group(0)}': {s[:70]}"))

        first = w[0].lower().strip(".,:;") if w else ""
        if first in IMPERATIVE_VERBS and MULTI_ACTION.search(s):
            v.append((where, "multi-action", f"one instruction per sentence: {s[:80]}"))
    return v


# ----------------------------------------------------------------- extraction

def literal_of(node) -> str | None:
    """Best-effort string for a message argument, including f-strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("{X}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = literal_of(node.left), literal_of(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def guard_messages(root: pathlib.Path):
    """Yield (location, message) for every report.fail/warn/info call."""
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("fail", "warn", "info") or len(node.args) < 2:
                continue
            msg = literal_of(node.args[1])
            if msg:
                tag = getattr(node.args[0], "value", "?")
                yield f"{path.name}:{node.lineno} [{tag}]", msg


def readme_prose(path: pathlib.Path):
    t = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    keep = [l for l in t.split("\n") if l.strip() and not l.startswith(("#", "|"))]
    return "\n".join(keep)


# ---------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true", help="exit 1 on any violation")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS_DEFAULT)
    ap.add_argument("--targets", default="guards,readme",
                    help="comma-separated: guards, readme")
    args = ap.parse_args(argv)

    repo = pathlib.Path(__file__).resolve().parents[1]
    targets = {t.strip() for t in args.targets.split(",")}
    violations = []
    counted = 0

    if "guards" in targets:
        for where, msg in guard_messages(repo / "src"):
            counted += 1
            violations += check_text(msg, where, args.max_words)

    if "readme" in targets:
        counted += 1
        violations += check_text(readme_prose(repo / "README.md"),
                                 "README.md", args.max_words)

    by_rule = {}
    for _, rule, _ in violations:
        by_rule[rule] = by_rule.get(rule, 0) + 1

    print(f"STE lint: {counted} text blocks checked, {len(violations)} violation(s)")
    if by_rule:
        for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
            print(f"  {rule:16s} {n}")
        print()
        for where, rule, detail in violations:
            print(f"  {where}\n    {rule}: {detail}")

    if violations and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
