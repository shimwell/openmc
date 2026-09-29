#!/usr/bin/env python3
"""Reject C++ alternative operator tokens in OpenMC's own sources.

``and``, ``or``, ``not`` and friends are standard C++, but MSVC only accepts
them when ``/permissive-`` is passed, which is not the default for a
CMake-driven build. A single ``and`` in a widely included header therefore
breaks the whole Windows build, and because the failure is a parse error the
resulting log is a wall of unrelated cascade errors that says nothing about the
real cause. clang-format does not rewrite these tokens, so nothing else in CI
catches them.

Vendored code under src/external/ is not ours to restyle and is skipped.

Run with no arguments to check the default directories::

    python tools/ci/check_alternative_tokens.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Spelled with concatenation so this file does not match its own check.
TOKENS = [
    "a" "nd", "a" "nd_eq", "bi" "tand", "bi" "tor", "co" "mpl",
    "n" "ot", "n" "ot_eq", "o" "r", "o" "r_eq", "x" "or", "x" "or_eq",
]

TOKEN_RE = re.compile(r"(?<!\w)(" + "|".join(TOKENS) + r")(?!\w)")

DEFAULT_PATHS = ["src", "include"]
EXCLUDED_DIRS = {"external"}
SUFFIXES = {".cpp", ".h", ".hh", ".hpp", ".cc"}


def blank_comments_and_literals(src: str) -> str:
    """Replace comments and string/char literals with spaces.

    Newlines are preserved so reported line numbers still line up with the
    original file.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if c == "/" and nxt == "/":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (src[i] == "*" and src[i + 1 : i + 2] == "/"):
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append("  ")
                i += 2
        elif c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < n and src[i] != quote:
                # Skip escaped characters so \" does not end the literal.
                if src[i] == "\\" and i + 1 < n:
                    out.append(" ")
                    i += 1
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append(" ")
                i += 1
        else:
            out.append(c)
            i += 1

    return "".join(out)


def iter_sources(paths: list[str]):
    for root in paths:
        for path in sorted(Path(root).rglob("*")):
            if path.suffix not in SUFFIXES:
                continue
            if EXCLUDED_DIRS.intersection(path.parts):
                continue
            yield path


def check(paths: list[str]) -> int:
    failures = 0
    for path in iter_sources(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        code = blank_comments_and_literals(text)
        original_lines = text.splitlines()
        for lineno, line in enumerate(code.splitlines(), start=1):
            for match in TOKEN_RE.finditer(line):
                failures += 1
                shown = original_lines[lineno - 1].strip()
                print(
                    f"{path}:{lineno}: alternative token "
                    f"'{match.group(1)}' -> use the symbolic operator\n"
                    f"    {shown}"
                )
    return failures


def main(argv: list[str]) -> int:
    paths = argv[1:] or DEFAULT_PATHS
    failures = check(paths)

    if failures:
        print(
            f"\n{failures} alternative operator token(s) found. MSVC rejects "
            "these without /permissive-, which breaks the Windows build.\n"
            "Replace them with the symbolic form (&& || ! & | ~ ^ etc.)."
        )
        return 1

    print("No C++ alternative operator tokens found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
