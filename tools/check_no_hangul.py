#!/usr/bin/env python
"""Fail if Hangul shows up in anything a CCpy user sees at runtime.

Why this exists
---------------
CCpy runs on cms2 / node99, and much of its output is produced inside SLURM
batch jobs whose stdout is captured with a C/POSIX locale. A Korean string in
a print() there either mojibakes the log or kills the job with
UnicodeEncodeError. On top of that, several CCpy tables are laid out with
str.ljust()/rjust() and f-string field widths, which count characters, not
display columns -- Hangul is double-width in a terminal, so any Korean inside a
padded field breaks the alignment of the whole table.

So: user-visible text is English (ASCII). Korean is still fine in comments,
where it never reaches a terminal.

What is checked
---------------
Hangul is reported when it appears in

  * a string handed to print(), sys.stdout.write(), input(), raw_input(),
    write_log(), file_writer(), or a logging call
  * an exception message (raise ValueError("..."), SystemExit("..."), ...)
  * argparse help= / usage= / description= / epilog= text
  * a module, class or function docstring

What is NOT checked
-------------------
  * ``#`` comments. Long Korean design-rationale blocks are deliberately kept.
  * string literals compared against user input
    (``if ans in ("y", "yes", "예", "네")``) -- those are accepted *input*
    aliases, not output, and dropping them would change behaviour.

Escape hatch
------------
Put ``# ko-ok`` at the end of a line to allow Hangul on it, e.g. when a message
is genuinely meant for a Korean-only audience.

Usage
-----
    python tools/check_no_hangul.py                 # whole repo, exit 1 on any finding
    python tools/check_no_hangul.py CCpy/VASP       # only that path
    python tools/check_no_hangul.py --comments      # also list Korean comments (never fails)
    python tools/check_no_hangul.py --ascii         # also fail on any non-ASCII output text

``--ascii`` widens the same check to every non-ASCII character, not just Hangul.
The encoding argument above is not specific to Korean: a degree sign, an angstrom
symbol or a "~=" glyph in a print() breaks a C-locale job just as reliably. It is
opt-in because a few plot labels legitimately use them (matplotlib writes to a
file, not to the terminal).

Suggested pre-commit hook (.git/hooks/pre-commit):

    #!/bin/sh
    python tools/check_no_hangul.py || {
        echo "Hangul found in user-visible text. Fix it, or add '# ko-ok'." >&2
        exit 1
    }
"""

import ast
import io
import os
import re
import sys
import tokenize

HANGUL = re.compile(r"[가-힣㄰-㆏]")
NON_ASCII = re.compile(r"[^\x00-\x7f]")

# Output sinks: a Hangul string reaching any of these is a finding.
OUTPUT_CALLS = {
    "print", "input", "raw_input", "write", "writelines",
    "write_log", "file_writer", "linux_command",
    "debug", "info", "warning", "warn", "error", "critical", "exception",
}

# argparse / help-text keywords.
HELP_KWARGS = {"help", "usage", "description", "epilog", "metavar", "prog"}

# Out of scope on purpose. These predate the English-output rule and are not
# maintained: CCpyATKAnal_hoijung.py is a 2020 personal script, and the two
# tools/ scripts are developer-only harnesses that never run on a queue.
DEFAULT_EXCLUDE = {
    "CCpy/ATK/CCpyATKAnal_hoijung.py",
    "tools/test_config_home.py",
    "tools/pymatgen_signature_sweep.py",
    "tools/check_no_hangul.py",
}


def call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def allowed_lines(src):
    """Line numbers carrying a '# ko-ok' escape hatch."""
    ok = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and "ko-ok" in tok.string:
                ok.add(tok.start[0])
    except (tokenize.TokenError, IndentationError):
        pass
    return ok


def comment_hits(src):
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and HANGUL.search(tok.string):
                out.append((tok.start[0], tok.string.strip()))
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def check_source(rel, src, pattern=HANGUL):
    """Return a list of (lineno, kind, text) findings."""
    findings = []
    skip = allowed_lines(src)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [(getattr(exc, "lineno", 0) or 0, "syntax-error", str(exc))]

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def classify(node):
        """Walk up from a string constant and decide what it is."""
        cur = node
        for _ in range(8):
            parent = parents.get(id(cur))
            if parent is None:
                return None
            if isinstance(parent, ast.Compare):
                return None            # accepted input alias -> allowed
            if isinstance(parent, ast.Raise):
                return "exception"
            if isinstance(parent, ast.keyword) and parent.arg in HELP_KWARGS:
                return "help-text"
            if isinstance(parent, ast.Call):
                name = call_name(parent)
                if name in OUTPUT_CALLS:
                    return "output"
                if name and (name.endswith("Error") or name in ("SystemExit", "Exception")):
                    return "exception"
                return None
            cur = parent
        return None

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if not pattern.search(node.value):
            continue
        if node.lineno in skip:
            continue
        kind = "docstring" if id(node) in docstrings else classify(node)
        if kind:
            text = node.value.strip().splitlines()[0] if node.value.strip() else node.value
            findings.append((node.lineno, kind, text[:100]))
    return findings


def iter_py(paths, root):
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def main(argv):
    show_comments = "--comments" in argv
    ascii_mode = "--ascii" in argv
    pattern = NON_ASCII if ascii_mode else HANGUL
    label = "non-ASCII" if ascii_mode else "Hangul"
    args = [a for a in argv if not a.startswith("--")]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args or [os.path.join(root, "CCpy"), os.path.join(root, "tools")]

    total = 0
    for path in iter_py(paths, root):
        rel = os.path.relpath(path, root)
        if rel in DEFAULT_EXCLUDE:
            continue
        if os.path.islink(path):
            continue               # CCpy/bin/*.py are symlinks to the modules
        try:
            src = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if not pattern.search(src):
            continue
        for lineno, kind, text in check_source(rel, src, pattern):
            print("%s:%d: [%s] %s" % (rel, lineno, kind, text))
            total += 1
        if show_comments:
            for lineno, text in comment_hits(src):
                print("%s:%d: (comment, allowed) %s" % (rel, lineno, text[:100]))

    if total:
        print("\n%d %s string(s) in user-visible text. "
              "Translate them, or mark the line '# ko-ok'." % (total, label))
        return 1
    print("OK: no %s in user-visible text." % label)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
