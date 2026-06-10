"""Verilog-A file discovery and lightweight parsing.

Finds *.va files in the workspace, extracts module name and `parameter`
declarations, and maps each file to one of the Python behavioral models by
filename/module heuristics.
"""

import os
import re
from dataclasses import dataclass, field

_RE_MODULE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)
_RE_PARAM = re.compile(
    r"parameter\s+(?:real|integer)?\s*([A-Za-z_]\w*)\s*=\s*([^;]+);")

# (model_key, keywords) - first match wins, checked against filename + module
MODEL_HINTS = [
    ("fefet", ("fefet", "ferro", "hzo")),
    ("v2", ("v2", "ecram", "practical")),
    ("v1", ("v1", "basic", "diffu", "drft", "memristor", "ecfet")),
]


@dataclass
class VaFile:
    path: str
    module: str = "?"
    params: dict = field(default_factory=dict)   # name -> float (parsable only)
    raw_params: dict = field(default_factory=dict)  # name -> source text
    model_key: str = ""                          # mapped python model ("" = none)
    mtime: float = 0.0
    error: str = ""

    @property
    def name(self):
        return os.path.basename(self.path)


def parse_va(path):
    va = VaFile(path=path)
    try:
        va.mtime = os.path.getmtime(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        va.error = str(e)
        return va

    # strip comments so commented-out parameters are ignored
    code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)

    m = _RE_MODULE.search(code)
    if m:
        va.module = m.group(1)
    for name, value in _RE_PARAM.findall(code):
        value = value.strip()
        # drop range qualifiers:  0.6 from (0:inf)
        value = re.split(r"\bfrom\b", value)[0].strip()
        va.raw_params[name] = value
        try:
            va.params[name] = float(value)
        except ValueError:
            pass

    hint_text = (va.name + " " + va.module).lower()
    for key, words in MODEL_HINTS:
        if any(w in hint_text for w in words):
            va.model_key = key
            break
    return va


def scan(workdir, max_depth=2):
    """Return [VaFile] for every .va under workdir (skips hidden/results dirs)."""
    found = []
    workdir = os.path.abspath(workdir)
    base_depth = workdir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(workdir):
        if root.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d not in
                   ("results", "__pycache__", "node_modules", "venv", ".git")]
        for fn in sorted(files):
            if fn.lower().endswith(".va"):
                found.append(parse_va(os.path.join(root, fn)))
    return found
