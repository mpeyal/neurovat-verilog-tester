"""Verilog-A file discovery and lightweight parsing.

Finds *.va files in the workspace (including the standard verilog/ folder),
extracts module name and `parameter`
declarations, and maps each file to one of the Python behavioral models by
filename/module heuristics.
"""

import os
import re
from dataclasses import dataclass, field

_RE_MODULE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)
_RE_PARAM = re.compile(
    r"parameter\s+(?:real|integer)?\s*([A-Za-z_]\w*)\s*=\s*([^;]+);")

# (model_key, keywords), checked against filename + module.  The MOST SPECIFIC
# (longest) matching keyword wins; equal lengths fall back to list order.
# Dynamic twins insert their va_keywords at the front at load time
# (registry._register_twin), so without the length rule the winner would depend
# on twin load order.  Generic device words ("memristor", "diffu"[sive]) are
# deliberately NOT ECFET keywords: an unrelated memristor .va must stay
# unmapped (the GUI then offers to build a twin) instead of silently
# simulating the ECFET v1 port.  The v1 source (basic_v1_Diffu_Drft_verilog.va)
# still maps through "v1"/"basic"/"drft".
MODEL_HINTS = [
    ("fefet", ("fefet", "ferro", "hzo")),
    ("v3", ("v3", "paper")),
    ("v2", ("v2", "ecram", "practical")),
    ("v1", ("v1", "basic", "drft", "ecfet")),
]


def match_hint(hint_text, hints=None):
    """Model key of the longest keyword found in hint_text ("" if none);
    ties go to the earlier entry in `hints`."""
    best_key, best_len = "", 0
    for key, words in (MODEL_HINTS if hints is None else hints):
        for w in words:
            if w and len(w) > best_len and w in hint_text:
                best_key, best_len = key, len(w)
    return best_key


@dataclass
class VaFile:
    path: str
    module: str = "?"
    params: dict = field(default_factory=dict)   # name -> float (parsable only)
    raw_params: dict = field(default_factory=dict)  # name -> source text
    model_key: str = ""                          # mapped python model ("" = none)
    mtime: float = 0.0
    error: str = ""
    role: str = "device"  # network primitives are source assets, not missing device twins

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
    if va.module == "ideal_signed_synapse":
        va.role = "network_primitive"
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
    va.model_key = match_hint(hint_text)
    # Guard versioned names: if the name says "vN" but we matched a DIFFERENT
    # version family (e.g. "ecfet_v4" falling through to the generic "ecfet"->v1
    # keyword), trust the name's version - remap to vN if that's a real twin,
    # else leave it UNMAPPED so the GUI prompts to build a twin (instead of
    # silently simulating the wrong model).
    known_versions = {k for k, _ in MODEL_HINTS if re.fullmatch(r"v\d+", k)}
    ver = re.search(r"(?<![a-z0-9])v(\d+)", hint_text)
    if ver and re.fullmatch(r"v\d+", va.model_key or ""):
        vkey = "v" + ver.group(1)
        if va.model_key != vkey:
            va.model_key = vkey if vkey in known_versions else ""
    return va


def scan(workdir, max_depth=2):
    """Discover verilog/ models and legacy root/plugin sources.

    Keep scanning older workspace layouts; hidden and results directories are
    excluded. The standard verilog/ folder fits within the default depth limit.
    """
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
