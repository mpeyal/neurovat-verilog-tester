"""Re-embed studio/ui_src.html into the bundled studio/index.html.

studio/index.html is a self-contained bundle: fonts + the app JS are base64
blobs, and the whole UI page is stored as ONE JSON-encoded string inside a
<script type="__bundler/template"> tag (the browser JSON.parses it at load and
swaps it in). That string is the only editable copy of the UI.

Workflow:
    1. edit studio/ui_src.html  (readable, real HTML/JS)
    2. python studio/rebuild_ui.py
    3. reload the web GUI

This replaces the template line in index.html with the current ui_src.html,
re-encoding it exactly the way the bundler does: a JSON string literal with
every "/" escaped as \\u002F so an inner "</script>" can't close the tag.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")
SRC = os.path.join(HERE, "ui_src.html")


def find_template_line(lines):
    """Index of the line holding the bundler template (the giant JSON string)."""
    for i, l in enumerate(lines):
        if len(l) > 100000 and "<!DOCTYPE" in l[:200] and l.lstrip().startswith('"'):
            return i
    raise SystemExit("could not locate the bundler template line in index.html")


def encode(inner):
    """JSON-encode like the bundler: escape every '/' as \\u002F (script-safe)."""
    s = json.dumps(inner, ensure_ascii=False)
    return s.replace("/", "\\u002F")


def main():
    inner = open(SRC, encoding="utf-8").read()
    raw = open(INDEX, encoding="utf-8").read()
    lines = raw.split("\n")
    idx = find_template_line(lines)

    # sanity: the new string must round-trip back to exactly what we read
    encoded = encode(inner)
    assert json.loads(encoded) == inner, "re-encode did not round-trip"

    lines[idx] = encoded
    out = "\n".join(lines)
    with open(INDEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print("rebuilt %s  (template line %d, %d chars UI)" % (INDEX, idx + 1, len(inner)))


if __name__ == "__main__":
    main()
