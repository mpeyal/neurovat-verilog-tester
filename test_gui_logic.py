"""Quick logic checks for the GUI support modules (no window needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vatester.agent import ClaudeAgent

txt = """Here is your pattern.
```json
{"type": "waveform", "kind": "current", "unit": "pA", "label": "poisson 50Hz",
 "pulses": [[0.01, 0.002, -100], [0.05, 0.002, -100], [0.08, 0.002, -100]]}
```
Load it with the button."""

w = ClaudeAgent.extract_waveform(txt)
assert w is not None, "no waveform extracted"
assert len(w["pulses"]) == 3 and w["unit"] == "pA" and w["kind"] == "current"

# bare-pulses variant without "type"
w2 = ClaudeAgent.extract_waveform('```json\n{"pulses": [[0, 0.01, 1.5]], "unit": "V", "kind": "voltage"}\n```')
assert w2 and w2["kind"] == "voltage" and w2["pulses"][0][2] == 1.5

# invalid blocks are skipped, valid earlier block still found
w3 = ClaudeAgent.extract_waveform(
    '```json\n{"type":"waveform","pulses":[[0,0.01,-5]]}\n```\n```json\n{broken\n```')
assert w3 and len(w3["pulses"]) == 1

print("agent extraction: all checks passed")
print("backend detected:", ClaudeAgent(os.getcwd()).backend_label())
