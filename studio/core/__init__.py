"""NeuroVAT Studio — backend core.

Pure-Python, dependency-free device physics + tool seams. Everything the HTML
front-end asks for routes through bridge.Bridge into this package. The default
implementation is a real behavioural twin (see twin.py) that works out of the
box; OpenVAF and Cadence Virtuoso are optional upgrades that light up
automatically when their tools are on PATH (see openvaf.py / virtuoso.py).
"""

from . import twin, openvaf, virtuoso  # noqa: F401
