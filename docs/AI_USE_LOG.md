# AI use log

IEEE (including TCAD) requires AI-generated content (text, figures, code) to be
disclosed in the Acknowledgments, naming the AI system and the sections it
affected. This log feeds that statement; add a row for each AI-assisted piece.
Early rows are reconstructed from `git log` subjects; the author must confirm
the extent of AI use for each.

| Date | AI system | What it did (code / figures / text / analysis) | Files or sections | Human review |
|------|-----------|-----------------------------------------------|-------------------|--------------|
| 2026-11-02 | Claude (Anthropic) via Claude Code | Initial test bench, ECFET models, setup scripts (code, text) | `simulation_core/`, `verilog/`, `vatester/`, setup scripts, README | author review pending |
| 2026-11-09 to 2026-11-10 | Claude (Anthropic) via Claude Code | neurovat core package, power and energy model, Power tab (code) | `src/neurovat/`, `vatester/` power views | author review pending |
| 2026-11-11 | Claude (Anthropic) via Claude Code | Masquelier 2008 benchmark, competitive learning, continuous-stream mode, column-scan encoding, ECFET STDP timing (code, text) | `src/neurovat/`, `vatester/network_stream*.py`, README | author review pending |
| 2026-11-12 | Claude (Anthropic) via Claude Code | Stream encodings (row scan, latency, frozen Poisson), robustness sweep, measured results tables (code, analysis, text) | `src/neurovat/encoding.py`, `vatester/stream_*.py`, `vatester/network_stream_robust.py`, `src/neurovat/README.md` | author review pending |
| 2026-11-13 | Claude (Anthropic) via Claude Code | User tutorial PDF (text, screenshots) | `docs/NeuroVAT_Tutorial_v1.3.1.pdf` | author review pending |
| 2026-11-13 | Claude (Anthropic) via Claude Code | Model catalog and assistant fixes: public model list in agent prompt, model menu placement, assistant model default (code) | `vatester/agent.py`, `vatester/` GUI | author review pending |
| 2026-11-13 | Claude (Anthropic) via Claude Code | Documentation consistency: selftest counts, validate_v3 docstring, ecfet_v2.va validation claim, this log (text) | README.md, MODEL_VS_PAPER.md, `tools/validate_v3.py`, `verilog/ecfet_v2.va`, `docs/AI_USE_LOG.md` | author review pending |
| 2026-11-14 | Claude (Anthropic) via Claude Code | VTEAM memristor Verilog-A model (MemTorch 1.1.6 variant) and its closed-form Python twin (code) | `verilog/vteam.va`, `twins/vteam.py` | author review pending |
| 2026-11-14 | Claude (Anthropic) via Claude Code | Engine fix for device synapses in physical time: pulse sub-stepping, per-device pulse scheduling, causal reads (code, tests) | `src/neurovat/devices/`, `src/neurovat/masquelier.py`, `tests/test_static_paths_unchanged.py`, CLAUDE.md | author review pending |
| 2026-11-14 | Claude (Anthropic) via Claude Code | MIT license files and license metadata (text) | `LICENSE`, `CITATION.cff`, `pyproject.toml`, README | author review pending |
