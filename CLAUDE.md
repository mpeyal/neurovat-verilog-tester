# CLAUDE.md — NeuroVAT

Guidance for working in this repo. **Read the Security section before touching
`vatester/agent.py`, `vatester/virtuoso.py`, `vatester/datasets.py`, or the
`twins/` / `patterns/` loaders** — those are the trust boundaries.

## What this is

NeuroVAT is a DearPyGui desktop test bench for neuromorphic Verilog-A device
models (ECFET / ECRAM / FeFET). Python "twins" of the `.va` models let you
iterate without Cadence. It also has an embedded LLM agent (Claude/Codex) that
can read and **edit** the `.va`/twin sources and run shell commands, a
crossbar+LIF **Trainer** studio, and an SSH **skillbridge to a live Cadence
Virtuoso** on a shared university host.

- Entry points: `run_gui.py` (GUI), `run_gui.py --web` (browser UI on
  localhost — NeuroVAT Studio in `studio/`, same `ecfet` engine),
  `run_ecfet.py` (CLI scenarios), `selftest.py` (35 sanity checks — **run this
  after any physics/model change**).
- Core engine: `ecfet/` (models v1/v2/v3 + FeFET, `simulator.py`, `signals.py`).
- GUI: `vatester/app.py` (single large App class; ~8k lines).
- Agent: `vatester/agent.py`. Virtuoso: `vatester/virtuoso.py`.
- Control bridge: `vatester/control_bridge.py` + client `tools/nvat_ctl.py`.
- Tools-menu utilities (GUI-free, testable, shared with the bridge):
  `vatester/tools_ext.py` (twin/.va consistency, OpenVAF compile, LTP/LTD fit).

## Dev workflow

- After editing any `ecfet/model_*.py`, run `python selftest.py` — it must stay
  `35 passed, 0 failed`. v3 checks are tallied separately and are guarded so a
  v3 regression can't mask v2.
- Keep each `.va` and its Python twin in sync — a change to one should be
  mirrored into the other (the models are meant to be equivalent).
- The models are used across runs via `simulate()`, which calls `model.reset()`
  at entry. Preserve that; don't introduce cross-run state leaks.

## Control bridge (drive the live app, skillbridge-style)

`vatester/control_bridge.py` lets an outside driver operate the *running* GUI -
the local analog of the Virtuoso skillbridge, except here the app is the server.
Start it with `python run_gui.py --bridge` (or `NVAT_BRIDGE=1`); a `control/`
file mailbox appears in the workspace. Drive it with the CLI client:

    python tools/nvat_ctl.py ping | state
    python tools/nvat_ctl.py set nt_epochs 25 / get nt_epochs / tab tab_results
    python tools/nvat_ctl.py run            # runs a transient, waits, dumps R/G + snapshot
    python tools/nvat_ctl.py stdp
    python tools/nvat_ctl.py snapshot       # -> results/agent_snapshot.json + neuro_snapshot.json
    python tools/nvat_ctl.py shot results/x.png    # screenshot the live window
    python tools/nvat_ctl.py reload         # hot-reload twins after editing ecfet/
    python tools/nvat_ctl.py nt_build | nt_train | nt_test
    python tools/nvat_ctl.py check_va       # twin <-> .va param consistency
    python tools/nvat_ctl.py openvaf        # compile enabled .va with OpenVAF
    python tools/nvat_ctl.py fit meas.csv   # fit LTP/LTD nonlinearity of a CSV
    python tools/nvat_ctl.py log            # -> results/session_log.txt (readable)

This enables the observe -> edit-code -> reload -> re-run loop against the live
app (the embedded agent's system prompt teaches it this loop).

### Real-time testing & debugging (Claude drives the software)

Claude can now run this software end-to-end for real-time testing: **launch** the
app, **read/write** its data, and **control the tools** (run sims, train, compile
`.va`, fit curves, screenshot), then fix the code and re-verify against the LIVE
app - not just against unit tests. Use this to reproduce a bug, observe it, edit
the twin/`.va`/GUI, `reload`, and re-run until the behaviour is correct.

**Generate and read your OWN log file - the bridge does not surface everything.**
Bridge JSON results and the in-app console (`log` command ->
`results/session_log.txt`) only capture `app.log()` lines. Python **tracebacks,
warnings, native stderr, and startup crashes** go to the process stdout/stderr,
which the bridge never sees. So when you launch the app to test it, redirect its
output to your own log and read that for debugging:

    python run_gui.py > results/app_run.log 2>&1 &   # YOUR process log (errors/warnings)
    # ... drive it via tools/nvat_ctl.py ...
    python tools/nvat_ctl.py log                      # in-app console -> results/session_log.txt

For any failure, **read both** and diagnose from the process log first:
  * `results/app_run.log`     - tracebacks / warnings / stderr (real errors live here)
  * `results/session_log.txt` - the in-app console (what the tools reported)
A bridge return value tells you a command's *outcome*, not *why* the app
misbehaved - always generate and read your own log file for proper analysis.

**Design constraints to preserve:** DearPyGui is single-threaded, so
`ControlBridge.poll()` runs on the render thread (called from `App.run()`) and
every command executes there - no locks, no off-thread DPG access. Long commands
(run/train) don't block a frame: the bridge starts the work, then watches
`sim_running`/`trainer_running` across frames and answers only when it finishes.
The command set is a **fixed whitelist** with no eval/exec of caller strings;
`set`/`get` only touch existing widget tags. If you add a command, keep it on the
render thread and keep it whitelisted.

## Security model — maintain these invariants

This app runs an LLM agent with **file-edit and shell** access and ingests
**untrusted content** (attached PDFs/CSVs/datasheets, downloaded datasets,
`.va` scan text, dataset URLs). The dominant risk is **indirect prompt
injection** → arbitrary code execution. The following defenses are in place —
**do not weaken them without a deliberate, documented decision:**

1. **Untrusted context is fenced, not trusted.** All agent backends wrap
   `context` with `_wrap_ctx()` in `vatester/agent.py` (BEGIN/END UNTRUSTED
   CONTEXT markers) and the system prompt instructs the model to treat that
   region as data, never instructions. If you add a new place that feeds
   scan/file/snapshot text to the model, wrap it with `_wrap_ctx()` too. Never
   concatenate raw external text directly into a system prompt.

2. **`np.load(..., allow_pickle=False)`** in `vatester/datasets.py::load_npz`.
   These datasets are numeric — never set `allow_pickle=True`; a `.npz` can come
   from a user/agent-supplied URL and a pickled object array = RCE on load.

3. **Autonomous mode is deliberately permissive.** `Agent.send(autonomous=True)`
   forces `allow_edits=allow_bash=True` and launches the CLI with
   `--permission-mode bypassPermissions` (Codex:
   `--dangerously-bypass-approvals-and-sandbox`). This is an intentional
   edit→run→verify loop, but it means a successful prompt injection can run
   shell commands unattended. Keep defense #1 strong. If you make autonomous
   mode more capable, consider workspace-confining Edit/Write and requiring
   confirmation for `Bash`.

4. **`twins/` and `patterns/` are executed on load (no sandbox).** The
   `twin_loader` / `pattern_loader` `importlib.exec_module` every `*.py` there,
   and the hot-reload watcher re-runs them on change. Since the agent's Write
   tool can create files there, this is a code-execution path — only load twins
   you trust, and don't extend auto-exec to agent-writable locations without a
   trust check.

5. **Virtuoso skillbridge** (`virtuoso.py`): SSH uses `BatchMode=yes` and does
   **not** disable host-key checking — keep it that way (never add
   `StrictHostKeyChecking=no` / `UserKnownHostsFile=/dev/null`). `write_source`
   overwrites files on a **shared multi-user host** with no undo — treat any
   agent-driven remote write as destructive; prefer diff/confirm/backup.

6. Don't commit secrets. Credentials come from env vars / the Account dialog
   only. The committed host/username/IP in `virtuoso.py` / `connect_test.py` are
   infrastructure config, not secrets, but avoid adding more.

7. **Studio web server** (`studio/server.py`, `run_gui.py --web`) binds
   **127.0.0.1 only** — keep it that way (never `0.0.0.0`; it has no auth and
   its write-back can edit the repo's `.va` files). Cell names from the browser
   are validated in `studio/bridge.py::_check_cell` and
   `studio/core/virtuoso.py::_safe_cell` (plain file stems, no path
   separators) — keep both checks; they block path traversal from the web UI.

## GUI threading discipline (don't break it)

`app.py` uses a strict pattern: long work runs in **daemon worker threads** that
communicate **only** through `self.q`, drained on the render thread in
`_process_queue`. Workers must **always** post a terminal queue message
(`results`/`chat`/`nt_done`/…) even on exception, or the corresponding busy flag
(`sim_running`, `chat_busy`, `trainer_running`) stays stuck and disables the UI
until restart. Render-loop ticks called directly from `run()` (e.g.
`_nt_tick_cell_anim`) must not let an exception escape — guard them.
