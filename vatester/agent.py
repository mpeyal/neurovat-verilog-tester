"""Agent backend for the GUI chat panel (Claude or OpenAI, user-switchable).

Claude provider (default):
  - `claude` CLI headless (`claude -p --output-format json`): full agentic
    backend with real file tools / shell; conversation via --resume.
  - Anthropic SDK fallback (claude-opus-4-8): chat + patterns only.

OpenAI provider:
  - `codex` CLI headless (`codex exec`) when installed: agentic backend with
    file/shell access (best-effort flag mapping).
  - OpenAI SDK fallback (gpt-5.1): chat + patterns + GUI-action blocks only -
    no file tools, so the autonomous edit loop is unavailable on this path.

The waveform / GUI-action json blocks are plain text conventions, so they
work identically with every backend.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

SYSTEM_PROMPT = """\
You are the embedded engineering agent of "NeuroVAT", a neuromorphic Verilog-A
device tester GUI (ECFET / ECRAM / FeFET synaptic models).

MODELS. The GUI plots Python behavioral twins of the workspace .va files:
  ecfet/model_v1.py    <-> basic_v1_Diffu_Drft_verilog.va   (basic ECFET port)
  ecfet/model_v2.py    <-> ecfet_v2.va                       (practical ECRAM)
  ecfet/model_fefet.py <-> fefet_v1.va                       (FeFET)
The plotted curves come from the PYTHON twins. To change what the GUI shows,
edit the matching twin; to change the Verilog model, edit the .va so the two
stay in sync. Each model exposes step(t, dt, I), .R, .G, reset(), observables().

REAL-TIME RULE - put DEVICE behavior in the TWINS and MEASUREMENT logic in the
analysis layer; both hot-reload LIVE. The GUI reloads ecfet/model_*.py and
vatester/analysis.py on every edit, so changes there appear with no restart:
  * ecfet/model_*.py     - device equations (R/G dynamics, the dR write-then-relax,
                           retention, noise). Fix model behavior here, mirror to .va.
  * vatester/analysis.py - how results are MEASURED: stdp_sweep (the anti-symmetric
                           STDP window) and per_pulse_samples (LTP/LTD sampling). Fix
                           how a curve is computed/sampled here.
The GUI does NOT reload its own shell: editing vatester/app.py (widgets, layout,
threads/queue plumbing) does NOTHING until the app is manually restarted, so DO
NOT put model or measurement logic there. If a measured curve is wrong: a wrong
DEVICE response (LTP/LTD imbalance, retention, dR magnitude) -> fix the twin and
mirror to .va; a wrong MEASUREMENT (sampling time, pairing definition, baseline
subtraction) -> fix vatester/analysis.py. Never patch app.py for either.

RUNNING THINGS IN THE GUI. You CANNOT click buttons. When the user asks you
to run, plot, or switch a view in the app ("run", "plot the STDP", "run STDP",
"show analysis as resistance", "fit the plots"), DO NOT write or run a script
and DO NOT produce your own PNG. Instead reply with one short sentence and
exactly ONE fenced action block - the GUI executes it instantly and shows the
result on its own plots:
```json
{"type": "action", "action": "plot_stdp"}
```
Valid actions: "run" (transient sim), "plot_stdp" (STDP sweep), "plot_polar"
(FeFET polarization P-V loop), "preview" (stimulus preview), "analyze_g" /
"analyze_r" (analysis quantity), "fit" (fit plot axes), "export_csv".

PARAMETER SWEEPS. To overlay multiple curves by sweeping model parameter(s) on
ONE graph, use the "sweep" action - the GUI runs the selected model once per
value (or per combination for several params) and overlays them:
```json
{"type": "action", "action": "sweep", "mode": "run",
 "sweeps": [{"param": "n_states", "values": [400, 650, 900]}]}
```
"mode" is "run" (transient), "stdp" (STDP window), or "polar" (P-V loop). Each
sweep entry is either explicit values ("values":[...]) or a range
({"param":"A_stdp","from":4e-6,"to":1.2e-5,"steps":5,"step_type":"Linear"}).
Add more entries to "sweeps" for a multi-parameter grid (e.g. n_states x nu_p).
"param" must be a real parameter of the SELECTED model (capped at a couple
dozen total curves).  Use this whenever the user asks to "sweep", "vary",
"compare across", or "see the effect of" a parameter.

Be fast on all actions - no file reads, no deliberation; just one sentence +
the action block. Only use tools.../the edit loop for tasks that change model
code.

WAVEFORM PATTERNS. When asked to design a spike pattern / stimulus, reply
briefly and include exactly ONE fenced json block:
```json
{"type": "waveform", "kind": "current", "unit": "pA", "label": "name",
 "pulses": [[t_start_s, width_s, amplitude], ...]}
```
kind "current" (ECFET) or "voltage" (FeFET); unit pA,nA,uA,mA,A,mV,V; times in
seconds; <=2000 pulses. ECFET sign convention is MODEL-DEPENDENT: the
paper-matched v2 (default) POTENTIATES on POSITIVE intercalation current (G up
/ R down); the legacy v1 port is the opposite (positive current raises R).
FeFET: positive gate voltage potentiates. The GUI turns that block into a
one-click "Load pattern" card.

AUTONOMOUS EDIT -> SIMULATE -> VERIFY -> FIX LOOP. When the user asks you to
change device behavior or fix a model (and file edits are enabled), DO THE FULL
LOOP yourself, in this turn, until it works - don't just describe the change:
  1. Read the relevant twin (ecfet/model_*.py) and its .va.
  2. Edit the twin to implement the change. Mirror the same change into the .va
     so the Verilog stays consistent.
  3. Run a fresh simulation by writing a spec file and calling the headless
     runner (this imports your edited twin, so it reflects your changes):
       write results/agent_spec.json:
         {"models":["v2"], "params":{"v2":{...optional overrides...}},
          "pulses":[[t_s,width_s,amp_SI], ...], "t_stop":2.0,
          "analysis":"G", "n_each":20}
       run:  python tools/agent_sim.py results/agent_spec.json results/agent_out.json
       (amplitudes are SI: amps in A, e.g. -100 pA = -100e-12; volts for FeFET)
  4. Read results/agent_out.json and CHECK it against the goal (e.g. monotonic
     LTP, asymmetric STDP, target dG/pulse, bounded R, no NaNs). Use the
     per_pulse curve + mean_delta_LTP/LTD and the R/G min/max/start/end stats.
  5. If it does not meet the goal, revise the twin and repeat from step 3.
     Keep iterating (typically up to ~5-8 rounds) until the result is correct,
     then mirror the final change into the .va and give a short summary of what
     you changed and the measured result.
You may also write and run ad-hoc Python that imports the ecfet package (see
my_test.py / README.md) when the runner is not flexible enough.

ATTACHED FILES. The context may list files the user attached in the GUI (PDF
papers, CSV measurement data, images of plots, datasheets). FIRST read every
attached file with your Read tool - it understands PDF, images and text/CSV.
Typical jobs: extract device physics, parameter values or measured curves from
a paper and apply them to the matching twin + .va (run the full
edit->simulate->verify loop against the extracted targets); or compare a
measured CSV against the simulated curves and report/fit the differences.
Quote the extracted numbers (with units and source page/figure) in your reply
so the user can check them.

OTHER TOOLS (when shell is enabled). You have a real shell. The project root has
Virtuoso connection helpers: virtuoso_connect.bat and connect_test.py (SSH
tunnel + skillbridge to Virtuoso on coen-cassia). Use them via bash only if the
task needs live Virtuoso access. You can run any project script, inspect git,
install nothing destructive, and read results/agent_snapshot.json for the data
currently on the GUI's plots.

Keep chat replies short and concrete: say what you changed, what you ran, and
the measured outcome. Do the work; don't ask for permission you already have.

LANGUAGE & TONE. Match the user's language when they ask (e.g. answer in Bangla
if they write in or ask for Bangla), but stay neutral and professional. Do NOT
open with a religious or culturally loaded greeting - no "নমস্কার"/"Namaskar",
no "আসসালামু আলাইকুম", no "Vanakkam", etc. Skip the greeting entirely, or use a
plain neutral one ("Hi", or "হ্যালো"). Keep technical terms - STDP, ECFET,
ECRAM, FeFET, LTP/LTD, R/G, P-V, .va, file names and parameter names - in
English/Latin even inside a non-English reply (they are proper nouns and the
user reads them that way). When you DO write Bangla (or any non-English),
write correct, natural, standard spelling and grammar - prefer simple common
words you are confident about over rare/ambiguous ones, and re-read the
sentence before sending; a clear simple sentence beats a fancy misspelled one.
CRITICAL: if you are not fully confident of the correct Bangla spelling of a
word, write that word in English instead of guessing - a correct English word
mixed into a Bangla sentence is far better than a misspelled Bangla one.
"""

_WAVEFORM_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _vscode_ext_claude():
    pats = [
        os.path.expanduser(r"~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe"),
        os.path.expanduser(r"~/.vscode-insiders/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe"),
    ]
    hits = []
    for p in pats:
        hits.extend(glob.glob(p))
    return sorted(hits)[-1] if hits else None


def find_claude_cli():
    """Locate the claude CLI: PATH, then known install spots."""
    exe = shutil.which("claude")
    if exe:
        return exe
    for cand in (os.path.expanduser(r"~/.local/bin/claude.exe"),
                 os.path.expanduser(r"~/.claude/local/claude.exe"),
                 _vscode_ext_claude()):
        if cand and os.path.isfile(cand):
            return cand
    return None


def find_codex_cli():
    """Locate OpenAI's codex CLI, if installed."""
    exe = shutil.which("codex")
    if exe:
        return exe
    cand = os.path.expanduser(r"~/.codex/bin/codex.exe")
    return cand if os.path.isfile(cand) else None

# appended to the system prompt on SDK-only paths (no tools available there)
SDK_CHAT_NOTE = """

NOTE - CHAT-ONLY MODE: this backend has NO file tools and NO shell. Skip the
autonomous edit loop entirely; do not claim to have edited or run anything.
You can still produce waveform json blocks, GUI action blocks, and advice.
"""


class ClaudeAgent:
    def __init__(self, workdir):
        self.workdir = workdir
        self.session_id = None
        self.total_cost = 0.0
        self.total_in = 0           # accumulated input/output tokens this session
        self.total_out = 0
        self._history = []          # Anthropic SDK conversation
        self.cli = find_claude_cli()
        # per-app credential override (does NOT touch the global CLI login)
        self.override_kind = None   # "api_key" | "oauth" | None
        self.override_value = None
        self._proc = None           # running CLI process (for Stop)
        self._stop_requested = False
        # provider switch (claude | openai)
        self.provider = "claude"
        self.codex = find_codex_cli()
        self.openai_key = None      # per-app override for OPENAI_API_KEY
        self._openai_history = []
        self._codex_started = False
        self._refresh_sdk_ok()
        self._refresh_openai_ok()

    # ---- provider ----------------------------------------------------

    def set_provider(self, name):
        name = (name or "").strip().lower()
        if name not in ("claude", "openai"):
            return False, f"unknown provider '{name}' (claude | openai)"
        if name != self.provider:
            self.provider = name
            self.reset()
        return True, self.backend_label()

    def _refresh_openai_ok(self):
        self.openai_ok = False
        try:
            import openai  # noqa: F401
        except ImportError:
            return
        self.openai_ok = bool(self.openai_key
                              or os.environ.get("OPENAI_API_KEY"))

    def set_openai_key(self, value):
        self.openai_key = (value or "").strip() or None
        self._refresh_openai_ok()
        if self.provider == "openai":
            self.reset()

    def stop(self):
        """Terminate the running agent process tree (Stop button)."""
        p = self._proc
        if p is None:
            return False
        self._stop_requested = True
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               capture_output=True)
            else:
                p.terminate()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        return True

    @property
    def busy(self):
        return self._proc is not None

    def _refresh_sdk_ok(self):
        self.sdk_ok = False
        if self.cli:
            return
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return
        self.sdk_ok = bool(
            (self.override_kind == "api_key" and self.override_value)
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    # ---- credentials -------------------------------------------------

    def _flags(self, new_console=False):
        if sys.platform != "win32":
            return 0
        return (subprocess.CREATE_NEW_CONSOLE if new_console
                else subprocess.CREATE_NO_WINDOW)

    def _env(self):
        """Environment for chat subprocesses, with the app override applied."""
        env = os.environ.copy()
        if self.override_value:
            if self.override_kind == "api_key":
                env["ANTHROPIC_API_KEY"] = self.override_value
                env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            else:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = self.override_value
                env.pop("ANTHROPIC_API_KEY", None)
        return env

    def set_override(self, kind, value):
        value = (value or "").strip()
        self.override_kind = kind if value else None
        self.override_value = value or None
        self._refresh_sdk_ok()
        self.reset()

    def auth_status(self):
        """Returns (logged_in: bool, human_text: str). Blocking - thread it."""
        if self.override_value:
            label = "API key" if self.override_kind == "api_key" \
                else "OAuth token"
            tail = self.override_value[-4:]
            return True, (f"App override active: {label} (...{tail}).\n"
                          "Your global Claude Code login is untouched.")
        if self.cli:
            try:
                proc = subprocess.run(
                    [self.cli, "auth", "status", "--text"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=30,
                    creationflags=self._flags())
                out = (proc.stdout or proc.stderr or "").strip()
                return proc.returncode == 0, out or "unknown"
            except Exception as e:
                return False, f"status check failed: {e}"
        if self.sdk_ok:
            return True, "Anthropic SDK using ANTHROPIC_API_KEY from the env."
        return False, "No backend. Install Claude Code or set ANTHROPIC_API_KEY."

    def login_interactive(self):
        """Open a console + browser for sign-in (non-blocking)."""
        if not self.cli:
            return False, "Login needs the Claude Code CLI installed."
        try:
            subprocess.Popen([self.cli, "auth", "login"],
                             creationflags=self._flags(new_console=True),
                             cwd=self.workdir)
            return True, ("Opened a sign-in window and browser. Complete it "
                          "there, then click Refresh.")
        except Exception as e:
            return False, f"failed to launch login: {e}"

    def logout(self):
        if not self.cli:
            return False, "Logout needs the Claude Code CLI."
        try:
            proc = subprocess.run(
                [self.cli, "auth", "logout"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
                creationflags=self._flags())
            return (proc.returncode == 0,
                    (proc.stdout or proc.stderr or "logged out").strip())
        except Exception as e:
            return False, f"logout failed: {e}"

    # ------------------------------------------------------------------

    @property
    def backend(self):
        if self.provider == "openai":
            if self.codex:
                return "cli"
            if self.openai_ok:
                return "sdk"
            return "none"
        if self.cli:
            return "cli"
        if self.sdk_ok:
            return "sdk"
        return "none"

    def backend_label(self):
        if self.provider == "openai":
            if self.codex:
                return "OpenAI codex CLI (agentic)"
            if self.openai_ok:
                return "OpenAI SDK (gpt-5.1) - chat only, no file tools"
            return ("OpenAI: no backend - install the codex CLI or set an "
                    "OPENAI_API_KEY (Account dialog)")
        if self.cli:
            return f"claude CLI ({os.path.basename(os.path.dirname(self.cli)) or 'PATH'})"
        if self.sdk_ok:
            return "Anthropic SDK (claude-opus-4-8)"
        return "no backend - install claude CLI or set ANTHROPIC_API_KEY"

    def reset(self):
        self.session_id = None
        self._history = []
        self._openai_history = []
        self._codex_started = False

    # ------------------------------------------------------------------

    def send(self, text, context="", allow_edits=False, allow_bash=False,
             model="default", timeout=600, autonomous=False):
        """Blocking - call from a worker thread.
        Returns {ok, text, error, cost, session_id}."""
        if autonomous:                  # full edit->run->verify->fix loop
            allow_edits = allow_bash = True
            timeout = max(timeout, 1800)
        if self.provider == "openai":
            if self.codex:
                return self._send_codex(text, context, allow_edits,
                                        model, timeout, autonomous)
            if self.openai_ok:
                return self._send_openai_sdk(text, context, model)
            return {"ok": False, "text": "",
                    "error": "No OpenAI backend. Install the codex CLI, or "
                             "`pip install openai` and set an OPENAI_API_KEY "
                             "(env var or the Account dialog). "
                             "/provider claude switches back."}
        if self.cli:
            return self._send_cli(text, context, allow_edits, allow_bash,
                                  model, timeout, autonomous)
        if self.sdk_ok:
            return self._send_sdk(text, context, model)
        return {"ok": False, "text": "",
                "error": "No agent backend. Install Claude Code "
                         "(https://claude.com/claude-code) or "
                         "`pip install anthropic` + set ANTHROPIC_API_KEY."}

    # ------------------------------------------------------------------

    def _send_cli(self, text, context, allow_edits, allow_bash, model, timeout,
                  autonomous=False):
        tools = ["Read", "Grep", "Glob", "TodoWrite"]
        if allow_edits:
            tools += ["Edit", "Write", "MultiEdit"]
        if allow_bash:
            tools.append("Bash")
        args = [self.cli, "-p", "--output-format", "json",
                "--append-system-prompt", SYSTEM_PROMPT + "\n" + context,
                "--allowedTools", ",".join(tools)]
        if autonomous:
            # auto-approve every tool call so the edit/run/verify loop is not
            # interrupted by permission prompts
            args += ["--permission-mode", "bypassPermissions"]
        elif allow_edits:
            args += ["--permission-mode", "acceptEdits"]
        if self.session_id:
            args += ["--resume", self.session_id]
        if model and model != "default":
            args += ["--model", model]

        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._stop_requested = False
        try:
            self._proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", cwd=self.workdir, creationflags=flags,
                env=self._env())
        except OSError as e:
            return {"ok": False, "text": "", "error": f"failed to launch CLI: {e}"}
        try:
            out, err = self._proc.communicate(input=text, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stop()
            self._proc = None
            return {"ok": False, "text": "",
                    "error": f"claude CLI timed out after {timeout}s"}
        rc = self._proc.returncode
        self._proc = None
        if self._stop_requested:
            return {"ok": False, "text": "",
                    "error": "stopped by user (agent run cancelled)"}

        out = (out or "").strip()
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            if rc != 0:
                msg = (err or out or "unknown CLI error").strip()
                return {"ok": False, "text": "", "error": msg[-2000:]}
            return {"ok": True, "text": out or "(empty reply)"}

        self.session_id = data.get("session_id") or self.session_id
        cost = data.get("total_cost_usd") or 0.0
        self.total_cost += cost
        usage = data.get("usage") or {}
        self.total_in += (usage.get("input_tokens") or 0)
        self.total_out += (usage.get("output_tokens") or 0)
        reply = data.get("result") or ""
        if data.get("is_error"):
            return {"ok": False, "text": reply,
                    "error": reply or "CLI reported an error",
                    "cost": cost, "session_id": self.session_id}
        return {"ok": True, "text": reply, "cost": cost,
                "session_id": self.session_id}

    # ------------------------------------------------------------------

    def _send_sdk(self, text, context, model="default"):
        import anthropic
        if self.override_kind == "api_key" and self.override_value:
            client = anthropic.Anthropic(api_key=self.override_value)
        else:
            client = anthropic.Anthropic()
        model_id = model if model and model != "default" else "claude-opus-4-8"
        self._history.append({"role": "user", "content": text})
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT + "\n" + context,
                messages=self._history,
            )
        except Exception as e:                      # surface any API error
            self._history.pop()
            return {"ok": False, "text": "", "error": f"{type(e).__name__}: {e}"}
        reply = "".join(b.text for b in resp.content if b.type == "text")
        self._history.append({"role": "assistant", "content": reply})
        return {"ok": True, "text": reply, "cost": 0.0}

    # ---- OpenAI provider ---------------------------------------------

    def _send_codex(self, text, context, allow_edits, model, timeout,
                    autonomous=False):
        """Headless OpenAI codex CLI (`codex exec`). Best-effort flag
        mapping; session continuity via `exec resume --last`."""
        if self._codex_started:
            args = [self.codex, "exec", "resume", "--last"]
            prompt = context + "\n\nUSER REQUEST:\n" + text
        else:
            args = [self.codex, "exec"]
            prompt = SYSTEM_PROMPT + "\n" + context \
                + "\n\nUSER REQUEST:\n" + text
        if autonomous:
            args.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            args += ["--sandbox",
                     "workspace-write" if allow_edits else "read-only"]
        if model and model != "default":
            args += ["-m", model]
        args.append("-")                      # prompt from stdin

        env = os.environ.copy()
        if self.openai_key:
            env["OPENAI_API_KEY"] = self.openai_key
        self._stop_requested = False
        try:
            self._proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", cwd=self.workdir,
                creationflags=self._flags(), env=env)
        except OSError as e:
            return {"ok": False, "text": "",
                    "error": f"failed to launch codex: {e}"}
        try:
            out, err = self._proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stop()
            self._proc = None
            return {"ok": False, "text": "",
                    "error": f"codex timed out after {timeout}s"}
        rc = self._proc.returncode
        self._proc = None
        if self._stop_requested:
            return {"ok": False, "text": "",
                    "error": "stopped by user (agent run cancelled)"}
        out = (out or "").strip()
        if rc != 0 and not out:
            return {"ok": False, "text": "",
                    "error": (err or "codex CLI error").strip()[-2000:]}
        self._codex_started = True
        return {"ok": True, "text": out or "(empty reply)", "cost": 0.0}

    def _send_openai_sdk(self, text, context, model="default"):
        """OpenAI chat completions - chat / pattern / action replies only."""
        try:
            from openai import OpenAI
        except ImportError:
            return {"ok": False, "text": "",
                    "error": "`pip install openai` to use the OpenAI provider"}
        client = OpenAI(api_key=self.openai_key) if self.openai_key \
            else OpenAI()
        model_id = model if model and model != "default" else "gpt-5.1"
        msgs = ([{"role": "system",
                  "content": SYSTEM_PROMPT + SDK_CHAT_NOTE + "\n" + context}]
                + self._openai_history
                + [{"role": "user", "content": text}])
        try:
            resp = client.chat.completions.create(model=model_id,
                                                  messages=msgs)
        except Exception as e:                  # surface any API error
            return {"ok": False, "text": "",
                    "error": f"{type(e).__name__}: {e}"}
        reply = (resp.choices[0].message.content or "") if resp.choices else ""
        self._openai_history.append({"role": "user", "content": text})
        self._openai_history.append({"role": "assistant", "content": reply})
        return {"ok": True, "text": reply, "cost": 0.0}

    # ------------------------------------------------------------------

    @staticmethod
    def extract_action(text):
        """Find a GUI-action control block in a reply -> the action dict
        ({"action": "...", ...other keys like sweeps/mode}) or None."""
        for blob in reversed(_WAVEFORM_RE.findall(text or "")):
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("type") == "action":
                a = str(d.get("action", "")).strip().lower()
                if a:
                    d["action"] = a
                    return d
        return None

    @staticmethod
    def extract_waveform(text):
        """Find the last valid waveform json block in a reply, or None."""
        for blob in reversed(_WAVEFORM_RE.findall(text or "")):
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            pulses = d.get("pulses")
            if d.get("type") not in (None, "waveform") or not isinstance(pulses, list):
                continue
            clean = []
            try:
                for p in pulses[:2000]:
                    t0, w, a = float(p[0]), float(p[1]), float(p[2])
                    if w <= 0:
                        continue
                    clean.append((t0, w, a))
            except (TypeError, ValueError, IndexError):
                continue
            if clean:
                return {"pulses": clean,
                        "unit": str(d.get("unit", "A")),
                        "kind": str(d.get("kind", "current")),
                        "label": str(d.get("label", "agent pattern"))}
        return None
