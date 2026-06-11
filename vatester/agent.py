"""Claude agent backend for the GUI chat panel.

Primary backend: the `claude` CLI in headless print mode
(`claude -p --output-format json`), which gives the agent real file tools so
it can read and (when allowed) modify the Verilog-A sources in the
workspace.  Conversation continuity uses --resume <session_id>.

Fallback: the Anthropic Python SDK (model claude-opus-4-8) when the CLI is
not installed but ANTHROPIC_API_KEY is set - chat + pattern generation only
(no file tools; the GUI inlines the selected .va source into the context).
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

REAL-TIME RULE - put ALL device and analysis behavior in the TWINS. The GUI
hot-reloads ecfet/model_*.py on every edit, so changes there appear LIVE with
no restart. It does NOT reload its own code: editing vatester/app.py (e.g. the
STDP worker, the analysis sampling, plotting) does NOTHING until the app is
manually restarted, so DO NOT put model fixes there. If a measured curve is
wrong (STDP not antisymmetric, tails not returning to 0, LTP/LTD imbalance,
baseline pedestal), the cause and the fix live in the device equations of the
twin - fix model_v2.py (and mirror to ecfet_v2.va), never patch app.py. Keep
the simulator/GUI untouched.

RUNNING THINGS IN THE GUI. You CANNOT click buttons. When the user asks you
to run, plot, or switch a view in the app ("run", "plot the STDP", "run STDP",
"show analysis as resistance", "fit the plots"), DO NOT write or run a script
and DO NOT produce your own PNG. Instead reply with one short sentence and
exactly ONE fenced action block - the GUI executes it instantly and shows the
result on its own plots:
```json
{"type": "action", "action": "plot_stdp"}
```
Valid actions: "run" (transient sim), "plot_stdp" (STDP sweep), "preview"
(stimulus preview), "analyze_g" / "analyze_r" (analysis quantity), "fit"
(fit plot axes), "export_csv". Be fast here - no file reads, no deliberation;
just the one sentence + the action block. Only use tools.../the edit loop for
tasks that actually change model code.

WAVEFORM PATTERNS. When asked to design a spike pattern / stimulus, reply
briefly and include exactly ONE fenced json block:
```json
{"type": "waveform", "kind": "current", "unit": "pA", "label": "name",
 "pulses": [[t_start_s, width_s, amplitude], ...]}
```
kind "current" (ECFET) or "voltage" (FeFET); unit pA,nA,uA,mA,A,mV,V; times in
seconds; <=2000 pulses. ECFET sign convention: positive gate current RAISES R
(depresses); potentiation uses negative current. FeFET: positive gate voltage
potentiates. The GUI turns that block into a one-click "Load pattern" card.

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

OTHER TOOLS (when shell is enabled). You have a real shell. The project root has
Virtuoso connection helpers: virtuoso_connect.bat and connect_test.py (SSH
tunnel + skillbridge to Virtuoso on coen-cassia). Use them via bash only if the
task needs live Virtuoso access. You can run any project script, inspect git,
install nothing destructive, and read results/agent_snapshot.json for the data
currently on the GUI's plots.

Keep chat replies short and concrete: say what you changed, what you ran, and
the measured outcome. Do the work; don't ask for permission you already have.
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


class ClaudeAgent:
    def __init__(self, workdir):
        self.workdir = workdir
        self.session_id = None
        self.total_cost = 0.0
        self._history = []          # SDK fallback conversation
        self.cli = find_claude_cli()
        # per-app credential override (does NOT touch the global CLI login)
        self.override_kind = None   # "api_key" | "oauth" | None
        self.override_value = None
        self._proc = None           # running CLI process (for Stop)
        self._stop_requested = False
        self._refresh_sdk_ok()

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
        if self.cli:
            return "cli"
        if self.sdk_ok:
            return "sdk"
        return "none"

    def backend_label(self):
        if self.cli:
            return f"claude CLI ({os.path.basename(os.path.dirname(self.cli)) or 'PATH'})"
        if self.sdk_ok:
            return "Anthropic SDK (claude-opus-4-8)"
        return "no backend - install claude CLI or set ANTHROPIC_API_KEY"

    def reset(self):
        self.session_id = None
        self._history = []

    # ------------------------------------------------------------------

    def send(self, text, context="", allow_edits=False, allow_bash=False,
             model="default", timeout=600, autonomous=False):
        """Blocking - call from a worker thread.
        Returns {ok, text, error, cost, session_id}."""
        if autonomous:                  # full edit->run->verify->fix loop
            allow_edits = allow_bash = True
            timeout = max(timeout, 1800)
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

    # ------------------------------------------------------------------

    @staticmethod
    def extract_action(text):
        """Find a GUI-action control block in a reply, or None."""
        for blob in reversed(_WAVEFORM_RE.findall(text or "")):
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("type") == "action":
                a = str(d.get("action", "")).strip().lower()
                if a:
                    return a
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
