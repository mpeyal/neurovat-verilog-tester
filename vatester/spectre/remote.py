"""Run Spectre on the Cadence host and bring the PSF-ASCII result back, over
ONE SSH login.

    with Session() as s:                     # one login (paramiko)
        runs = run_remote_batch(decks, va_path, session=s)
    run = run_remote(deck_text, va_path, name="s1")      # opens and closes one

Why one login.  University security monitoring treats many SSH logins in a
short time as a brute-force attack and resets the connections (this happened
on 2026-11-18, when every step - mkdir, upload, start, each status poll, each
download, each retry - opened a new login).  So:

* a Session authenticates once; upload (SFTP), start, status polls (every
  POLL_S >= 30 s), downloads and cleanup are channels of that one connection;
* nothing is retried: a refused, reset or dropped connection stops the run
  with a clear error;
* LoginGuard keeps a local record of logins (all NeuroVAT processes on this
  computer) and refuses a new one beyond MAX_LOGINS in WINDOW_S; after a
  refused or reset connection it refuses every login for COOLDOWN_S.

The host is the Virtuoso login (NVAT_VIRTUOSO_HOST or the gitignored
virtuoso.local); host keys must already be in ~/.ssh/known_hosts (unknown or
changed keys are refused).  Each run gets its own new folder under
~/.neurovat_runs/ on the host, removed afterwards unless keep=True.  The
Cadence environment comes from NVAT_SPECTRE_SETUP (default "module load
cadence") in a login shell, the binary from NVAT_SPECTRE (default "spectre").
"""

import io
import json
import os
import shlex
import socket
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..virtuoso import _load_host

REMOTE_ROOT = ".neurovat_runs"            # under the remote home directory
POLL_S = 30.0                             # status poll interval (same connection)
MAX_LOGINS = 4                            # new SSH logins allowed ...
WINDOW_S = 600.0                          # ... per 10 minutes, all processes
COOLDOWN_S = 1800.0                       # no logins for 30 min after a reset
GUARD_FILE = Path(os.path.expanduser("~")) / ".neurovat" / "ssh_logins.json"


class SpectreRunError(RuntimeError):
    pass


@dataclass
class RemoteRun:
    raw_dir: Path
    log_text: str
    remote_dir: str
    wall_s: float


# ---------------------------------------------------------------- login guard
class LoginGuard:
    """Local, cross-process record of SSH logins to the Cadence host."""

    def __init__(self, path=GUARD_FILE, max_logins=MAX_LOGINS, window_s=WINDOW_S,
                 cooldown_s=COOLDOWN_S):
        self.path = Path(path)
        self.max_logins, self.window_s, self.cooldown_s = max_logins, window_s, cooldown_s

    def _load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return {"logins": [float(x) for x in d.get("logins", [])],
                    "blocked_until": float(d.get("blocked_until", 0.0)),
                    "reason": str(d.get("reason", ""))}
        except (OSError, ValueError):
            return {"logins": [], "blocked_until": 0.0, "reason": ""}

    def _save(self, d):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        os.replace(tmp, self.path)

    def check_and_record(self, now=None):
        """Raise SpectreRunError if a login is not allowed now; else record it."""
        now = time.time() if now is None else now
        d = self._load()
        if d["blocked_until"] > now:
            mins = (d["blocked_until"] - now) / 60
            raise SpectreRunError(
                f"SSH to the Cadence host is paused for {mins:.0f} more min after "
                f"'{d['reason']}'. Repeated logins can be flagged as an attack; "
                f"wait, or delete {self.path} once IT confirms the account is fine.")
        recent = [t for t in d["logins"] if now - t < self.window_s]
        if len(recent) >= self.max_logins:
            wait = (min(recent) + self.window_s - now) / 60
            raise SpectreRunError(
                f"{len(recent)} SSH logins in the last {self.window_s / 60:.0f} min: "
                f"the limit is {self.max_logins}. Try again in {wait:.0f} min.")
        recent.append(now)
        d["logins"] = recent
        self._save(d)

    def block(self, reason, now=None):
        now = time.time() if now is None else now
        d = self._load()
        d["blocked_until"] = now + self.cooldown_s
        d["reason"] = reason[:200]
        self._save(d)


# ---------------------------------------------------------------- one session
def _split_host(host):
    user, _, h = host.rpartition("@")
    if not h:
        raise SpectreRunError(f"host must be 'user@host', got {host!r}")
    return (user or None), h


class Session:
    """One authenticated SSH connection to the Cadence host (paramiko).
    Host keys come from ~/.ssh/known_hosts; an unknown key is refused."""

    def __init__(self, host=None, guard=None, connect_timeout=20):
        self.host = host or _load_host()
        if not self.host:
            raise SpectreRunError("no host: set NVAT_VIRTUOSO_HOST or write "
                                  "'user@host' into virtuoso.local")
        self.guard = guard or LoginGuard()
        self.connect_timeout = connect_timeout
        self.client = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self):
        if self.client is not None:
            return self
        try:
            import paramiko
        except ImportError as e:
            raise SpectreRunError("paramiko is needed (pip install paramiko)") from e
        self.guard.check_and_record()
        user, hostname = _split_host(self.host)
        c = paramiko.SSHClient()
        c.load_system_host_keys()
        kh = Path(os.path.expanduser("~")) / ".ssh" / "known_hosts"
        if kh.is_file():
            c.load_host_keys(str(kh))
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            c.connect(hostname, username=user, timeout=self.connect_timeout,
                      banner_timeout=self.connect_timeout, auth_timeout=self.connect_timeout,
                      allow_agent=True, look_for_keys=True)
        except paramiko.AuthenticationException as e:
            self.guard.block(f"authentication failed: {e}")
            raise SpectreRunError(f"SSH authentication to the Cadence host failed ({e}); "
                                  "no retry. Check the key.") from e
        except (paramiko.SSHException, socket.error, OSError) as e:
            self.guard.block(f"connection failed: {type(e).__name__}: {e}")
            raise SpectreRunError(f"could not connect to the Cadence host "
                                  f"({type(e).__name__}: {e}); no retry.") from e
        c.get_transport().set_keepalive(30)
        self.client = c
        return self

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None

    def _lost(self, e):
        self.guard.block(f"connection lost: {type(e).__name__}: {e}")
        self.close()
        return SpectreRunError(f"lost the connection to the Cadence host "
                               f"({type(e).__name__}: {e}); no retry.")

    def exec(self, cmd, timeout=120, binary=False):
        """Run a command over the open connection -> (exit code, stdout, stderr)."""
        if self.client is None:
            self.open()
        try:
            _, out, err = self.client.exec_command(cmd, timeout=timeout)
            data = out.read()
            code = out.channel.recv_exit_status()
            e = err.read().decode(errors="replace")
        except (OSError, EOFError, socket.error) as ex:
            raise self._lost(ex) from ex
        except Exception as ex:                     # paramiko.SSHException et al.
            if type(ex).__module__.startswith("paramiko"):
                raise self._lost(ex) from ex
            raise
        return code, (data if binary else data.decode(errors="replace")), e

    def put(self, local, remote):
        if self.client is None:
            self.open()
        try:
            sftp = self.client.open_sftp()
            try:
                sftp.put(str(local), remote)
            finally:
                sftp.close()
        except (OSError, EOFError) as ex:
            raise self._lost(ex) from ex
        except Exception as ex:
            if type(ex).__module__.startswith("paramiko"):
                raise self._lost(ex) from ex
            raise


# ---------------------------------------------------------------- runs
def _safe_name(name):
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"run name must be letters, digits, _ or -: {name!r}")
    return name


def run_remote(deck_text, va_path, name, host=None, timeout=3600, keep=False,
               out_dir=None, setup=None, spectre=None, session=None):
    """Upload the deck and the .va, run Spectre, download <name>.raw and the log.
    The deck must ahdl_include the .va by its bare file name (make_deck does)."""
    runs = run_remote_batch({name: deck_text}, va_path, host=host, timeout=timeout,
                            keep=keep, out_dir=out_dir, setup=setup, spectre=spectre,
                            session=session)
    return runs[name]


def run_remote_batch(decks, va_path, host=None, timeout=3600, keep=False,
                     out_dir=None, setup=None, spectre=None, progress=None,
                     session=None):
    """Run several decks {name: text} that include the same .va, in one remote
    folder, over one SSH connection (`session`, or one opened and closed here).
    Returns {name: RemoteRun}; raises SpectreRunError naming every deck that
    failed.  progress(done, total) is called at each status poll; it may raise
    to stop."""
    names = [_safe_name(n) for n in decks]
    va_path = Path(va_path)
    if not va_path.is_file():
        raise SpectreRunError(f"no such .va file: {va_path}")
    own = session is None
    s = session or Session(host)
    try:
        return _batch(s, decks, names, va_path, timeout, keep, out_dir, setup,
                      spectre, progress)
    finally:
        if own:
            s.close()


def _batch(s, decks, names, va_path, timeout, keep, out_dir, setup, spectre, progress):
    setup = setup if setup is not None else os.environ.get("NVAT_SPECTRE_SETUP",
                                                           "module load cadence")
    spectre = spectre or os.environ.get("NVAT_SPECTRE", "spectre")
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    rdir = f"{REMOTE_ROOT}/{run_id}"        # relative to the remote home
    q = shlex.quote
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="nvat_spectre_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    codes = {}
    code, _, err = s.exec(f"mkdir -p {q(rdir)}", 60)
    if code != 0:
        raise SpectreRunError(f"remote mkdir failed: {err.strip()[-400:]}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for n in names:
                pth = Path(tmp) / f"{n}.scs"
                pth.write_text(decks[n], encoding="utf-8", newline="\n")
                s.put(pth, f"{rdir}/{n}.scs")
        s.put(va_path, f"{rdir}/{va_path.name}")
        cmds = [f"{spectre} +escchars -format psfascii -raw {q(n + '.raw')} "
                f"+log {q(n + '.log')} {q(n + '.scs')} >/dev/null 2>&1; "
                f"echo EXIT_{n}=$? >> status.txt" for n in names]
        job = f"{setup} >/dev/null 2>&1; " + "; ".join(cmds) + "; echo DONE >> status.txt"
        # detached on the host, so a dropped connection does not kill Spectre
        start = (f"cd {q(rdir)} && setsid nohup bash -lc {q(job)} > job.out 2>&1 "
                 f"< /dev/null & echo started")
        code, _, err = s.exec(start, 60)
        if code != 0:
            raise SpectreRunError(f"spectre start failed: {err.strip()[-400:]}")
        out = _wait_status(s, rdir, timeout, len(names), progress)
        for ln in out.splitlines():
            if ln.startswith("EXIT_") and "=" in ln:
                k, v = ln[5:].split("=", 1)
                codes[k] = v.strip()
        fetch = []
        for n in names:
            if codes.get(n) == "0":
                fetch.append(f"{n}.raw")
            fetch.append(f"{n}.log")
        _download_tar(s, rdir, fetch, out_dir)
    finally:
        if not keep and s.client is not None:
            try:
                s.exec(f"rm -rf {q(rdir)}", 60)
            except SpectreRunError:
                pass
    wall = time.perf_counter() - t0
    runs, failed = {}, []
    for n in names:
        log_path = out_dir / f"{n}.log"
        log_text = (log_path.read_text(encoding="utf-8", errors="replace")
                    if log_path.exists() else "")
        if codes.get(n) != "0":
            tail = "\n".join(log_text.splitlines()[-25:])
            failed.append(f"{n}: spectre exited with {codes.get(n, '?')}; log tail:\n{tail}")
            continue
        runs[n] = RemoteRun(raw_dir=out_dir / f"{n}.raw", log_text=log_text,
                            remote_dir=rdir if keep else "", wall_s=wall)
    if failed:
        raise SpectreRunError("\n\n".join(failed))
    return runs


def _wait_status(s, rdir, timeout, total=0, progress=None, poll=POLL_S):
    """Read the job's status.txt over the open connection every `poll` s until
    DONE.  No new logins; a dropped connection ends the run."""
    q = shlex.quote
    t_end = time.monotonic() + timeout
    first = True
    while time.monotonic() < t_end:
        time.sleep(min(10.0, poll) if first else poll)
        first = False
        _, out, _ = s.exec(f"cat {q(rdir)}/status.txt 2>/dev/null; true", 60)
        if progress is not None:
            progress(sum(1 for ln in out.splitlines() if ln.startswith("EXIT_")), total)
        if "DONE" in out.split():
            return out
    raise SpectreRunError(f"Spectre did not finish within {timeout} s")


def _download_tar(s, rdir, members, out_dir, timeout=1800):
    """Bring `members` of the run folder back as one gzip tar over the open
    connection and unpack them into out_dir (no absolute paths, links or '..')."""
    q = shlex.quote
    cmd = (f"cd {q(rdir)} && tar czf - --ignore-failed-read "
           + " ".join(q(m) for m in members))
    code, data, err = s.exec(cmd, timeout, binary=True)
    if code != 0 and not data:
        raise SpectreRunError(f"download failed: {err.strip()[-400:]}")
    unpack_tar(data, out_dir)


def unpack_tar(data, out_dir):
    out_dir = Path(out_dir).resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            target = (out_dir / m.name).resolve()
            if (m.issym() or m.islnk() or m.name.startswith(("/", "\\"))
                    or (out_dir not in target.parents and target != out_dir)):
                raise SpectreRunError(f"unsafe path in the result archive: {m.name}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())


def read_via_skillbridge(link, remote_path):
    """Read a text result file (PSF-ASCII / CSV) on the Cadence host through a
    connected VirtuosoLink, e.g. a file an ADE/Maestro run wrote there."""
    return link.read_file(remote_path)
