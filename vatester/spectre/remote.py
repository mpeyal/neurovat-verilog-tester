"""Run Spectre on the Cadence host over SSH and bring the PSF-ASCII result back.

    run = run_remote(deck_text, va_path, name="s1", host=None)
    run.raw_dir      # local folder with the PSF-ASCII files (<analysis>.tran.tran ...)
    run.log_text     # the Spectre log

The host is the same login the Virtuoso link uses (NVAT_VIRTUOSO_HOST or the
gitignored virtuoso.local file); nothing is hard-coded.  SSH runs with
BatchMode=yes and keeps host-key checking on.  Each run gets its own new
folder under ~/.neurovat_runs/ on the host, so nothing else there is touched;
the folder is removed afterwards unless keep=True.

The Cadence environment comes from NVAT_SPECTRE_SETUP (default
"module load cadence") in a login shell, and the binary from NVAT_SPECTRE
(default "spectre").
"""

import os
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..virtuoso import _load_host

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]
REMOTE_ROOT = ".neurovat_runs"            # under the remote home directory


class SpectreRunError(RuntimeError):
    pass


@dataclass
class RemoteRun:
    raw_dir: Path
    log_text: str
    remote_dir: str
    wall_s: float


def _run(cmd, timeout, what, retries=1):
    """Run an ssh/scp command; `retries` > 1 only for steps that are safe to repeat
    (exit code 255 = connection-level failure is retried after a short wait)."""
    for attempt in range(retries):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            if attempt + 1 < retries:
                continue
            raise SpectreRunError(f"{what} timed out after {timeout} s") from e
        except FileNotFoundError as e:
            raise SpectreRunError(f"{what}: {cmd[0]} not found (install OpenSSH)") from e
        if p.returncode == 0:
            return p.stdout
        if p.returncode == 255 and attempt + 1 < retries:
            time.sleep(5 * (attempt + 1))
            continue
        raise SpectreRunError(f"{what} failed ({p.returncode}): "
                              f"{(p.stderr or p.stdout).strip()[-800:]}")


def _safe_name(name):
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"run name must be letters, digits, _ or -: {name!r}")
    return name


def run_remote(deck_text, va_path, name, host=None, timeout=3600, keep=False,
               out_dir=None, setup=None, spectre=None):
    """Upload the deck and the .va, run Spectre, download <name>.raw and the log.

    The deck must ahdl_include the .va by its bare file name (make_deck does).
    """
    runs = run_remote_batch({name: deck_text}, va_path, host=host, timeout=timeout,
                            keep=keep, out_dir=out_dir, setup=setup, spectre=spectre)
    return runs[name]


def run_remote_batch(decks, va_path, host=None, timeout=3600, keep=False,
                     out_dir=None, setup=None, spectre=None):
    """Run several decks {name: text} that include the same .va, in one remote
    folder and one SSH session per step; returns {name: RemoteRun}.  Raises
    SpectreRunError naming every deck that failed."""
    host = host or _load_host()
    if not host:
        raise SpectreRunError("no host: set NVAT_VIRTUOSO_HOST or write "
                              "'user@host' into virtuoso.local")
    names = [_safe_name(n) for n in decks]
    va_path = Path(va_path)
    if not va_path.is_file():
        raise SpectreRunError(f"no such .va file: {va_path}")
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
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for n in names:
            pth = Path(tmp) / f"{n}.scs"
            pth.write_text(decks[n], encoding="utf-8", newline="\n")
            files.append(str(pth))
        _run(["ssh", *SSH_OPTS, host, f"mkdir -p {q(rdir)}"], 60, "remote mkdir", 4)
        try:
            _run(["scp", "-q", "-B", *SSH_OPTS, *files, str(va_path),
                  f"{host}:{rdir}/"], 300, "upload", 4)
            cmds = [f"{spectre} +escchars -format psfascii -raw {q(n + '.raw')} "
                    f"+log {q(n + '.log')} {q(n + '.scs')} >/dev/null 2>&1; "
                    f"echo EXIT_{n}=$? >> status.txt" for n in names]
            job = (f"{setup} >/dev/null 2>&1; " + "; ".join(cmds)
                   + "; echo DONE >> status.txt")
            # detached on the host: a dropped SSH connection does not stop it
            # the 'started' marker makes a retried start a no-op
            start = (f"cd {q(rdir)} && if [ ! -e started ]; then touch started; "
                     f"setsid nohup bash -lc {q(job)} > job.out 2>&1 < /dev/null & "
                     f"fi; echo started")
            _run(["ssh", *SSH_OPTS, host, start], 60, "spectre start", 4)
            out = _wait_status(host, rdir, timeout)
            for ln in out.splitlines():
                if ln.startswith("EXIT_") and "=" in ln:
                    k, v = ln[5:].split("=", 1)
                    codes[k] = v.strip()
            fetch = []
            for n in names:
                if codes.get(n) == "0":
                    fetch.append(f"{n}.raw")
                fetch.append(f"{n}.log")
            for k in range(0, len(fetch), 2):      # one deck (raw + log) per stream
                _download_tar(host, rdir, fetch[k:k + 2], out_dir)
        finally:
            if not keep:
                try:
                    _run(["ssh", *SSH_OPTS, host, f"rm -rf {q(rdir)}"], 60,
                         "remote cleanup")
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




def _wait_status(host, rdir, timeout, poll=5.0):
    """Poll the detached job's status.txt with short SSH calls until DONE.
    A failed poll (dropped connection) is retried; gives up after `timeout` s."""
    q = shlex.quote
    t_end = time.monotonic() + timeout
    fails = 0
    while time.monotonic() < t_end:
        time.sleep(poll)
        try:
            p = subprocess.run(["ssh", *SSH_OPTS, host,
                                f"cat {q(rdir)}/status.txt 2>/dev/null; true"],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            fails += 1
            continue
        if p.returncode != 0:
            fails += 1
            if fails > 20:
                raise SpectreRunError("lost the connection to the host: "
                                      + (p.stderr or "").strip()[-400:])
            continue
        fails = 0
        if "DONE" in p.stdout.split():
            return p.stdout
        poll = min(poll * 1.2, 20.0)
    raise SpectreRunError(f"Spectre did not finish within {timeout} s")

def _download_tar(host, rdir, members, out_dir, timeout=600, retries=3):
    """Stream `members` of the remote run folder back as one gzip tar over SSH
    and unpack them into out_dir (no absolute paths, links or '..' accepted).
    A dropped connection is retried."""
    import io
    import tarfile
    q = shlex.quote
    cmd = (f"cd {q(rdir)} && tar czf - --ignore-failed-read "
           + " ".join(q(m) for m in members))
    err = ""
    for attempt in range(retries):
        try:
            p = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            err = f"timed out after {timeout} s"
            continue
        if p.returncode == 0:
            break
        err = p.stderr.decode(errors="replace").strip()[-800:]
        time.sleep(3 * (attempt + 1))
    else:
        raise SpectreRunError(f"download failed after {retries} attempts: {err}")
    out_dir = Path(out_dir).resolve()
    with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as tf:
        for m in tf.getmembers():
            target = (out_dir / m.name).resolve()
            if (m.issym() or m.islnk() or m.name.startswith(("/", "\\"))
                    or out_dir not in target.parents and target != out_dir):
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
