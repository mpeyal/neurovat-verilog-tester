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

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
REMOTE_ROOT = ".neurovat_runs"            # under the remote home directory


class SpectreRunError(RuntimeError):
    pass


@dataclass
class RemoteRun:
    raw_dir: Path
    log_text: str
    remote_dir: str
    wall_s: float


def _run(cmd, timeout, what):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise SpectreRunError(f"{what} timed out after {timeout} s") from e
    except FileNotFoundError as e:
        raise SpectreRunError(f"{what}: {cmd[0]} not found (install OpenSSH)") from e
    if p.returncode != 0:
        raise SpectreRunError(f"{what} failed ({p.returncode}): "
                              f"{(p.stderr or p.stdout).strip()[-800:]}")
    return p.stdout


def _safe_name(name):
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"run name must be letters, digits, _ or -: {name!r}")
    return name


def run_remote(deck_text, va_path, name, host=None, timeout=3600, keep=False,
               out_dir=None, setup=None, spectre=None):
    """Upload the deck and the .va, run Spectre, download <name>.raw and the log.

    The deck must ahdl_include the .va by its bare file name (make_deck does).
    """
    host = host or _load_host()
    if not host:
        raise SpectreRunError("no host: set NVAT_VIRTUOSO_HOST or write "
                              "'user@host' into virtuoso.local")
    name = _safe_name(name)
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
    with tempfile.TemporaryDirectory() as tmp:
        deck_path = Path(tmp) / f"{name}.scs"
        deck_path.write_text(deck_text, encoding="utf-8", newline="\n")
        _run(["ssh", *SSH_OPTS, host, f"mkdir -p {q(rdir)}"], 60, "remote mkdir")
        try:
            _run(["scp", "-q", "-B", *SSH_OPTS, str(deck_path), str(va_path),
                  f"{host}:{rdir}/"], 300, "upload")
            inner = (f"{setup} >/dev/null 2>&1; cd {q(rdir)} && "
                     f"{spectre} +escchars -format psfascii -raw {q(name + '.raw')} "
                     f"+log {q(name + '.log')} {q(name + '.scs')} >/dev/null 2>&1; "
                     f"echo EXIT=$?")
            out = _run(["ssh", *SSH_OPTS, host, f"bash -lc {q(inner)}"],
                       timeout, "spectre run")
            code = next((ln.split("=", 1)[1] for ln in out.splitlines()
                         if ln.startswith("EXIT=")), "?")
            _run(["scp", "-q", "-B", "-r", *SSH_OPTS,
                  f"{host}:{rdir}/{name}.raw", f"{host}:{rdir}/{name}.log",
                  str(out_dir) + "/"], 600, "download")
        finally:
            if not keep:
                try:
                    _run(["ssh", *SSH_OPTS, host, f"rm -rf {q(rdir)}"], 60,
                         "remote cleanup")
                except SpectreRunError:
                    pass
    log_path = out_dir / f"{name}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if code != "0":
        tail = "\n".join(log_text.splitlines()[-25:])
        raise SpectreRunError(f"spectre exited with {code}; log tail:\n{tail}")
    return RemoteRun(raw_dir=out_dir / f"{name}.raw", log_text=log_text,
                     remote_dir=rdir if keep else "", wall_s=time.perf_counter() - t0)


def read_via_skillbridge(link, remote_path):
    """Read a text result file (PSF-ASCII / CSV) on the Cadence host through a
    connected VirtuosoLink, e.g. a file an ADE/Maestro run wrote there."""
    return link.read_file(remote_path)
