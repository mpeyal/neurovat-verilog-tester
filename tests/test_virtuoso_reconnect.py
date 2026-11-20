"""VirtuosoLink.connect after a failed attempt opens a FRESH skillbridge
connection (skillbridge caches Workspace.open(); a dead cached one used to
fail every later Connect until the app restarted).  No network: the tunnel
and the skillbridge channel are replaced by fakes."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

sbw = pytest.importorskip("skillbridge.client.workspace")

from vatester import virtuoso as V  # noqa: E402


class FakeWs:
    opened = []

    def __init__(self, alive):
        self.alive, self.closed = alive, False
        FakeWs.opened.append(self)

    def __getitem__(self, name):
        def call(*a):
            if not self.alive:
                raise ConnectionResetError(10054, "forcibly closed by the remote host")
            return "virtuoso version IC25.1-64b"
        return call

    @property
    def dd(self):
        class DD:
            @staticmethod
            def get_lib_list():
                return []
        return DD

    def close(self, log_exception=True):
        self.closed = True


def test_connect_again_after_failure_opens_fresh(monkeypatch):
    FakeWs.opened.clear()
    alive = iter([False, True])

    def fake_open(cls=None, workspace_id=None, **kw):     # mimics the cache
        if workspace_id not in sbw._open_workspaces:
            sbw._open_workspaces[workspace_id] = FakeWs(next(alive))
        return sbw._open_workspaces[workspace_id]

    monkeypatch.setattr(sbw.Workspace, "open", classmethod(fake_open))
    monkeypatch.setattr(sbw, "_open_workspaces", {})
    link = V.VirtuosoLink(host="user@example.invalid")
    tunnels = []
    monkeypatch.setattr(link, "_ensure_tunnel", lambda: tunnels.append(1) or "reused")
    monkeypatch.setattr(link, "_ensure_helpers", lambda: None)

    with pytest.raises(RuntimeError, match="skillbridge failed"):
        link.connect()
    assert FakeWs.opened[0].closed and None not in sbw._open_workspaces
    info = link.connect()                                 # the server is up now
    assert link.connected and "IC25.1" in info["version"]
    assert len(FakeWs.opened) == 2 and link.ws is FakeWs.opened[1]
