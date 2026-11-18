"""vatester.spectre: PSF-ASCII import, deck, twin drive and the pass rule.

No Cadence needed: the PSF text below has the layout Spectre 24.1 writes with
`-format psfascii` (HEADER / TYPE / SWEEP / TRACE / VALUE), and the
end-to-end check writes a twin run in that layout and reads it back.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vatester.spectre import compare as C  # noqa: E402
from vatester.spectre.deck import make_deck  # noqa: E402
from vatester.spectre.psf import (PSFError, find_signal, parse_waveform,  # noqa: E402
                                  read_csv_export, read_psf_ascii)
from vatester.spectre.stimulus import Stimulus  # noqa: E402
from vatester.spectre.twinrun import load_twin, run_twin  # noqa: E402


def psf_text(t, i, v, version="24.1.0.078"):
    lines = ["HEADER", '"PSFversion" "1.00"', '"simulator" "spectre"',
             f'"version" "{version}"', '"analysis type" "tran"',
             "TYPE", '"sweep" FLOAT DOUBLE PROP(', '"key" "sweep"', ")",
             '"V" FLOAT DOUBLE PROP(', '"units" "V"', ")",
             '"I" FLOAT DOUBLE PROP(', '"units" "A"', ")",
             "SWEEP", '"time" "sweep" PROP(', '"units" "s"', ")",
             "TRACE", '"X1:p" "I" PROP(', '"units" "A"', ")", '"n_p" "V"',
             "VALUE"]
    for a, b, c in zip(t, i, v):
        lines += [f'"time" {a:.15e}', f'"X1:p" {b:.15e}', f'"n_p" {c:.15e}']
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_read_psf_ascii_layout():
    t = np.linspace(0, 1e-9, 11)
    sig, info = read_psf_ascii(psf_text(t, 2 * t, 3 * t))
    assert info == {"simulator": "spectre", "version": "24.1.0.078"}
    assert set(sig) == {"time", "X1:p", "n_p"}
    np.testing.assert_allclose(sig["X1:p"], 2 * t)
    np.testing.assert_allclose(find_signal(sig, "V(n_p)"), 3 * t)
    np.testing.assert_allclose(find_signal(sig, "x1:P"), 2 * t)


def test_read_psf_ascii_group():
    text = "\n".join([
        "HEADER", '"simulator" "spectre"', "TYPE", "SWEEP", '"time" "sweep"',
        "TRACE", '"group" GROUP 2', '"a" "V"', '"b" "I"',
        "VALUE", '"time" 0', '"group" 1 2', '"time" 1e-9', '"group" 3 4', "END"])
    sig, _ = read_psf_ascii(text)
    np.testing.assert_allclose(sig["a"], [1, 3])
    np.testing.assert_allclose(sig["b"], [2, 4])


def test_not_psf_raises():
    with pytest.raises(PSFError):
        read_psf_ascii("hello\nworld\n")


def test_csv_export_and_dispatch():
    text = "time,V(n_p),I(X1:p)\n0,0,0\n1e-9,0.5,1e-3\n2e-9,0.5,2e-3\n"
    sig, _ = parse_waveform(text)
    assert list(sig) == ["time", "V(n_p)", "I(X1:p)"]
    np.testing.assert_allclose(find_signal(sig, "n_p"), [0, 0.5, 0.5])
    viva = "a X,a Y,b X,b Y\n0,1,0,2\n1,3,1,4\n2,5,2,6\n"
    sig, _ = read_csv_export(viva)
    np.testing.assert_allclose(sig["b"], [2, 4, 6])


def test_stimulus_grid_and_reads():
    st = Stimulus.pulses([0.5, -0.25], 10e-9, 3e-9, read=-0.1)
    g = st.grid(1e-9)
    assert set(np.round(st.t, 18)) <= set(np.round(g, 18))      # every corner
    assert np.max(np.diff(g)) <= 1e-9 + 1e-18
    assert len(st.read_times(-0.1)) == 2
    with pytest.raises(ValueError):
        Stimulus(np.array([0.0, 1.0, 1.0]), np.zeros(3))
    s2 = Stimulus.from_text("0 0 1e-9 0.5 2e-9 0.5")
    np.testing.assert_allclose(s2.value(0.5e-9), 0.25)


def test_deck():
    st = Stimulus.from_text("0 0 1e-9 0.5 2e-9 0.5", name="s1")
    d = make_deck("vteam.va", "vteam", st, 1e-11, {"x_init": 1e-9})
    assert 'ahdl_include "vteam.va"' in d
    assert "X1 (n_p 0) vteam x_init=1e-09" in d
    assert "vin (n_p 0) vsource type=pwl wave=[ 0 0 1e-09 0.5 2e-09 0.5 ]" in d
    assert "tran tran stop=2e-09 maxstep=1e-11" in d
    assert "save n_p X1:p" in d
    st_i = Stimulus.from_text("0 0 1e-3 1e-9", kind="current")
    assert "iin (0 n_p) isource type=pwl" in make_deck("a.va", "a", st_i, 1e-6)
    for bad in (("../x.va", "m"), ("x.va", "m;rm"), ("x.va", "m")):
        with pytest.raises(ValueError):
            make_deck(*bad, st, 1e-11, {"a b": 1} if bad[1] == "m" else None)


def test_metrics_and_verdict():
    t = np.linspace(0, 1, 101)
    y = np.sin(2 * np.pi * t)
    m = C.waveform_metrics(t, y, t, y)
    assert m["nrmse"] == 0 and m["maxerr"] == 0
    m = C.waveform_metrics(t, y + 0.019, t, y)         # range 2 -> 0.95 %
    assert m["nrmse"] == pytest.approx(0.0095) and m["maxerr"] == pytest.approx(0.0095)
    assert C.verdict(m) == "PASS"
    m = C.waveform_metrics(t, y + 0.08, t, y)          # 4 % > 3 %
    assert C.verdict(m) == "FAIL"


def test_end_to_end_without_spectre():
    """VTEAM twin -> PSF-ASCII text -> import -> compare: PASS for the same
    parameters, FAIL when the twin's k_off is 20 % off."""
    spec = load_twin("vteam", ROOT)
    st = Stimulus.pulses([0.5, 0.8, -0.22, -0.25], 10e-9, 3e-9, read=-0.1)
    ref = run_twin(spec, st, 2e-11)
    sig, info = read_psf_ascii(psf_text(ref["time"], ref["I"], ref["V"]))
    r = {"time": sig["time"], "V": sig["n_p"], "I": sig["X1:p"]}

    tw = run_twin(spec, st, 1e-11)
    m = C.waveform_metrics(tw["time"], tw["I"], r["time"], r["I"])
    rd = C.read_metrics(tw, r, st.read_times(-0.1))
    assert rd["n_reads"] == 4 and np.ptp(rd["g_ref"]) > 0   # the state moved
    assert C.verdict(m, rd, "reads") == "PASS"

    bad = run_twin(spec, st, 1e-11, {"k_off": 6e-4})
    m = C.waveform_metrics(bad["time"], bad["I"], r["time"], r["I"])
    rd = C.read_metrics(bad, r, st.read_times(-0.1))
    assert C.verdict(m, rd, "reads") == "FAIL"


def test_twin_drive_kind_and_params_checked():
    spec = load_twin("vteam", ROOT)
    with pytest.raises(ValueError):
        run_twin(spec, Stimulus.from_text("0 0 1e-9 1e-6", kind="current"), 1e-10)
    with pytest.raises(KeyError):
        run_twin(spec, Stimulus.from_text("0 0 1e-9 0.5"), 1e-10, {"nope": 1})


def test_remote_needs_host_and_safe_name(tmp_path, monkeypatch):
    from vatester.spectre import remote
    va = tmp_path / "a.va"
    va.write_text("module a(p, n); endmodule\n")
    monkeypatch.setattr(remote, "_load_host", lambda: "")
    with pytest.raises(remote.SpectreRunError):
        remote.run_remote("x", va, "ok")
    with pytest.raises(ValueError):
        remote.run_remote("x", va, "bad name;rm", host="u@h")
