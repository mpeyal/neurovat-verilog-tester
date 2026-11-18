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


# ---------------------------------------------------------------- ECFET, layers 2 and 3
def test_ecfet_deck_and_builtin_twin():
    from vatester.spectre.deck import make_ecfet_deck
    from vatester.spectre.layer1 import reference, score
    st = Stimulus.from_text("0 0 0.01 0 0.010001 5e-11 0.02 5e-11 0.020001 0 0.5 0",
                            kind="current", name="single")
    d = make_ecfet_deck("ecfet_v3.va", "ecfet_v3", st, 1e-3, {"Rinit": 4400})
    assert "X1 (n_in 0 n_igsc n_rmem) ecfet_v3 Rinit=4400" in d
    assert "igate (0 n_igsc) isource type=pwl" in d and "ic=node" in d
    with pytest.raises(ValueError):
        make_ecfet_deck("ecfet_v3.va", "ecfet_v3", Stimulus.from_text("0 0 1 1"), 1e-3)
    spec = load_twin("v3", ROOT)                       # built-in, not in twins/
    tw = run_twin(spec, st, 1e-3)
    assert tw["G"][-1] > tw["G"][0]                    # +50 pA potentiates
    ref = reference({"time": tw["time"], "n_rmem": 1.0 / tw["G"]}, "ecfet")
    s = score(tw, ref, "ecfet", st)
    assert s["compared"] == "conductance" and s["verdict"] == "PASS"
    bad = run_twin(spec, st, 1e-3, {"Q_full": 0.4e-9})
    assert score(bad, ref, "ecfet", st)["verdict"] == "FAIL"


def test_convergence_table():
    from vatester.spectre.converge import convergence
    spec = load_twin("vteam", ROOT)
    st = Stimulus.pulses([0.5, -0.25], 10e-9, 3e-9, read=-0.1)
    fine = run_twin(spec, st, 1e-11)
    good = dict(fine)
    bad = dict(fine, I=fine["I"] * 1.1)                # a coarse step that is 10 % off
    twins = {d: run_twin(spec, st, d) for d in (1e-10, 5e-11, 2.5e-11)}
    rows, s = convergence({1e-12: fine, 1e-11: good, 1e-10: bad}, st, "2t", twins)
    assert s["unconverged_steps_s"] == [1e-10]
    assert s["coarsest_converged_step_s"] == 1e-11
    assert s["twin_vs_finest"] == "PASS"


def test_replay_recording_waves_and_score():
    from vatester.spectre.replay import TICK, gate_waves, record_training, score_replay
    rec = record_training(afferents=4, seconds=0.6, seed=2)
    assert rec.n == 4 and len(rec.pulses) > 0 and rec.reads_g.shape[1] == 4
    waves, t_stop = gate_waves(rec)
    assert len(waves) == 4 and t_stop > rec.t_end
    for k, w in enumerate(waves):                       # PWL charge = schedule charge
        nums = [float(x) for x in w.strip("[] ").replace("\\", " ").split()]
        t, i = np.array(nums[0::2]), np.array(nums[1::2])
        q_pwl = np.trapezoid(i, t) if hasattr(np, "trapezoid") else np.trapz(i, t)
        q_sch = sum(a * wd for c, s, wd, a in rec.pulses if c == k)
        assert q_pwl == pytest.approx(q_sch, rel=1e-6, abs=1e-24)
        assert np.all(np.diff(np.round(t / TICK)) > 0)
    sig = {"time": rec.reads_t, **{f"n_rm{k}": 1.0 / rec.reads_g[:, k] for k in range(4)}}
    assert all(r["verdict"] == "PASS" for r in score_replay(rec, sig))


# ---------------------------------------------------------------- automatic validation
def _fake_runner(spec, scale=1.0):
    """Stands in for Spectre: reads the PWL and maxstep from each deck, runs
    the twin and returns the signals the deck saves (current scaled by `scale`)."""
    import re as _re

    def run(decks, va_path, out_dir):
        out = {}
        for name, text in decks.items():
            body = _re.search(r"wave=\[(.*?)\]", text, _re.S).group(1).replace("\\", " ")
            nums = [float(x) for x in body.split()]
            st = Stimulus(np.array(nums[0::2]), np.array(nums[1::2]))
            step = float(_re.search(r"maxstep=([0-9.eE+-]+)", text).group(1))
            tw = run_twin(spec, st, step)
            out[name] = ({"time": tw["time"], "n_p": tw["V"], "X1:p": scale * tw["I"]},
                         {"simulator": "fake", "version": "0"})
        return out
    return run


def test_suites_are_fixed_and_scaled():
    from vatester.spectre.suites import build_suite, profile_of
    vt = load_twin("vteam", ROOT)
    su = build_suite(vt)
    assert profile_of(vt) == "memristor" and len(su.stimuli) == 5
    assert su.stimuli[0].name == "train"
    assert 0.02 < su.calibration["V_switch_pos"] < 0.2      # VTEAM v_off = 0.02 V
    assert -0.3 < su.calibration["V_switch_neg"] < -0.2     # VTEAM v_on = -0.2 V
    assert [s.name for s in build_suite(vt).stimuli] == [s.name for s in su.stimuli]
    e = build_suite(load_twin("v3", ROOT))
    assert e.profile == "ecfet" and len(e.stimuli) == 8 and e.calibration["I_unit_A"] > 0


def test_va_interface_and_twin_match():
    from vatester.spectre.deck import transistor_roles, va_interface
    from vatester.spectre.validate import match_twin
    mod, ports, params, bound = va_interface(open(os.path.join(ROOT, "verilog", "vteam.va"),
                                                  encoding="utf-8").read())
    assert (mod, ports) == ("vteam", ["p", "n"]) and "r_on" in params and bound is None
    assert match_twin("vteam.va", "vteam", ROOT) == "vteam"
    assert match_twin("x.va", "ecfet_v3", ROOT) == "v3"
    roles, mon = transistor_roles(["vdrain", "vgate", "vsource", "vbody", "p_node"])
    assert roles == {"drain": "vdrain", "gate": "vgate", "source": "vsource",
                     "body": "vbody"} and mon == ["p_node"]
    roles, mon = transistor_roles(["d", "g", "s"])
    assert roles == {"drain": "d", "gate": "g", "source": "s"} and mon == []


def test_validate_end_to_end_with_fake_spectre(tmp_path):
    from vatester.spectre.validate import validate
    spec = load_twin("vteam", ROOT)
    va = os.path.join(ROOT, "verilog", "vteam.va")
    rep, out = validate(va, None, ROOT, tmp_path / "ok", runner=_fake_runner(spec),
                        open_report=False, log=lambda *a: None)
    assert rep["verdict"] == "PASS" and rep["profile"] == "memristor"
    assert len(rep["layer1"]) == 5 and (out / "report.html").exists()
    assert (out / "layer2_convergence.png").exists() and (out / "l1_train.png").exists()
    assert "r_on" in rep["shared_params"]          # twin values passed to the instance
    rep, _ = validate(va, None, ROOT, tmp_path / "bad", runner=_fake_runner(spec, 1.05),
                      open_report=False, log=lambda *a: None)
    assert rep["verdict"] == "FAIL"                # 5 % current error is caught


# ---------------------------------------------------------------- setup questions
def test_parse_number_and_read_gate_suggestion():
    from vatester.spectre.setup import parse_number, suggest_read_gate
    assert parse_number("10n") == pytest.approx(1e-8)
    assert parse_number("1.5m") == pytest.approx(1.5e-3)
    assert parse_number("-3V") == -3.0 and parse_number("50p") == pytest.approx(5e-11)
    assert parse_number("1e-6s") == pytest.approx(1e-6)
    vg = np.linspace(-3, 6, 37)
    idd = np.where(vg > 0.5, (vg - 0.5) * 8e-6, 0.0)
    v = suggest_read_gate(vg, idd, 3.0)
    assert 0.5 < v <= 1.5                                # conducts, below half of 3 V
    assert suggest_read_gate(vg, np.zeros_like(vg), 3.0) is None


def test_interactive_setup_saves_answers(tmp_path):
    import shutil
    from vatester.spectre.setup import interactive_setup, load_answers
    va = tmp_path / "vteam.va"
    shutil.copy(os.path.join(ROOT, "verilog", "vteam.va"), va)
    replies = iter(["", "", "0.3", "-0.25", "20n", "nonsense", "-0.1"])
    a = interactive_setup(va, ROOT, ask=lambda q: next(replies), say=lambda *x: None)
    assert a["twin"] == "vteam" and a["profile"] == "memristor"
    assert a["amp_pos"] == 0.3 and a["width"] == pytest.approx(20e-9)
    assert a["read_level"] == -0.1                      # "nonsense" was asked again
    assert load_answers(va) == a
    from vatester.spectre.suites import build_suite
    su = build_suite(load_twin("vteam", ROOT), None, a["profile"], None, a["read_level"],
                     {k: a[k] for k in ("amp_pos", "amp_neg", "width")})
    assert su.calibration["V_switch_pos"] == 0.3 and su.width == pytest.approx(20e-9)
