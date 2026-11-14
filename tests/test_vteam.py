"""VTEAM memristor: twin equations, .va/twin parameter sync, MemTorch-order
Euler mode, vteam-1 stimuli, decks (--step) and the Spectre importer."""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.vteam import core, make_decks, scrub  # noqa: E402
from validation.vteam import stimuli as S  # noqa: E402

PINNED = "3103a410f6b1223a"      # bump STIMULUS_SET_VERSION + this when the set changes
m = core.twin_module()

# MemTorch 1.1.6 memtorch/bh/memristor/VTEAM.py __init__ defaults (= paper
# Table II, column "TEAM [9]")
MEMTORCH_DEFAULTS = dict(r_off=1000, r_on=50, d=3e-9, k_on=-10, k_off=5e-4, alpha_on=3,
                         alpha_off=1, v_on=-0.2, v_off=0.02, x_on=0, x_off=3e-9)


def _run(pts, **over):
    d = m.VTEAM(m.VTEAMParams(**over))
    for (ta, va), (tb, vb) in zip(pts[:-1], pts[1:]):
        d.step_linear(ta, tb - ta, va, vb)
    return d


def test_defaults_are_memtorch_defaults():
    P = m.VTEAMParams()
    for k, v in MEMTORCH_DEFAULTS.items():
        assert getattr(P, k) == pytest.approx(v, rel=1e-15), k
    assert P.x_init == P.x_on
    assert P.v_on < P.V_read < P.v_off


def test_va_parameters_match_twin():
    src = (ROOT / "verilog" / "vteam.va").read_text(encoding="utf-8")
    va = {n: float(v) for n, v in re.findall(
        r"parameter\s+real\s+(\w+)\s*=\s*([-+0-9.eE]+)", src)}
    P = m.VTEAMParams()
    tw = {f: getattr(P, f) for f in P.__dataclass_fields__}
    assert set(va) == set(tw) - {"V_read", "integrator"}
    for k, v in va.items():
        assert tw[k] == pytest.approx(v, rel=1e-15), k
    code = "\n".join(ln.split("//")[0] for ln in src.splitlines())
    assert "$bound_step" not in code and "idt(" in code


def test_twin_spec_loads():
    from vatester import twin_loader
    got = {Path(p).name: (payload, err) for p, payload, err in
           twin_loader.load_twins(str(ROOT / "twins"))}
    payload, err = got["vteam.py"]
    assert err is None
    spec = payload[1]
    assert spec["key"] == "vteam" and spec["input_kind"] == "voltage"
    d = spec["model_class"](spec["params_class"]())
    d.step(0.0, 1e-9, 0.5)
    assert 0 < d.x < d.p.d and d.G > 0 and "x (nm)" in d.observables()


def test_constant_voltage_closed_form():
    P = m.VTEAMParams()
    d = _run([(0, 0.8), (10e-9, 0.8)])
    assert d.x == pytest.approx(5e-4 * (0.8 / 0.02 - 1) * 10e-9, rel=1e-13)
    d = _run([(0, -0.225), (10e-9, -0.225)], x_init=P.d)
    assert P.d - d.x == pytest.approx(10 * 0.125 ** 3 * 10e-9, rel=1e-12)


def test_ramp_through_threshold_matches_fine_euler():
    # 0 -> 0.8 V in one step: crosses v_off inside the step
    d = _run([(0, 0.0), (1e-9, 0.8)])
    n = 200000
    t = (np.arange(n) + 0.5) / n * 1e-9
    v = 0.8 * t / 1e-9
    r = np.where(v > 0.02, 5e-4 * (v / 0.02 - 1), 0.0)
    assert d.x == pytest.approx(float(np.sum(r) * 1e-9 / n), rel=1e-8)
    # same for the cubic ON side
    d = _run([(0, 0.0), (1e-9, -0.4)], x_init=3e-9)
    v = -0.4 * t / 1e-9
    r = np.where(v < -0.2, -10 * (v / -0.2 - 1) ** 3, 0.0)
    assert 3e-9 - d.x == pytest.approx(float(-np.sum(r) * 1e-9 / n), rel=1e-7)


def test_result_independent_of_step_split():
    pts = [(0, 0), (1e-10, 0.9), (5e-9, 0.9), (5.1e-9, 0), (6e-9, -0.3), (9e-9, -0.1)]
    a = _run(pts)
    fine = []
    for (ta, va), (tb, vb) in zip(pts[:-1], pts[1:]):
        for k in range(37):
            fine.append((ta + (tb - ta) * k / 37, va + (vb - va) * k / 37))
    fine.append(pts[-1])
    b = _run(fine)
    assert b.x == pytest.approx(a.x, rel=1e-11, abs=1e-24)


def test_dead_zone_and_read_do_not_move_state():
    for v in (0.0, 0.02, 0.01, -0.1, -0.2, S.V_READ):
        d = _run([(0, v), (1e-6, v)], x_init=1.3e-9)
        assert d.x == 1.3e-9


def test_window_bounds_and_no_windup():
    P = m.VTEAMParams()
    d = m.VTEAM(P)
    d.step(0, 1e-6, 1.0)                    # 8x the travel needed to reach d
    assert d.p.d * (1 - 1e-12) < d.x <= d.p.d
    x_sat = d.x
    d.step(0, 10e-9, -0.22)                 # reverse at once: 3.3 % of d
    assert x_sat - d.x == pytest.approx(10 * 0.1 ** 3 * 10e-9, rel=1e-3)
    d.step(0, 1e-6, -1.0)
    assert 0 <= d.x < 1e-12 * P.d
    # inside the edge band the approach is exponential (= the .va window)
    d = m.VTEAM(m.VTEAMParams(x_init=P.d * (1 - 0.5 * P.w_eps)))
    d.step(0, 1e-9, 0.8)
    q = 5e-4 * 39 * 1e-9 / P.d
    assert 1 - d.x / P.d == pytest.approx(0.5 * P.w_eps * math.exp(-q / P.w_eps), rel=1e-9)
    # w_eps = 0 is the hard clip
    d = m.VTEAM(m.VTEAMParams(w_eps=0.0))
    d.step(0, 1e-6, 1.0)
    assert d.x == P.d


def _memtorch_restated(P, x0, V, dt):
    """MemTorch VTEAM.simulate() restated: x += dxdt(v) dt; clip to [0, d];
    current from the new x."""
    x, out = x0, []
    for v in V:
        if v >= P.v_off:
            r = P.k_off * ((v / P.v_off) - 1) ** P.alpha_off
        elif v <= P.v_on:
            r = P.k_on * ((v / P.v_on) - 1) ** P.alpha_on
        else:
            r = 0
        x = x + r * dt
        if x >= P.d or x <= 0:
            x = P.d if x >= P.d else 0
        out.append(v / (P.r_off * x / P.d + P.r_on * (1 - x / P.d)))
    return np.array(out)


def test_euler_mode_is_memtorch_order():
    st = S.BY_NAME["s2_vteam"]
    dt = 1e-10
    n = int(round(st.t_stop / dt))
    V = st.v((np.arange(n) + 0.5) * dt)
    P = core.params(st)
    ref = _memtorch_restated(P, P.x_init, V, dt)
    d = core.new_twin(st, integrator=1)
    got = []
    for k, v in enumerate(V):
        d.step(k * dt, dt, v)
        got.append(d.current(v))
    assert np.max(np.abs(np.array(got) - ref)) <= 1e-12 * np.max(np.abs(ref))
    try:
        import memtorch.bh.memristor as mb
    except Exception:            # noqa: BLE001 - optional GPL-3 dependency
        return
    mt = mb.VTEAM(time_series_resolution=dt, **MEMTORCH_DEFAULTS)
    old = np.geterr()
    try:
        i = np.asarray(mt.simulate(list(V), return_current=True))
    finally:
        np.seterr(**old)
    assert np.max(np.abs(i - ref)) <= 1e-12 * np.max(np.abs(ref))


# ---- stimuli / decks / importer ------------------------------------------
def test_fingerprint_pinned_and_stable():
    assert S.STIMULUS_SET_VERSION == "vteam-1"
    assert S.fingerprint(S.build_stimuli()) == S.fingerprint()
    assert S.fingerprint()[:16] == PINNED


@pytest.mark.parametrize("st", S.STIMULI, ids=lambda s: s.name)
def test_pwl_round_trip_and_reads(st):
    back = S.parse_spectre_pwl(st.spectre_pwl())
    assert np.allclose(back, st.pwl_points(), rtol=1e-12, atol=0)
    if len(st.reads):
        assert np.allclose(st.v(st.read_times), S.V_READ)
    ts = core.grid(st)
    assert set(np.round(st.t_points / 1e-15)) <= set(np.round(ts / 1e-15))


def test_twin_self_check_is_exact():
    st = S.BY_NAME["s1_vteam"]
    a, _, _ = core.run_twin(st)
    b, _, _ = core.run_twin(st, refine=3)
    assert np.max(np.abs(a["s_state"] - b["s_state"][::3])) < 1e-12
    r = core.score(a, b, st)
    assert core.verdict(r) == "PASS" and r["g_read_max_pct"] < 1e-8


def test_decks_generated_equal_committed(tmp_path):
    out = tmp_path / "decks"
    make_decks.main(["--out", str(out), "--va-dir", "../../../verilog"])
    for st in S.STIMULI:
        assert (out / f"{st.name}.scs").read_text() == \
            (core.DECKS / f"{st.name}.scs").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_make_decks_step_variant(tmp_path):
    out = tmp_path / "fine"
    make_decks.main(["--step", "1e-12", "--out", str(out)])
    txt = (out / "s1_vteam.scs").read_text()
    assert "maxstep=1e-12" in txt and "reltol=1e-06" in txt
    assert "x_init" not in txt.split("X1 (n_p 0)")[1].splitlines()[0]
    on = (out / "on_train.scs").read_text()
    assert "x_init=3e-09" in on
    assert make_decks.step_tag(1e-11) == "10ps" and make_decks.step_tag(1e-12) == "1ps"
    assert make_decks.step_tag(1e-13) == "01ps"


def test_scrub_canonical_and_leak_check():
    st = S.BY_NAME["s1_vteam"]
    tw, _, _ = core.run_twin(st)
    sig = {"time": tw["t_s"], "n_p": tw["V_V"], "X1:p": tw["I_A"], "X1.ws": tw["s_state"]}
    cols = scrub.to_canonical(sig, st)
    assert np.allclose(cols["x_m"], tw["x_m"], rtol=0, atol=1e-20)
    sig2 = {"time": tw["t_s"], "n_p": tw["V_V"], "vin:p": -tw["I_A"]}
    c2 = scrub.to_canonical(sig2, st)
    ok = np.isfinite(c2["x_m"])
    assert ok.sum() > 100 and np.allclose(c2["x_m"][ok], tw["x_m"][ok], atol=1e-15)
    lines = scrub.render(cols, {"version": "24.1.0"})
    scrub.check_clean(lines)
    bad = list(lines)
    bad[0] = bad[0].replace("model: vteam", "model: vteam host")
    with pytest.raises(scrub.ScrubError):
        scrub.check_clean(bad)
    sig["X1:p"] = -tw["I_A"]
    with pytest.raises(scrub.ScrubError):
        scrub.to_canonical(sig, st)
