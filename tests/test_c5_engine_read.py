"""Acceptance of the physical-time engine fix against Spectre: the committed
C5 tier-1 schedule (16 v3 synapses, 4 s, 3608 pulses) replayed through the
engine's device code (program at each write time, peek every 1 ms) reads
what Spectre computes, inside and outside write pulses: Gate A rule
(NRMSE <= 1 %, max <= 3 % of each synapse's Spectre G range) on 16/16."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "validation" / "enginefix")]


@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    import replay_c5 as rc
    out = tmp_path_factory.mktemp("c5_fixed")
    rc.replay(out=out)
    try:
        return rc.score(out=out, figures=False)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_engine_full_read_passes_gate_a_on_every_synapse(accepted):
    rows, summ = accepted
    assert summ["synapses"] == 16 and summ["engine_pass"] == 16
    for r in rows:
        assert r["engine_nrmse_pct"] <= 1.0 and r["engine_max_pct"] <= 3.0
        # in-pulse and out-of-pulse reads separately (C5 measured 52 % and
        # 4.9 % max before the fix); the replay is far inside the rule
        assert r["engine_in_max_pct"] <= 0.05 and r["engine_out_max_pct"] <= 0.05
        assert r["in_pulse_reads"] > 1000
        assert abs(r["final_G_engine_err_pct"]) <= 0.05


def test_twin_replay_still_matches_spectre(accepted):
    rows, summ = accepted
    assert summ["twin_pass"] == 16
