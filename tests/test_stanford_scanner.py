"""va_scan keyword mapping: most specific keyword wins, independent of the
order in which twins register their hints (registry inserts at the front)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vatester import va_scan  # noqa: E402

BUILTIN = [("fefet", ("fefet", "ferro", "hzo")),
           ("v3", ("v3", "paper")),
           ("v2", ("v2", "ecram", "practical")),
           ("v1", ("v1", "basic", "drft", "ecfet"))]
TWINS = [("rram_demo", ("rram", "reram")),
         ("fecap_nls", ("fecap", "ferrocap", "fe_cap")),
         ("fecap_nls_dr", ("fecap_v2_dr",)),
         ("stanford_pku", ("stanford_pku", "stanford_rram", "pku_rram", "rram_v_1_0_0",
                           "rram_random_v_1_0_0"))]

CASES = [
    # (file name, module name, expected key after the version guard)
    ("stanford_pku_rram.va", "rram_v_1_0_0", "stanford_pku"),
    ("rram_v_1_0_0.va", "rram_v_1_0_0", "stanford_pku"),          # nanoHUB name
    ("rram_random_v_1_0_0.va", "rram_v_1_0_0", "stanford_pku"),
    ("example_rram.va", "example_rram", "rram_demo"),
    ("ecfet_v2.va", "ecfet_v2", "v2"),
    ("ecfet_v3.va", "ecfet_v3", "v3"),
    ("ecfet_v4.va", "ecfet_v4", ""),                              # unknown version
    ("basic_v1_Diffu_Drft_verilog.va", "basic_v1", "v1"),
    ("FeFET.va", "FeFET", "fefet"),
    ("FeCAP_v2.va", "FeCAP_v2", "fecap_nls"),
    ("FeCAP_v2_DR.va", "FeCAP_v2_DR", "fecap_nls_dr"),
    # generic memristor files must NOT fall onto the ECFET v1 port
    ("asic2_memristor_model.va", "Memristor", ""),
    ("diffusive_memristor.va", "diffusive_memristor", ""),
    ("vteam.va", "memristor", ""),
]


def _write(tmp_path, fname, module):
    p = tmp_path / fname
    p.write_text(f"`include \"disciplines.vams\"\nmodule {module}(p, n);\n"
                 "parameter real a = 1.0;\nendmodule\n", encoding="utf-8")
    return p


@pytest.mark.parametrize("order", ["twins_first", "twins_last", "twins_reversed"])
@pytest.mark.parametrize("fname,module,expected", CASES)
def test_mapping_independent_of_hint_order(monkeypatch, tmp_path, order, fname, module,
                                           expected):
    if order == "twins_first":
        hints = TWINS + BUILTIN
    elif order == "twins_last":
        hints = BUILTIN + TWINS
    else:
        hints = TWINS[::-1] + BUILTIN
    monkeypatch.setattr(va_scan, "MODEL_HINTS", list(hints))
    va = va_scan.parse_va(str(_write(tmp_path, fname, module)))
    assert va.model_key == expected


def test_match_hint_longest_and_tie_order():
    hints = [("a", ("rram",)), ("b", ("rram_v_1_0_0",)), ("c", ("xxxx",))]
    assert va_scan.match_hint("rram_v_1_0_0.va", hints) == "b"
    assert va_scan.match_hint("my_rram.va", hints) == "a"
    assert va_scan.match_hint("rram xxxx", hints) == "a"          # tie -> earlier entry
    assert va_scan.match_hint("nothing.va", hints) == ""


def test_builtin_hints_drop_generic_memristor_words():
    words = {w for _k, ws in va_scan.MODEL_HINTS for w in ws}
    assert "memristor" not in words and "diffu" not in words


def test_stanford_twin_spec_keywords():
    from validation.stanford import core
    spec = core.twin_module().TWIN_SPEC
    assert spec["key"] == "stanford_pku"
    assert "rram_v_1_0_0" in spec["va_keywords"]
    assert all(len(w) > len("rram") for w in spec["va_keywords"])
