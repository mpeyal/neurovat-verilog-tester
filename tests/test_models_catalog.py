"""GUI-free model catalogue: newest-Opus pin and its refresh rule."""
from vatester import models_catalog as mc


def _rows(*ids):
    return [("Default", None, "default")] + [(i, i, "x") for i in ids]


def test_newest_picks_opus_5_5():
    rows = _rows("claude-opus-4-8", "claude-opus-5", "claude-opus-5-5",
                 "claude-sonnet-5-5", "claude-fable-5-2")
    assert mc.newest(rows, "opus")[1] == "claude-opus-5-5"


def test_newest_numeric_and_future_major():
    assert mc.newest(_rows("claude-opus-5-9", "claude-opus-5-10"),
                     "opus")[1] == "claude-opus-5-10"
    assert mc.newest(_rows("claude-opus-5-5", "claude-opus-6"),
                     "opus")[1] == "claude-opus-6"


def test_newest_ignores_dated_snapshots():
    rows = _rows("claude-opus-5-5", "claude-opus-4-20250514",
                 "claude-opus-6-20270101")
    assert mc.newest(rows, "opus")[1] == "claude-opus-5-5"
    # a date is never read as a minor version, and dated ids are the
    # fallback only when nothing undated exists
    only_dated = _rows("claude-opus-4-20250514", "claude-opus-4-1-20250805")
    assert mc.newest(only_dated, "opus")[1] == "claude-opus-4-1-20250805"


def test_newest_none_when_family_missing():
    assert mc.newest(_rows("claude-sonnet-5-5"), "opus") is None
    assert mc.newest([], "opus") is None


def test_builtin_pin_is_opus_5_5():
    row = mc.newest(mc.BUILTIN["claude"], "opus")
    assert row[:2] == ("Opus 5.5", "claude-opus-5-5")
    assert mc.BUILTIN["claude"][0][1] is None        # Default stays first
    assert mc.resolve_pin(("Default", None), mc.BUILTIN["claude"],
                          False) == ("Opus 5.5", "claude-opus-5-5")


def test_live_list_moves_pin_only_without_hand_pick():
    live = mc.BUILTIN["claude"] + [("Opus 6", "claude-opus-6", "new")]
    cur = ("Opus 5.5", "claude-opus-5-5")
    assert mc.resolve_pin(cur, live, user_picked=False) == \
        ("Opus 6", "claude-opus-6")
    assert mc.resolve_pin(cur, live, user_picked=True) == cur
    picked = ("Sonnet 5.5", "claude-sonnet-5-5")
    assert mc.resolve_pin(picked, live, user_picked=True) == picked


def test_unlisted_selection_falls_back_to_default():
    rows = _rows("claude-sonnet-5-5")
    assert mc.resolve_pin(("Opus 5", "claude-opus-5"), rows, True) == \
        ("Default", None)
    # not picked, but the list has no Opus: same fallback rule
    assert mc.resolve_pin(("Opus 5.5", "claude-opus-5-5"), rows, False) == \
        ("Default", None)
    assert mc.resolve_pin(("Default", None), rows, True) == ("Default", None)


def test_failed_fetch_degrades_to_builtin(tmp_path, monkeypatch):
    def boom(api_key=None):
        raise RuntimeError("no credentials")
    monkeypatch.setitem(mc._FETCHERS, "claude", boom)
    rows, src = mc.list_models("claude", str(tmp_path), force=True)
    assert src == "builtin" and rows == mc.BUILTIN["claude"]
    assert "built-in list used" in mc.source_note(src)
