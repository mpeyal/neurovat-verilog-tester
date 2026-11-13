"""Live model catalogue for the agent chat's model picker.

The hardcoded model lists in app.py go stale every time a provider ships a new
model - the user has to wait for an app update to pick it. This module asks each
provider what it *actually* offers right now, caches the answer on disk, and
falls back to a built-in list when offline or unauthenticated.

    choices, source = list_models("claude", workdir)
    # choices = [(label, model_id_or_None, blurb), ...]   source = live|cache|builtin

Design notes:
  * Every fetch is best-effort and MUST NOT raise - the picker has to open even
    with no network, no API key and no SDK installed. Failures degrade
    live -> cache -> builtin silently (the `source` return says which).
  * Nothing here touches DearPyGui, so it is unit-testable headless and safe to
    call from a worker thread (app.py must not call it on the render thread -
    it does network I/O).
  * The "Default" entry (model id None) always stays first: it means "whatever
    the CLI/SDK is already configured to use", which is the safest choice and
    never goes stale.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

CACHE_TTL = 24 * 3600           # refetch at most once a day
_FETCH_TIMEOUT = 15

# Built-in fallbacks. These are the LAST resort - correct at the time of
# writing, and deliberately short. The whole point of this module is that the
# live list, not this one, is what the user normally sees.
BUILTIN = {
    "claude": [
        ("Default", None, "your Claude Code default"),
        ("Opus (latest)", "opus", "newest Opus your Claude Code knows"),
        ("Opus 5.5", "claude-opus-5-5", "most capable everyday"),
        ("Fable 5.2", "claude-fable-5-2", "most capable, priciest"),
        ("Sonnet 5.5", "claude-sonnet-5-5", "fast + smart"),
        ("Haiku 5.5", "claude-haiku-5-5", "fastest, cheapest"),
        ("Opus 5", "claude-opus-5", "previous Opus"),
    ],
    "openai": [
        ("Default", None, "your codex default"),
        ("GPT-5.1", "gpt-5.1", "flagship"),
        ("GPT-5.1 mini", "gpt-5.1-mini", "fast, cheap"),
    ],
    "gemini": [
        ("Default", None, "your gemini CLI default"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro", "most capable"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash", "fast, cheap"),
    ],
}

PROVIDERS = tuple(BUILTIN)


# ------------------------------------------------------- default-model pin
# The app pins Claude to the newest Opus it can see, so the pin follows the
# list (builtin, cache or live) instead of a version hard-coded in app.py.

_VERSION_RE = re.compile(
    r"^claude-(?P<fam>[a-z]+)-(?P<major>\d{1,3})(?:-(?P<minor>\d{1,3}))?"
    r"(?:-(?P<date>\d{8}))?$")


def _parse_id(mid):
    """'claude-opus-5-5' -> ('opus', (5, 5), None); dated snapshot ids keep
    their 8-digit date separately so it is never read as a minor version."""
    m = _VERSION_RE.match(mid or "")
    if not m:
        return None
    ver = (int(m.group("major")), int(m.group("minor") or 0))
    return m.group("fam"), ver, m.group("date")


def newest(rows, family="opus"):
    """Row (label, id, blurb) with the highest-versioned claude-<family> id,
    or None. Versions compare numerically (5-10 > 5-9, 6 > 5-5). Dated
    snapshot ids are used only when no undated id of that family exists."""
    plain, dated = [], []
    for row in rows or ():
        p = _parse_id(row[1] if len(row) > 1 else None)
        if not p or p[0] != family:
            continue
        (dated if p[2] else plain).append((p[1], p[2] or "", row))
    pool = plain or dated
    if not pool:
        return None
    return max(pool, key=lambda t: (t[0], t[1]))[2]


def resolve_pin(current, rows, user_picked, family="opus"):
    """Selection (label, id-or-None) after a model list arrives.

    Not picked by hand this session -> the `family` alias row when the list
    has one (the built-in list: the Claude CLI resolves the alias "opus" to
    the newest Opus it knows, and the CLI updates itself, so a new Opus is
    picked up with no app change), else the newest `family` model in `rows`
    (a live list, which is already current).
    Otherwise (or when the list has none) keep `current` while the list still
    offers it, and fall back to Default when it no longer does - never send
    an id the provider stopped listing."""
    if not user_picked:
        alias = next((r for r in rows or () if len(r) > 1 and r[1] == family),
                     None)
        best = alias or newest(rows, family)
        if best:
            return best[0], best[1]
    label, mid = current or ("Default", None)
    if mid is None or any(len(r) > 1 and r[1] == mid for r in rows or ()):
        return label, mid
    return "Default", None


ALIASES = ("opus", "sonnet", "haiku", "fable")


def api_model_id(model, api_key=None, fallback="claude-opus-5-5"):
    """Concrete id for the Messages API, which does not accept CLI aliases.

    'opus' (or Default) -> the newest Opus the Models API lists right now;
    an explicit id passes through. Never raises: offline or unauthenticated
    it returns the newest Opus in the built-in list, then `fallback`."""
    if model and model != "default" and model not in ALIASES:
        return model
    family = model if model in ALIASES else "opus"
    try:
        row = newest(_fetch_claude(api_key), family)
        if row:
            return row[1]
    except Exception:
        pass
    row = newest(BUILTIN["claude"], family)
    return row[1] if row else fallback


def source_note(source, provider="claude"):
    """Human-readable reason for which model list is in use (for the log)."""
    no_auth = ("no API key / ant profile" if provider == "claude"
               else "no API key")
    return {
        "live": "live list from the provider",
        "cache": "cached provider list",
        "builtin": f"{no_auth}; built-in list used",
    }.get(source, source)


def _cache_file(workdir):
    return os.path.join(workdir, "results", "model_cache.json")


def _load_cache(workdir):
    try:
        with open(_cache_file(workdir), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(workdir, cache):
    path = _cache_file(workdir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1)
    except OSError:
        pass                    # a read-only workspace just means no caching


# ---------------------------------------------------------------- fetchers
# Each returns [(label, id, blurb)] WITHOUT the Default row, or raises. The
# caller catches everything, so these can be written for the happy path.

def _fetch_claude(api_key=None):
    """Anthropic Models API. Needs a key or an `ant auth login` profile."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key) if api_key \
        else anthropic.Anthropic()
    out = []
    for m in client.models.list():          # SDK auto-paginates
        label = getattr(m, "display_name", None) or m.id
        ctx = getattr(m, "max_input_tokens", None)
        blurb = f"{ctx // 1000}K context" if ctx else m.id
        out.append((label, m.id, blurb))
    return out


def _fetch_openai(api_key=None):
    """OpenAI Models API, filtered to chat-capable families."""
    import openai
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    out = []
    for m in client.models.list():
        mid = m.id
        # keep chat models; drop embeddings/audio/image/moderation endpoints
        if not mid.startswith(("gpt-", "o1", "o3", "o4")):
            continue
        if any(k in mid for k in ("embedding", "audio", "realtime", "image",
                                  "tts", "whisper", "moderation", "transcribe")):
            continue
        out.append((mid, mid, "openai"))
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def _fetch_gemini(api_key=None):
    """Google Generative Language models endpoint (plain REST, no SDK dep)."""
    key = api_key or os.environ.get("GEMINI_API_KEY") \
        or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY / GOOGLE_API_KEY")
    url = ("https://generativelanguage.googleapis.com/v1beta/models"
           "?key=" + urllib.parse.quote(key))
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for m in data.get("models", []):
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        mid = (m.get("name") or "").replace("models/", "")
        if not mid:
            continue
        out.append((m.get("displayName") or mid, mid,
                    m.get("description", "")[:40] or "gemini"))
    return out


_FETCHERS = {"claude": _fetch_claude, "openai": _fetch_openai,
             "gemini": _fetch_gemini}


# ------------------------------------------------------------------- api

def list_models(provider, workdir, api_key=None, force=False):
    """Model choices for `provider`, newest info available.

    Returns (choices, source) with source one of "live" / "cache" / "builtin".
    Never raises: a provider that can't be reached degrades to the cache, then
    to the built-in list.
    """
    provider = (provider or "").lower()
    if provider not in BUILTIN:
        return [], "builtin"
    default_row = BUILTIN[provider][0]
    cache = _load_cache(workdir)
    entry = cache.get(provider) or {}
    fresh = (time.time() - entry.get("ts", 0)) < CACHE_TTL

    if not force and fresh and entry.get("models"):
        return [default_row] + [tuple(r) for r in entry["models"]], "cache"

    try:
        rows = _FETCHERS[provider](api_key)
    except Exception:
        rows = None
    if rows:
        cache[provider] = {"ts": time.time(), "models": [list(r) for r in rows]}
        _save_cache(workdir, cache)
        return [default_row] + rows, "live"

    if entry.get("models"):                 # stale cache beats the builtin list
        return [default_row] + [tuple(r) for r in entry["models"]], "cache"
    return list(BUILTIN[provider]), "builtin"


def refresh_all(workdir, keys=None):
    """Force-refetch every provider. Returns {provider: source}."""
    keys = keys or {}
    return {p: list_models(p, workdir, api_key=keys.get(p), force=True)[1]
            for p in PROVIDERS}
