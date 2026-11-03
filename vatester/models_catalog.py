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
        ("Opus 5", "claude-opus-5", "most capable"),
        ("Sonnet 5", "claude-sonnet-5", "fast + smart"),
        ("Haiku 4.5", "claude-haiku-4-5", "fastest, cheapest"),
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
