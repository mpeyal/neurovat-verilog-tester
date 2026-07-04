# NeuroVAT Studio

Web/desktop front-end for the NeuroVAT synaptic-device tester. One HTML UI, one
shared Python backend, two ways to run it.

**Connected mode (this repo):** when studio/ sits inside the NeuroVAT repo —
as it does here — the backend automatically uses the REAL app engine:

* `run_sim` runs the actual `ecfet` models (EcfetV2 / EcfetV3 / FeFET), the
  same physics the Dear PyGui app simulates (see `core/engine.py`).
* `compile_va` uses the app's OpenVAF locator (`vatester.tools_ext`), so it
  finds the same binary the desktop Tools menu uses.
* the Virtuoso panel's **Load source / Write back** read and write the repo's
  real root `.va` files (`ecfet_v2.va`, `ecfet_v3.va`, `fefet_v1.va`,
  `ECFET_Synapse.va`) — the same sources the desktop app scans.

Launch it from the repo root with:

```bash
python run_gui.py --web            # http://127.0.0.1:8000, opens browser
python run_gui.py --web --port 9000 --no-open
```

If studio/ is copied out on its own, everything below still works — each seam
falls back to the bundled standalone implementation in `core/`.

```
studio/
├── index.html        ← the whole UI, self-contained (no internet needed)
├── run.py            ← launcher  →  python run.py
├── server.py         ← zero-dependency web server (stdlib only)
├── desktop.py        ← optional native window (pywebview)
├── bridge.py         ← the API the UI calls  (run_sim, compile_va, …)
├── core/
│   ├── twin.py       ← default physics engine (real Python, works out of the box)
│   ├── openvaf.py    ← OpenVAF compile seam  (auto-used if `openvaf` on PATH)
│   └── virtuoso.py   ← Cadence Virtuoso seam (auto-used if skillbridge reachable)
├── va/ecfet_v2.va    ← sample Verilog-A source
└── requirements.txt
```

## Editing the UI

`index.html` is a self-contained **bundle**: the fonts and app JS are base64
blobs and the whole UI page is stored as one JSON-encoded string inside a
`<script type="__bundler/template">` tag. Don't hand-edit that string. Instead:

1. edit **`ui_src.html`** (the readable UI source — HTML + the `DCLogic`
   component),
2. run **`python rebuild_ui.py`** (re-embeds `ui_src.html` into `index.html`),
3. reload the page.

`ui_src.html` is the source of truth; `index.html` is generated.

## Run it (no installs)

You only need **Python 3.7+**. From this folder:

```bash
python run.py
```

That starts a local server on `http://127.0.0.1:8000/` and opens your browser.
Every button, slider and plot is now driven by the Python backend. Stop with
Ctrl+C.

Options:

```bash
python run.py --port 9000     # different port
python run.py --no-open       # don't auto-open the browser
python run.py --desktop       # native window instead (needs: pip install pywebview)
```

## How it fits together

The UI talks to Python through a single seam — `backend.call("<name>", args)` —
which resolves to:

* **desktop:** an in-process call to `pywebview`'s `js_api` (no server), or
* **server:** `POST /api/<name>` handled by `server.py`.

Either way it lands on the matching method in `bridge.py`. If the page is opened
with **no** backend at all (double-click `index.html`), it falls back to an
in-page JavaScript twin so it still runs as a demo.

### The four calls the UI makes

| UI action                     | Bridge method          | Returns                                   |
|-------------------------------|------------------------|-------------------------------------------|
| **Run**                       | `run_sim`              | `{stim, gts, ana, Gfinal}` (plot data)    |
| **Compile .va / Build & run** | `compile_va`           | `{ok, params, osdi, engine}`              |
| Virtuoso ▸ **Load source**    | `load_from_virtuoso`   | `{source}` (Verilog-A text)               |
| Virtuoso ▸ **Write back**     | `write_to_virtuoso`    | `{ok}`                                     |

The exact shapes are pinned in the front-end next to each call as
`// CONTRACT:` comments, and implemented in `core/`.

## Making the plots come from *your* equations (OpenVAF)

Out of the box, `run_sim` uses the behavioural twin in `core/twin.py` — real
Python, but analytic. To drive the plots from the actual compiled `.va`:

1. Install OpenVAF (standalone binary) and put `openvaf` on your PATH.
   `compile_va` will then build a real `.osdi` (see `core/openvaf.py`).
2. Implement `core/openvaf.py:run_osdi()` against a Verilog-A–capable simulator
   (ngspice+OSDI, Xyce, or Spectre), returning the same dict shape as
   `twin.simulate()`.
3. In `bridge.py:run_sim`, prefer that compiled path and fall back to the twin
   only when no `.osdi` exists (there's a `TODO` marking the spot).

Nothing in `index.html` changes — the moment `run_sim` returns real data, the
plots are real.

## Cadence Virtuoso

`core/virtuoso.py` uses `skillbridge` when a Virtuoso workspace is reachable and
otherwise reads/writes the local `va/` folder, so the Load/Write-back buttons
work with or without Cadence.

## Keeping the classic Dear PyGui UI too

`core/` has no UI in it, so your existing Dear PyGui app can `import core` and
share the exact same physics/compile/Virtuoso code. Two front-ends, one brain:
run the DPG app as before, or `python run.py` for this one.
