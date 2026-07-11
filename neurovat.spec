# -*- mode: python ; coding: utf-8 -*-
# PyInstaller ONE-FOLDER build of NeuroVAT (Windows).
#   Build:  pyinstaller --noconfirm --clean neurovat.spec
#   Output: dist/NeuroVAT/NeuroVAT.exe  (+ an _internal/ folder)
#
# The runtime plugin/data dirs (twins/, patterns/, studio/, root *.va) are the
# app's WORKSPACE - it looks for them next to the exe (cwd), not inside the
# bundle - so the release workflow copies them alongside the exe after this
# build. (Everything imported as a module, incl. ecfet/vatester, IS bundled.)
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# DearPyGui ships a native .pyd + DLL - pull all of it so the exe can render.
for pkg in ("dearpygui",):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# our own packages (some modules are imported lazily / by string name)
hiddenimports += collect_submodules("ecfet") + collect_submodules("vatester")

# optional deps - bundle if the build env has them, ignore if not
for opt in ("uharfbuzz", "freetype", "anthropic", "openai"):
    try:
        d, b, h = collect_all(opt)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["run_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="NeuroVAT",
    console=True,          # keep the console: tracebacks / agent output go here
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="NeuroVAT",
)
