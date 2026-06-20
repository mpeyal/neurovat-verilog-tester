"""Complex-script text shaping for the chat (Bengali, Devanagari, ...).

DearPyGui / ImGui render glyphs left-to-right with no shaping, so Indic scripts
come out with detached vowel signs and unformed conjuncts. This module shapes a
string with HarfBuzz, rasterises the shaped glyphs with FreeType, word-wraps to
a width, and returns an RGBA image the GUI can show as a texture instead.

Everything degrades gracefully: if uharfbuzz / freetype / a covering font are
missing, TextShaper.ok is False and the caller falls back to plain dpg.add_text.
"""

import os

try:
    import numpy as np
    import uharfbuzz as hb
    import freetype
    _DEPS = True
except Exception:                       # noqa: BLE001 - any import problem
    _DEPS = False

# scripts that need shaping (ImGui can't): Arabic..Hangul-Jamo + CJK and beyond.
# Plain Latin/digits stay on the fast dpg.add_text path.
def needs_shaping(s):
    return any(0x0600 <= ord(c) <= 0x10FF or ord(c) >= 0x1100 for c in (s or ""))


_FONT_CANDIDATES = ("Nirmala.ttc", "Nirmala.ttf", "kalpurush.ttf",
                    "NirmalaS.ttf")


def _find_font():
    fdir = r"C:\Windows\Fonts"
    for fn in _FONT_CANDIDATES:
        p = os.path.join(fdir, fn)
        if os.path.isfile(p):
            return p
    return None


class TextShaper:
    def __init__(self, size=17, font_path=None, face_index=0):
        self.ok = False
        if not _DEPS:
            return
        path = font_path or _find_font()
        if not path:
            return
        try:
            blob = hb.Blob.from_file_path(path)
            self._hbface = hb.Face(blob, face_index)
            self._hbfont = hb.Font(self._hbface)
            self._hbfont.scale = (size * 64, size * 64)
            self._ft = freetype.Face(path, face_index)
            self._ft.set_pixel_sizes(0, size)
            self._ascent = self._ft.size.ascender >> 6
            self._descent = (-self._ft.size.descender) >> 6
            self._line_h = self._ascent + self._descent + 2
            self.ok = True
        except Exception:               # noqa: BLE001
            self.ok = False

    # -- shaping helpers ------------------------------------------------
    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self._hbfont, buf)
        return buf.glyph_infos, buf.glyph_positions

    def _width_px(self, text):
        if not text:
            return 0
        _, pos = self._shape(text)
        return int(sum(p.x_advance for p in pos) / 64)

    def _wrap(self, text, max_width):
        lines = []
        for para in text.split("\n"):
            if not para.strip():
                lines.append("")
                continue
            cur = ""
            for word in para.split(" "):
                cand = word if not cur else cur + " " + word
                if not cur or self._width_px(cand) <= max_width:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    def _blit_line(self, canvas, line, y_top):
        if not line:
            return
        infos, positions = self._shape(line)
        pen = 2.0
        baseline = y_top + self._ascent
        H, W = canvas.shape
        for info, pos in zip(infos, positions):
            self._ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            bmp = self._ft.glyph.bitmap
            w, h = bmp.width, bmp.rows
            if w and h:
                arr = np.array(bmp.buffer, dtype=np.uint8).reshape(h, w)
                x = int(pen + pos.x_offset / 64 + self._ft.glyph.bitmap_left)
                y = int(baseline - pos.y_offset / 64 - self._ft.glyph.bitmap_top)
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 > x0 and y1 > y0:
                    canvas[y0:y1, x0:x1] = np.maximum(
                        canvas[y0:y1, x0:x1],
                        arr[y0 - y:y1 - y, x0 - x:x1 - x])
            pen += pos.x_advance / 64

    def render(self, text, color=(225, 228, 236), max_width=300):
        """Shape+rasterise `text`, wrapped to max_width. Returns
        (rgba_float32_flat, width, height) for a DPG static texture, or None."""
        if not self.ok or not text:
            return None
        try:
            max_width = max(40, int(max_width))
            lines = self._wrap(text, max_width)
            W = min(max_width, max((self._width_px(ln) for ln in lines),
                                   default=1) + 4) or 1
            W = max(W, 1)
            H = max(1, len(lines) * self._line_h + 2)
            canvas = np.zeros((H, W), dtype=np.uint8)
            for i, ln in enumerate(lines):
                self._blit_line(canvas, ln, i * self._line_h + 1)
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[..., 0] = color[0] / 255.0
            rgba[..., 1] = color[1] / 255.0
            rgba[..., 2] = color[2] / 255.0
            rgba[..., 3] = canvas.astype(np.float32) / 255.0
            return rgba.reshape(-1), W, H
        except Exception:               # noqa: BLE001
            return None
