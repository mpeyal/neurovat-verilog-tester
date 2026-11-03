"""Network Studio data controls, spike previews and pattern preparation.

App supplies widget helpers, trainer configuration and the event queue. Dataset
workers only post queue messages; rendering stays on the main thread. File
validation and download policy remain in datasets.py and the plugin loaders.
"""
import glob
import os
import threading

import dearpygui.dearpygui as dpg
import numpy as np

from . import neuro
from .registry import BUILTIN_PATSETS, PATTERN_PLUGINS, register_pattern_plugins
from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2


class NetworkDataMixin:
    def _nt_patterns_tab(self):
        _pre = lambda *_: self._nt_encoding_preview()
        with dpg.child_window(border=False, tag="nt_data_scroll", no_scrollbar=True, no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                dpg.add_combo(["Pattern set", "Custom painted", "Imported"], tag="nt_source_selector",
                              default_value="Pattern set", width=140,
                              callback=lambda s, v: self._nt_set_source({"Pattern set": "set", "Custom painted": "custom", "Imported": "dataset"}[v]))
                source = dpg.add_button(label="Configure source", tag="nt_source_setup")
                noise = dpg.add_button(label="Noise settings", tag="nt_noise_setup")
            with self.desktop.popup(source, tag="nt_source_popup"):
                with self._card("Custom pattern", tag="nt_custom_card", width=540, height=300,
                                tip="Click cells to paint a pattern, then "
                                    "'Add pattern' to store it. Pick 'Custom "
                                    "painted' above to train on them."):
                    self._small("click cells to paint - stored patterns "
                                "become the training set", color=C_MUTED)
                    with dpg.group(horizontal=True):
                        dl = dpg.add_drawlist(width=200, height=200,
                                              tag="nt_pat_editor")
                        dpg.add_draw_layer(parent=dl, tag="nt_pat_layer")
                        dpg.add_spacer(width=6)
                        with dpg.group():
                            dpg.add_button(label="New grid", width=150,
                                           callback=self._nt_paint_new)
                            dpg.add_button(label="Clear", width=150,
                                           callback=self._nt_paint_clear)
                            b = dpg.add_button(label="Add pattern", width=150,
                                               callback=self._nt_paint_add)
                            dpg.bind_item_theme(b, self.themes["primary"])
                            dpg.add_button(label="Drop last", width=150,
                                           callback=self._nt_paint_drop)
                            dpg.add_text("custom patterns: 0",
                                         tag="nt_custom_txt", color=C_TEXT2)
                    self._small("left-click / drag paints ON, right-click "
                                "erases", color=C_MUTED)
                with self._card("Import dataset", tag="nt_dataset_card",
                                width=540, height=300,
                                tip="Load MNIST or your own images, then "
                                    "pick 'Imported' above to train on it."):
                    self._small("load a real dataset to train the crossbar "
                                "on", color=C_MUTED)
                    with dpg.group(horizontal=True):
                        b = dpg.add_button(label=" Download MNIST ",
                                           callback=self.on_nt_download_mnist)
                        dpg.bind_item_theme(b, self.themes["primary"])
                        dpg.add_button(label=" Load file... ",
                                       callback=lambda: dpg.show_item(
                                           "nt_ds_file_dialog"))
                        dpg.add_button(label=" Load folder... ",
                                       callback=lambda: dpg.show_item(
                                           "nt_ds_dir_dialog"))
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(
                            tag="nt_ds_url", width=-86,
                            hint="https://... raw GitHub / URL to .npz / "
                                 ".csv / .idx.gz")
                        b = dpg.add_button(label="  Fetch  ",
                                           callback=self.on_nt_fetch_url)
                        dpg.bind_item_theme(b, self.themes["primary"])
                    with dpg.group(horizontal=True):
                        with self._ctl_table(label_w=86):
                            self._ctl_row("resolution", "nt_ds_res", "int",
                                          14, "Square grid images are "
                                          "downsampled to (also sets GRID "
                                          "H/W on build).")
                            self._ctl_row("img / class", "nt_ds_perclass",
                                          "int", 12, "Images per class to "
                                          "sample for the training set.")
                        dpg.add_spacer(width=10)
                        with dpg.group():
                            self._caption("INVERT")
                            dpg.add_checkbox(tag="nt_ds_invert",
                                             default_value=False)
                    self._small("MNIST .idx/.gz, keras .npz, CSV (label,"
                                "pixels), or one image;  folder: one "
                                "subfolder per class.", color=C_MUTED)
                    dpg.add_text("no dataset loaded", tag="nt_ds_txt",
                                 wrap=430, color=C_TEXT2)
                    with dpg.child_window(height=8, border=False,
                                          show=False, tag="nt_ds_holder"):
                        pass
            with self.desktop.popup(noise, tag="nt_noise_popup"):
                with dpg.child_window(width=390, height=340, border=False):
                    with self._ctl_table(label_w=124):
                        self._ctl_row("input noise", "nt_input_noise", "double", 0., "Input intensity noise (0-1).", callback=_pre)
                        self._ctl_row("jitter (ms)", "nt_jitter", "double", 0., "Spike timing jitter.", callback=_pre)
                    with self._ctl_table(label_w=124):
                        self._ctl_row("bg rate", "nt_bg_rate", "double",
                                      0.0, "Spontaneous background firing on "
                                      "EVERY input.", callback=_pre, unit="freq")
                        self._ctl_row("signal afferents", "nt_signal_frac",
                                      "double", 1.0, "Fraction of inputs "
                                      "carrying the REAL pattern; the rest fire "
                                      "only bg noise.", callback=_pre)
                        self._ctl_row("pattern", "nt_pat_ms", "double", 0,
                                      "How long the pattern shows within "
                                      "present ms (centred). 0 = whole window.",
                                      callback=_pre, unit="time")
                        self._ctl_row("membrane noise", "nt_vnoise", "double",
                                      0.0, "Gaussian noise on each neuron's "
                                      "membrane every step (0.02-0.08 useful).")
                        self._ctl_row("write noise", "nt_wnoise", "double", 0.0,
                                      "Device cycle-to-cycle write noise "
                                      "(needs ECFET v2).")
            self._small("", tag="nt_source_summary", color=C_TEXT2)
            with dpg.group(horizontal=True):
                th = dpg.add_drawlist(width=70, height=70, tag="nt_pat_thumb")
                dpg.add_draw_layer(parent=th, tag="nt_pat_thumb_layer")
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchSame):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        self._small("Pattern", color=C_TEXT2)
                        self._small("Encoding", color=C_TEXT2)
                    with dpg.table_row():
                        dpg.add_combo(self._patset_items(), tag="nt_patset", default_value="bars", width=-1,
                                      callback=lambda *_: (self._nt_refresh_source(), self._nt_encoding_preview()))
                        dpg.add_combo(["rate (Poisson)", "latency (TTFS)", "temporal (phase)"], tag="nt_encoding",
                                      default_value="rate (Poisson)", width=-1, callback=_pre)
            with dpg.subplots(1, 2, tag="nt_data_plots", width=-1, height=-1):
                with dpg.plot(label="Clean encoding", tag="nt_enc_clean", no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)", tag="nt_enc_clean_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="afferent #", tag="nt_enc_clean_y")
                with dpg.plot(label="After noise + jitter", tag="nt_enc_noisy", no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)", tag="nt_enc_noisy_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="afferent #", tag="nt_enc_noisy_y")
            dpg.add_text("building preview...", tag="nt_enc_status", color=C_GREEN, wrap=400)
        self._nt_refresh_source()

    # ---- encoding preview (clean vs noisy rasters) ------------------

    def _nt_preview_vec(self, cfg):
        """A representative input pattern (flat, gh*gw) for the raster preview."""
        n = cfg.grid_h * cfg.grid_w
        src = cfg.pattern_set
        try:
            if src == "custom" and self._nt_custom:
                v = np.asarray(self._nt_custom[0][1], float).reshape(-1)
                if v.size == n:
                    return v
            elif src == "dataset" and getattr(self, "_nt_ds_sample", None) \
                    is not None:
                v = np.asarray(self._nt_ds_sample, float).reshape(-1)
                if v.size == n:
                    return v
            elif src in ("bars", "letters", "digits", "random"):
                pats = neuro.make_patterns(src, cfg.grid_h, cfg.grid_w)
                if pats:
                    return pats[min(1, len(pats) - 1)][1]
            elif src in PATTERN_PLUGINS:
                pats, _t = PATTERN_PLUGINS[src]["make"](cfg)
                if pats:
                    v = np.asarray(pats[0], float).reshape(-1)
                    if v.size == n:
                        return v
        except Exception:                                   # noqa: BLE001
            pass
        g = np.zeros((cfg.grid_h, cfg.grid_w))              # fallback: a plus
        g[cfg.grid_h // 2, :] = 1.0
        g[:, cfg.grid_w // 2] = 1.0
        return g.reshape(-1)

    def _nt_encoding_preview(self, *_):
        """Encode one pattern clean + noisy and draw the two rasters + status."""
        if not dpg.does_item_exist("nt_enc_clean_y"):
            return
        try:
            N, S, cfg = self._nt_read_params()
            cfg.n_out, cfg.hidden_layers = 1, ()      # cheap: encoding only
            dev = self._nt_dev or self._nt_device_factory()
            vec = self._nt_preview_vec(cfg)
            tr = neuro.Trainer(dev["make"], dev["kind"], N, S, cfg)
            clean, noisy = tr.compare_spikes(vec)
        except Exception as e:                              # noqa: BLE001
            self.log(f"[neuro] encoding preview failed: {e}")
            return
        dt = cfg.dt_ms
        n_in = cfg.grid_h * cfg.grid_w
        for mat, yax, attr, theme in (
                (clean, "nt_enc_clean_y", "_nt_enc_clean_ser", "dot_clean"),
                (noisy, "nt_enc_noisy_y", "_nt_enc_noisy_ser", "dot_noise")):
            ti, ii = np.nonzero(mat)
            old = getattr(self, attr, None)
            if old and dpg.does_item_exist(old):
                dpg.delete_item(old)
            ser = dpg.add_scatter_series((ti * dt).tolist(), ii.tolist(),
                                         parent=yax)
            dpg.bind_item_theme(ser, self.themes[theme])
            setattr(self, attr, ser)
        for xax, yax in (("nt_enc_clean_x", "nt_enc_clean_y"),
                         ("nt_enc_noisy_x", "nt_enc_noisy_y")):
            dpg.set_axis_limits(xax, 0.0, cfg.present_ms)
            dpg.set_axis_limits(yax, -0.5, n_in - 0.5)
        dpg.configure_item("nt_enc_clean", label=f"Clean encoding - {int(clean.sum())} spikes")
        dpg.configure_item("nt_enc_noisy", label=f"After noise + jitter - {int(noisy.sum())} spikes")
        enc = {"rate": "rate-encoded", "latency": "latency-encoded",
               "temporal": "temporal phase-encoded"}.get(cfg.encoding, cfg.encoding)
        dpg.set_value("nt_enc_status",
                      f"Active: {cfg.pattern_set} pattern set  -  {enc}  -  "
                      f"noise {int(round(cfg.input_noise * 100))}%  -  jitter "
                      f"{cfg.jitter_ms:.0f} ms  -  {cfg.present_ms:.0f} ms window")
        self._nt_draw_pat_thumb(vec, cfg.grid_h, cfg.grid_w)

    def _nt_draw_pat_thumb(self, vec, gh, gw):
        layer = "nt_pat_thumb_layer"
        if not dpg.does_item_exist(layer):
            return
        dpg.delete_item(layer, children_only=True)
        v = np.asarray(vec, float).reshape(gh, gw)
        W = H = 70
        cw, ch = W / gw, H / gh
        for r in range(gh):
            for c in range(gw):
                val = float(v[r, c])
                col = self._cmap_color(val) if val > 0 else (26, 30, 40, 255)
                dpg.draw_rectangle((c * cw, r * ch),
                                   ((c + 1) * cw, (r + 1) * ch), fill=col,
                                   color=(50, 56, 70, 255), parent=layer)

    def _nt_paint_grid_dims(self):
        gh = min(max(int(self._nt_get("nt_gh", 5)), 2), 14)
        gw = min(max(int(self._nt_get("nt_gw", 5)), 2), 14)
        return gh, gw

    def _nt_paint_new(self, *_):
        gh, gw = self._nt_paint_grid_dims()
        if self._nt_custom and (gh, gw) != (self._nt_paint_gh, self._nt_paint_gw):
            self._nt_custom = []            # grid changed -> old ones incompatible
            if dpg.does_item_exist("nt_custom_txt"):
                dpg.set_value("nt_custom_txt", "custom patterns: 0")
        self._nt_paint = np.zeros((gh, gw), np.float32)
        self._nt_paint_gh, self._nt_paint_gw = gh, gw
        self._nt_paint_redraw()

    def _nt_paint_clear(self, *_):
        if self._nt_paint is not None:
            self._nt_paint[:] = 0.0
            self._nt_paint_redraw()

    def _nt_paint_add(self, *_):
        if self._nt_paint is None or self._nt_paint.sum() <= 0:
            self._nt_status("paint some pixels first", C_AMBER)
            return
        n = len(self._nt_custom)
        self._nt_custom.append((f"custom {n}", self._nt_paint.copy()))
        if dpg.does_item_exist("nt_custom_txt"):
            dpg.set_value("nt_custom_txt",
                          f"custom patterns: {len(self._nt_custom)}")
        self._nt_paint_clear()
        self._nt_refresh_source()
        self._nt_status(f"added custom pattern ({len(self._nt_custom)} total)",
                        C_GREEN)

    def _nt_paint_drop(self, *_):
        if self._nt_custom:
            self._nt_custom.pop()
            dpg.set_value("nt_custom_txt",
                          f"custom patterns: {len(self._nt_custom)}")
            self._nt_refresh_source()

    def _nt_paint_redraw(self):
        layer = "nt_pat_layer"
        if not dpg.does_item_exist(layer) or self._nt_paint is None:
            return
        dpg.delete_item(layer, children_only=True)
        gh, gw = self._nt_paint_gh, self._nt_paint_gw
        W = H = 200                              # matches the nt_pat_editor size
        cw, ch = W / gw, H / gh
        for r in range(gh):
            for c in range(gw):
                v = float(self._nt_paint[r, c])
                col = self._cmap_color(v) if v > 0 else (26, 30, 40, 255)
                dpg.draw_rectangle((c * cw, r * ch), ((c + 1) * cw, (r + 1) * ch),
                                   fill=col, color=(50, 56, 70, 255),
                                   parent=layer)

    def _nt_tick_paint(self):
        """Drag-paint the custom-pattern editor: left held = set, right = erase.
        Driven from the render loop so click-and-drag works smoothly."""
        if (self._nt_paint is None or not dpg.does_item_exist("nt_pat_editor")
                or not dpg.is_item_hovered("nt_pat_editor")):
            return
        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        right = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
        if not (left or right):
            return
        gx, gy = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min("nt_pat_editor")
        lx, ly = gx - rmin[0], gy - rmin[1]
        gh, gw = self._nt_paint_gh, self._nt_paint_gw
        c = int(lx / (200.0 / gw))
        r = int(ly / (200.0 / gh))
        if 0 <= r < gh and 0 <= c < gw:
            nv = 1.0 if left else 0.0
            if self._nt_paint[r, c] != nv:
                self._nt_paint[r, c] = nv
                self._nt_paint_redraw()

    # ---- dataset loading --------------------------------------------

    def on_nt_download_mnist(self, *_):
        if self._nt_busy_load:
            return
        self._nt_busy_load = True
        self._nt_status("downloading MNIST (~11 MB)...", C_AMBER)
        self.log("[neuro] downloading MNIST from the keras mirror...")
        threading.Thread(target=self._nt_download_worker, daemon=True).start()

    def _nt_download_worker(self):
        try:
            from . import datasets as nds
            path = nds.download_mnist(os.path.join(self.workdir, "datasets"))
            self.q.put(("nt_dataset", path, None))
        except Exception as e:                          # noqa: BLE001
            self.q.put(("nt_dataset", None, f"download failed: {e}"))

    def on_nt_fetch_url(self, *_):
        """Download a dataset straight from a URL (raw GitHub, a mirror, ...)
        then load it through the normal pipeline (load_any dispatches by type)."""
        url = (dpg.get_value("nt_ds_url") or "").strip()
        if not url:
            self._nt_status("paste a dataset URL first", C_AMBER)
            return
        if self._nt_busy_load:
            return
        self._nt_busy_load = True
        self._nt_status("downloading from URL...", C_AMBER)
        self.log(f"[neuro] fetching dataset from {url}")
        threading.Thread(target=self._nt_url_worker, args=(url,),
                         daemon=True).start()

    def _nt_url_worker(self, url):
        try:
            from . import datasets as nds
            path = nds.download_url(url, os.path.join(self.workdir, "datasets"))
            self.q.put(("nt_dataset", path, None))      # -> _on_nt_dataset -> load
        except Exception as e:                          # noqa: BLE001
            self.q.put(("nt_dataset", None, f"URL download failed: {e}"))

    def _on_nt_ds_file(self, sender, app_data):
        sel = app_data.get("file_path_name") or ""
        paths = list((app_data.get("selections") or {}).values())
        path = paths[0] if paths else sel
        if path:
            self._nt_load_dataset(path)

    def _on_nt_ds_dir(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if path:
            self._nt_load_dataset(path)

    def _nt_load_dataset(self, path):
        if self._nt_busy_load:
            return
        self._nt_busy_load = True
        self._nt_status("loading dataset...", C_AMBER)
        threading.Thread(target=self._nt_load_worker, args=(path,),
                         daemon=True).start()

    def _nt_load_worker(self, path):
        try:
            from . import datasets as nds
            imgs, labels, names = nds.load_any(path)
            n = len(imgs)
            classes = sorted(set((labels if labels is not None
                                  else np.zeros(n, int)).tolist()))
            self.q.put(("nt_dataset_loaded",
                        {"images": imgs, "labels": labels, "names": names,
                         "path": path, "n": n, "classes": classes}, None))
        except Exception as e:                          # noqa: BLE001
            self.q.put(("nt_dataset", None, f"load failed: {e!r}"))

    def _on_nt_dataset(self, path, err):
        self._nt_busy_load = False
        if err:
            self._nt_status(err, C_RED)
            self.log(f"[neuro] {err}")
            return
        if path:                                        # downloaded -> now load
            self._nt_load_dataset(path)

    def _on_nt_dataset_loaded(self, ds, err):
        self._nt_busy_load = False
        if err or not ds:
            self._nt_status(err or "dataset load failed", C_RED)
            return
        self._nt_ds = ds
        nclass = len(ds["classes"])
        dpg.set_value("nt_ds_txt",
                      f"loaded {ds['n']} images, {nclass} classes "
                      f"({os.path.basename(ds['path'])}). Set PATTERNS = "
                      f"dataset and Build.")
        dpg.set_value("nt_patset", "dataset")
        if nclass > 1:
            dpg.set_value("nt_nout", nclass)            # one neuron per class
        # remember a sample + label for the source card + encoding preview
        try:
            from . import datasets as nds
            res = min(max(int(self._nt_get("nt_ds_res", 14)), 2), 32)
            dpg.set_value("nt_gh", res)
            dpg.set_value("nt_gw", res)
            g = nds._resize(ds["images"][0], res, res).astype(float)
            g = g - g.min()
            self._nt_ds_sample = g / (g.max() or 1.0)
            self._nt_ds_label = (f"{os.path.basename(ds['path'])}  -  "
                                 f"{ds['n']} imgs / {nclass} cls")
        except Exception:                                   # noqa: BLE001
            self._nt_ds_sample, self._nt_ds_label = None, None
        self._nt_dataset_preview(ds)
        self._nt_refresh_source()
        self._nt_encoding_preview()
        self._nt_status(f"dataset ready: {ds['n']} imgs / {nclass} classes",
                        C_GREEN)
        self.log(f"[neuro] dataset loaded: {ds['n']} images, {nclass} classes")

    def _nt_dataset_preview(self, ds):
        """Montage of one example image per class into the preview strip."""
        if not dpg.does_item_exist("nt_ds_holder"):
            return
        for tag in ("nt_ds_prev_img", "nt_ds_prev_tex"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        from . import datasets as nds
        imgs, labels, classes = ds["images"], ds["labels"], ds["classes"]
        labels = (labels if labels is not None
                  else np.zeros(len(imgs), int))
        cell = 40
        cols = min(len(classes), 10)
        lut = self._cmap_lut()
        mont = np.full((cell, cols * cell, 4), 0.08, np.float32)
        mont[..., 3] = 1.0
        for k, c in enumerate(classes[:cols]):
            idx = np.where(np.asarray(labels) == c)[0]
            if not len(idx):
                continue
            g = nds._resize(imgs[idx[0]], cell, cell).astype(np.float32)
            g = g - g.min()
            g = g / (g.max() or 1.0)
            mont[:, k * cell:(k + 1) * cell, :3] = lut[
                np.clip((g * 255).astype(int), 0, 255)]
        dpg.add_raw_texture(cols * cell, cell, mont.reshape(-1),
                            format=dpg.mvFormat_Float_rgba, parent="nt_tex_reg",
                            tag="nt_ds_prev_tex")
        dpg.add_image("nt_ds_prev_tex", tag="nt_ds_prev_img",
                      parent="nt_ds_holder", width=cols * cell * 2,
                      height=cell * 2)

    # ---- pattern-source dispatch (for on_nt_build) ------------------

    def _patset_items(self):
        """PATTERNS dropdown items: built-ins + hot-loaded plugins + the two
        special sources (custom paint / loaded dataset)."""
        return BUILTIN_PATSETS + list(PATTERN_PLUGINS) + ["custom", "dataset"]

    def _nt_watch_patterns(self):
        """Poll the patterns/ folder; when a plugin is added / edited / removed
        (stable for one poll), re-register and refresh the PATTERNS dropdown
        LIVE - so new gates appear with no restart."""
        self._pat_tick = getattr(self, "_pat_tick", 0) + 1
        if self._pat_tick % 30 != 0:              # ~ every 0.5 s at 60 fps
            return
        pdir = os.path.join(self.workdir, "patterns")
        cur = {}
        try:
            for p in glob.glob(os.path.join(pdir, "*.py")):
                if not os.path.basename(p).startswith("_"):
                    cur[p] = os.path.getmtime(p)
        except OSError:
            return
        if self._pat_mtimes is None:
            self._pat_mtimes = self._pat_prev_mtimes = cur
            return
        # reload when the folder has been STABLE for a poll AND differs from the
        # currently-registered set (a new / edited / removed plugin file)
        if cur == self._pat_prev_mtimes and cur != self._pat_mtimes:
            keys, errs = register_pattern_plugins(self.workdir)
            for name, err in errs:
                self.log(f"[patterns] {name}: {err}")
            if dpg.does_item_exist("nt_patset"):
                sel = dpg.get_value("nt_patset")
                items = self._patset_items()
                dpg.configure_item("nt_patset", items=items)
                if sel not in items:              # its file was removed
                    dpg.set_value("nt_patset", "bars")
            self.log(f"[patterns] reloaded - {len(keys)} plugin(s): {keys}")
            self._pat_mtimes = cur
        self._pat_prev_mtimes = cur

    def _nt_build_patterns(self, cfg):
        """Return (patterns, targets) for the selected PATTERNS source, or
        (None, None) to use the built-in bank.  May adjust cfg in place
        (grid + n_out for a dataset)."""
        if self._nt_loaded_patterns is not None:  # restoring a saved model
            pats, tgts = self._nt_loaded_patterns
            self._nt_loaded_patterns = None
            return pats, tgts
        src = cfg.pattern_set
        if src in PATTERN_PLUGINS:               # a patterns/ plugin (hot-loaded)
            pats, targets = PATTERN_PLUGINS[src]["make"](cfg)
            return pats, targets
        if src == "nand":
            # 2-input NAND logic gate -> output neuron 0 ("0") / 1 ("1").
            # COMPLEMENTARY coding: each input becomes a pixel pair [A, ~A] so the
            # afferent grid is [A, ~A, B, ~B] on a 2x2 field.  The softmax/argmax
            # readout has no per-neuron bias, so a shared "always-on" afferent
            # cancels out; the complement pixels instead give each class a wide,
            # bias-free margin (class "0" reads A&B, class "1" reads ~A|~B), and
            # a single crossbar of positive conductances learns it reliably.
            # NAND is 3:1 imbalanced, so the (1,1)->0 minority is oversampled 3x
            # and interleaved with the majority -> a balanced 3-vs-3 train set.
            cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
            def _v(a, b):
                return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)
            table = [((0, 0), 1), ((1, 1), 0), ((0, 1), 1),
                     ((1, 1), 0), ((1, 0), 1), ((1, 1), 0)]
            pats = [(f"A={a} B={b} -> {y}", _v(a, b)) for (a, b), y in table]
            targets = [y for _, y in table]
            return pats, targets
        if src == "xor":
            # 2-input XOR -> output neuron 0 ("0") / 1 ("1"). XOR is NOT linearly
            # separable, so a SINGLE crossbar can never learn it: build with a
            # hidden layer (nt_hidden, e.g. "8") and the Surrogate-grad (BPTT)
            # rule so the gradient flows through the hidden crossbar. Same
            # complementary [A, ~A, B, ~B] 2x2 coding as nand; XOR is already
            # 2-vs-2 balanced, repeated 2x for a fuller train set.
            cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
            def _v(a, b):
                return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)
            table = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0),
                     ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
            pats = [(f"A={a} B={b} -> {y}", _v(a, b)) for (a, b), y in table]
            targets = [y for _, y in table]
            return pats, targets
        if src == "custom":
            if not self._nt_custom:
                raise ValueError("no custom patterns - paint some in the "
                                 "Data tab (Configure source > Add pattern)")
            gh, gw = self._nt_paint_gh or cfg.grid_h, self._nt_paint_gw or cfg.grid_w
            cfg.grid_h, cfg.grid_w = gh, gw
            pats = [(lbl, g.reshape(-1).astype(np.float32))
                    for lbl, g in self._nt_custom]
            targets = [k % cfg.n_out for k in range(len(pats))]
            return pats, targets
        if src == "dataset":
            if not self._nt_ds:
                raise ValueError("no dataset loaded - use Download MNIST / Load "
                                 "in Data > Configure source")
            res = min(max(int(self._nt_get("nt_ds_res", 14)), 6), 28)
            per = min(max(int(self._nt_get("nt_ds_perclass", 12)), 2), 60)
            cfg.grid_h = cfg.grid_w = res
            from . import datasets as nds
            tr_p, tr_t, te_p, te_t, names = nds.to_patterns_split(
                self._nt_ds["images"], self._nt_ds["labels"], res, res,
                train_per_class=per, test_per_class=max(4, per // 2),
                seed=cfg.seed,
                invert=bool(self._nt_get("nt_ds_invert", False)))
            cfg.n_out = max(cfg.n_out, len(names))
            self._nt_pending_test = (te_p, te_t, names)   # held-out test set
            return tr_p, tr_t
        return None, None
