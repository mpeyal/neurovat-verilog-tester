"""Network Studio rendering: crossbars, cells, weights and activity.

App owns the trainer and UI state. Render methods run on the main thread.
The snapshot and montage helpers operate on data only and are also used by
training workers; keep DearPyGui calls out of those helpers.
"""
import dearpygui.dearpygui as dpg
import numpy as np

from .ui_constants import C_MUTED


class NetworkVisualizationMixin:

    # viridis-style colour ramp, expanded once to a 256-row [0,1] rgb LUT
    _CMAP_STOPS = [(0.0, (68, 1, 84)), (0.22, (65, 68, 135)),
                   (0.45, (42, 120, 142)), (0.66, (34, 168, 132)),
                   (0.84, (122, 209, 81)), (1.0, (253, 231, 37))]

    def _cmap_lut(self):
        if self._cmap_lut_cache is None:
            xs = np.array([s for s, _ in self._CMAP_STOPS])
            cols = np.array([c for _, c in self._CMAP_STOPS], float) / 255.0
            g = np.linspace(0, 1, 256)
            self._cmap_lut_cache = np.stack(
                [np.interp(g, xs, cols[:, k]) for k in range(3)], axis=1)
        return self._cmap_lut_cache

    def _cmap_color(self, v, alpha=255):
        lut = self._cmap_lut()
        r, g, b = lut[int(min(max(v, 0.0), 1.0) * 255)] * 255.0
        return (int(r), int(g), int(b), alpha)

    def _nt_setup_rf_texture(self, n_out, gh, gw):
        cell = max(8, min(30, 130 // max(gw, 1)))
        gap = 6
        tw = n_out * gw * cell + (n_out - 1) * gap
        th = gh * cell
        if self._rf_tex and dpg.does_item_exist(self._rf_tex):
            dpg.delete_item(self._rf_tex)
        blank = self._blank_rgba(tw, th)
        self._rf_tex = dpg.add_raw_texture(
            tw, th, blank, format=dpg.mvFormat_Float_rgba, parent="nt_tex_reg")
        self._rf_layout = (n_out, gh, gw, cell, gap, tw, th)
        if dpg.does_item_exist("nt_rf_image"):
            dpg.delete_item("nt_rf_image")
        dpg.add_image(self._rf_tex, tag="nt_rf_image", parent="nt_rf_holder",
                      width=tw, height=th)

    @staticmethod
    def _blank_rgba(tw, th):
        img = np.empty((th * tw, 4), np.float32)
        img[:] = (0.08, 0.09, 0.12, 1.0)
        return img.reshape(-1)

    def _nt_montage(self, Wn):
        n_out, gh, gw, cell, gap, tw, th = self._rf_layout
        lut = self._cmap_lut()
        img = np.empty((th, tw, 4), np.float32)
        img[:] = (0.08, 0.09, 0.12, 1.0)
        for j in range(min(n_out, Wn.shape[0])):
            rf = Wn[j].reshape(gh, gw)
            rgb = lut[np.clip((rf * 255).astype(int), 0, 255)]    # (gh,gw,3)
            block = np.repeat(np.repeat(rgb, cell, 0), cell, 1)
            x0 = j * (gw * cell + gap)
            img[:, x0:x0 + gw * cell, :3] = block
        return img.reshape(-1)

    # ---- snapshot (pure data, safe to build off-thread) -------------

    def _nt_snapshot(self, epoch, epochs):
        tr = self.trainer
        K = tr.n_layers
        Wn_all = [tr.weights_norm(c) for c in range(K)]
        W_uS_all = [tr.weights_uS(c) for c in range(K)]
        last = W_uS_all[-1]                 # incoming weights of the output layer
        return {
            "epoch": epoch, "epochs": epochs,
            "Wn0": Wn_all[0], "Wn_all": Wn_all, "W_uS_all": W_uS_all,
            "rf_rgba": self._nt_montage(Wn_all[0]),
            "mean_w": [float(last[j].mean()) for j in range(tr.n_out)],
            "present": self._last_present,
            "spikes": int(tr.spike_count.sum()),
        }

    # ---- snapshot rendering (main thread) ---------------------------

    def _apply_nt_snapshot(self, snap):
        if (self._rf_tex and dpg.does_item_exist(self._rf_tex)
                and snap.get("rf_rgba") is not None):
            dpg.set_value(self._rf_tex, snap["rf_rgba"])
        self._nt_wuS_all = snap["W_uS_all"]            # for the crossbar selector
        # _nt_xbar_sel_idx is a plain attribute (set by the crossbar selector),
        # NOT a DPG item - reading it via _nt_get always returned the default 0.
        c = min(int(getattr(self, "_nt_xbar_sel_idx", 0)),
                len(snap["W_uS_all"]) - 1)
        self._nt_draw_weight_matrix(snap["W_uS_all"][c])
        self._nt_draw_diagram(snap["Wn_all"], snap.get("present"))
        if snap["epoch"] > 0:
            self._wevo_epochs.append(snap["epoch"])
            for j, mw in enumerate(snap["mean_w"]):
                self._wevo.setdefault(j, []).append(mw)
            self._nt_draw_wevo()
            # record this epoch's full weights for the per-synapse cell trace
            self._nt_whist["epoch"].append(snap["epoch"])
            self._nt_whist["W"].append([np.asarray(w).copy()
                                        for w in snap["W_uS_all"]])
            if len(self._nt_whist["epoch"]) > 400:     # bound memory
                self._nt_whist["epoch"].pop(0)
                self._nt_whist["W"].pop(0)
            self._nt_draw_cell_trace()
        if snap.get("present"):
            self._nt_draw_activity(*snap["present"])
            self._nt_draw_raster(*snap["present"])

    def _nt_clear(self, group):
        for s in self._nt_series.get(group, []):
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._nt_series[group] = []

    def _on_nt_xbar_sel(self, sender, value):
        labels = getattr(self, "_nt_xbar_labels", [])
        idx = labels.index(value) if value in labels else 0
        self._nt_xbar_sel_idx = idx
        wall = getattr(self, "_nt_wuS_all", None)
        if wall:
            self._nt_draw_weight_matrix(wall[min(idx, len(wall) - 1)])

    def _nt_draw_weight_matrix(self, W_uS):
        if not dpg.does_item_exist("nt_wm_y"):
            return
        self._nt_clear("wm")
        n_out, n_in = W_uS.shape
        lo, hi = float(W_uS.min()), float(W_uS.max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        s = dpg.add_heat_series(
            W_uS.reshape(-1).tolist(), n_out, n_in, scale_min=lo, scale_max=hi,
            bounds_min=(0.0, 0.0), bounds_max=(float(n_in), float(n_out)),
            format="", parent="nt_wm_y")
        self._nt_series["wm"] = [s]
        dpg.configure_item("nt_wm_cbar", min_scale=lo, max_scale=hi)
        dpg.fit_axis_data("nt_wm_x")
        dpg.fit_axis_data("nt_wm_y")

    def _nt_draw_diagram(self, Wn_all, present):
        """Dispatch: one crossbar -> the detailed array schematic; a stack of
        crossbars -> the layered multi-crossbar network."""
        layer = "nt_diag_layer"
        if not dpg.does_item_exist(layer) or self.trainer is None:
            return
        dpg.delete_item(layer, children_only=True)
        if self.trainer.n_layers > 1:
            self._nt_draw_layered(Wn_all, present)
        else:
            self._nt_draw_xbar_array(Wn_all[0], present)

    def _nt_draw_xbar_array(self, Wn, present):
        """A single crossbar: input neurons drive horizontal wordlines that
        cross vertical bitlines; a synapse device sits at every crossing
        (colour = conductance/weight); each bitline is summed by an output
        neuron at the bottom. Input spikes light their wordline, output spikes
        light their neuron, the winner is ringed."""
        layer = "nt_diag_layer"
        self._nt_cell_hits = []                       # clickable device cells
        n_out, gh, gw = self._rf_layout[:3]
        n_in = gh * gw
        Wd, Hd = self._nt_diag_size
        vec = present[1] if present else None
        counts = present[2]["n_out_spikes"] if present else None
        winner = present[2]["winner"] if present else -1
        sig = self.trainer.signal_mask if self.trainer is not None else None
        inh = self.trainer.inh0 if self.trainer is not None else None
        di = getattr(self.trainer, "device_binding", self._nt_dev or {})
        kind = di.get("kind", "current")            # current=ECFET, voltage=FeFET
        tt = di.get("three_terminal", True)         # 3-terminal (gate) vs 2-term
        dname = "ECFET" if kind == "current" else ("FeFET" if tt else "memristor")

        # cap the number of wordlines drawn on a big grid (sample evenly)
        N_show = min(n_in, 56)
        rows = (np.linspace(0, n_in - 1, N_show).round().astype(int)
                if n_in > N_show else np.arange(n_in))
        nr = len(rows)
        ow = max(2, len(str(max(n_out - 1, 0))))      # zero-pad widths for IDs
        iw = max(2, len(str(max(n_in - 1, 0))))
        label_inputs = nr <= 30                       # skip on big grids (clutter)

        thumb = float(min(120.0, Hd * 0.30))
        ix = 18 + thumb + 28                       # input-neuron column x
        ax0 = ix + 50                              # array left (first bitline)
        ax1 = Wd - 182                             # array right (last bitline)
        ay0 = 52.0                                 # array top (room for legend)
        ay1 = Hd - 84.0                            # array bottom
        oy = Hd - 42.0                             # output-neuron row y
        row_y = lambda r: ay0 + (r + 0.5) * (ay1 - ay0) / max(nr, 1)
        col_x = lambda j: ax0 + (j + 0.5) * (ax1 - ax0) / max(n_out, 1)
        dev = max(2.5, min((ax1 - ax0) / max(n_out, 1) * 0.20,
                           (ay1 - ay0) / max(nr, 1) * 0.42))
        detail = dev >= 9.0                        # big cells -> 3-terminal view
        pitch_r = (ay1 - ay0) / max(nr, 1)         # row spacing (pre drivers)

        # input pattern thumbnail (the 2-D image driving the wordlines)
        if vec is not None and gw and gh:
            cw, ch = thumb / gw, thumb / gh
            for r in range(gh):
                for c in range(gw):
                    v = float(vec[r * gw + c])
                    g = int(40 + 205 * v)
                    dpg.draw_rectangle((18 + c * cw, 30 + r * ch),
                                       (18 + (c + 1) * cw, 30 + (r + 1) * ch),
                                       fill=(g, g, min(255, g + 25), 255),
                                       color=(40, 46, 60, 110), parent=layer)
        dpg.draw_text((18, 12), "input pattern", size=13,
                      color=(140, 150, 172), parent=layer)

        # wordlines + input neurons (one per shown afferent), coloured by type:
        # EXCITATORY afferents add to the membrane (EPSP +), INHIBITORY ones
        # subtract (IPSP -); ~1/6 of inputs are inhibitory when ipsp_gain > 0
        C_EXC, C_INH = (95, 215, 165), (245, 115, 115)
        C_PRE = (180, 150, 225)                       # PRE program / driver
        prw = min(22.0, 26.0)
        prh = min(10.0, pitch_r * 0.5)
        for r, i in enumerate(rows):
            i = int(i)
            y = row_y(r)
            inten = float(vec[i]) if vec is not None else 0.18
            active = inten > 0.25
            is_inh = inh is not None and i < len(inh) and inh[i] > 0
            ring = (C_INH if is_inh else C_EXC) + (255,)
            g = int(55 + 190 * inten)
            dpg.draw_circle((ix, y), 4.6, fill=(g, g, min(255, g + 30), 255),
                            color=ring, thickness=1.7, parent=layer)
            if label_inputs:                        # neuron id, e.g. x00, x01
                dpg.draw_text((ix - 16 - 6 * iw, y - 6), f"x{i:0{iw}d}",
                              size=10, color=(124, 136, 156), parent=layer)
            if is_inh:                              # inhibitory wordline -> red
                wl = (240, 120, 120, 220) if active else (120, 70, 70, 150)
            else:
                wl = (150, 185, 255, 210) if active else (66, 76, 96, 150)
            dpg.draw_line((ix + 5, y), (col_x(n_out - 1) + dev + 4, y),
                          color=wl, thickness=1.8 if active else 0.8,
                          parent=layer)
            # PRE write DRIVER, SIDE-BY-SIDE with the input neuron (mirrors the
            # post driver): it taps the pre spike and drives THIS row's gate /
            # program line. The wordline READ still rides straight into the array.
            if label_inputs:
                pry = y - (dev * 1.5 if (tt and detail) else 4.0)
                dpg.draw_line((ix + 5, y), (ix + 9, pry), color=C_PRE + (215,),
                              thickness=1.3, parent=layer)      # neuron -> driver
                dpg.draw_rectangle((ix + 9, pry - prh / 2),
                                   (ix + 9 + prw, pry + prh / 2),
                                   fill=(38, 32, 54, 240), color=C_PRE + (235,),
                                   thickness=1.1, rounding=2, parent=layer)
                if tt:                 # 3-terminal: drives a separate GATE line
                    dpg.draw_line((ix + 9 + prw, pry),
                                  (col_x(n_out - 1) + dev, pry),
                                  color=C_PRE + (150,), thickness=1.0, parent=layer)
                else:                  # 2-terminal: the write rides the WORDLINE
                    dpg.draw_line((ix + 9 + prw, pry), (ix + 9 + prw + 8, y),
                                  color=C_PRE + (150,), thickness=1.0, parent=layer)
            # signal-vs-noise (Masquelier) kept as a small orange side dot
            if sig is not None and sig[i] == 0:
                dpg.draw_circle((ix - 9, y), 2.0, fill=(255, 170, 90, 235),
                                parent=layer)

        # bitlines + device cells + output neurons.  When the cells are big
        # enough (small grid or zoomed in) each synapse is drawn as a real
        # 3-TERMINAL device: a source-drain channel (colour = conductance) with
        # a separate GATE (write) terminal tapping a per-row gate/program line.
        cmax = max(counts) if counts else 1
        for j in range(n_out):
            x = col_x(j)
            dpg.draw_line((x, ay0 - 6), (x, oy - 13), color=(82, 94, 122, 170),
                          thickness=1.3, parent=layer)
            for r, i in enumerate(rows):
                w = float(Wn[j][int(i)])
                y = row_y(r)
                sel = self._nt_cell == (0, j, int(i))
                ecol = (255, 220, 120, 255) if sel else (18, 22, 30, 160)
                if detail:
                    bw, bh = dev * 1.25, dev * 0.78
                    # source-drain channel (the stored weight)
                    dpg.draw_rectangle((x - bw, y - bh), (x + bw, y + bh),
                                       fill=self._cmap_color(w, 255), color=ecol,
                                       thickness=2.6 if sel else 1.2,
                                       rounding=2.5, parent=layer)
                    # source contact taps the wordline (left), drain the bitline
                    dpg.draw_circle((x - bw, y), max(1.4, dev * 0.13),
                                    fill=(220, 224, 236, 230), parent=layer)
                    dpg.draw_circle((x, y - bh), max(1.4, dev * 0.13),
                                    fill=(150, 170, 210, 230), parent=layer)
                    if tt:        # 3-terminal only: a GATE stub up to its rail
                        gy = y - dev * 1.5
                        dpg.draw_line((x, y - bh), (x, gy),
                                      color=(170, 150, 200, 200), thickness=1.4,
                                      parent=layer)
                        dpg.draw_line((x - bw * 0.5, gy + 2),
                                      (x + bw * 0.5, gy + 2),
                                      color=(196, 168, 226, 230), thickness=2.0,
                                      parent=layer)
                else:
                    dpg.draw_rectangle((x - dev, y - dev), (x + dev, y + dev),
                                       fill=self._cmap_color(w, 255), color=ecol,
                                       thickness=2.5 if sel else 1.0,
                                       rounding=1.5, parent=layer)
                self._nt_cell_hits.append((0, j, int(i), x, y))
            act = (counts[j] / cmax) if (counts and cmax) else 0.0
            fill = self._cmap_color(0.15 + 0.85 * act, 255)
            oring = (255, 178, 90, 255) if j == winner else (90, 100, 124, 230)
            dpg.draw_circle((x, oy), 12.0, fill=fill, color=oring,
                            thickness=2.5, parent=layer)
            lbl = f"N{j:0{ow}d}" + (f" {counts[j]}sp" if counts else "")
            dpg.draw_text((x - 4 * len(lbl), oy + 15), lbl, size=12,
                          color=(200, 208, 224), parent=layer)

        # ---- DIRECT, LOCAL STDP wiring (NO central driver): every synapse is
        # programmed by ITS OWN pre + post neuron. The PRE runs along its ROW
        # (the purple program rail, fed by input neuron i); the POST feeds back
        # UP its COLUMN (amber, from output neuron Nj). They meet at the
        # cross-point and that coincidence sets that one device's gate pulse. ---
        C_FB = (240, 220, 70)                         # POST feedback (yellow)
        pitch_c = (ax1 - ax0) / max(n_out, 1)
        dw = min(30.0, max(12.0, pitch_c * 0.5))      # driver width (fits pitch)
        for j in range(n_out):
            nx = col_x(j)
            fx = nx + 5                               # post line, just off bitline
            # per-neuron WRITE DRIVER, SIDE-BY-SIDE with the neuron: it only TAPS
            # the post spike. The drain-source READ path (bitline) shorts STRAIGHT
            # into the neuron - it does NOT pass through the driver.
            dx0 = nx + 13
            dpg.draw_line((nx + 12, oy), (dx0, oy), color=C_FB + (225,),
                          thickness=1.6, parent=layer)         # neuron -> driver
            dpg.draw_rectangle((dx0, oy - 8), (dx0 + dw, oy + 8),
                               fill=(50, 46, 16, 242), color=C_FB + (240,),
                               thickness=1.4, rounding=3, parent=layer)
            if dw >= 24:
                dpg.draw_text((dx0 + 4, oy - 7), "drv", size=11,
                              color=C_FB + (255,), parent=layer)
            if tt:        # 3-terminal: drives a separate GATE line up the column
                dpg.draw_line((dx0, oy - 8), (fx, oy - 18), color=C_FB + (230,),
                              thickness=1.5, parent=layer)
                dpg.draw_line((fx, oy - 18), (fx, ay0 - 4), color=C_FB + (185,),
                              thickness=1.4, parent=layer)
                dpg.draw_circle((fx, ay0 - 4), 2.4, fill=C_FB + (240,),
                                parent=layer)
                if detail:                            # tap each gate in the column
                    for r in range(nr):
                        yr = row_y(r) - dev * 1.5
                        dpg.draw_line((nx, yr), (fx, yr), color=C_FB + (175,),
                                      thickness=1.2, parent=layer)
            else:         # 2-terminal: the write rides the BITLINE itself
                dpg.draw_line((dx0, oy - 8), (nx, oy - 16), color=C_FB + (210,),
                              thickness=1.4, parent=layer)
        # ---- top: one-line explainer + a single horizontal legend strip ------
        if tt:
            wq = "current" if kind == "current" else "voltage"
            explain = (f"3-TERMINAL {dname} - GATE {wq} pulse programs each cell "
                       "(read on source-drain); PRE drives the ROW gate, POST "
                       "the COLUMN gate.")
        else:
            explain = ("2-TERMINAL memristor - SET/RESET voltage across the SAME "
                       "word & bit lines (no gate); PRE drives the ROW, POST the "
                       "COLUMN.")
        dpg.draw_text((ix - 30, 6), explain,
                      size=11, color=(206, 196, 150), parent=layer)
        leg = [(C_EXC, "exc"), (C_INH, "inh"), (C_PRE, "PRE drv"),
               (C_FB, "POST drv"), ((240, 120, 120), "WTA")]
        lxp = ix - 30
        for col, name in leg:
            dpg.draw_circle((lxp + 5, 27), 4.2, fill=col + (255,), parent=layer)
            dpg.draw_text((lxp + 13, 20), name, size=12, color=col + (255,),
                          parent=layer)
            lxp += 30 + 7 * len(name)

        # region labels
        dpg.draw_text((ix - 30, ay0 - 16), "input neurons", size=13,
                      color=(140, 150, 172), parent=layer)
        dpg.draw_text((0.5 * (ax0 + ax1) - 70, ay0 - 16),
                      (f"crossbar array  ({'3' if tt else '2'}-terminal "
                       f"{dname} per cross-point)" if detail else
                       f"crossbar array  (zoom in for the {dname} cell view)"),
                      size=13, color=(150, 165, 210), parent=layer)
        dpg.draw_text((col_x(0) - 16, oy + 30), "output neurons", size=13,
                      color=(140, 150, 172), parent=layer)
        if n_in > N_show:
            dpg.draw_text((ix - 30, ay1 + 8),
                          f"(showing {N_show} of {n_in} wordlines)", size=12,
                          color=C_MUTED, parent=layer)
        # lateral inhibition (winner-take-all) between the output neurons
        if (n_out > 1 and self.trainer is not None
                and self.trainer.N.inhibition > 0):
            ybar = ay1 + 10                        # just below the array
            dpg.draw_line((col_x(0), ybar), (col_x(n_out - 1), ybar),
                          color=(240, 120, 120, 170), thickness=1.4, parent=layer)
            for j in range(n_out):                 # drop-ticks toward each neuron
                dpg.draw_line((col_x(j), ybar), (col_x(j), oy - 12),
                              color=(240, 120, 120, 130), thickness=1.2,
                              parent=layer)
            dpg.draw_text((col_x(0) - 4, ybar - 15),
                          "lateral inhibition (WTA)", size=11,
                          color=(245, 140, 140, 230), parent=layer)
        if present and winner >= 0:
            dpg.draw_text((Wd - 150, oy - 6), f"output -> N{winner}", size=15,
                          color=(255, 190, 110, 255), parent=layer)

    def _nt_draw_layered(self, Wn_all, present):
        """A STACK of crossbars: each layer is a column of LIF neurons, joined
        to the next by its own crossbar (weighted connections, colour =
        conductance).  Signal flows input -> crossbar -> hidden -> ... ->
        output.  The detailed conductance array of each crossbar is on the
        Weights tab (use the crossbar selector)."""
        layer = "nt_diag_layer"
        L = self.trainer.layer_sizes
        nL = len(L)
        Wd, Hd = self._nt_diag_size
        vec = present[1] if present else None
        counts = present[2]["n_out_spikes"] if present else None
        winner = present[2]["winner"] if present else -1
        lspk = present[2].get("layer_spikes") if present else None

        padx, ptop, pbot = 64, 40, 56
        NS = 28                                # max nodes shown per layer
        # reserve room on the left for the 2-D input-pattern thumbnail (shown in
        # the single-crossbar view too), then lay the layers out to its right
        gh = self.trainer.cfg.grid_h if self.trainer is not None else 0
        gw = self.trainer.cfg.grid_w if self.trainer is not None else 0
        thumb = float(min(110.0, Hd * 0.26))
        loff = (thumb + 30) if (vec is not None and gw and gh) else 0.0
        xs = [padx + loff + s * (Wd - 2 * padx - loff) / max(nL - 1, 1)
              for s in range(nL)]
        pos, shown = [], []
        for sz in L:
            ns = min(sz, NS)
            idx = (np.linspace(0, sz - 1, ns).round().astype(int)
                   if sz > NS else np.arange(sz))
            ys = [ptop + (r + 0.5) * (Hd - ptop - pbot) / max(ns, 1)
                  for r in range(ns)]
            pos.append(list(zip([xs[len(pos)]] * ns, ys)))
            shown.append(idx)

        # the input-pattern thumbnail (2-D image driving the input layer)
        if vec is not None and gw and gh:
            cw, ch = thumb / gw, thumb / gh
            y0 = max(34.0, ptop - 6)
            for r in range(gh):
                for c in range(gw):
                    v = float(vec[r * gw + c])
                    g = int(40 + 205 * v)
                    dpg.draw_rectangle((18 + c * cw, y0 + r * ch),
                                       (18 + (c + 1) * cw, y0 + (r + 1) * ch),
                                       fill=(g, g, min(255, g + 25), 255),
                                       color=(40, 46, 60, 110), parent=layer)
            dpg.draw_text((18, y0 - 18), "input pattern", size=13,
                          color=(140, 150, 172), parent=layer)

        # crossbar connections (top-K per post neuron, colour = weight)
        for c in range(nL - 1):
            W = Wn_all[c]
            src, dst = shown[c], shown[c + 1]
            kk = min(len(src), 10)
            for dj, j in enumerate(dst):
                row = W[int(j)][src]
                for oi in np.argsort(row)[::-1][:kk]:
                    w = float(row[oi])
                    if w < 0.06:
                        continue
                    dpg.draw_line(pos[c][oi], pos[c + 1][dj],
                                  color=self._cmap_color(w, int(40 + 175 * w)),
                                  thickness=0.5 + 2.0 * w, parent=layer)
            mx = 0.5 * (xs[c] + xs[c + 1])
            dpg.draw_text((mx - 28, ptop - 38), f"crossbar {c}", size=12,
                          color=(150, 165, 210), parent=layer)
            dpg.draw_text((mx - 22, Hd - pbot + 24), f"{L[c]}x{L[c + 1]}",
                          size=12, color=C_MUTED, parent=layer)

        # neuron nodes per layer
        for s in range(nL):
            sz, idx = L[s], shown[s]
            cmax = (max(counts) or 1) if counts else 1
            smax = (max((len(x) for x in lspk[s]), default=1) or 1) \
                if (lspk and s < len(lspk)) else 1
            rad = 5.5 if sz > NS else max(5.0, min(13.0, 95.0 / max(sz, 1)))
            lw = max(2, len(str(max(sz - 1, 0))))     # id zero-pad width
            if s == 0:                                 # input -> x00, x01 ...
                pre, w = "x", lw
            elif s == nL - 1:                          # output -> N00, N01 ...
                pre, w = "N", lw
            else:                                      # hidden: encode the layer
                pre = f"h{s - 1}"                      # 1st hidden h0.., 2nd h1..
                w = max(1, len(str(max(sz - 1, 0))))
            label_nodes = len(idx) <= NS              # skip when too crowded
            for r, i in enumerate(idx):
                i = int(i)
                x, y = pos[s][r]
                if s == 0:
                    a = float(vec[i]) if vec is not None else 0.18
                    g = int(45 + 200 * a)
                    inh = self.trainer.inh0
                    is_inh = inh is not None and i < len(inh) and inh[i] > 0
                    fill = (g, g, min(255, g + 25), 255)
                    ring = (245, 115, 115, 255) if is_inh else (95, 215, 165, 255)
                else:
                    if s == nL - 1 and counts:
                        a = counts[i] / cmax
                    elif lspk and s < len(lspk):
                        a = len(lspk[s][i]) / smax
                    else:
                        a = 0.2
                    fill = self._cmap_color(0.15 + 0.85 * a, 255)
                    ring = ((255, 178, 90, 255) if (s == nL - 1 and i == winner)
                            else (90, 100, 124, 220))
                dpg.draw_circle((x, y), rad, fill=fill, color=ring,
                                thickness=1.6, parent=layer)
                if label_nodes:                     # neuron id (x / h<L> / N)
                    tag = f"{pre}{i:0{w}d}"
                    tx = (x - rad - 4 - 6 * len(tag)) if s == 0 else (x + rad + 4)
                    dpg.draw_text((tx, y - 6), tag, size=10,
                                  color=(150, 160, 182), parent=layer)
            name = ("input" if s == 0 else
                    "output" if s == nL - 1 else f"hidden {s}")
            dpg.draw_text((xs[s] - 16, ptop - 22), name, size=13,
                          color=(140, 150, 172), parent=layer)
            if sz > NS:
                dpg.draw_text((xs[s] - 22, Hd - pbot + 8),
                              f"{sz} (showing {len(idx)})", size=11,
                              color=C_MUTED, parent=layer)
        # E/I legend for the input layer
        lx, ly = 12, 10
        dpg.draw_circle((lx + 5, ly + 5), 4.4, fill=(0, 0, 0, 0),
                        color=(95, 215, 165, 255), thickness=1.6, parent=layer)
        dpg.draw_text((lx + 14, ly - 2), "excitatory in", size=11,
                      color=(95, 215, 165, 255), parent=layer)
        dpg.draw_circle((lx + 5, ly + 22), 4.4, fill=(0, 0, 0, 0),
                        color=(245, 115, 115, 255), thickness=1.6, parent=layer)
        dpg.draw_text((lx + 14, ly + 15), "inhibitory in", size=11,
                      color=(245, 115, 115, 255), parent=layer)
        if present and winner >= 0:
            dpg.draw_text((Wd - 150, Hd - pbot + 24), f"output -> N{winner}",
                          size=14, color=(255, 190, 110, 255), parent=layer)

    # ---- synapse cell inspector (animated read/write) ----------------

    def _nt_pick_cell(self):
        """Select the crossbar cell nearest the click on the Canvas; jump to
        the Cell tab."""
        gx, gy = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min("nt_diagram")
        lx, ly = gx - rmin[0], gy - rmin[1]
        best = None
        for c, j, i, x, y in self._nt_cell_hits:
            d2 = (lx - x) ** 2 + (ly - y) ** 2
            if best is None or d2 < best[0]:
                best = (d2, (c, j, i))
        if best and best[0] < 26 ** 2:
            self._nt_cell = best[1]
            self._nt_anim_t = 0.0
            self._nt_draw_cell_trace()
            if dpg.does_item_exist("nt_viz_tabs"):
                dpg.set_value("nt_viz_tabs", "nt_tab_cell")

    def _nt_tick_cell_anim(self):
        """Advance + redraw the cell-inspector animation while its tab shows."""
        if (self.trainer is None or not dpg.does_item_exist("nt_cell_draw")
                or not dpg.is_item_visible("nt_cell_draw")):
            return
        size = dpg.get_item_rect_size("nt_cell_holder")
        if size and size[0] > 60 and (
                int(size[0]), int(size[1])) != getattr(self, "_nt_cell_sz", None):
            self._nt_cell_sz = (int(size[0]), int(size[1]))
            dpg.configure_item("nt_cell_draw", width=max(360, int(size[0]) - 8),
                               height=max(220, int(size[1]) - 8))
        if dpg.get_value("nt_cell_play"):
            self._nt_anim_t += 0.02 * dpg.get_value("nt_cell_speed")
        try:
            self._nt_draw_cell()
        except Exception as e:      # never let a render-loop tick crash the app
            self.log(f"[nt] cell inspector draw skipped: {e}")

    CELL_CYCLE = 5.0                 # seconds: READ (0..0.62) then WRITE (..1)

    def _nt_draw_cell(self):
        layer = "nt_cell_layer"
        if not dpg.does_item_exist(layer) or self.trainer is None:
            return
        tr = self.trainer
        # default to the strongest synapse of neuron 0 if nothing picked
        if (self._nt_cell is None
                or self._nt_cell[0] >= tr.n_layers):
            W0 = tr.weights_norm(0)
            self._nt_cell = (0, 0, int(np.argmax(W0[0])))
        c, j, i = self._nt_cell
        Wn = tr.weights_norm(c)
        # a rebuild to a smaller crossbar can leave j/i past the new dims
        if j >= len(Wn) or int(i) >= len(Wn[0]):
            self._nt_cell = (c, 0, int(np.argmax(Wn[0])))
            c, j, i = self._nt_cell
        g = float(Wn[j][int(i)])
        g_uS = float(tr.weights_uS(c)[j][int(i)])
        kind = (self._nt_dev or {}).get("kind", "current")
        unit = "pA gate current" if kind == "current" else "V gate voltage"
        pre_name = "input pixel" if c == 0 else f"layer-{c} neuron"
        vec = self._last_present[1] if self._last_present else None
        inten = (float(vec[int(i)]) if (vec is not None and c == 0
                 and int(i) < len(vec)) else 0.7)
        pot, strg = tr.last_write.get((c, j, i), (g >= 0.5, 0.0))

        dpg.delete_item(layer, children_only=True)
        sz = getattr(self, "_nt_cell_sz", None)
        W, H = (max(360, sz[0] - 8), max(220, sz[1] - 8)) if sz else (900, 360)
        L = layer

        # ---- PLAYBACK: replay this synapse's REAL per-epoch history (the gate
        # pulses that arrived + the channel conductance changing, start -> now)-
        h = self._nt_whist
        traj = []
        for ep, Wt in zip(h["epoch"], h["W"]):
            if c < len(Wt) and j < Wt[c].shape[0] and int(i) < Wt[c].shape[1]:
                traj.append((int(ep), float(Wt[c][j][int(i)])))
        playback = len(traj) >= 2
        Gmin_uS, Gmax_uS = tr.g_min * 1e6, tr.g_max * 1e6
        spanS = (Gmax_uS - Gmin_uS) or 1.0
        _norm = lambda u: min(max((u - Gmin_uS) / spanS, 0.0), 1.0)
        cur_ep = None
        if playback:
            E = len(traj)
            BEAT = 0.95                          # seconds of animation per epoch
            pos = (self._nt_anim_t % (E * BEAT)) / (E * BEAT)
            fe = pos * E
            e = min(int(fe), E - 1)
            beat_p = fe - e
            cur_ep, G_cur = traj[e]
            G_nxt = traj[e + 1][1] if e + 1 < E else G_cur
            split = 0.55
            reading = beat_p < split
            wfrac = 0.0 if reading else min((beat_p - split) / (1 - split), 1.0)
            g_uS = G_cur + (G_nxt - G_cur) * wfrac        # eased displayed G
            g = _norm(g_uS)
            pot = (G_nxt - G_cur) >= 0
            p = beat_p
        else:
            split = 0.62
            p = (self._nt_anim_t % self.CELL_CYCLE) / self.CELL_CYCLE
            reading = p < split
        read_prog = (p / split) if reading else 0.0
        write_prog = 0.0 if reading else (p - split) / (1 - split)

        # ---- fixed geometry (3-terminal FET: GATE top, SOURCE/DRAIN sides);
        # the top strip is reserved for the gate pulse-train timeline ----------
        ymid = H * 0.54
        ix = W * 0.085                     # input (pre) neuron
        cx = W * 0.45                       # device channel centre
        cw, ch = 92.0, 66.0
        sx = cx - cw / 2 - 22               # SOURCE pin (wordline side)
        dx = cx + cw / 2 + 22               # DRAIN pin (bitline side)
        gate_top = max(ymid - ch / 2 - 64, 86)   # GATE pin (write electrode)
        ox = W * 0.85                       # output neuron
        oy = H * 0.86

        # real numbers: signed gate pulse amplitude, channel R, read current
        amp_si = tr.S.pot_amp if pot else tr.S.dep_amp
        direction = tr.pot_sign if pot else -tr.pot_sign
        gsign = "+" if direction > 0 else "-"
        if kind == "current":
            amp_str = f"{gsign}{abs(amp_si) / 1e-12:.0f} pA"
            gate_unit = "gate current"
        else:
            amp_str = f"{gsign}{abs(amp_si):.2g} V"
            gate_unit = "gate voltage"
        R_ohm = 1.0e6 / max(g_uS, 1e-6)     # uS -> ohm
        R_str = f"{R_ohm/1000:.1f} kohm" if R_ohm >= 1000 else f"{R_ohm:.0f} ohm"
        Vread = 0.1
        I_read = g_uS * Vread               # G(uS)*Vread -> uA

        # phase banner (bottom - the top is the pulse-train timeline)
        dpg.draw_text((W * 0.04, H - 18),
                      ("READ  -  read voltage on SOURCE -> current through the "
                       "channel (set by G) -> DRAIN -> bitline -> neuron"
                       if reading else
                       "WRITE  -  a GATE pulse programs the channel conductance "
                       "(the stored weight)"),
                      size=14, color=((120, 200, 255) if reading
                                      else (255, 190, 110)), parent=L)

        # ---- gate pulse-train timeline (playback): one tick per epoch, up =
        # potentiate / down = depress, height ~ |dG|; a playhead scrubs it -----
        if playback:
            sy = 52.0
            sx0, sx1 = W * 0.07, W * 0.93
            deltas = [traj[k + 1][1] - traj[k][1] for k in range(E - 1)]
            dmax = max((abs(d) for d in deltas), default=1.0) or 1.0
            dpg.draw_text((sx0, 8),
                          "gate pulse history (green up = potentiate, red down "
                          "= depress)", size=12, color=(170, 180, 200), parent=L)
            dpg.draw_text((sx1 - 92, 8), f"epoch {cur_ep} / {traj[-1][0]}",
                          size=13, color=(235, 205, 120), parent=L)
            dpg.draw_line((sx0, sy), (sx1, sy), color=(70, 80, 100, 160),
                          thickness=1.0, parent=L)
            for k, d in enumerate(deltas):
                xk = sx0 + (k + 0.5) * (sx1 - sx0) / max(E - 1, 1)
                hh = 2.0 + 15.0 * min(abs(d) / dmax, 1.0)
                up = d >= 0
                col = (120, 235, 150, 235) if up else (255, 130, 130, 235)
                y2 = sy - hh if up else sy + hh
                dpg.draw_line((xk, sy), (xk, y2), color=col, thickness=2.2,
                              parent=L)
                dpg.draw_circle((xk, y2), 2.0, fill=col, parent=L)
            phx = sx0 + min(e + beat_p, E - 1) * (sx1 - sx0) / max(E - 1, 1)
            dpg.draw_line((phx, sy - 24), (phx, sy + 24),
                          color=(255, 220, 120, 235), thickness=1.6, parent=L)

        # ---- GATE pin (write electrode) - always shows its signed amplitude ----
        gate_on = (not reading)
        gcol = ((120, 235, 150) if pot else (255, 130, 130)) if gate_on \
            else (110, 120, 150)
        dpg.draw_line((cx, gate_top), (cx, ymid - ch / 2), color=gcol,
                      thickness=3.0, parent=L)
        dpg.draw_circle((cx, gate_top), 9, fill=(40, 46, 60, 255), color=gcol,
                        thickness=2.0, parent=L)
        dpg.draw_text((cx + 16, gate_top - 9), "G  (gate)", size=14,
                      color=(180, 190, 210), parent=L)
        dpg.draw_text((cx + 16, gate_top + 11), f"pulse = {amp_str}",
                      size=14, color=gcol if gate_on else (160, 170, 190),
                      parent=L)
        dpg.draw_text((cx + 16, gate_top + 30),
                      f"({gate_unit}; {'POTENTIATE - G up' if pot else 'DEPRESS - G down'})",
                      size=12, color=(150, 160, 180), parent=L)
        if gate_on:                        # the write pulse travelling down
            yp = gate_top + min(write_prog / 0.5, 1.0) * (ymid - ch / 2 - gate_top)
            dpg.draw_circle((cx, yp), 7, fill=gcol, parent=L)

        # ---- channel (the weight): g is already eased through the real
        # trajectory in playback; in non-playback it nudges to show the change --
        gd = g
        if gate_on and not playback:
            gd = min(max(g + (0.12 if pot else -0.12)
                         * max(write_prog - 0.5, 0) * 2, 0), 1)
        dpg.draw_rectangle((cx - cw / 2, ymid - ch / 2), (cx + cw / 2, ymid + ch / 2),
                           fill=self._cmap_color(gd, 255),
                           color=(230, 235, 245, 210), rounding=8, thickness=2.0,
                           parent=L)
        dpg.draw_text((cx - cw / 2 - 6, ymid + ch / 2 + 10),
                      f"channel:  G = {g_uS:.0f} uS    R = {R_str}    "
                      f"I_read = {I_read:.1f} uA  (Vread {Vread:g} V)",
                      size=14, color=(210, 218, 232), parent=L)

        # ---- SOURCE pin (left) + DRAIN pin (right) ----
        for px, lbl, full in ((sx, "S", "source"), (dx, "D", "drain")):
            edge = cx - cw / 2 if px < cx else cx + cw / 2
            dpg.draw_line((edge, ymid), (px, ymid), color=(215, 220, 235, 230),
                          thickness=2.6, parent=L)
            dpg.draw_circle((px, ymid), 5, fill=(215, 220, 235, 255), parent=L)
            dpg.draw_text((px - 4, ymid - 28), lbl, size=15,
                          color=(235, 205, 120), parent=L)
            dpg.draw_text((px - 16, ymid - 46), full, size=11,
                          color=(150, 160, 180), parent=L)

        # ---- input (pre) neuron + wordline to the SOURCE pin ----
        dpg.draw_circle((ix, ymid), 16, fill=(50, 60, 84, 255),
                        color=(120, 160, 255, 255), thickness=2.0, parent=L)
        dpg.draw_text((ix - 28, ymid + 22), f"{pre_name} {i}", size=13,
                      color=(170, 180, 200), parent=L)
        dpg.draw_line((ix + 16, ymid), (sx, ymid),
                      color=(150, 185, 255, 210) if reading else (70, 80, 100, 150),
                      thickness=2.4, parent=L)
        dpg.draw_text(((ix + sx) / 2 - 36, ymid - 24), "wordline (read V)",
                      size=12, color=(130, 150, 190), parent=L)

        # ---- bitline: from the DRAIN pin down to the output neuron ----
        dpg.draw_line((dx, ymid), (dx, oy), color=(82, 94, 122, 190),
                      thickness=1.9, parent=L)
        dpg.draw_line((dx, oy), (ox - 18, oy), color=(82, 94, 122, 190),
                      thickness=1.9, parent=L)
        dpg.draw_text((dx + 8, (ymid + oy) / 2 - 12),
                      f"bitline\nI_read = {I_read:.1f} uA", size=12,
                      color=(130, 150, 190), parent=L)

        # ---- READ animation: dots input -> SOURCE -> channel -> DRAIN ->
        # bitline -> neuron (the read current path) ----
        if reading:
            rp = read_prog
            pts = [(ix + 16, ymid), (sx, ymid), (dx, ymid), (dx, oy),
                   (ox - 18, oy)]
            seglen = [((pts[k + 1][0] - pts[k][0]) ** 2
                       + (pts[k + 1][1] - pts[k][1]) ** 2) ** 0.5
                      for k in range(len(pts) - 1)]
            total = sum(seglen) or 1.0
            ndots = max(2, int(2 + inten * 5))
            for d in range(ndots):
                frac = (rp + d / ndots) % 1.0
                dist = frac * total
                k = 0
                while k < len(seglen) and dist > seglen[k]:
                    dist -= seglen[k]; k += 1
                k = min(k, len(seglen) - 1)
                t = dist / (seglen[k] or 1.0)
                px = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t
                py = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t
                # past the DRAIN the read current is scaled by the conductance G
                after = (px > dx - 1)
                a = int(110 + 140 * (g if after else 1.0))
                dpg.draw_circle((px, py), 4.5,
                                fill=(120, 200, 255, a), parent=L)

        # ---- output neuron with a membrane fill bar ----
        counts = (self._last_present[2]["n_out_spikes"]
                  if self._last_present else None)
        # neuron j here is the post of crossbar c; membrane illustrative
        mem = (0.2 + 0.8 * read_prog * g) if reading else 0.15
        fired = reading and mem > 0.85
        nfill = self._cmap_color(0.15 + 0.85 * (1.0 if fired else mem), 255)
        ring = (255, 178, 90, 255) if fired else (90, 110, 140, 230)
        dpg.draw_circle((ox, oy), 22, fill=(36, 42, 56, 255), color=ring,
                        thickness=2.5, parent=L)
        dpg.draw_circle((ox, oy), 22 * (0.35 + 0.6 * mem), fill=nfill, parent=L)
        post = "output neuron" if c == tr.n_layers - 1 else f"hidden neuron"
        lbl = f"{post} {j}"
        if counts and c == tr.n_layers - 1 and j < len(counts):
            lbl += f"  ({counts[j]} sp)"
        dpg.draw_text((ox - 34, oy + 28), lbl, size=13, color=(200, 208, 224),
                      parent=L)
        if fired:
            dpg.draw_text((ox + 26, oy - 8), "spike!", size=14,
                          color=(255, 200, 120, 255), parent=L)

        if dpg.does_item_exist("nt_cell_title"):
            scrub = (f"replaying epoch {cur_ep}/{traj[-1][0]}   |   "
                     if playback else "")
            dpg.set_value("nt_cell_title",
                          f"crossbar {c} · {pre_name} {i} -> neuron {j}   |   "
                          + scrub +
                          f"G = {g_uS:.0f} uS  /  R = {R_str}  "
                          f"({g*100:.0f}% of range)   |   gate pulse "
                          f"{amp_str}  ({'potentiate' if pot else 'depress'})")

    def _nt_draw_cell_trace(self):
        """Transient plot of the selected synapse's conductance (its weight)
        across training epochs - the start-to-end write trajectory."""
        if (not dpg.does_item_exist("nt_cell_py") or self.trainer is None
                or self._nt_cell is None):
            return
        c, j, i = self._nt_cell
        h = self._nt_whist
        ys = []
        for W in h["W"]:
            if (c < len(W) and j < W[c].shape[0] and i < W[c].shape[1]):
                ys.append(float(W[c][j][i]))
        self._nt_clear("celltr")
        if len(ys) >= 1:
            xs = h["epoch"][:len(ys)]
            s = dpg.add_line_series(xs, ys, label=f"input {i} -> N{j}",
                                    parent="nt_cell_py")
            dpg.bind_item_theme(s, self.themes["markers"])
            self._nt_series["celltr"] = [s]
            dpg.fit_axis_data("nt_cell_px")
            dpg.fit_axis_data("nt_cell_py")

    def _nt_trace_theme(self, c, j, n_post):
        """A thin, semi-transparent line theme coloured by the destination
        neuron j (cached per (crossbar, neuron))."""
        key = (c, j)
        th = self._nt_trace_themes.get(key)
        if th is None or not dpg.does_item_exist(th):
            col = self._cmap_color(j / max(n_post - 1, 1), 70)
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.3,
                                        category=dpg.mvThemeCat_Plots)
            self._nt_trace_themes[key] = th
        return th

    def _nt_draw_weight_traces(self, *_):
        """Plot EVERY synapse's conductance over training in one figure - the
        whole crossbar's learning trajectory, coloured by destination neuron."""
        if not dpg.does_item_exist("nt_wtrace_y") or self.trainer is None:
            return
        self._nt_clear("wtrace")
        h = self._nt_whist
        if not h["epoch"]:
            if dpg.does_item_exist("nt_wtrace_info"):
                dpg.set_value("nt_wtrace_info", "train to populate")
            return
        xs = list(h["epoch"])
        E = len(xs)
        CAP = 700                                     # max lines actually drawn
        cols = []                                     # (c, j, n_post, ys)
        for c in range(self.trainer.n_layers):
            try:
                stack = np.stack([h["W"][e][c] for e in range(E)])   # (E,post,pre)
            except Exception:
                continue
            n_post, n_pre = stack.shape[1], stack.shape[2]
            for j in range(n_post):
                for i in range(n_pre):
                    cols.append((c, j, n_post, stack[:, j, i]))
        n_total = len(cols)
        if n_total > CAP:                             # subsample evenly
            idx = np.linspace(0, n_total - 1, CAP).astype(int)
            cols = [cols[k] for k in idx]
        series = []
        for c, j, n_post, ys in cols:
            s = dpg.add_line_series(xs, ys.tolist(), label="",
                                    parent="nt_wtrace_y")
            dpg.bind_item_theme(s, self._nt_trace_theme(c, j, n_post))
            series.append(s)
        self._nt_series["wtrace"] = series
        dpg.fit_axis_data("nt_wtrace_x")
        dpg.fit_axis_data("nt_wtrace_y")
        nx = self.trainer.n_layers
        info = (f"{n_total} synapses"
                + (f"  (showing {len(cols)} sampled)" if len(cols) < n_total
                   else "")
                + f"  ·  {E} epochs"
                + (f"  ·  {nx} crossbars" if nx > 1 else "")
                + "  ·  colour = destination neuron")
        if dpg.does_item_exist("nt_wtrace_info"):
            dpg.set_value("nt_wtrace_info", info)

    def _nt_draw_wevo(self):
        if not dpg.does_item_exist("nt_wevo_y"):
            return
        self._nt_clear("wevo")
        xs = self._wevo_epochs
        series = []
        for j in sorted(self._wevo):
            ys = self._wevo[j]
            if not ys:
                continue
            series.append(dpg.add_line_series(
                xs[:len(ys)], ys, label=f"neuron {j}", parent="nt_wevo_y"))
        self._nt_series["wevo"] = series
        dpg.fit_axis_data("nt_wevo_x")
        dpg.fit_axis_data("nt_wevo_y")

    def _nt_draw_activity(self, label, vec, res):
        n_in = self.trainer.n_in if self.trainer else 0
        if dpg.does_item_exist("nt_act_title"):
            dpg.set_value("nt_act_title",
                          f"pattern '{label}'  ->  winner neuron "
                          f"{res['winner']}   (output spikes: "
                          f"{res['n_out_spikes']})")
        # raster
        if dpg.does_item_exist("nt_rast_y"):
            self._nt_clear("rast")
            ser = []
            in_sp = res["in_spikes"]
            if in_sp:
                xs = [t for t, _ in in_sp]
                ys = [i for _, i in in_sp]
                ser.append(dpg.add_scatter_series(xs, ys, label="inputs",
                                                  parent="nt_rast_y"))
            xs2, ys2 = [], []
            for j, sp in enumerate(res["out_spikes"]):
                xs2 += list(sp)
                ys2 += [n_in + j] * len(sp)
            if xs2:
                ser.append(dpg.add_scatter_series(
                    xs2, ys2, label="neuron spikes", parent="nt_rast_y"))
            self._nt_series["rast"] = ser
            dpg.fit_axis_data("nt_rast_x")
            dpg.fit_axis_data("nt_rast_y")
        # membrane V(t) of the output neurons: a prominent threshold line, the
        # WINNER neuron drawn bold (others muted), and a spike marker each time a
        # neuron crosses threshold - so you SEE when + how a neuron fires.
        # (v_trace already carries the spike peak on the firing step.)
        if dpg.does_item_exist("nt_mem_y"):
            self._nt_clear("mem")
            v = res["v_trace"]
            dt = res["dt_ms"]
            tarr = [k * dt for k in range(res["n_steps"])]
            vth = self.trainer.N.v_threshold if self.trainer else 1.0
            winner = res.get("winner", -1)
            # threshold line (amber, behind the traces)
            s_th = dpg.add_line_series(
                [tarr[0], tarr[-1]] if tarr else [0, 1], [vth, vth],
                label=f"threshold ({vth:.2g})", parent="nt_mem_y")
            dpg.bind_item_theme(s_th, self.themes["ln_amber"])
            ser = [s_th]
            for j in range(v.shape[1]):
                lbl = f"N{j}" + ("  (winner)" if j == winner else "")
                s = dpg.add_line_series(tarr, v[:, j].tolist(), label=lbl,
                                        parent="nt_mem_y")
                dpg.bind_item_theme(s, self.themes[
                    "ln_blue" if j == winner else "ln_muted"])
                ser.append(s)
            # spike markers: a dot at every output-neuron firing time
            peak = float(getattr(self.trainer.N, "v_spike_peak", 1.6)
                         if self.trainer else 1.6)
            sx = [t for sp in res["out_spikes"] for t in sp]
            if sx:
                sm = dpg.add_scatter_series(sx, [peak] * len(sx), label="spikes",
                                            parent="nt_mem_y")
                dpg.bind_item_theme(sm, self.themes["dot_noise"])
                ser.append(sm)
            self._nt_series["mem"] = ser
            dpg.fit_axis_data("nt_mem_x")
            dpg.fit_axis_data("nt_mem_y")
